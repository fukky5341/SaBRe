## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00506088


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0038478, -0.0003729, -0.0038478, -0.0003729, -0.0034749, 0.0034749)
1: (-0.0043582, 0.0039877, -0.0043582, 0.0039877, -0.0063742, 0.0063742)
2: (0.0030878, 0.0098350, 0.0030878, 0.0098350, -0.0049656, 0.0049656)
3: (-0.0045331, -0.0035076, -0.0045331, -0.0035076, -0.0010255, 0.0010255)
4: (0.0018617, 0.0077431, 0.0018617, 0.0077431, -0.0053495, 0.0053495)
5: (-0.0029239, 0.0030384, -0.0029239, 0.0030384, -0.0052306, 0.0052306)
6: (-0.0067578, -0.0034269, -0.0067578, -0.0034269, -0.0032173, 0.0032173)
7: (-0.0020702, 0.0036866, -0.0020702, 0.0036866, -0.0055112, 0.0055112)
8: (-0.0008525, 0.0000612, -0.0008525, 0.0000612, -0.0009138, 0.0009138)
9: (0.9976376, 1.0118670, 0.9976376, 1.0118670, -0.0109396, 0.0109396)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.56 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0062972, upper bound: 0.0062972

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057922, upper bound: 0.0059470
time: 0.77 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059822, upper bound: 0.0059822
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 9, lower bound: -0.0057922, upper bound: 0.0059470
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 9, lower bound: -0.0059822, upper bound: 0.0059822

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0034275, -0.0003845, -0.0036998, -0.0003817, -0.0030458, 0.0033153
1: -0.0044366, 0.0029725, -0.0043522, 0.0036235, -0.0055892, 0.0051354
2: 0.0030036, 0.0091853, 0.0030927, 0.0096028, -0.0045336, 0.0041590
3: -0.0044560, -0.0035174, -0.0045065, -0.0035125, -0.0009435, 0.0009891
4: 0.0021536, 0.0077341, 0.0019629, 0.0077090, -0.0050217, 0.0052135
5: -0.0029282, 0.0026508, -0.0028933, 0.0029024, -0.0050497, 0.0048090
6: -0.0067184, -0.0035254, -0.0067320, -0.0034619, -0.0031403, 0.0030940
7: -0.0017778, 0.0037158, -0.0019646, 0.0036630, -0.0051332, 0.0053572
8: -0.0008373, 0.0000053, -0.0008462, 0.0000424, -0.0008797, 0.0008515
9: 0.9989018, 1.0119768, 0.9980832, 1.0118326, -0.0095190, 0.0103112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0057665
time: 0.69 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055961, upper bound: 0.0057665
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0037348, -0.0003812, -0.0038294, -0.0003742, -0.0032328, 0.0034482
1: -0.0043525, 0.0037081, -0.0043573, 0.0039409, -0.0063391, 0.0050243
2: 0.0030923, 0.0096306, 0.0030885, 0.0098024, -0.0049325, 0.0042264
3: -0.0045091, -0.0035121, -0.0045294, -0.0035083, -0.0010009, 0.0010172
4: 0.0019767, 0.0077126, 0.0018799, 0.0077384, -0.0051984, 0.0052961
5: -0.0028967, 0.0029046, -0.0029197, 0.0030177, -0.0051787, 0.0049913
6: -0.0067351, -0.0034721, -0.0067543, -0.0034339, -0.0031859, 0.0031732
7: -0.0019528, 0.0036652, -0.0020522, 0.0036833, -0.0053307, 0.0054575
8: -0.0008469, 0.0000438, -0.0008517, 0.0000586, -0.0009055, 0.0008954
9: 0.9980375, 1.0118358, 0.9976996, 1.0118623, -0.0098323, 0.0108495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059470, upper bound: 0.0057919
time: 0.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0059470, upper bound: 0.0059822
time: 0.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.76 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0057665
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 9, lower bound: -0.0055961, upper bound: 0.0057665
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 9, lower bound: -0.0059470, upper bound: 0.0057919
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 9, lower bound: -0.0059470, upper bound: 0.0059822

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0034245, -0.0004340, -0.0036389, -0.0005548, -0.0028696, 0.0032049
1: -0.0043011, 0.0029681, -0.0038837, 0.0034216, -0.0050844, 0.0046797
2: 0.0031269, 0.0091782, 0.0035046, 0.0094213, -0.0041218, 0.0037681
3: -0.0044550, -0.0035328, -0.0044951, -0.0035662, -0.0008888, 0.0009623
4: 0.0021609, 0.0076351, 0.0021123, 0.0073818, -0.0046998, 0.0049490
5: -0.0028139, 0.0026448, -0.0025210, 0.0027210, -0.0047064, 0.0044271
6: -0.0066686, -0.0035297, -0.0065680, -0.0035330, -0.0030247, 0.0029349
7: -0.0017669, 0.0036401, -0.0018607, 0.0034063, -0.0048895, 0.0051750
8: -0.0008182, 0.0000046, -0.0007827, 0.0000223, -0.0008405, 0.0007873
9: 0.9989170, 1.0117009, 0.9985397, 1.0109057, -0.0086496, 0.0094179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0053905
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0057665
time: 0.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0034253, -0.0004075, -0.0036902, -0.0004676, -0.0029577, 0.0032827
1: -0.0043805, 0.0029690, -0.0041484, 0.0036080, -0.0054849, 0.0045136
2: 0.0030508, 0.0091807, 0.0032647, 0.0095855, -0.0044782, 0.0036291
3: -0.0044553, -0.0035252, -0.0045034, -0.0035423, -0.0009129, 0.0009782
4: 0.0021580, 0.0076889, 0.0019802, 0.0075508, -0.0047366, 0.0051533
5: -0.0028794, 0.0026468, -0.0027314, 0.0028868, -0.0049867, 0.0044172
6: -0.0066952, -0.0035279, -0.0066493, -0.0034720, -0.0031068, 0.0029764
7: -0.0017716, 0.0036769, -0.0019405, 0.0035268, -0.0049282, 0.0052941
8: -0.0008280, 0.0000047, -0.0008126, 0.0000402, -0.0008681, 0.0007908
9: 0.9989119, 1.0118668, 0.9981219, 1.0114567, -0.0084431, 0.0101779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055961, upper bound: 0.0053905
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055961, upper bound: 0.0057665
time: 0.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0037348, -0.0003812, -0.0034275, -0.0003845, -0.0033503, 0.0030463
1: -0.0043525, 0.0037081, -0.0044366, 0.0029725, -0.0051331, 0.0058765
2: 0.0030923, 0.0096306, 0.0030036, 0.0091853, -0.0041601, 0.0047081
3: -0.0045091, -0.0035121, -0.0044560, -0.0035174, -0.0009918, 0.0009439
4: 0.0019767, 0.0077126, 0.0021536, 0.0077341, -0.0051979, 0.0050261
5: -0.0028967, 0.0029046, -0.0029282, 0.0026508, -0.0048129, 0.0050580
6: -0.0067351, -0.0034721, -0.0067184, -0.0035254, -0.0030971, 0.0031265
7: -0.0019528, 0.0036652, -0.0017778, 0.0037158, -0.0053499, 0.0051369
8: -0.0008469, 0.0000438, -0.0008373, 0.0000053, -0.0008522, 0.0008811
9: 0.9980375, 1.0118358, 0.9989018, 1.0119768, -0.0105057, 0.0095229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057665, upper bound: 0.0052388
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057665, upper bound: 0.0055961
time: 0.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0037348, -0.0003812, -0.0037348, -0.0003812, -0.0032242, 0.0032242
1: -0.0043525, 0.0037081, -0.0043525, 0.0037081, -0.0050182, 0.0050182
2: 0.0030923, 0.0096306, 0.0030923, 0.0096306, -0.0042131, 0.0042131
3: -0.0045091, -0.0035121, -0.0045091, -0.0035121, -0.0009970, 0.0009970
4: 0.0019767, 0.0077126, 0.0019767, 0.0077126, -0.0051664, 0.0051664
5: -0.0028967, 0.0029046, -0.0028967, 0.0029046, -0.0049611, 0.0049611
6: -0.0067351, -0.0034721, -0.0067351, -0.0034721, -0.0031522, 0.0031522
7: -0.0019528, 0.0036652, -0.0019528, 0.0036652, -0.0052999, 0.0052999
8: -0.0008469, 0.0000438, -0.0008469, 0.0000438, -0.0008907, 0.0008907
9: 0.9980375, 1.0118358, 0.9980375, 1.0118358, -0.0097911, 0.0097911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057665, upper bound: 0.0052784
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0057665, upper bound: 0.0056404
time: 1.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.13 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0053905
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0057665
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 9, lower bound: -0.0055961, upper bound: 0.0053905
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 9, lower bound: -0.0055961, upper bound: 0.0057665
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 9, lower bound: -0.0057665, upper bound: 0.0052388
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 9, lower bound: -0.0057665, upper bound: 0.0055961
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 9, lower bound: -0.0057665, upper bound: 0.0052784
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.13
Output dim: 9, lower bound: -0.0057665, upper bound: 0.0056404

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033736, -0.0005578, -0.0036389, -0.0005548, -0.0028188, 0.0030812
1: -0.0039735, 0.0027799, -0.0038837, 0.0034216, -0.0047490, 0.0043148
2: 0.0034156, 0.0090027, 0.0035046, 0.0094213, -0.0038373, 0.0034769
3: -0.0044454, -0.0035706, -0.0044951, -0.0035662, -0.0008791, 0.0009245
4: 0.0022974, 0.0073970, 0.0021123, 0.0073818, -0.0045528, 0.0047202
5: -0.0025454, 0.0024783, -0.0025210, 0.0027210, -0.0044366, 0.0042182
6: -0.0065483, -0.0035941, -0.0065680, -0.0035330, -0.0029115, 0.0028736
7: -0.0016789, 0.0034567, -0.0018607, 0.0034063, -0.0047856, 0.0049974
8: -0.0007721, -0.0000149, -0.0007827, 0.0000223, -0.0007944, 0.0007678
9: 0.9993390, 1.0110426, 0.9985397, 1.0109057, -0.0080525, 0.0087888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0053368
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0054864
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0034195, -0.0004704, -0.0036389, -0.0005548, -0.0028647, 0.0031685
1: -0.0042291, 0.0029594, -0.0038837, 0.0034216, -0.0051158, 0.0046141
2: 0.0031793, 0.0091678, 0.0035046, 0.0094213, -0.0041347, 0.0037578
3: -0.0044532, -0.0035465, -0.0044951, -0.0035662, -0.0008870, 0.0009486
4: 0.0021704, 0.0075688, 0.0021123, 0.0073818, -0.0046914, 0.0048885
5: -0.0027532, 0.0026356, -0.0025210, 0.0027210, -0.0046637, 0.0044181
6: -0.0066328, -0.0035351, -0.0065680, -0.0035330, -0.0029865, 0.0029298
7: -0.0017544, 0.0035739, -0.0018607, 0.0034063, -0.0048780, 0.0051075
8: -0.0008035, 0.0000030, -0.0007827, 0.0000223, -0.0008258, 0.0007857
9: 0.9989398, 1.0115755, 0.9985397, 1.0109057, -0.0086294, 0.0093901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0055961
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0057665
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0033736, -0.0005578, -0.0036902, -0.0004676, -0.0029060, 0.0031324
1: -0.0039735, 0.0027799, -0.0041484, 0.0036080, -0.0050274, 0.0045744
2: 0.0034156, 0.0090027, 0.0032647, 0.0095855, -0.0040964, 0.0037008
3: -0.0044454, -0.0035706, -0.0045034, -0.0035423, -0.0009030, 0.0009328
4: 0.0022974, 0.0073970, 0.0019802, 0.0075508, -0.0047087, 0.0048683
5: -0.0025454, 0.0024783, -0.0027314, 0.0028868, -0.0046468, 0.0044227
6: -0.0065483, -0.0035941, -0.0066493, -0.0034720, -0.0029698, 0.0029435
7: -0.0016789, 0.0034567, -0.0019405, 0.0035268, -0.0048724, 0.0050828
8: -0.0007721, -0.0000149, -0.0008126, 0.0000402, -0.0008123, 0.0007977
9: 0.9993390, 1.0110426, 0.9981219, 1.0114567, -0.0085530, 0.0093578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0052388
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0053905
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0034195, -0.0004704, -0.0036902, -0.0004676, -0.0029518, 0.0032093
1: -0.0042291, 0.0029594, -0.0041484, 0.0036080, -0.0049055, 0.0044779
2: 0.0031793, 0.0091678, 0.0032647, 0.0095855, -0.0039848, 0.0036175
3: -0.0044532, -0.0035465, -0.0045034, -0.0035423, -0.0009109, 0.0009569
4: 0.0021704, 0.0075688, 0.0019802, 0.0075508, -0.0047257, 0.0049067
5: -0.0027532, 0.0026356, -0.0027314, 0.0028868, -0.0046384, 0.0044066
6: -0.0066328, -0.0035351, -0.0066493, -0.0034720, -0.0030134, 0.0029695
7: -0.0017544, 0.0035739, -0.0019405, 0.0035268, -0.0049119, 0.0051338
8: -0.0008035, 0.0000030, -0.0008126, 0.0000402, -0.0008107, 0.0007850
9: 0.9989398, 1.0115755, 0.9981219, 1.0114567, -0.0084210, 0.0091783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0055961
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0057665
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036775, -0.0005540, -0.0034245, -0.0004340, -0.0032435, 0.0028704
1: -0.0038843, 0.0035062, -0.0043011, 0.0029681, -0.0046769, 0.0054633
2: 0.0035039, 0.0094506, 0.0031269, 0.0091782, -0.0037703, 0.0043650
3: -0.0044993, -0.0035658, -0.0044550, -0.0035328, -0.0009665, 0.0008893
4: 0.0021228, 0.0073880, 0.0021609, 0.0076351, -0.0049511, 0.0047070
5: -0.0025269, 0.0027194, -0.0028139, 0.0026448, -0.0044341, 0.0047378
6: -0.0065731, -0.0035443, -0.0066686, -0.0035297, -0.0029402, 0.0030124
7: -0.0018377, 0.0034105, -0.0017669, 0.0036401, -0.0051860, 0.0048961
8: -0.0007838, 0.0000238, -0.0008182, 0.0000046, -0.0007885, 0.0008419
9: 0.9985001, 1.0109119, 0.9989170, 1.0117009, -0.0097210, 0.0086571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053905, upper bound: 0.0052399
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053905, upper bound: 0.0052399
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0037257, -0.0004672, -0.0034253, -0.0004075, -0.0033182, 0.0029582
1: -0.0041487, 0.0036930, -0.0043805, 0.0029690, -0.0045107, 0.0057763
2: 0.0032644, 0.0096131, 0.0030508, 0.0091807, -0.0036303, 0.0046511
3: -0.0045060, -0.0035420, -0.0044553, -0.0035252, -0.0009808, 0.0009132
4: 0.0019945, 0.0075545, 0.0021580, 0.0076889, -0.0051372, 0.0047411
5: -0.0027347, 0.0028886, -0.0028794, 0.0026468, -0.0044212, 0.0049939
6: -0.0066523, -0.0034823, -0.0066952, -0.0035279, -0.0029796, 0.0030927
7: -0.0019273, 0.0035289, -0.0017716, 0.0036769, -0.0052859, 0.0049321
8: -0.0008133, 0.0000416, -0.0008280, 0.0000047, -0.0007908, 0.0008695
9: 0.9980774, 1.0114597, 0.9989119, 1.0118668, -0.0103702, 0.0084472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053905, upper bound: 0.0055961
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053905, upper bound: 0.0055961
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036775, -0.0005540, -0.0037314, -0.0004308, -0.0030610, 0.0030312
1: -0.0038843, 0.0035062, -0.0042157, 0.0037031, -0.0045110, 0.0044890
2: 0.0035039, 0.0094506, 0.0032155, 0.0096240, -0.0037957, 0.0037720
3: -0.0044993, -0.0035658, -0.0045080, -0.0035276, -0.0009717, 0.0009423
4: 0.0021228, 0.0073880, 0.0019838, 0.0076153, -0.0049101, 0.0048455
5: -0.0025269, 0.0027194, -0.0027836, 0.0028985, -0.0045835, 0.0046243
6: -0.0065731, -0.0035443, -0.0066867, -0.0034763, -0.0029985, 0.0030405
7: -0.0018377, 0.0034105, -0.0019419, 0.0035899, -0.0051125, 0.0050533
8: -0.0007838, 0.0000238, -0.0008282, 0.0000431, -0.0008269, 0.0008360
9: 0.9985001, 1.0109119, 0.9980525, 1.0115551, -0.0088767, 0.0088844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054199, upper bound: 0.0052784
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054199, upper bound: 0.0052784
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0037257, -0.0004672, -0.0037324, -0.0004045, -0.0031135, 0.0030868
1: -0.0041487, 0.0036930, -0.0042961, 0.0037041, -0.0043627, 0.0049123
2: 0.0032644, 0.0096131, 0.0031393, 0.0096260, -0.0036662, 0.0041583
3: -0.0045060, -0.0035420, -0.0045083, -0.0035202, -0.0009858, 0.0009663
4: 0.0019945, 0.0075545, 0.0019814, 0.0076689, -0.0051071, 0.0048825
5: -0.0027347, 0.0028886, -0.0028522, 0.0029004, -0.0045645, 0.0048989
6: -0.0066523, -0.0034823, -0.0067128, -0.0034747, -0.0030366, 0.0031193
7: -0.0019273, 0.0035289, -0.0019460, 0.0036281, -0.0052386, 0.0050977
8: -0.0008133, 0.0000416, -0.0008376, 0.0000432, -0.0008104, 0.0008792
9: 0.9980774, 1.0114597, 0.9980480, 1.0117323, -0.0096632, 0.0086923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054199, upper bound: 0.0056404
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054199, upper bound: 0.0056403
time: 1.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.10 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0053368
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0054864
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0055961
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0057665
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0052388
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0053905
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0055961
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0052399, upper bound: 0.0057665
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0053905, upper bound: 0.0052399
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0053905, upper bound: 0.0052399
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0053905, upper bound: 0.0055961
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0053905, upper bound: 0.0055961
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0054199, upper bound: 0.0052784
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0054199, upper bound: 0.0052784
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0054199, upper bound: 0.0056404
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 9, lower bound: -0.0054199, upper bound: 0.0056403

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033736, -0.0005578, -0.0033736, -0.0005578, -0.0028158, 0.0028158
1: -0.0039735, 0.0027799, -0.0039735, 0.0027799, -0.0039313, 0.0039313
2: 0.0034156, 0.0090027, 0.0034156, 0.0090027, -0.0033173, 0.0033173
3: -0.0044454, -0.0035706, -0.0044454, -0.0035706, -0.0008747, 0.0008747
4: 0.0022974, 0.0073970, 0.0022974, 0.0073970, -0.0045445, 0.0045445
5: -0.0025454, 0.0024783, -0.0025454, 0.0024783, -0.0041994, 0.0041994
6: -0.0065483, -0.0035941, -0.0065483, -0.0035941, -0.0028517, 0.0028517
7: -0.0016789, 0.0034567, -0.0016789, 0.0034567, -0.0047738, 0.0047738
8: -0.0007721, -0.0000149, -0.0007721, -0.0000149, -0.0007540, 0.0007540
9: 0.9993390, 1.0110426, 0.9993390, 1.0110426, -0.0079066, 0.0079066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052239, upper bound: 0.0051332
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052239, upper bound: 0.0052232
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033736, -0.0005578, -0.0036775, -0.0005540, -0.0028195, 0.0031197
1: -0.0039735, 0.0027799, -0.0038843, 0.0035062, -0.0051278, 0.0043120
2: 0.0034156, 0.0090027, 0.0035039, 0.0094506, -0.0040806, 0.0034790
3: -0.0044454, -0.0035706, -0.0044993, -0.0035658, -0.0008796, 0.0009287
4: 0.0022974, 0.0073970, 0.0021228, 0.0073880, -0.0045600, 0.0047223
5: -0.0025454, 0.0024783, -0.0025269, 0.0027194, -0.0044680, 0.0042251
6: -0.0065483, -0.0035941, -0.0065731, -0.0035443, -0.0028993, 0.0028789
7: -0.0016789, 0.0034567, -0.0018377, 0.0034105, -0.0047922, 0.0050084
8: -0.0007721, -0.0000149, -0.0007838, 0.0000238, -0.0007959, 0.0007690
9: 0.9993390, 1.0110426, 0.9985001, 1.0109119, -0.0080600, 0.0090920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052239, upper bound: 0.0053034
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052239, upper bound: 0.0053703
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0034195, -0.0004704, -0.0033736, -0.0005578, -0.0028507, 0.0028882
1: -0.0042291, 0.0029594, -0.0039735, 0.0027799, -0.0042981, 0.0042306
2: 0.0031793, 0.0091678, 0.0034156, 0.0090027, -0.0036146, 0.0035982
3: -0.0044532, -0.0035465, -0.0044454, -0.0035706, -0.0008826, 0.0008989
4: 0.0021704, 0.0075688, 0.0022974, 0.0073970, -0.0046832, 0.0047129
5: -0.0027532, 0.0026356, -0.0025454, 0.0024783, -0.0044265, 0.0043993
6: -0.0066328, -0.0035351, -0.0065483, -0.0035941, -0.0029268, 0.0029080
7: -0.0017544, 0.0035739, -0.0016789, 0.0034567, -0.0048662, 0.0048839
8: -0.0008035, 0.0000030, -0.0007721, -0.0000149, -0.0007886, 0.0007752
9: 0.9989398, 1.0115755, 0.9993390, 1.0110426, -0.0084835, 0.0085079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051285, upper bound: 0.0054127
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051285, upper bound: 0.0054847
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034195, -0.0004704, -0.0036775, -0.0005540, -0.0028654, 0.0032071
1: -0.0042291, 0.0029594, -0.0038843, 0.0035062, -0.0054946, 0.0046113
2: 0.0031793, 0.0091678, 0.0035039, 0.0094506, -0.0043780, 0.0037600
3: -0.0044532, -0.0035465, -0.0044993, -0.0035658, -0.0008875, 0.0009528
4: 0.0021704, 0.0075688, 0.0021228, 0.0073880, -0.0046986, 0.0048906
5: -0.0027532, 0.0026356, -0.0025269, 0.0027194, -0.0046951, 0.0044251
6: -0.0066328, -0.0035351, -0.0065731, -0.0035443, -0.0029743, 0.0029352
7: -0.0017544, 0.0035739, -0.0018377, 0.0034105, -0.0048846, 0.0051185
8: -0.0008035, 0.0000030, -0.0007838, 0.0000238, -0.0008272, 0.0007869
9: 0.9989398, 1.0115755, 0.9985001, 1.0109119, -0.0086369, 0.0096933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051285, upper bound: 0.0055942
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051285, upper bound: 0.0056541
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033736, -0.0005578, -0.0034195, -0.0004704, -0.0028882, 0.0028507
1: -0.0039735, 0.0027799, -0.0042291, 0.0029594, -0.0042306, 0.0042981
2: 0.0034156, 0.0090027, 0.0031793, 0.0091678, -0.0035982, 0.0036146
3: -0.0044454, -0.0035706, -0.0044532, -0.0035465, -0.0008989, 0.0008826
4: 0.0022974, 0.0073970, 0.0021704, 0.0075688, -0.0047129, 0.0046832
5: -0.0025454, 0.0024783, -0.0027532, 0.0026356, -0.0043993, 0.0044265
6: -0.0065483, -0.0035941, -0.0066328, -0.0035351, -0.0029080, 0.0029268
7: -0.0016789, 0.0034567, -0.0017544, 0.0035739, -0.0048839, 0.0048662
8: -0.0007721, -0.0000149, -0.0008035, 0.0000030, -0.0007752, 0.0007886
9: 0.9993390, 1.0110426, 0.9989398, 1.0115755, -0.0085079, 0.0084835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054847, upper bound: 0.0050371
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054847, upper bound: 0.0051281
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033736, -0.0005578, -0.0037257, -0.0004672, -0.0029064, 0.0031679
1: -0.0039735, 0.0027799, -0.0041487, 0.0036930, -0.0053188, 0.0045719
2: 0.0034156, 0.0090027, 0.0032644, 0.0096131, -0.0042693, 0.0037019
3: -0.0044454, -0.0035706, -0.0045060, -0.0035420, -0.0009033, 0.0009354
4: 0.0022974, 0.0073970, 0.0019945, 0.0075545, -0.0047131, 0.0048522
5: -0.0025454, 0.0024783, -0.0027347, 0.0028886, -0.0046540, 0.0044267
6: -0.0065483, -0.0035941, -0.0066523, -0.0034823, -0.0029557, 0.0029467
7: -0.0016789, 0.0034567, -0.0019273, 0.0035289, -0.0048763, 0.0050747
8: -0.0007721, -0.0000149, -0.0008133, 0.0000416, -0.0008137, 0.0007985
9: 0.9993390, 1.0110426, 0.9980774, 1.0114597, -0.0085569, 0.0095501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054847, upper bound: 0.0051939
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054847, upper bound: 0.0052793
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0034195, -0.0004704, -0.0034195, -0.0004704, -0.0029304, 0.0029304
1: -0.0042291, 0.0029594, -0.0042291, 0.0029594, -0.0040962, 0.0040962
2: 0.0031793, 0.0091678, 0.0031793, 0.0091678, -0.0034683, 0.0034683
3: -0.0044532, -0.0035465, -0.0044532, -0.0035465, -0.0009067, 0.0009067
4: 0.0021704, 0.0075688, 0.0021704, 0.0075688, -0.0047292, 0.0047292
5: -0.0027532, 0.0026356, -0.0027532, 0.0026356, -0.0043985, 0.0043985
6: -0.0066328, -0.0035351, -0.0066328, -0.0035351, -0.0029521, 0.0029521
7: -0.0017544, 0.0035739, -0.0017544, 0.0035739, -0.0049163, 0.0049163
8: -0.0008035, 0.0000030, -0.0008035, 0.0000030, -0.0007693, 0.0007693
9: 0.9989398, 1.0115755, 0.9989398, 1.0115755, -0.0083004, 0.0083004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051290, upper bound: 0.0054127
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051290, upper bound: 0.0054847
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034195, -0.0004704, -0.0037257, -0.0004672, -0.0029523, 0.0032553
1: -0.0042291, 0.0029594, -0.0041487, 0.0036930, -0.0052835, 0.0044750
2: 0.0031793, 0.0091678, 0.0032644, 0.0096131, -0.0042246, 0.0036188
3: -0.0044532, -0.0035465, -0.0045060, -0.0035420, -0.0009112, 0.0009596
4: 0.0021704, 0.0075688, 0.0019945, 0.0075545, -0.0047302, 0.0048946
5: -0.0027532, 0.0026356, -0.0027347, 0.0028886, -0.0046622, 0.0044106
6: -0.0066328, -0.0035351, -0.0066523, -0.0034823, -0.0030009, 0.0029728
7: -0.0017544, 0.0035739, -0.0019273, 0.0035289, -0.0049158, 0.0051310
8: -0.0008035, 0.0000030, -0.0008133, 0.0000416, -0.0008151, 0.0007849
9: 0.9989398, 1.0115755, 0.9980774, 1.0114597, -0.0084251, 0.0094572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051290, upper bound: 0.0055942
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051290, upper bound: 0.0056541
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036775, -0.0005540, -0.0033736, -0.0005578, -0.0031197, 0.0028195
1: -0.0038843, 0.0035062, -0.0039735, 0.0027799, -0.0043120, 0.0051278
2: 0.0035039, 0.0094506, 0.0034156, 0.0090027, -0.0034790, 0.0040806
3: -0.0044993, -0.0035658, -0.0044454, -0.0035706, -0.0009287, 0.0008796
4: 0.0021228, 0.0073880, 0.0022974, 0.0073970, -0.0047223, 0.0045600
5: -0.0025269, 0.0027194, -0.0025454, 0.0024783, -0.0042251, 0.0044680
6: -0.0065731, -0.0035443, -0.0065483, -0.0035941, -0.0028789, 0.0028993
7: -0.0018377, 0.0034105, -0.0016789, 0.0034567, -0.0050084, 0.0047922
8: -0.0007838, 0.0000238, -0.0007721, -0.0000149, -0.0007690, 0.0007959
9: 0.9985001, 1.0109119, 0.9993390, 1.0110426, -0.0090920, 0.0080600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053703, upper bound: 0.0050371
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053703, upper bound: 0.0051285
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036775, -0.0005540, -0.0034195, -0.0004704, -0.0032071, 0.0028654
1: -0.0038843, 0.0035062, -0.0042291, 0.0029594, -0.0046113, 0.0054946
2: 0.0035039, 0.0094506, 0.0031793, 0.0091678, -0.0037600, 0.0043780
3: -0.0044993, -0.0035658, -0.0044532, -0.0035465, -0.0009528, 0.0008875
4: 0.0021228, 0.0073880, 0.0021704, 0.0075688, -0.0048906, 0.0046986
5: -0.0025269, 0.0027194, -0.0027532, 0.0026356, -0.0044251, 0.0046951
6: -0.0065731, -0.0035443, -0.0066328, -0.0035351, -0.0029352, 0.0029743
7: -0.0018377, 0.0034105, -0.0017544, 0.0035739, -0.0051185, 0.0048846
8: -0.0007838, 0.0000238, -0.0008035, 0.0000030, -0.0007869, 0.0008272
9: 0.9985001, 1.0109119, 0.9989398, 1.0115755, -0.0096933, 0.0086369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053703, upper bound: 0.0050371
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053703, upper bound: 0.0051285
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0037257, -0.0004672, -0.0033736, -0.0005578, -0.0031679, 0.0029064
1: -0.0041487, 0.0036930, -0.0039735, 0.0027799, -0.0045719, 0.0053188
2: 0.0032644, 0.0096131, 0.0034156, 0.0090027, -0.0037019, 0.0042693
3: -0.0045060, -0.0035420, -0.0044454, -0.0035706, -0.0009354, 0.0009033
4: 0.0019945, 0.0075545, 0.0022974, 0.0073970, -0.0048522, 0.0047131
5: -0.0027347, 0.0028886, -0.0025454, 0.0024783, -0.0044267, 0.0046540
6: -0.0066523, -0.0034823, -0.0065483, -0.0035941, -0.0029467, 0.0029557
7: -0.0019273, 0.0035289, -0.0016789, 0.0034567, -0.0050747, 0.0048763
8: -0.0008133, 0.0000416, -0.0007721, -0.0000149, -0.0007985, 0.0008137
9: 0.9980774, 1.0114597, 0.9993390, 1.0110426, -0.0095501, 0.0085569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052793, upper bound: 0.0054127
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052793, upper bound: 0.0054847
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0037257, -0.0004672, -0.0034195, -0.0004704, -0.0032553, 0.0029523
1: -0.0041487, 0.0036930, -0.0042291, 0.0029594, -0.0044750, 0.0052835
2: 0.0032644, 0.0096131, 0.0031793, 0.0091678, -0.0036188, 0.0042246
3: -0.0045060, -0.0035420, -0.0044532, -0.0035465, -0.0009596, 0.0009112
4: 0.0019945, 0.0075545, 0.0021704, 0.0075688, -0.0048946, 0.0047302
5: -0.0027347, 0.0028886, -0.0027532, 0.0026356, -0.0044106, 0.0046622
6: -0.0066523, -0.0034823, -0.0066328, -0.0035351, -0.0029728, 0.0030009
7: -0.0019273, 0.0035289, -0.0017544, 0.0035739, -0.0051310, 0.0049158
8: -0.0008133, 0.0000416, -0.0008035, 0.0000030, -0.0007849, 0.0008151
9: 0.9980774, 1.0114597, 0.9989398, 1.0115755, -0.0094572, 0.0084251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052793, upper bound: 0.0054127
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052793, upper bound: 0.0054847
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036775, -0.0005540, -0.0036775, -0.0005540, -0.0029371, 0.0029371
1: -0.0038843, 0.0035062, -0.0038843, 0.0035062, -0.0041544, 0.0041544
2: 0.0035039, 0.0094506, 0.0035039, 0.0094506, -0.0034927, 0.0034927
3: -0.0044993, -0.0035658, -0.0044993, -0.0035658, -0.0009336, 0.0009336
4: 0.0021228, 0.0073880, 0.0021228, 0.0073880, -0.0046932, 0.0046932
5: -0.0025269, 0.0027194, -0.0025269, 0.0027194, -0.0043670, 0.0043670
6: -0.0065731, -0.0035443, -0.0065731, -0.0035443, -0.0029358, 0.0029358
7: -0.0018377, 0.0034105, -0.0018377, 0.0034105, -0.0049470, 0.0049470
8: -0.0007838, 0.0000238, -0.0007838, 0.0000238, -0.0007903, 0.0007903
9: 0.9985001, 1.0109119, 0.9985001, 1.0109119, -0.0082652, 0.0082652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053927, upper bound: 0.0050709
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053927, upper bound: 0.0051603
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036775, -0.0005540, -0.0037257, -0.0004672, -0.0030054, 0.0029675
1: -0.0038843, 0.0035062, -0.0041487, 0.0036930, -0.0044551, 0.0045219
2: 0.0035039, 0.0094506, 0.0032644, 0.0096131, -0.0037840, 0.0037813
3: -0.0044993, -0.0035658, -0.0045060, -0.0035420, -0.0009573, 0.0009403
4: 0.0021228, 0.0073880, 0.0019945, 0.0075545, -0.0048474, 0.0048359
5: -0.0025269, 0.0027194, -0.0027347, 0.0028886, -0.0045737, 0.0045787
6: -0.0065731, -0.0035443, -0.0066523, -0.0034823, -0.0029925, 0.0030027
7: -0.0018377, 0.0034105, -0.0019273, 0.0035289, -0.0050463, 0.0050391
8: -0.0007838, 0.0000238, -0.0008133, 0.0000416, -0.0008217, 0.0008300
9: 0.9985001, 1.0109119, 0.9980774, 1.0114597, -0.0088421, 0.0088624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053927, upper bound: 0.0050709
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053927, upper bound: 0.0051602
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0037257, -0.0004672, -0.0036775, -0.0005540, -0.0029675, 0.0030054
1: -0.0041487, 0.0036930, -0.0038843, 0.0035062, -0.0045219, 0.0044551
2: 0.0032644, 0.0096131, 0.0035039, 0.0094506, -0.0037813, 0.0037840
3: -0.0045060, -0.0035420, -0.0044993, -0.0035658, -0.0009403, 0.0009573
4: 0.0019945, 0.0075545, 0.0021228, 0.0073880, -0.0048359, 0.0048474
5: -0.0027347, 0.0028886, -0.0025269, 0.0027194, -0.0045787, 0.0045737
6: -0.0066523, -0.0034823, -0.0065731, -0.0035443, -0.0030027, 0.0029925
7: -0.0019273, 0.0035289, -0.0018377, 0.0034105, -0.0050391, 0.0050463
8: -0.0008133, 0.0000416, -0.0007838, 0.0000238, -0.0008300, 0.0008217
9: 0.9980774, 1.0114597, 0.9985001, 1.0109119, -0.0088624, 0.0088421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052996, upper bound: 0.0054521
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052996, upper bound: 0.0055223
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0037257, -0.0004672, -0.0037257, -0.0004672, -0.0030458, 0.0030458
1: -0.0041487, 0.0036930, -0.0041487, 0.0036930, -0.0043292, 0.0043292
2: 0.0032644, 0.0096131, 0.0032644, 0.0096131, -0.0036534, 0.0036534
3: -0.0045060, -0.0035420, -0.0045060, -0.0035420, -0.0009640, 0.0009640
4: 0.0019945, 0.0075545, 0.0019945, 0.0075545, -0.0048707, 0.0048707
5: -0.0027347, 0.0028886, -0.0027347, 0.0028886, -0.0045533, 0.0045533
6: -0.0066523, -0.0034823, -0.0066523, -0.0034823, -0.0030291, 0.0030291
7: -0.0019273, 0.0035289, -0.0019273, 0.0035289, -0.0050794, 0.0050794
8: -0.0008133, 0.0000416, -0.0008133, 0.0000416, -0.0008049, 0.0008049
9: 0.9980774, 1.0114597, 0.9980774, 1.0114597, -0.0086685, 0.0086685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052996, upper bound: 0.0054521
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052996, upper bound: 0.0055223
time: 0.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.96 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052239, upper bound: 0.0051332
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052239, upper bound: 0.0052232
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052239, upper bound: 0.0053034
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052239, upper bound: 0.0053703
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0051285, upper bound: 0.0054127
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0051285, upper bound: 0.0054847
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0051285, upper bound: 0.0055942
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0051285, upper bound: 0.0056541
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0054847, upper bound: 0.0050371
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0054847, upper bound: 0.0051281
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0054847, upper bound: 0.0051939
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0054847, upper bound: 0.0052793
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0051290, upper bound: 0.0054127
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0051290, upper bound: 0.0054847
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0051290, upper bound: 0.0055942
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0051290, upper bound: 0.0056541
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0053703, upper bound: 0.0050371
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0053703, upper bound: 0.0051285
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0053703, upper bound: 0.0050371
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0053703, upper bound: 0.0051285
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052793, upper bound: 0.0054127
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052793, upper bound: 0.0054847
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052793, upper bound: 0.0054127
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052793, upper bound: 0.0054847
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0053927, upper bound: 0.0050709
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0053927, upper bound: 0.0051603
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0053927, upper bound: 0.0050709
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0053927, upper bound: 0.0051602
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052996, upper bound: 0.0054521
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052996, upper bound: 0.0055223
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052996, upper bound: 0.0054521
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 9, lower bound: -0.0052996, upper bound: 0.0055223

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0033682, -0.0005596, -0.0027406, 0.0027933
1: -0.0039139, 0.0027547, -0.0039673, 0.0027774, -0.0038822, 0.0038800
2: 0.0035188, 0.0089843, 0.0034264, 0.0090008, -0.0031961, 0.0032816
3: -0.0044068, -0.0035774, -0.0044412, -0.0035714, -0.0008355, 0.0008638
4: 0.0023096, 0.0072495, 0.0022986, 0.0073814, -0.0045213, 0.0044018
5: -0.0024314, 0.0024635, -0.0025332, 0.0024768, -0.0040899, 0.0041775
6: -0.0064537, -0.0035992, -0.0065380, -0.0035946, -0.0027564, 0.0028367
7: -0.0016628, 0.0032800, -0.0016772, 0.0034382, -0.0047297, 0.0045830
8: -0.0007646, -0.0000391, -0.0007713, -0.0000174, -0.0007447, 0.0007210
9: 0.9993789, 1.0108052, 0.9993432, 1.0110176, -0.0078485, 0.0076622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051337, upper bound: 0.0051337
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051337, upper bound: 0.0051337
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0033333, -0.0005338, -0.0033464, -0.0005652, -0.0027636, 0.0028126
1: -0.0039051, 0.0028545, -0.0039445, 0.0027699, -0.0038941, 0.0039317
2: 0.0035436, 0.0091325, 0.0034628, 0.0089948, -0.0032263, 0.0034432
3: -0.0044122, -0.0035194, -0.0044307, -0.0035737, -0.0008385, 0.0009113
4: 0.0021643, 0.0072807, 0.0023029, 0.0073319, -0.0046150, 0.0044428
5: -0.0024637, 0.0025546, -0.0024944, 0.0024718, -0.0041297, 0.0042317
6: -0.0064812, -0.0035014, -0.0065079, -0.0035966, -0.0027848, 0.0029027
7: -0.0019011, 0.0032890, -0.0016716, 0.0033830, -0.0049503, 0.0046334
8: -0.0008063, -0.0000452, -0.0007691, -0.0000257, -0.0007807, 0.0007238
9: 0.9991218, 1.0107985, 0.9993563, 1.0109340, -0.0080383, 0.0077191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0051420
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050774, upper bound: 0.0050774
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0036725, -0.0005558, -0.0027674, 0.0030976
1: -0.0039139, 0.0027547, -0.0038783, 0.0035033, -0.0050795, 0.0042597
2: 0.0035188, 0.0089843, 0.0035142, 0.0094488, -0.0039586, 0.0034444
3: -0.0044068, -0.0035774, -0.0044951, -0.0035665, -0.0008403, 0.0009177
4: 0.0023096, 0.0072495, 0.0021241, 0.0073727, -0.0045369, 0.0045793
5: -0.0024314, 0.0024635, -0.0025149, 0.0027179, -0.0043582, 0.0042036
6: -0.0064537, -0.0035992, -0.0065631, -0.0035448, -0.0028039, 0.0028640
7: -0.0016628, 0.0032800, -0.0018358, 0.0033920, -0.0047486, 0.0048174
8: -0.0007646, -0.0000391, -0.0007830, 0.0000212, -0.0007858, 0.0007385
9: 0.9993789, 1.0108052, 0.9985046, 1.0108873, -0.0080025, 0.0088460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051337, upper bound: 0.0053034
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051337, upper bound: 0.0053034
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0033333, -0.0005338, -0.0036500, -0.0005616, -0.0027717, 0.0031163
1: -0.0039051, 0.0028545, -0.0038554, 0.0034950, -0.0050922, 0.0043131
2: 0.0035436, 0.0091325, 0.0035544, 0.0094430, -0.0039862, 0.0035950
3: -0.0044122, -0.0035194, -0.0044855, -0.0035691, -0.0008431, 0.0009660
4: 0.0021643, 0.0072807, 0.0021286, 0.0073243, -0.0046306, 0.0046195
5: -0.0024637, 0.0025546, -0.0024771, 0.0027131, -0.0043972, 0.0042582
6: -0.0064812, -0.0035014, -0.0065325, -0.0035469, -0.0028322, 0.0029297
7: -0.0019011, 0.0032890, -0.0018296, 0.0033385, -0.0049658, 0.0048668
8: -0.0008063, -0.0000452, -0.0007806, 0.0000134, -0.0008197, 0.0007354
9: 0.9991218, 1.0107985, 0.9985181, 1.0108021, -0.0081920, 0.0088986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0052843
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050774, upper bound: 0.0052458
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0033682, -0.0005596, -0.0027713, 0.0028660
1: -0.0041685, 0.0029345, -0.0039673, 0.0027774, -0.0042498, 0.0041800
2: 0.0032797, 0.0091512, 0.0034264, 0.0090008, -0.0035002, 0.0035598
3: -0.0044138, -0.0035538, -0.0044412, -0.0035714, -0.0008425, 0.0008873
4: 0.0021825, 0.0074231, 0.0022986, 0.0073814, -0.0046595, 0.0045743
5: -0.0026427, 0.0026214, -0.0025332, 0.0024768, -0.0043225, 0.0043759
6: -0.0065378, -0.0035401, -0.0065380, -0.0035946, -0.0028312, 0.0028929
7: -0.0017391, 0.0034059, -0.0016772, 0.0034382, -0.0048222, 0.0046988
8: -0.0007959, -0.0000212, -0.0007713, -0.0000174, -0.0007785, 0.0007501
9: 0.9989762, 1.0113478, 0.9993432, 1.0110176, -0.0084200, 0.0082806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050371, upper bound: 0.0054127
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050371, upper bound: 0.0054127
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0033777, -0.0004513, -0.0033464, -0.0005652, -0.0027913, 0.0028951
1: -0.0041575, 0.0030323, -0.0039445, 0.0027699, -0.0042591, 0.0042142
2: 0.0033229, 0.0092811, 0.0034628, 0.0089948, -0.0035006, 0.0036845
3: -0.0044203, -0.0034953, -0.0044307, -0.0035737, -0.0008466, 0.0009354
4: 0.0020487, 0.0074385, 0.0023029, 0.0073319, -0.0047416, 0.0045916
5: -0.0026567, 0.0027090, -0.0024944, 0.0024718, -0.0043360, 0.0044223
6: -0.0065492, -0.0034466, -0.0065079, -0.0035966, -0.0028472, 0.0029522
7: -0.0019553, 0.0033748, -0.0016716, 0.0033830, -0.0050005, 0.0047188
8: -0.0008380, -0.0000260, -0.0007691, -0.0000257, -0.0008123, 0.0007431
9: 0.9987509, 1.0112903, 0.9993563, 1.0109340, -0.0085806, 0.0082848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053887
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049692, upper bound: 0.0053356
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0036725, -0.0005558, -0.0028149, 0.0031836
1: -0.0041685, 0.0029345, -0.0038783, 0.0035033, -0.0054470, 0.0045597
2: 0.0032797, 0.0091512, 0.0035142, 0.0094488, -0.0042626, 0.0037226
3: -0.0044138, -0.0035538, -0.0044951, -0.0035665, -0.0008473, 0.0009412
4: 0.0021825, 0.0074231, 0.0021241, 0.0073727, -0.0046750, 0.0047518
5: -0.0026427, 0.0026214, -0.0025149, 0.0027179, -0.0045908, 0.0044020
6: -0.0065378, -0.0035401, -0.0065631, -0.0035448, -0.0028787, 0.0029202
7: -0.0017391, 0.0034059, -0.0018358, 0.0033920, -0.0048410, 0.0049332
8: -0.0007959, -0.0000212, -0.0007830, 0.0000212, -0.0008171, 0.0007619
9: 0.9989762, 1.0113478, 0.9985046, 1.0108873, -0.0085739, 0.0094644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050371, upper bound: 0.0055942
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050371, upper bound: 0.0055942
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0033777, -0.0004513, -0.0036500, -0.0005616, -0.0028162, 0.0031988
1: -0.0041575, 0.0030323, -0.0038554, 0.0034950, -0.0054572, 0.0045955
2: 0.0033229, 0.0092811, 0.0035544, 0.0094430, -0.0042605, 0.0038363
3: -0.0044203, -0.0034953, -0.0044855, -0.0035691, -0.0008512, 0.0009901
4: 0.0020487, 0.0074385, 0.0021286, 0.0073243, -0.0047572, 0.0047683
5: -0.0026567, 0.0027090, -0.0024771, 0.0027131, -0.0046035, 0.0044487
6: -0.0065492, -0.0034466, -0.0065325, -0.0035469, -0.0028946, 0.0029792
7: -0.0019553, 0.0033748, -0.0018296, 0.0033385, -0.0050160, 0.0049521
8: -0.0008380, -0.0000260, -0.0007806, 0.0000134, -0.0008514, 0.0007547
9: 0.9987509, 1.0112903, 0.9985181, 1.0108021, -0.0087342, 0.0094643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055551
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049692, upper bound: 0.0055205
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0034142, -0.0004723, -0.0028092, 0.0028311
1: -0.0039139, 0.0027547, -0.0042228, 0.0029569, -0.0041816, 0.0042466
2: 0.0035188, 0.0089843, 0.0031898, 0.0091661, -0.0034768, 0.0035802
3: -0.0044068, -0.0035774, -0.0044490, -0.0035473, -0.0008596, 0.0008716
4: 0.0023096, 0.0072495, 0.0021717, 0.0075533, -0.0046899, 0.0045404
5: -0.0024314, 0.0024635, -0.0027414, 0.0026341, -0.0042897, 0.0044052
6: -0.0064537, -0.0035992, -0.0066225, -0.0035356, -0.0028126, 0.0029117
7: -0.0016628, 0.0032800, -0.0017528, 0.0035561, -0.0048406, 0.0046754
8: -0.0007646, -0.0000391, -0.0008027, 0.0000005, -0.0007651, 0.0007635
9: 0.9993789, 1.0108052, 0.9989437, 1.0115513, -0.0084520, 0.0082385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054127, upper bound: 0.0050371
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054127, upper bound: 0.0050371
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0033333, -0.0005338, -0.0033924, -0.0004786, -0.0028305, 0.0028587
1: -0.0039051, 0.0028545, -0.0041997, 0.0029493, -0.0041940, 0.0042931
2: 0.0035436, 0.0091325, 0.0032299, 0.0091605, -0.0035059, 0.0037289
3: -0.0044122, -0.0035194, -0.0044389, -0.0035498, -0.0008623, 0.0009195
4: 0.0021643, 0.0072807, 0.0021761, 0.0075046, -0.0047811, 0.0045811
5: -0.0024637, 0.0025546, -0.0027013, 0.0026291, -0.0043290, 0.0044557
6: -0.0064812, -0.0035014, -0.0065905, -0.0035376, -0.0028410, 0.0029765
7: -0.0019011, 0.0032890, -0.0017472, 0.0034934, -0.0050555, 0.0047257
8: -0.0008063, -0.0000452, -0.0008003, -0.0000076, -0.0007988, 0.0007551
9: 0.9991218, 1.0107985, 0.9989561, 1.0114590, -0.0086294, 0.0082934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053086, upper bound: 0.0050529
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053356, upper bound: 0.0049693
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0037207, -0.0004692, -0.0028541, 0.0031458
1: -0.0039139, 0.0027547, -0.0041426, 0.0036903, -0.0052704, 0.0045200
2: 0.0035188, 0.0089843, 0.0032747, 0.0096114, -0.0041473, 0.0036676
3: -0.0044068, -0.0035774, -0.0045018, -0.0035429, -0.0008640, 0.0009244
4: 0.0023096, 0.0072495, 0.0019959, 0.0075393, -0.0046906, 0.0047092
5: -0.0024314, 0.0024635, -0.0027231, 0.0028871, -0.0045441, 0.0044056
6: -0.0064537, -0.0035992, -0.0066423, -0.0034828, -0.0028603, 0.0029319
7: -0.0016628, 0.0032800, -0.0019256, 0.0035109, -0.0048334, 0.0048836
8: -0.0007646, -0.0000391, -0.0008125, 0.0000391, -0.0008036, 0.0007734
9: 0.9993789, 1.0108052, 0.9980815, 1.0114361, -0.0085006, 0.0093041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054127, upper bound: 0.0051939
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054127, upper bound: 0.0051939
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0033333, -0.0005338, -0.0036990, -0.0004759, -0.0028573, 0.0031652
1: -0.0039051, 0.0028545, -0.0041185, 0.0036820, -0.0052835, 0.0045676
2: 0.0035436, 0.0091325, 0.0033158, 0.0096057, -0.0041750, 0.0038151
3: -0.0044122, -0.0035194, -0.0044923, -0.0035457, -0.0008665, 0.0009729
4: 0.0021643, 0.0072807, 0.0020007, 0.0074905, -0.0047801, 0.0047492
5: -0.0024637, 0.0025546, -0.0026832, 0.0028819, -0.0045826, 0.0044557
6: -0.0064812, -0.0035014, -0.0066100, -0.0034849, -0.0028885, 0.0029963
7: -0.0019011, 0.0032890, -0.0019195, 0.0034506, -0.0050451, 0.0049330
8: -0.0008063, -0.0000452, -0.0008100, 0.0000310, -0.0008374, 0.0007648
9: 0.9991218, 1.0107985, 0.9980952, 1.0113436, -0.0086802, 0.0093559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0052011
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053356, upper bound: 0.0051407
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0034142, -0.0004723, -0.0028527, 0.0029108
1: -0.0041685, 0.0029345, -0.0042228, 0.0029569, -0.0040460, 0.0040457
2: 0.0032797, 0.0091512, 0.0031898, 0.0091661, -0.0033502, 0.0034337
3: -0.0044138, -0.0035538, -0.0044490, -0.0035473, -0.0008666, 0.0008951
4: 0.0021825, 0.0074231, 0.0021717, 0.0075533, -0.0047061, 0.0045876
5: -0.0026427, 0.0026214, -0.0027414, 0.0026341, -0.0042893, 0.0043770
6: -0.0065378, -0.0035401, -0.0066225, -0.0035356, -0.0028562, 0.0029371
7: -0.0017391, 0.0034059, -0.0017528, 0.0035561, -0.0048730, 0.0047280
8: -0.0007959, -0.0000212, -0.0008027, 0.0000005, -0.0007600, 0.0007366
9: 0.9989762, 1.0113478, 0.9989437, 1.0115513, -0.0082438, 0.0080633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050379, upper bound: 0.0054127
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050379, upper bound: 0.0054127
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0033777, -0.0004513, -0.0033924, -0.0004786, -0.0028649, 0.0029412
1: -0.0041575, 0.0030323, -0.0041997, 0.0029493, -0.0040534, 0.0040895
2: 0.0033229, 0.0092811, 0.0032299, 0.0091605, -0.0033608, 0.0035760
3: -0.0044203, -0.0034953, -0.0044389, -0.0035498, -0.0008704, 0.0009436
4: 0.0020487, 0.0074385, 0.0021761, 0.0075046, -0.0047939, 0.0046116
5: -0.0026567, 0.0027090, -0.0027013, 0.0026291, -0.0043123, 0.0044288
6: -0.0065492, -0.0034466, -0.0065905, -0.0035376, -0.0028737, 0.0029988
7: -0.0019553, 0.0033748, -0.0017472, 0.0034934, -0.0050598, 0.0047536
8: -0.0008380, -0.0000260, -0.0008003, -0.0000076, -0.0008103, 0.0007399
9: 0.9987509, 1.0112903, 0.9989561, 1.0114590, -0.0084166, 0.0080837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053887
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049692, upper bound: 0.0053356
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0037207, -0.0004692, -0.0029014, 0.0032317
1: -0.0041685, 0.0029345, -0.0041426, 0.0036903, -0.0052342, 0.0044236
2: 0.0032797, 0.0091512, 0.0032747, 0.0096114, -0.0041056, 0.0035848
3: -0.0044138, -0.0035538, -0.0045018, -0.0035429, -0.0008710, 0.0009480
4: 0.0021825, 0.0074231, 0.0019959, 0.0075393, -0.0047073, 0.0047528
5: -0.0026427, 0.0026214, -0.0027231, 0.0028871, -0.0045526, 0.0043891
6: -0.0065378, -0.0035401, -0.0066423, -0.0034828, -0.0029049, 0.0029579
7: -0.0017391, 0.0034059, -0.0019256, 0.0035109, -0.0048734, 0.0049424
8: -0.0007959, -0.0000212, -0.0008125, 0.0000391, -0.0008059, 0.0007522
9: 0.9989762, 1.0113478, 0.9980815, 1.0114361, -0.0083682, 0.0092187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050379, upper bound: 0.0055942
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050379, upper bound: 0.0055942
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0033777, -0.0004513, -0.0036990, -0.0004759, -0.0029018, 0.0032477
1: -0.0041575, 0.0030323, -0.0041185, 0.0036820, -0.0052426, 0.0044693
2: 0.0033229, 0.0092811, 0.0033158, 0.0096057, -0.0041134, 0.0037213
3: -0.0044203, -0.0034953, -0.0044923, -0.0035457, -0.0008746, 0.0009970
4: 0.0020487, 0.0074385, 0.0020007, 0.0074905, -0.0047946, 0.0047760
5: -0.0026567, 0.0027090, -0.0026832, 0.0028819, -0.0045746, 0.0044421
6: -0.0065492, -0.0034466, -0.0066100, -0.0034849, -0.0029223, 0.0030194
7: -0.0019553, 0.0033748, -0.0019195, 0.0034506, -0.0050573, 0.0049670
8: -0.0008380, -0.0000260, -0.0008100, 0.0000310, -0.0008562, 0.0007553
9: 0.9987509, 1.0112903, 0.9980952, 1.0113436, -0.0085450, 0.0092346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055551
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049692, upper bound: 0.0055206
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0033682, -0.0005596, -0.0030717, 0.0027972
1: -0.0038272, 0.0034772, -0.0039673, 0.0027774, -0.0042529, 0.0050825
2: 0.0036027, 0.0094329, 0.0034264, 0.0090008, -0.0033615, 0.0040368
3: -0.0044598, -0.0035728, -0.0044412, -0.0035714, -0.0008884, 0.0008683
4: 0.0021356, 0.0072439, 0.0022986, 0.0073814, -0.0046968, 0.0044189
5: -0.0024147, 0.0027050, -0.0025332, 0.0024768, -0.0041172, 0.0044431
6: -0.0064802, -0.0035495, -0.0065380, -0.0035946, -0.0027847, 0.0028840
7: -0.0018203, 0.0032344, -0.0016772, 0.0034382, -0.0049618, 0.0046017
8: -0.0007762, -0.0000005, -0.0007713, -0.0000174, -0.0007589, 0.0007680
9: 0.9985415, 1.0106769, 0.9993432, 1.0110176, -0.0090206, 0.0078206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053034, upper bound: 0.0051337
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053034, upper bound: 0.0051337
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036387, -0.0005363, -0.0033464, -0.0005652, -0.0030734, 0.0028101
1: -0.0038154, 0.0035825, -0.0039445, 0.0027699, -0.0042651, 0.0051571
2: 0.0036428, 0.0095646, 0.0034628, 0.0089948, -0.0033677, 0.0041801
3: -0.0044676, -0.0035181, -0.0044307, -0.0035737, -0.0008939, 0.0009126
4: 0.0019980, 0.0072638, 0.0023029, 0.0073319, -0.0047843, 0.0044464
5: -0.0024372, 0.0027910, -0.0024944, 0.0024718, -0.0041480, 0.0044984
6: -0.0064982, -0.0034548, -0.0065079, -0.0035966, -0.0028061, 0.0029501
7: -0.0020380, 0.0032337, -0.0016716, 0.0033830, -0.0051513, 0.0046369
8: -0.0008161, -0.0000061, -0.0007691, -0.0000257, -0.0007905, 0.0007630
9: 0.9983037, 1.0106512, 0.9993563, 1.0109340, -0.0092213, 0.0078551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052286, upper bound: 0.0051420
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052458, upper bound: 0.0050774
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0034142, -0.0004723, -0.0031589, 0.0028432
1: -0.0038272, 0.0034772, -0.0042228, 0.0029569, -0.0045523, 0.0054492
2: 0.0036027, 0.0094329, 0.0031898, 0.0091661, -0.0036421, 0.0043354
3: -0.0044598, -0.0035728, -0.0044490, -0.0035473, -0.0009125, 0.0008761
4: 0.0021356, 0.0072439, 0.0021717, 0.0075533, -0.0048653, 0.0045575
5: -0.0024147, 0.0027050, -0.0027414, 0.0026341, -0.0043170, 0.0046708
6: -0.0064802, -0.0035495, -0.0066225, -0.0035356, -0.0028409, 0.0029590
7: -0.0018203, 0.0032344, -0.0017528, 0.0035561, -0.0050727, 0.0046941
8: -0.0007762, -0.0000005, -0.0008027, 0.0000005, -0.0007767, 0.0008022
9: 0.9985415, 1.0106769, 0.9989437, 1.0115513, -0.0096240, 0.0083969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055942, upper bound: 0.0050371
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055942, upper bound: 0.0050371
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0036387, -0.0005363, -0.0033924, -0.0004786, -0.0031600, 0.0028561
1: -0.0038154, 0.0035825, -0.0041997, 0.0029493, -0.0045650, 0.0055186
2: 0.0036428, 0.0095646, 0.0032299, 0.0091605, -0.0036473, 0.0044658
3: -0.0044676, -0.0035181, -0.0044389, -0.0035498, -0.0009177, 0.0009208
4: 0.0019980, 0.0072638, 0.0021761, 0.0075046, -0.0049504, 0.0045847
5: -0.0024372, 0.0027910, -0.0027013, 0.0026291, -0.0043473, 0.0047224
6: -0.0064982, -0.0034548, -0.0065905, -0.0035376, -0.0028623, 0.0030238
7: -0.0020380, 0.0032337, -0.0017472, 0.0034934, -0.0052565, 0.0047292
8: -0.0008161, -0.0000061, -0.0008003, -0.0000076, -0.0008085, 0.0007942
9: 0.9983037, 1.0106512, 0.9989561, 1.0114590, -0.0098124, 0.0084294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055087, upper bound: 0.0050529
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0049693
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0033682, -0.0005596, -0.0031196, 0.0028816
1: -0.0040906, 0.0036655, -0.0039673, 0.0027774, -0.0045168, 0.0052733
2: 0.0033639, 0.0095964, 0.0034264, 0.0090008, -0.0035885, 0.0042263
3: -0.0044669, -0.0035499, -0.0044412, -0.0035714, -0.0008955, 0.0008913
4: 0.0020077, 0.0074119, 0.0022986, 0.0073814, -0.0048264, 0.0045773
5: -0.0026266, 0.0028738, -0.0025332, 0.0024768, -0.0043245, 0.0046281
6: -0.0065589, -0.0034874, -0.0065380, -0.0035946, -0.0028538, 0.0029403
7: -0.0019106, 0.0033605, -0.0016772, 0.0034382, -0.0050282, 0.0046961
8: -0.0008055, 0.0000177, -0.0007713, -0.0000174, -0.0007882, 0.0007890
9: 0.9981174, 1.0112350, 0.9993432, 1.0110176, -0.0094771, 0.0083275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051939, upper bound: 0.0054127
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051939, upper bound: 0.0054127
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036869, -0.0004558, -0.0033464, -0.0005652, -0.0031217, 0.0028906
1: -0.0040733, 0.0037617, -0.0039445, 0.0027699, -0.0045259, 0.0053276
2: 0.0034106, 0.0097233, 0.0034628, 0.0089948, -0.0035820, 0.0043435
3: -0.0044745, -0.0034944, -0.0044307, -0.0035737, -0.0009008, 0.0009363
4: 0.0018748, 0.0074206, 0.0023029, 0.0073319, -0.0049042, 0.0045872
5: -0.0026369, 0.0029579, -0.0024944, 0.0024718, -0.0043317, 0.0046704
6: -0.0065684, -0.0033950, -0.0065079, -0.0035966, -0.0028665, 0.0029975
7: -0.0021094, 0.0033303, -0.0016716, 0.0033830, -0.0051946, 0.0047036
8: -0.0008459, 0.0000125, -0.0007691, -0.0000257, -0.0008202, 0.0007815
9: 0.9978935, 1.0111725, 0.9993563, 1.0109340, -0.0096434, 0.0083249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050937, upper bound: 0.0053887
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053356
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0034142, -0.0004723, -0.0031923, 0.0029276
1: -0.0040906, 0.0036655, -0.0042228, 0.0029569, -0.0044156, 0.0052406
2: 0.0033639, 0.0095964, 0.0031898, 0.0091661, -0.0035013, 0.0041817
3: -0.0044669, -0.0035499, -0.0044490, -0.0035473, -0.0009196, 0.0008991
4: 0.0020077, 0.0074119, 0.0021717, 0.0075533, -0.0048692, 0.0045907
5: -0.0026266, 0.0028738, -0.0027414, 0.0026341, -0.0043026, 0.0046373
6: -0.0065589, -0.0034874, -0.0066225, -0.0035356, -0.0028793, 0.0029855
7: -0.0019106, 0.0033605, -0.0017528, 0.0035561, -0.0050851, 0.0047316
8: -0.0008055, 0.0000177, -0.0008027, 0.0000005, -0.0007751, 0.0007841
9: 0.9981174, 1.0112350, 0.9989437, 1.0115513, -0.0093870, 0.0081893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051944, upper bound: 0.0054127
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051944, upper bound: 0.0054127
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0036869, -0.0004558, -0.0033924, -0.0004786, -0.0032036, 0.0029366
1: -0.0040733, 0.0037617, -0.0041997, 0.0029493, -0.0044228, 0.0053118
2: 0.0034106, 0.0097233, 0.0032299, 0.0091605, -0.0035003, 0.0043119
3: -0.0044745, -0.0034944, -0.0044389, -0.0035498, -0.0009246, 0.0009445
4: 0.0018748, 0.0074206, 0.0021761, 0.0075046, -0.0049558, 0.0046084
5: -0.0026369, 0.0029579, -0.0027013, 0.0026291, -0.0043226, 0.0046881
6: -0.0065684, -0.0033950, -0.0065905, -0.0035376, -0.0028916, 0.0030430
7: -0.0021094, 0.0033303, -0.0017472, 0.0034934, -0.0052566, 0.0047462
8: -0.0008459, 0.0000125, -0.0008003, -0.0000076, -0.0008245, 0.0007860
9: 0.9978935, 1.0111725, 0.9989561, 1.0114590, -0.0095673, 0.0082031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050938, upper bound: 0.0053887
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053356
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0036725, -0.0005558, -0.0028607, 0.0029178
1: -0.0038272, 0.0034772, -0.0038783, 0.0035033, -0.0041051, 0.0041046
2: 0.0036027, 0.0094329, 0.0035142, 0.0094488, -0.0033746, 0.0034569
3: -0.0044598, -0.0035728, -0.0044951, -0.0035665, -0.0008933, 0.0009223
4: 0.0021356, 0.0072439, 0.0021241, 0.0073727, -0.0046703, 0.0045540
5: -0.0024147, 0.0027050, -0.0025149, 0.0027179, -0.0042612, 0.0043454
6: -0.0064802, -0.0035495, -0.0065631, -0.0035448, -0.0028430, 0.0029210
7: -0.0018203, 0.0032344, -0.0018358, 0.0033920, -0.0049024, 0.0047620
8: -0.0007762, -0.0000005, -0.0007830, 0.0000212, -0.0007810, 0.0007574
9: 0.9985415, 1.0106769, 0.9985046, 1.0108873, -0.0082066, 0.0080260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053267, upper bound: 0.0051682
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053267, upper bound: 0.0051682
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036387, -0.0005363, -0.0036500, -0.0005616, -0.0028806, 0.0029642
1: -0.0038154, 0.0035825, -0.0038554, 0.0034950, -0.0041152, 0.0041566
2: 0.0036428, 0.0095646, 0.0035544, 0.0094430, -0.0033966, 0.0036135
3: -0.0044676, -0.0035181, -0.0044855, -0.0035691, -0.0008985, 0.0009673
4: 0.0019980, 0.0072638, 0.0021286, 0.0073243, -0.0047641, 0.0045889
5: -0.0024372, 0.0027910, -0.0024771, 0.0027131, -0.0042947, 0.0043987
6: -0.0064982, -0.0034548, -0.0065325, -0.0035469, -0.0028670, 0.0029862
7: -0.0020380, 0.0032337, -0.0018296, 0.0033385, -0.0051183, 0.0047982
8: -0.0008161, -0.0000061, -0.0007806, 0.0000134, -0.0008295, 0.0007615
9: 0.9983037, 1.0106512, 0.9985181, 1.0108021, -0.0084004, 0.0080721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052514, upper bound: 0.0051795
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052738, upper bound: 0.0051268
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0037207, -0.0004692, -0.0029287, 0.0029481
1: -0.0038272, 0.0034772, -0.0041426, 0.0036903, -0.0044059, 0.0044722
2: 0.0036027, 0.0094329, 0.0032747, 0.0096114, -0.0036655, 0.0037464
3: -0.0044598, -0.0035728, -0.0045018, -0.0035429, -0.0009169, 0.0009290
4: 0.0021356, 0.0072439, 0.0019959, 0.0075393, -0.0048244, 0.0046966
5: -0.0024147, 0.0027050, -0.0027231, 0.0028871, -0.0044677, 0.0045574
6: -0.0064802, -0.0035495, -0.0066423, -0.0034828, -0.0028996, 0.0029878
7: -0.0018203, 0.0032344, -0.0019256, 0.0035109, -0.0050023, 0.0048541
8: -0.0007762, -0.0000005, -0.0008125, 0.0000391, -0.0008124, 0.0007970
9: 0.9985415, 1.0106769, 0.9980815, 1.0114361, -0.0087850, 0.0086226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056207, upper bound: 0.0050708
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0056207, upper bound: 0.0050709
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0036387, -0.0005363, -0.0036990, -0.0004759, -0.0029469, 0.0029947
1: -0.0038154, 0.0035825, -0.0041185, 0.0036820, -0.0044166, 0.0045187
2: 0.0036428, 0.0095646, 0.0033158, 0.0096057, -0.0036864, 0.0038919
3: -0.0044676, -0.0035181, -0.0044923, -0.0035457, -0.0009219, 0.0009742
4: 0.0019980, 0.0072638, 0.0020007, 0.0074905, -0.0049167, 0.0047313
5: -0.0024372, 0.0027910, -0.0026832, 0.0028819, -0.0045007, 0.0046075
6: -0.0064982, -0.0034548, -0.0066100, -0.0034849, -0.0029236, 0.0030519
7: -0.0020380, 0.0032337, -0.0019195, 0.0034506, -0.0052131, 0.0048903
8: -0.0008161, -0.0000061, -0.0008100, 0.0000310, -0.0008471, 0.0008008
9: 0.9983037, 1.0106512, 0.9980952, 1.0113436, -0.0089681, 0.0086668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055381, upper bound: 0.0050888
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055490, upper bound: 0.0050184
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0036725, -0.0005558, -0.0028900, 0.0029835
1: -0.0040906, 0.0036655, -0.0038783, 0.0035033, -0.0044737, 0.0044066
2: 0.0033639, 0.0095964, 0.0035142, 0.0094488, -0.0036678, 0.0037452
3: -0.0044669, -0.0035499, -0.0044951, -0.0035665, -0.0009003, 0.0009452
4: 0.0020077, 0.0074119, 0.0021241, 0.0073727, -0.0048122, 0.0047109
5: -0.0026266, 0.0028738, -0.0025149, 0.0027179, -0.0044765, 0.0045507
6: -0.0065589, -0.0034874, -0.0065631, -0.0035448, -0.0029097, 0.0029775
7: -0.0019106, 0.0033605, -0.0018358, 0.0033920, -0.0049946, 0.0048643
8: -0.0008055, 0.0000177, -0.0007830, 0.0000212, -0.0008197, 0.0007891
9: 0.9981174, 1.0112350, 0.9985046, 1.0108873, -0.0087985, 0.0086185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052159, upper bound: 0.0054521
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052159, upper bound: 0.0054521
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036869, -0.0004558, -0.0036500, -0.0005616, -0.0029085, 0.0030345
1: -0.0040733, 0.0037617, -0.0038554, 0.0034950, -0.0044810, 0.0044395
2: 0.0034106, 0.0097233, 0.0035544, 0.0094430, -0.0036652, 0.0038660
3: -0.0044745, -0.0034944, -0.0044855, -0.0035691, -0.0009054, 0.0009910
4: 0.0018748, 0.0074206, 0.0021286, 0.0073243, -0.0048940, 0.0047295
5: -0.0026369, 0.0029579, -0.0024771, 0.0027131, -0.0044889, 0.0045963
6: -0.0065684, -0.0033950, -0.0065325, -0.0035469, -0.0029239, 0.0030352
7: -0.0021094, 0.0033303, -0.0018296, 0.0033385, -0.0051686, 0.0048775
8: -0.0008459, 0.0000125, -0.0007806, 0.0000134, -0.0008593, 0.0007931
9: 0.9978935, 1.0111725, 0.9985181, 1.0108021, -0.0089559, 0.0086207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0054297
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053850
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0037207, -0.0004692, -0.0029707, 0.0030264
1: -0.0040906, 0.0036655, -0.0041426, 0.0036903, -0.0042791, 0.0042805
2: 0.0033639, 0.0095964, 0.0032747, 0.0096114, -0.0035365, 0.0036188
3: -0.0044669, -0.0035499, -0.0045018, -0.0035429, -0.0009240, 0.0009520
4: 0.0020077, 0.0074119, 0.0019959, 0.0075393, -0.0048478, 0.0047330
5: -0.0026266, 0.0028738, -0.0027231, 0.0028871, -0.0044478, 0.0045320
6: -0.0065589, -0.0034874, -0.0066423, -0.0034828, -0.0029366, 0.0030141
7: -0.0019106, 0.0033605, -0.0019256, 0.0035109, -0.0050356, 0.0048944
8: -0.0008055, 0.0000177, -0.0008125, 0.0000391, -0.0007957, 0.0007724
9: 0.9981174, 1.0112350, 0.9980815, 1.0114361, -0.0086117, 0.0084370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052176, upper bound: 0.0054521
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052176, upper bound: 0.0054127
time: 2.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0036869, -0.0004558, -0.0036990, -0.0004759, -0.0029816, 0.0030672
1: -0.0040733, 0.0037617, -0.0041185, 0.0036820, -0.0042829, 0.0043269
2: 0.0034106, 0.0097233, 0.0033158, 0.0096057, -0.0035418, 0.0037539
3: -0.0044745, -0.0034944, -0.0044923, -0.0035457, -0.0009288, 0.0009979
4: 0.0018748, 0.0074206, 0.0020007, 0.0074905, -0.0049332, 0.0047528
5: -0.0026369, 0.0029579, -0.0026832, 0.0028819, -0.0044691, 0.0045836
6: -0.0065684, -0.0033950, -0.0066100, -0.0034849, -0.0029517, 0.0030750
7: -0.0021094, 0.0033303, -0.0019195, 0.0034506, -0.0052194, 0.0049140
8: -0.0008459, 0.0000125, -0.0008100, 0.0000310, -0.0008465, 0.0007755
9: 0.9978935, 1.0111725, 0.9980952, 1.0113436, -0.0087816, 0.0084513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0054297
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053850
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.08 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051337, upper bound: 0.0051337
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051337, upper bound: 0.0051337
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0051420
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050774, upper bound: 0.0050774
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051337, upper bound: 0.0053034
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051337, upper bound: 0.0053034
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0052843
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050774, upper bound: 0.0052458
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050371, upper bound: 0.0054127
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050371, upper bound: 0.0054127
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053887
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0049692, upper bound: 0.0053356
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050371, upper bound: 0.0055942
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050371, upper bound: 0.0055942
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055551
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0049692, upper bound: 0.0055205
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0054127, upper bound: 0.0050371
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0054127, upper bound: 0.0050371
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0053086, upper bound: 0.0050529
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0053356, upper bound: 0.0049693
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0054127, upper bound: 0.0051939
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0054127, upper bound: 0.0051939
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0052011
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0053356, upper bound: 0.0051407
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050379, upper bound: 0.0054127
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050379, upper bound: 0.0054127
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053887
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0049692, upper bound: 0.0053356
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050379, upper bound: 0.0055942
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050379, upper bound: 0.0055942
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055551
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0049692, upper bound: 0.0055206
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0053034, upper bound: 0.0051337
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0053034, upper bound: 0.0051337
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0052286, upper bound: 0.0051420
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0052458, upper bound: 0.0050774
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0055942, upper bound: 0.0050371
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0055942, upper bound: 0.0050371
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0055087, upper bound: 0.0050529
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0049693
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051939, upper bound: 0.0054127
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051939, upper bound: 0.0054127
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050937, upper bound: 0.0053887
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053356
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051944, upper bound: 0.0054127
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051944, upper bound: 0.0054127
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0050938, upper bound: 0.0053887
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053356
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0053267, upper bound: 0.0051682
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0053267, upper bound: 0.0051682
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0052514, upper bound: 0.0051795
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0052738, upper bound: 0.0051268
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0056207, upper bound: 0.0050708
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0056207, upper bound: 0.0050709
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0055381, upper bound: 0.0050888
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0055490, upper bound: 0.0050184
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0052159, upper bound: 0.0054521
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0052159, upper bound: 0.0054521
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0054297
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053850
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0052176, upper bound: 0.0054521
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0052176, upper bound: 0.0054127
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0054297
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053850

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0033232, -0.0005749, -0.0027306, 0.0027306
1: -0.0039139, 0.0027547, -0.0039139, 0.0027547, -0.0038407, 0.0038407
2: 0.0035188, 0.0089843, 0.0035188, 0.0089843, -0.0031759, 0.0031759
3: -0.0044068, -0.0035774, -0.0044068, -0.0035774, -0.0008295, 0.0008295
4: 0.0023096, 0.0072495, 0.0023096, 0.0072495, -0.0043946, 0.0043946
5: -0.0024314, 0.0024635, -0.0024314, 0.0024635, -0.0040809, 0.0040809
6: -0.0064537, -0.0035992, -0.0064537, -0.0035992, -0.0027520, 0.0027520
7: -0.0016628, 0.0032800, -0.0016628, 0.0032800, -0.0045618, 0.0045618
8: -0.0007646, -0.0000391, -0.0007646, -0.0000391, -0.0007158, 0.0007158
9: 0.9993789, 1.0108052, 0.9993789, 1.0108052, -0.0076333, 0.0076333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051120, upper bound: 0.0049500
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050547, upper bound: 0.0049968
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0033333, -0.0005338, -0.0027801, 0.0027584
1: -0.0039139, 0.0027547, -0.0039051, 0.0028545, -0.0039166, 0.0038164
2: 0.0035188, 0.0089843, 0.0035436, 0.0091325, -0.0033654, 0.0032116
3: -0.0044068, -0.0035774, -0.0044122, -0.0035194, -0.0008874, 0.0008348
4: 0.0023096, 0.0072495, 0.0021643, 0.0072807, -0.0044350, 0.0045348
5: -0.0024314, 0.0024635, -0.0024637, 0.0025546, -0.0041713, 0.0041224
6: -0.0064537, -0.0035992, -0.0064812, -0.0035014, -0.0028484, 0.0027809
7: -0.0016628, 0.0032800, -0.0019011, 0.0032890, -0.0046108, 0.0048358
8: -0.0007646, -0.0000391, -0.0008063, -0.0000452, -0.0007194, 0.0007672
9: 0.9993789, 1.0108052, 0.9991218, 1.0107985, -0.0076769, 0.0078994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051120, upper bound: 0.0049500
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050547, upper bound: 0.0049968
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033266, -0.0005351, -0.0032741, -0.0005805, -0.0027448, 0.0027390
1: -0.0039036, 0.0028462, -0.0039274, 0.0026765, -0.0037556, 0.0039065
2: 0.0035445, 0.0091288, 0.0034727, 0.0089552, -0.0031778, 0.0034102
3: -0.0044085, -0.0035199, -0.0043905, -0.0035790, -0.0008295, 0.0008706
4: 0.0021667, 0.0072778, 0.0023307, 0.0072982, -0.0045456, 0.0043968
5: -0.0024611, 0.0025500, -0.0024637, 0.0024213, -0.0040602, 0.0041681
6: -0.0064781, -0.0035023, -0.0064743, -0.0036065, -0.0027650, 0.0028577
7: -0.0018995, 0.0032829, -0.0016544, 0.0033157, -0.0048694, 0.0046018
8: -0.0008058, -0.0000483, -0.0007635, -0.0000603, -0.0007456, 0.0007148
9: 0.9991314, 1.0107950, 0.9994619, 1.0108936, -0.0079347, 0.0075835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0050194
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0050774
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0032904, -0.0005422, -0.0032259, -0.0005107, -0.0027797, 0.0026837
1: -0.0038946, 0.0027733, -0.0039718, 0.0025366, -0.0037078, 0.0040239
2: 0.0035496, 0.0090957, 0.0034661, 0.0088936, -0.0031615, 0.0033672
3: -0.0043986, -0.0035221, -0.0043937, -0.0035120, -0.0008866, 0.0008717
4: 0.0021858, 0.0072596, 0.0023433, 0.0072933, -0.0045076, 0.0044040
5: -0.0024446, 0.0025100, -0.0024757, 0.0023506, -0.0040598, 0.0041401
6: -0.0064623, -0.0035068, -0.0064659, -0.0035860, -0.0027753, 0.0028374
7: -0.0018928, 0.0032608, -0.0017444, 0.0033059, -0.0048514, 0.0046625
8: -0.0008034, -0.0000569, -0.0008245, -0.0000594, -0.0007440, 0.0007676
9: 0.9992200, 1.0107740, 0.9996260, 1.0109274, -0.0078573, 0.0075455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039866, upper bound: 0.0034043
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050455, upper bound: 0.0050455
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0036313, -0.0005710, -0.0027523, 0.0030564
1: -0.0039139, 0.0027547, -0.0038272, 0.0034772, -0.0050433, 0.0042114
2: 0.0035188, 0.0089843, 0.0036027, 0.0094329, -0.0039310, 0.0033413
3: -0.0044068, -0.0035774, -0.0044598, -0.0035728, -0.0008340, 0.0008824
4: 0.0023096, 0.0072495, 0.0021356, 0.0072439, -0.0044116, 0.0045700
5: -0.0024314, 0.0024635, -0.0024147, 0.0027050, -0.0043465, 0.0041082
6: -0.0064537, -0.0035992, -0.0064802, -0.0035495, -0.0027993, 0.0027804
7: -0.0016628, 0.0032800, -0.0018203, 0.0032344, -0.0045805, 0.0047939
8: -0.0007646, -0.0000391, -0.0007762, -0.0000005, -0.0007627, 0.0007327
9: 0.9993789, 1.0108052, 0.9985415, 1.0106769, -0.0077917, 0.0088053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051120, upper bound: 0.0051781
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050547, upper bound: 0.0051884
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0036387, -0.0005363, -0.0027870, 0.0030638
1: -0.0039139, 0.0027547, -0.0038154, 0.0035825, -0.0051420, 0.0041996
2: 0.0035188, 0.0089843, 0.0036428, 0.0095646, -0.0041024, 0.0033311
3: -0.0044068, -0.0035774, -0.0044676, -0.0035181, -0.0008887, 0.0008902
4: 0.0023096, 0.0072495, 0.0019980, 0.0072638, -0.0044382, 0.0047041
5: -0.0024314, 0.0024635, -0.0024372, 0.0027910, -0.0044380, 0.0041406
6: -0.0064537, -0.0035992, -0.0064982, -0.0034548, -0.0028958, 0.0028018
7: -0.0016628, 0.0032800, -0.0020380, 0.0032337, -0.0046098, 0.0050368
8: -0.0007646, -0.0000391, -0.0008161, -0.0000061, -0.0007585, 0.0007770
9: 0.9993789, 1.0108052, 0.9983037, 1.0106512, -0.0078163, 0.0090823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051120, upper bound: 0.0051781
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050547, upper bound: 0.0051884
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033266, -0.0005351, -0.0035705, -0.0005767, -0.0027499, 0.0030354
1: -0.0039036, 0.0028462, -0.0038394, 0.0033845, -0.0049553, 0.0042863
2: 0.0035445, 0.0091288, 0.0035640, 0.0093964, -0.0039217, 0.0035646
3: -0.0044085, -0.0035199, -0.0044440, -0.0035743, -0.0008341, 0.0009241
4: 0.0021667, 0.0072778, 0.0021587, 0.0072936, -0.0045608, 0.0045713
5: -0.0024611, 0.0025500, -0.0024499, 0.0026615, -0.0043233, 0.0041957
6: -0.0064781, -0.0035023, -0.0064974, -0.0035568, -0.0028125, 0.0028834
7: -0.0018995, 0.0032829, -0.0018110, 0.0032697, -0.0048853, 0.0048329
8: -0.0008058, -0.0000483, -0.0007755, -0.0000208, -0.0007851, 0.0007272
9: 0.9991314, 1.0107950, 0.9986401, 1.0107657, -0.0080934, 0.0087441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0052287
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0052458
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0032904, -0.0005422, -0.0035327, -0.0004941, -0.0027963, 0.0029905
1: -0.0038946, 0.0027733, -0.0039247, 0.0032870, -0.0049005, 0.0043819
2: 0.0035496, 0.0090957, 0.0035427, 0.0093514, -0.0038961, 0.0035427
3: -0.0043986, -0.0035221, -0.0044472, -0.0035057, -0.0008929, 0.0009251
4: 0.0021858, 0.0072596, 0.0021699, 0.0072936, -0.0045291, 0.0045775
5: -0.0024446, 0.0025100, -0.0024665, 0.0025930, -0.0043201, 0.0041796
6: -0.0064623, -0.0035068, -0.0064901, -0.0035363, -0.0028235, 0.0028645
7: -0.0018928, 0.0032608, -0.0019014, 0.0032620, -0.0048686, 0.0048792
8: -0.0008034, -0.0000569, -0.0008356, -0.0000212, -0.0007821, 0.0007788
9: 0.9992200, 1.0107740, 0.9987696, 1.0108156, -0.0080407, 0.0086964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039866, upper bound: 0.0037923
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050455, upper bound: 0.0052126
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0033232, -0.0005749, -0.0027613, 0.0027968
1: -0.0041685, 0.0029345, -0.0039139, 0.0027547, -0.0042083, 0.0041408
2: 0.0032797, 0.0091512, 0.0035188, 0.0089843, -0.0034799, 0.0034541
3: -0.0044138, -0.0035538, -0.0044068, -0.0035774, -0.0008365, 0.0008530
4: 0.0021825, 0.0074231, 0.0023096, 0.0072495, -0.0045327, 0.0045671
5: -0.0026427, 0.0026214, -0.0024314, 0.0024635, -0.0043135, 0.0042792
6: -0.0065378, -0.0035401, -0.0064537, -0.0035992, -0.0028269, 0.0028082
7: -0.0017391, 0.0034059, -0.0016628, 0.0032800, -0.0046543, 0.0046776
8: -0.0007959, -0.0000212, -0.0007646, -0.0000391, -0.0007567, 0.0007434
9: 0.9989762, 1.0113478, 0.9993789, 1.0108052, -0.0082047, 0.0082516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050197, upper bound: 0.0052440
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0052690
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0033333, -0.0005338, -0.0028108, 0.0028274
1: -0.0041685, 0.0029345, -0.0039051, 0.0028545, -0.0042842, 0.0041164
2: 0.0032797, 0.0091512, 0.0035436, 0.0091325, -0.0036695, 0.0034898
3: -0.0044138, -0.0035538, -0.0044122, -0.0035194, -0.0008944, 0.0008583
4: 0.0021825, 0.0074231, 0.0021643, 0.0072807, -0.0045732, 0.0047073
5: -0.0026427, 0.0026214, -0.0024637, 0.0025546, -0.0044039, 0.0043208
6: -0.0065378, -0.0035401, -0.0064812, -0.0035014, -0.0029233, 0.0028370
7: -0.0017391, 0.0034059, -0.0019011, 0.0032890, -0.0047032, 0.0049516
8: -0.0007959, -0.0000212, -0.0008063, -0.0000452, -0.0007506, 0.0007851
9: 0.9989762, 1.0113478, 0.9991218, 1.0107985, -0.0082484, 0.0085178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050197, upper bound: 0.0052440
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0052691
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033709, -0.0004526, -0.0032741, -0.0005805, -0.0027722, 0.0028202
1: -0.0041560, 0.0030233, -0.0039274, 0.0026765, -0.0041205, 0.0041884
2: 0.0033238, 0.0092772, 0.0034727, 0.0089552, -0.0034524, 0.0036494
3: -0.0044165, -0.0034958, -0.0043905, -0.0035790, -0.0008376, 0.0008947
4: 0.0020514, 0.0074356, 0.0023307, 0.0072982, -0.0046716, 0.0045458
5: -0.0026542, 0.0027041, -0.0024637, 0.0024213, -0.0042669, 0.0043579
6: -0.0065462, -0.0034476, -0.0064743, -0.0036065, -0.0028274, 0.0029070
7: -0.0019537, 0.0033690, -0.0016544, 0.0033157, -0.0049194, 0.0046876
8: -0.0008375, -0.0000292, -0.0007635, -0.0000603, -0.0007773, 0.0007343
9: 0.9987615, 1.0112870, 0.9994619, 1.0108936, -0.0084738, 0.0081499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053086
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053356
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0033348, -0.0004596, -0.0032259, -0.0005107, -0.0028241, 0.0027662
1: -0.0041477, 0.0029533, -0.0039718, 0.0025366, -0.0040730, 0.0043072
2: 0.0033287, 0.0092457, 0.0034661, 0.0088936, -0.0034357, 0.0036184
3: -0.0044070, -0.0034980, -0.0043937, -0.0035120, -0.0008949, 0.0008958
4: 0.0020698, 0.0074194, 0.0023433, 0.0072933, -0.0046365, 0.0045539
5: -0.0026399, 0.0026661, -0.0024757, 0.0023506, -0.0042674, 0.0043333
6: -0.0065310, -0.0034520, -0.0064659, -0.0035860, -0.0028382, 0.0028871
7: -0.0019470, 0.0033445, -0.0017444, 0.0033059, -0.0049015, 0.0047462
8: -0.0008352, -0.0000374, -0.0008245, -0.0000594, -0.0007758, 0.0007871
9: 0.9988433, 1.0112675, 0.9996260, 1.0109274, -0.0084112, 0.0081116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0036646, upper bound: 0.0032867
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0053061
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0036313, -0.0005710, -0.0027998, 0.0031423
1: -0.0041685, 0.0029345, -0.0038272, 0.0034772, -0.0054108, 0.0045115
2: 0.0032797, 0.0091512, 0.0036027, 0.0094329, -0.0042351, 0.0036195
3: -0.0044138, -0.0035538, -0.0044598, -0.0035728, -0.0008410, 0.0009059
4: 0.0021825, 0.0074231, 0.0021356, 0.0072439, -0.0045498, 0.0047425
5: -0.0026427, 0.0026214, -0.0024147, 0.0027050, -0.0045791, 0.0043066
6: -0.0065378, -0.0035401, -0.0064802, -0.0035495, -0.0028741, 0.0028365
7: -0.0017391, 0.0034059, -0.0018203, 0.0032344, -0.0046729, 0.0049097
8: -0.0007959, -0.0000212, -0.0007762, -0.0000005, -0.0007954, 0.0007550
9: 0.9989762, 1.0113478, 0.9985415, 1.0106769, -0.0083631, 0.0094237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050197, upper bound: 0.0054678
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0054692
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0036387, -0.0005363, -0.0028345, 0.0031497
1: -0.0041685, 0.0029345, -0.0038154, 0.0035825, -0.0055096, 0.0044996
2: 0.0032797, 0.0091512, 0.0036428, 0.0095646, -0.0044064, 0.0036093
3: -0.0044138, -0.0035538, -0.0044676, -0.0035181, -0.0008957, 0.0009137
4: 0.0021825, 0.0074231, 0.0019980, 0.0072638, -0.0045763, 0.0048766
5: -0.0026427, 0.0026214, -0.0024372, 0.0027910, -0.0046706, 0.0043390
6: -0.0065378, -0.0035401, -0.0064982, -0.0034548, -0.0029706, 0.0028580
7: -0.0017391, 0.0034059, -0.0020380, 0.0032337, -0.0047022, 0.0051526
8: -0.0007959, -0.0000212, -0.0008161, -0.0000061, -0.0007898, 0.0007949
9: 0.9989762, 1.0113478, 0.9983037, 1.0106512, -0.0083878, 0.0097007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050197, upper bound: 0.0054678
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0054692
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033709, -0.0004526, -0.0035705, -0.0005767, -0.0027942, 0.0031179
1: -0.0041560, 0.0030233, -0.0038394, 0.0033845, -0.0053202, 0.0045682
2: 0.0033238, 0.0092772, 0.0035640, 0.0093964, -0.0041963, 0.0038039
3: -0.0044165, -0.0034958, -0.0044440, -0.0035743, -0.0008422, 0.0009482
4: 0.0020514, 0.0074356, 0.0021587, 0.0072936, -0.0046868, 0.0047203
5: -0.0026542, 0.0027041, -0.0024499, 0.0026615, -0.0045299, 0.0043855
6: -0.0065462, -0.0034476, -0.0064974, -0.0035568, -0.0028749, 0.0029328
7: -0.0019537, 0.0033690, -0.0018110, 0.0032697, -0.0049352, 0.0049187
8: -0.0008375, -0.0000292, -0.0007755, -0.0000208, -0.0008167, 0.0007464
9: 0.9987615, 1.0112870, 0.9986401, 1.0107657, -0.0086326, 0.0093106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055087
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055205
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0033348, -0.0004596, -0.0035327, -0.0004941, -0.0028407, 0.0030731
1: -0.0041477, 0.0029533, -0.0039247, 0.0032870, -0.0052658, 0.0046652
2: 0.0033287, 0.0092457, 0.0035427, 0.0093514, -0.0041704, 0.0037939
3: -0.0044070, -0.0034980, -0.0044472, -0.0035057, -0.0009012, 0.0009492
4: 0.0020698, 0.0074194, 0.0021699, 0.0072936, -0.0046579, 0.0047274
5: -0.0026399, 0.0026661, -0.0024665, 0.0025930, -0.0045276, 0.0043727
6: -0.0065310, -0.0034520, -0.0064901, -0.0035363, -0.0028863, 0.0029143
7: -0.0019470, 0.0033445, -0.0019014, 0.0032620, -0.0049187, 0.0049630
8: -0.0008352, -0.0000374, -0.0008356, -0.0000212, -0.0008140, 0.0007982
9: 0.9988433, 1.0112675, 0.9987696, 1.0108156, -0.0085946, 0.0092625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0036654, upper bound: 0.0036584
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0054900
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0033708, -0.0004890, -0.0027968, 0.0027613
1: -0.0039139, 0.0027547, -0.0041685, 0.0029345, -0.0041408, 0.0042083
2: 0.0035188, 0.0089843, 0.0032797, 0.0091512, -0.0034541, 0.0034799
3: -0.0044068, -0.0035774, -0.0044138, -0.0035538, -0.0008530, 0.0008365
4: 0.0023096, 0.0072495, 0.0021825, 0.0074231, -0.0045671, 0.0045327
5: -0.0024314, 0.0024635, -0.0026427, 0.0026214, -0.0042792, 0.0043135
6: -0.0064537, -0.0035992, -0.0065378, -0.0035401, -0.0028082, 0.0028269
7: -0.0016628, 0.0032800, -0.0017391, 0.0034059, -0.0046776, 0.0046543
8: -0.0007646, -0.0000391, -0.0007959, -0.0000212, -0.0007434, 0.0007567
9: 0.9993789, 1.0108052, 0.9989762, 1.0113478, -0.0082516, 0.0082047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053667, upper bound: 0.0048355
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053142, upper bound: 0.0048925
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0033777, -0.0004513, -0.0028491, 0.0027890
1: -0.0039139, 0.0027547, -0.0041575, 0.0030323, -0.0041991, 0.0041658
2: 0.0035188, 0.0089843, 0.0033229, 0.0092811, -0.0036068, 0.0034674
3: -0.0044068, -0.0035774, -0.0044203, -0.0034953, -0.0009115, 0.0008429
4: 0.0023096, 0.0072495, 0.0020487, 0.0074385, -0.0045838, 0.0046614
5: -0.0024314, 0.0024635, -0.0026567, 0.0027090, -0.0043618, 0.0043283
6: -0.0064537, -0.0035992, -0.0065492, -0.0034466, -0.0028979, 0.0028430
7: -0.0016628, 0.0032800, -0.0019553, 0.0033748, -0.0046894, 0.0048860
8: -0.0007646, -0.0000391, -0.0008380, -0.0000260, -0.0007386, 0.0007989
9: 0.9993789, 1.0108052, 0.9987509, 1.0112903, -0.0082361, 0.0084416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053667, upper bound: 0.0048355
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053142, upper bound: 0.0048925
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033266, -0.0005351, -0.0033174, -0.0004942, -0.0028104, 0.0027772
1: -0.0039036, 0.0028462, -0.0041829, 0.0028498, -0.0040496, 0.0042671
2: 0.0035445, 0.0091288, 0.0032395, 0.0091180, -0.0034407, 0.0036998
3: -0.0044085, -0.0035199, -0.0043977, -0.0035553, -0.0008532, 0.0008778
4: 0.0021667, 0.0072778, 0.0022062, 0.0074727, -0.0047152, 0.0045298
5: -0.0024611, 0.0025500, -0.0026734, 0.0025751, -0.0042502, 0.0043967
6: -0.0064781, -0.0035023, -0.0065582, -0.0035483, -0.0028197, 0.0029332
7: -0.0018995, 0.0032829, -0.0017292, 0.0034301, -0.0049787, 0.0046912
8: -0.0008058, -0.0000483, -0.0007948, -0.0000422, -0.0007636, 0.0007465
9: 0.9991314, 1.0107950, 0.9990715, 1.0114216, -0.0085359, 0.0081274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053086, upper bound: 0.0048953
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053086, upper bound: 0.0049693
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0032904, -0.0005422, -0.0032717, -0.0004212, -0.0028692, 0.0027295
1: -0.0038946, 0.0027733, -0.0042401, 0.0027191, -0.0040052, 0.0044013
2: 0.0035496, 0.0090957, 0.0032216, 0.0090675, -0.0034179, 0.0036948
3: -0.0043986, -0.0035221, -0.0044028, -0.0034854, -0.0009132, 0.0008807
4: 0.0021858, 0.0072596, 0.0022166, 0.0074748, -0.0046847, 0.0045319
5: -0.0024446, 0.0025100, -0.0027010, 0.0025135, -0.0042412, 0.0043918
6: -0.0064623, -0.0035068, -0.0065513, -0.0035298, -0.0028279, 0.0029149
7: -0.0018928, 0.0032608, -0.0018149, 0.0034146, -0.0049579, 0.0047402
8: -0.0008034, -0.0000569, -0.0008594, -0.0000410, -0.0007624, 0.0008026
9: 0.9992200, 1.0107740, 0.9992105, 1.0114871, -0.0085212, 0.0080715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039981, upper bound: 0.0032163
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053060, upper bound: 0.0049377
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0036792, -0.0004866, -0.0028367, 0.0031043
1: -0.0039139, 0.0027547, -0.0040906, 0.0036655, -0.0052341, 0.0044753
2: 0.0035188, 0.0089843, 0.0033639, 0.0095964, -0.0041206, 0.0035683
3: -0.0044068, -0.0035774, -0.0044669, -0.0035499, -0.0008570, 0.0008895
4: 0.0023096, 0.0072495, 0.0020077, 0.0074119, -0.0045701, 0.0046996
5: -0.0024314, 0.0024635, -0.0026266, 0.0028738, -0.0045314, 0.0043155
6: -0.0064537, -0.0035992, -0.0065589, -0.0034874, -0.0028556, 0.0028495
7: -0.0016628, 0.0032800, -0.0019106, 0.0033605, -0.0046749, 0.0048603
8: -0.0007646, -0.0000391, -0.0008055, 0.0000177, -0.0007823, 0.0007664
9: 0.9993789, 1.0108052, 0.9981174, 1.0112350, -0.0082985, 0.0092618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053667, upper bound: 0.0050451
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053142, upper bound: 0.0050661
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033232, -0.0005749, -0.0036869, -0.0004558, -0.0028675, 0.0031120
1: -0.0039139, 0.0027547, -0.0040733, 0.0037617, -0.0053125, 0.0044409
2: 0.0035188, 0.0089843, 0.0034106, 0.0097233, -0.0042657, 0.0035431
3: -0.0044068, -0.0035774, -0.0044745, -0.0034944, -0.0009124, 0.0008971
4: 0.0023096, 0.0072495, 0.0018748, 0.0074206, -0.0045785, 0.0048240
5: -0.0024314, 0.0024635, -0.0026369, 0.0029579, -0.0046100, 0.0043234
6: -0.0064537, -0.0035992, -0.0065684, -0.0033950, -0.0029432, 0.0028623
7: -0.0016628, 0.0032800, -0.0021094, 0.0033303, -0.0046743, 0.0050800
8: -0.0007646, -0.0000391, -0.0008459, 0.0000125, -0.0007771, 0.0008067
9: 0.9993789, 1.0108052, 0.9978935, 1.0111725, -0.0082750, 0.0095045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053667, upper bound: 0.0050451
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053142, upper bound: 0.0050661
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033266, -0.0005351, -0.0036152, -0.0004921, -0.0028346, 0.0030801
1: -0.0039036, 0.0028462, -0.0041026, 0.0035671, -0.0051343, 0.0045408
2: 0.0035445, 0.0091288, 0.0033249, 0.0095576, -0.0041000, 0.0037871
3: -0.0044085, -0.0035199, -0.0044507, -0.0035512, -0.0008573, 0.0009308
4: 0.0021667, 0.0072778, 0.0020335, 0.0074607, -0.0047133, 0.0046950
5: -0.0024611, 0.0025500, -0.0026573, 0.0028254, -0.0045001, 0.0043965
6: -0.0064781, -0.0035023, -0.0065760, -0.0034958, -0.0028671, 0.0029508
7: -0.0018995, 0.0032829, -0.0019001, 0.0033862, -0.0049674, 0.0048965
8: -0.0008058, -0.0000483, -0.0008048, -0.0000030, -0.0008028, 0.0007565
9: 0.9991314, 1.0107950, 0.9982234, 1.0113095, -0.0085899, 0.0091781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0050937
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0051407
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0032904, -0.0005422, -0.0035823, -0.0004043, -0.0028861, 0.0030401
1: -0.0038946, 0.0027733, -0.0041958, 0.0034754, -0.0050888, 0.0046701
2: 0.0035496, 0.0090957, 0.0032977, 0.0095160, -0.0040728, 0.0037852
3: -0.0043986, -0.0035221, -0.0044564, -0.0034786, -0.0009201, 0.0009344
4: 0.0021858, 0.0072596, 0.0020420, 0.0074682, -0.0046887, 0.0046947
5: -0.0024446, 0.0025100, -0.0026906, 0.0027678, -0.0044915, 0.0043961
6: -0.0064623, -0.0035068, -0.0065714, -0.0034767, -0.0028761, 0.0029345
7: -0.0018928, 0.0032608, -0.0019830, 0.0033746, -0.0049506, 0.0049366
8: -0.0008034, -0.0000569, -0.0008673, -0.0000032, -0.0008002, 0.0008104
9: 0.9992200, 1.0107740, 0.9983398, 1.0113953, -0.0085998, 0.0091211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039981, upper bound: 0.0035810
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053060, upper bound: 0.0051086
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0033708, -0.0004890, -0.0028424, 0.0028424
1: -0.0041685, 0.0029345, -0.0041685, 0.0029345, -0.0040054, 0.0040054
2: 0.0032797, 0.0091512, 0.0032797, 0.0091512, -0.0033307, 0.0033307
3: -0.0044138, -0.0035538, -0.0044138, -0.0035538, -0.0008600, 0.0008600
4: 0.0021825, 0.0074231, 0.0021825, 0.0074231, -0.0045805, 0.0045805
5: -0.0026427, 0.0026214, -0.0026427, 0.0026214, -0.0042805, 0.0042805
6: -0.0065378, -0.0035401, -0.0065378, -0.0035401, -0.0028519, 0.0028519
7: -0.0017391, 0.0034059, -0.0017391, 0.0034059, -0.0047073, 0.0047073
8: -0.0007959, -0.0000212, -0.0007959, -0.0000212, -0.0007313, 0.0007313
9: 0.9989762, 1.0113478, 0.9989762, 1.0113478, -0.0080348, 0.0080348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050198, upper bound: 0.0052440
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0052690
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0033777, -0.0004513, -0.0028892, 0.0028622
1: -0.0041685, 0.0029345, -0.0041575, 0.0030323, -0.0040735, 0.0039828
2: 0.0032797, 0.0091512, 0.0033229, 0.0092811, -0.0035067, 0.0033471
3: -0.0044138, -0.0035538, -0.0044203, -0.0034953, -0.0009185, 0.0008664
4: 0.0021825, 0.0074231, 0.0020487, 0.0074385, -0.0046044, 0.0047154
5: -0.0026427, 0.0026214, -0.0026567, 0.0027090, -0.0043702, 0.0043053
6: -0.0065378, -0.0035401, -0.0065492, -0.0034466, -0.0029446, 0.0028698
7: -0.0017391, 0.0034059, -0.0019553, 0.0033748, -0.0047262, 0.0049513
8: -0.0007959, -0.0000212, -0.0008380, -0.0000260, -0.0007393, 0.0007881
9: 0.9989762, 1.0113478, 0.9987509, 1.0112903, -0.0080422, 0.0082915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050198, upper bound: 0.0052440
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0052691
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033709, -0.0004526, -0.0033174, -0.0004942, -0.0028455, 0.0028591
1: -0.0041560, 0.0030233, -0.0041829, 0.0028498, -0.0039200, 0.0040638
2: 0.0033238, 0.0092772, 0.0032395, 0.0091180, -0.0033091, 0.0035433
3: -0.0044165, -0.0034958, -0.0043977, -0.0035553, -0.0008613, 0.0009019
4: 0.0020514, 0.0074356, 0.0022062, 0.0074727, -0.0047260, 0.0045616
5: -0.0026542, 0.0027041, -0.0026734, 0.0025751, -0.0042384, 0.0043665
6: -0.0065462, -0.0034476, -0.0065582, -0.0035483, -0.0028527, 0.0029546
7: -0.0019537, 0.0033690, -0.0017292, 0.0034301, -0.0049807, 0.0047201
8: -0.0008375, -0.0000292, -0.0007948, -0.0000422, -0.0007717, 0.0007296
9: 0.9987615, 1.0112870, 0.9990715, 1.0114216, -0.0083161, 0.0079362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053086
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053356
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0033348, -0.0004596, -0.0032717, -0.0004212, -0.0029136, 0.0028121
1: -0.0041477, 0.0029533, -0.0042401, 0.0027191, -0.0038847, 0.0041722
2: 0.0033287, 0.0092457, 0.0032216, 0.0090675, -0.0032987, 0.0035143
3: -0.0044070, -0.0034980, -0.0044028, -0.0034854, -0.0009215, 0.0009048
4: 0.0020698, 0.0074194, 0.0022166, 0.0074748, -0.0046965, 0.0045705
5: -0.0026399, 0.0026661, -0.0027010, 0.0025135, -0.0042389, 0.0043464
6: -0.0065310, -0.0034520, -0.0065513, -0.0035298, -0.0028648, 0.0029366
7: -0.0019470, 0.0033445, -0.0018149, 0.0034146, -0.0049616, 0.0047712
8: -0.0008352, -0.0000374, -0.0008594, -0.0000410, -0.0007760, 0.0007934
9: 0.9988433, 1.0112675, 0.9992105, 1.0114871, -0.0082700, 0.0079086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0036646, upper bound: 0.0032084
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0053061
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0036792, -0.0004866, -0.0028842, 0.0031821
1: -0.0041685, 0.0029345, -0.0040906, 0.0036655, -0.0052003, 0.0043749
2: 0.0032797, 0.0091512, 0.0033639, 0.0095964, -0.0040787, 0.0034818
3: -0.0044138, -0.0035538, -0.0044669, -0.0035499, -0.0008640, 0.0009130
4: 0.0021825, 0.0074231, 0.0020077, 0.0074119, -0.0045836, 0.0047435
5: -0.0026427, 0.0026214, -0.0026266, 0.0028738, -0.0045408, 0.0042938
6: -0.0065378, -0.0035401, -0.0065589, -0.0034874, -0.0029003, 0.0028749
7: -0.0017391, 0.0034059, -0.0019106, 0.0033605, -0.0047108, 0.0049194
8: -0.0007959, -0.0000212, -0.0008055, 0.0000177, -0.0007788, 0.0007464
9: 0.9989762, 1.0113478, 0.9981174, 1.0112350, -0.0081607, 0.0091780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050198, upper bound: 0.0054678
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0054692
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0033708, -0.0004890, -0.0036869, -0.0004558, -0.0029150, 0.0031980
1: -0.0041685, 0.0029345, -0.0040733, 0.0037617, -0.0052957, 0.0043632
2: 0.0032797, 0.0091512, 0.0034106, 0.0097233, -0.0042425, 0.0034670
3: -0.0044138, -0.0035538, -0.0044745, -0.0034944, -0.0009194, 0.0009206
4: 0.0021825, 0.0074231, 0.0018748, 0.0074206, -0.0046006, 0.0048774
5: -0.0026427, 0.0026214, -0.0026369, 0.0029579, -0.0046295, 0.0043156
6: -0.0065378, -0.0035401, -0.0065684, -0.0033950, -0.0029888, 0.0028874
7: -0.0017391, 0.0034059, -0.0021094, 0.0033303, -0.0047171, 0.0051481
8: -0.0007959, -0.0000212, -0.0008459, 0.0000125, -0.0007860, 0.0008023
9: 0.9989762, 1.0113478, 0.9978935, 1.0111725, -0.0081618, 0.0094422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050198, upper bound: 0.0054678
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0054692
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0033709, -0.0004526, -0.0036152, -0.0004921, -0.0028788, 0.0031625
1: -0.0041560, 0.0030233, -0.0041026, 0.0035671, -0.0051079, 0.0044422
2: 0.0033238, 0.0092772, 0.0033249, 0.0095576, -0.0040455, 0.0036910
3: -0.0044165, -0.0034958, -0.0044507, -0.0035512, -0.0008653, 0.0009549
4: 0.0020514, 0.0074356, 0.0020335, 0.0074607, -0.0047256, 0.0047250
5: -0.0026542, 0.0027041, -0.0026573, 0.0028254, -0.0044988, 0.0043800
6: -0.0065462, -0.0034476, -0.0065760, -0.0034958, -0.0029014, 0.0029736
7: -0.0019537, 0.0033690, -0.0019001, 0.0033862, -0.0049775, 0.0049317
8: -0.0008375, -0.0000292, -0.0008048, -0.0000030, -0.0008182, 0.0007450
9: 0.9987615, 1.0112870, 0.9982234, 1.0113095, -0.0084459, 0.0090787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055087
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055206
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0033348, -0.0004596, -0.0035823, -0.0004043, -0.0029305, 0.0031227
1: -0.0041477, 0.0029533, -0.0041958, 0.0034754, -0.0050687, 0.0045328
2: 0.0033287, 0.0092457, 0.0032977, 0.0095160, -0.0040275, 0.0036802
3: -0.0044070, -0.0034980, -0.0044564, -0.0034786, -0.0009284, 0.0009584
4: 0.0020698, 0.0074194, 0.0020420, 0.0074682, -0.0047019, 0.0047344
5: -0.0026399, 0.0026661, -0.0026906, 0.0027678, -0.0044981, 0.0043699
6: -0.0065310, -0.0034520, -0.0065714, -0.0034767, -0.0029129, 0.0029572
7: -0.0019470, 0.0033445, -0.0019830, 0.0033746, -0.0049619, 0.0049743
8: -0.0008352, -0.0000374, -0.0008673, -0.0000032, -0.0008227, 0.0008088
9: 0.9988433, 1.0112675, 0.9983398, 1.0113953, -0.0084179, 0.0090485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0036654, upper bound: 0.0035831
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0054901
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0033232, -0.0005749, -0.0030564, 0.0027523
1: -0.0038272, 0.0034772, -0.0039139, 0.0027547, -0.0042114, 0.0050433
2: 0.0036027, 0.0094329, 0.0035188, 0.0089843, -0.0033413, 0.0039310
3: -0.0044598, -0.0035728, -0.0044068, -0.0035774, -0.0008824, 0.0008340
4: 0.0021356, 0.0072439, 0.0023096, 0.0072495, -0.0045700, 0.0044116
5: -0.0024147, 0.0027050, -0.0024314, 0.0024635, -0.0041082, 0.0043465
6: -0.0064802, -0.0035495, -0.0064537, -0.0035992, -0.0027804, 0.0027993
7: -0.0018203, 0.0032344, -0.0016628, 0.0032800, -0.0047939, 0.0045805
8: -0.0007762, -0.0000005, -0.0007646, -0.0000391, -0.0007327, 0.0007627
9: 0.9985415, 1.0106769, 0.9993789, 1.0108052, -0.0088053, 0.0077917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052587, upper bound: 0.0049500
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052227, upper bound: 0.0049968
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0033333, -0.0005338, -0.0030975, 0.0027623
1: -0.0038272, 0.0034772, -0.0039051, 0.0028545, -0.0042873, 0.0050190
2: 0.0036027, 0.0094329, 0.0035436, 0.0091325, -0.0035308, 0.0039667
3: -0.0044598, -0.0035728, -0.0044122, -0.0035194, -0.0009404, 0.0008394
4: 0.0021356, 0.0072439, 0.0021643, 0.0072807, -0.0046105, 0.0045519
5: -0.0024147, 0.0027050, -0.0024637, 0.0025546, -0.0041986, 0.0043880
6: -0.0064802, -0.0035495, -0.0064812, -0.0035014, -0.0028768, 0.0028282
7: -0.0018203, 0.0032344, -0.0019011, 0.0032890, -0.0048428, 0.0048544
8: -0.0007762, -0.0000005, -0.0008063, -0.0000452, -0.0007310, 0.0008059
9: 0.9985415, 1.0106769, 0.9991218, 1.0107985, -0.0088490, 0.0080578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052587, upper bound: 0.0049500
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052227, upper bound: 0.0049968
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036314, -0.0005377, -0.0032741, -0.0005805, -0.0030508, 0.0027364
1: -0.0038139, 0.0035724, -0.0039274, 0.0026765, -0.0041265, 0.0051323
2: 0.0036437, 0.0095603, 0.0034727, 0.0089552, -0.0033194, 0.0041452
3: -0.0044638, -0.0035186, -0.0043905, -0.0035790, -0.0008849, 0.0008720
4: 0.0020007, 0.0072610, 0.0023307, 0.0072982, -0.0047146, 0.0044004
5: -0.0024348, 0.0027862, -0.0024637, 0.0024213, -0.0040785, 0.0044344
6: -0.0064949, -0.0034557, -0.0064743, -0.0036065, -0.0027861, 0.0029050
7: -0.0020364, 0.0032274, -0.0016544, 0.0033157, -0.0050702, 0.0046053
8: -0.0008156, -0.0000092, -0.0007635, -0.0000603, -0.0007554, 0.0007543
9: 0.9983147, 1.0106479, 0.9994619, 1.0108936, -0.0091153, 0.0077197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052287, upper bound: 0.0050194
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052287, upper bound: 0.0050774
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0035965, -0.0005449, -0.0032259, -0.0005107, -0.0030859, 0.0026810
1: -0.0038056, 0.0035105, -0.0039718, 0.0025366, -0.0040779, 0.0052406
2: 0.0036486, 0.0095299, 0.0034661, 0.0088936, -0.0033034, 0.0041085
3: -0.0044536, -0.0035206, -0.0043937, -0.0035120, -0.0009416, 0.0008731
4: 0.0020201, 0.0072449, 0.0023433, 0.0072933, -0.0046783, 0.0044084
5: -0.0024205, 0.0027468, -0.0024757, 0.0023506, -0.0040786, 0.0044078
6: -0.0064789, -0.0034600, -0.0064659, -0.0035860, -0.0027959, 0.0028851
7: -0.0020297, 0.0032048, -0.0017444, 0.0033059, -0.0050521, 0.0046643
8: -0.0008135, -0.0000178, -0.0008245, -0.0000594, -0.0007541, 0.0008067
9: 0.9983972, 1.0106291, 0.9996260, 1.0109274, -0.0090431, 0.0076832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052458, upper bound: 0.0050194
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052458, upper bound: 0.0050774
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0033708, -0.0004890, -0.0031423, 0.0027998
1: -0.0038272, 0.0034772, -0.0041685, 0.0029345, -0.0045115, 0.0054108
2: 0.0036027, 0.0094329, 0.0032797, 0.0091512, -0.0036195, 0.0042351
3: -0.0044598, -0.0035728, -0.0044138, -0.0035538, -0.0009059, 0.0008410
4: 0.0021356, 0.0072439, 0.0021825, 0.0074231, -0.0047425, 0.0045498
5: -0.0024147, 0.0027050, -0.0026427, 0.0026214, -0.0043066, 0.0045791
6: -0.0064802, -0.0035495, -0.0065378, -0.0035401, -0.0028365, 0.0028741
7: -0.0018203, 0.0032344, -0.0017391, 0.0034059, -0.0049097, 0.0046729
8: -0.0007762, -0.0000005, -0.0007959, -0.0000212, -0.0007550, 0.0007954
9: 0.9985415, 1.0106769, 0.9989762, 1.0113478, -0.0094237, 0.0083631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055314, upper bound: 0.0048330
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055010, upper bound: 0.0048925
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0033777, -0.0004513, -0.0031800, 0.0028068
1: -0.0038272, 0.0034772, -0.0041575, 0.0030323, -0.0045698, 0.0053684
2: 0.0036027, 0.0094329, 0.0033229, 0.0092811, -0.0037722, 0.0042226
3: -0.0044598, -0.0035728, -0.0044203, -0.0034953, -0.0009645, 0.0008474
4: 0.0021356, 0.0072439, 0.0020487, 0.0074385, -0.0047592, 0.0046785
5: -0.0024147, 0.0027050, -0.0026567, 0.0027090, -0.0043892, 0.0045939
6: -0.0064802, -0.0035495, -0.0065492, -0.0034466, -0.0029263, 0.0028902
7: -0.0018203, 0.0032344, -0.0019553, 0.0033748, -0.0049215, 0.0049047
8: -0.0007762, -0.0000005, -0.0008380, -0.0000260, -0.0007502, 0.0008375
9: 0.9985415, 1.0106769, 0.9987509, 1.0112903, -0.0094081, 0.0086001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055314, upper bound: 0.0048330
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055010, upper bound: 0.0048925
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036314, -0.0005377, -0.0033174, -0.0004942, -0.0031371, 0.0027797
1: -0.0038139, 0.0035724, -0.0041829, 0.0028498, -0.0044204, 0.0054929
2: 0.0036437, 0.0095603, 0.0032395, 0.0091180, -0.0035823, 0.0044348
3: -0.0044638, -0.0035186, -0.0043977, -0.0035553, -0.0009086, 0.0008791
4: 0.0020007, 0.0072610, 0.0022062, 0.0074727, -0.0048842, 0.0045334
5: -0.0024348, 0.0027862, -0.0026734, 0.0025751, -0.0042684, 0.0046630
6: -0.0064949, -0.0034557, -0.0065582, -0.0035483, -0.0028408, 0.0029805
7: -0.0020364, 0.0032274, -0.0017292, 0.0034301, -0.0051795, 0.0046947
8: -0.0008156, -0.0000092, -0.0007948, -0.0000422, -0.0007734, 0.0007856
9: 0.9983147, 1.0106479, 0.9990715, 1.0114216, -0.0097165, 0.0082636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055087, upper bound: 0.0048952
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055087, upper bound: 0.0049693
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0035965, -0.0005449, -0.0032717, -0.0004212, -0.0031754, 0.0027268
1: -0.0038056, 0.0035105, -0.0042401, 0.0027191, -0.0043753, 0.0056180
2: 0.0036486, 0.0095299, 0.0032216, 0.0090675, -0.0035598, 0.0044361
3: -0.0044536, -0.0035206, -0.0044028, -0.0034854, -0.0009682, 0.0008822
4: 0.0020201, 0.0072449, 0.0022166, 0.0074748, -0.0048554, 0.0045362
5: -0.0024205, 0.0027468, -0.0027010, 0.0025135, -0.0042600, 0.0046594
6: -0.0064789, -0.0034600, -0.0065513, -0.0035298, -0.0028485, 0.0029626
7: -0.0020297, 0.0032048, -0.0018149, 0.0034146, -0.0051586, 0.0047420
8: -0.0008135, -0.0000178, -0.0008594, -0.0000410, -0.0007725, 0.0008416
9: 0.9983972, 1.0106291, 0.9992105, 1.0114871, -0.0097071, 0.0082092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0048952
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0049693
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0033232, -0.0005749, -0.0031043, 0.0028367
1: -0.0040906, 0.0036655, -0.0039139, 0.0027547, -0.0044753, 0.0052341
2: 0.0033639, 0.0095964, 0.0035188, 0.0089843, -0.0035683, 0.0041206
3: -0.0044669, -0.0035499, -0.0044068, -0.0035774, -0.0008895, 0.0008570
4: 0.0020077, 0.0074119, 0.0023096, 0.0072495, -0.0046996, 0.0045701
5: -0.0026266, 0.0028738, -0.0024314, 0.0024635, -0.0043155, 0.0045314
6: -0.0065589, -0.0034874, -0.0064537, -0.0035992, -0.0028495, 0.0028556
7: -0.0019106, 0.0033605, -0.0016628, 0.0032800, -0.0048603, 0.0046749
8: -0.0008055, 0.0000177, -0.0007646, -0.0000391, -0.0007664, 0.0007823
9: 0.9981174, 1.0112350, 0.9993789, 1.0108052, -0.0092618, 0.0082985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0052437
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051144, upper bound: 0.0052690
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0033333, -0.0005338, -0.0031454, 0.0028467
1: -0.0040906, 0.0036655, -0.0039051, 0.0028545, -0.0045512, 0.0052098
2: 0.0033639, 0.0095964, 0.0035436, 0.0091325, -0.0037578, 0.0041563
3: -0.0044669, -0.0035499, -0.0044122, -0.0035194, -0.0009474, 0.0008623
4: 0.0020077, 0.0074119, 0.0021643, 0.0072807, -0.0047401, 0.0047103
5: -0.0026266, 0.0028738, -0.0024637, 0.0025546, -0.0044059, 0.0045730
6: -0.0065589, -0.0034874, -0.0064812, -0.0035014, -0.0029459, 0.0028844
7: -0.0019106, 0.0033605, -0.0019011, 0.0032890, -0.0049093, 0.0049488
8: -0.0008055, 0.0000177, -0.0008063, -0.0000452, -0.0007603, 0.0008240
9: 0.9981174, 1.0112350, 0.9991218, 1.0107985, -0.0093055, 0.0085646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0052437
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051144, upper bound: 0.0052690
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036793, -0.0004573, -0.0032741, -0.0005805, -0.0030988, 0.0028168
1: -0.0040719, 0.0037512, -0.0039274, 0.0026765, -0.0043873, 0.0053013
2: 0.0034115, 0.0097190, 0.0034727, 0.0089552, -0.0035339, 0.0043078
3: -0.0044706, -0.0034949, -0.0043905, -0.0035790, -0.0008917, 0.0008956
4: 0.0018776, 0.0074179, 0.0023307, 0.0072982, -0.0048341, 0.0045414
5: -0.0026345, 0.0029528, -0.0024637, 0.0024213, -0.0042625, 0.0046058
6: -0.0065652, -0.0033960, -0.0064743, -0.0036065, -0.0028466, 0.0029523
7: -0.0021077, 0.0033243, -0.0016544, 0.0033157, -0.0051132, 0.0046721
8: -0.0008454, 0.0000094, -0.0007635, -0.0000603, -0.0007852, 0.0007728
9: 0.9979050, 1.0111693, 0.9994619, 1.0108936, -0.0095355, 0.0081902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050937, upper bound: 0.0053085
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050937, upper bound: 0.0053356
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0036457, -0.0004644, -0.0032259, -0.0005107, -0.0031350, 0.0027615
1: -0.0040642, 0.0036916, -0.0039718, 0.0025366, -0.0043394, 0.0054190
2: 0.0034161, 0.0096898, 0.0034661, 0.0088936, -0.0035179, 0.0042764
3: -0.0044612, -0.0034970, -0.0043937, -0.0035120, -0.0009492, 0.0008967
4: 0.0018963, 0.0074031, 0.0023433, 0.0072933, -0.0047995, 0.0045490
5: -0.0026217, 0.0029152, -0.0024757, 0.0023506, -0.0042629, 0.0045824
6: -0.0065499, -0.0034002, -0.0064659, -0.0035860, -0.0028571, 0.0029329
7: -0.0021012, 0.0033011, -0.0017444, 0.0033059, -0.0050953, 0.0047302
8: -0.0008434, 0.0000007, -0.0008245, -0.0000594, -0.0007840, 0.0008251
9: 0.9979852, 1.0111523, 0.9996260, 1.0109274, -0.0094765, 0.0081523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053085
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053356
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0033708, -0.0004890, -0.0031821, 0.0028842
1: -0.0040906, 0.0036655, -0.0041685, 0.0029345, -0.0043749, 0.0052003
2: 0.0033639, 0.0095964, 0.0032797, 0.0091512, -0.0034818, 0.0040787
3: -0.0044669, -0.0035499, -0.0044138, -0.0035538, -0.0009130, 0.0008640
4: 0.0020077, 0.0074119, 0.0021825, 0.0074231, -0.0047435, 0.0045836
5: -0.0026266, 0.0028738, -0.0026427, 0.0026214, -0.0042938, 0.0045408
6: -0.0065589, -0.0034874, -0.0065378, -0.0035401, -0.0028749, 0.0029003
7: -0.0019106, 0.0033605, -0.0017391, 0.0034059, -0.0049194, 0.0047108
8: -0.0008055, 0.0000177, -0.0007959, -0.0000212, -0.0007464, 0.0007788
9: 0.9981174, 1.0112350, 0.9989762, 1.0113478, -0.0091780, 0.0081607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0052437
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051144, upper bound: 0.0052690
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0033777, -0.0004513, -0.0032279, 0.0028912
1: -0.0040906, 0.0036655, -0.0041575, 0.0030323, -0.0044430, 0.0051778
2: 0.0033639, 0.0095964, 0.0033229, 0.0092811, -0.0036578, 0.0040951
3: -0.0044669, -0.0035499, -0.0044203, -0.0034953, -0.0009716, 0.0008704
4: 0.0020077, 0.0074119, 0.0020487, 0.0074385, -0.0047675, 0.0047185
5: -0.0026266, 0.0028738, -0.0026567, 0.0027090, -0.0043835, 0.0045657
6: -0.0065589, -0.0034874, -0.0065492, -0.0034466, -0.0029676, 0.0029183
7: -0.0019106, 0.0033605, -0.0019553, 0.0033748, -0.0049383, 0.0049548
8: -0.0008055, 0.0000177, -0.0008380, -0.0000260, -0.0007544, 0.0008356
9: 0.9981174, 1.0112350, 0.9987509, 1.0112903, -0.0091854, 0.0084175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0052437
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051144, upper bound: 0.0052690
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036793, -0.0004573, -0.0033174, -0.0004942, -0.0031842, 0.0028601
1: -0.0040719, 0.0037512, -0.0041829, 0.0028498, -0.0042892, 0.0052860
2: 0.0034115, 0.0097190, 0.0032395, 0.0091180, -0.0034488, 0.0042774
3: -0.0044706, -0.0034949, -0.0043977, -0.0035553, -0.0009154, 0.0009028
4: 0.0018776, 0.0074179, 0.0022062, 0.0074727, -0.0048877, 0.0045582
5: -0.0026345, 0.0029528, -0.0026734, 0.0025751, -0.0042486, 0.0046256
6: -0.0065652, -0.0033960, -0.0065582, -0.0035483, -0.0028704, 0.0029988
7: -0.0021077, 0.0033243, -0.0017292, 0.0034301, -0.0051773, 0.0047126
8: -0.0008454, 0.0000094, -0.0007948, -0.0000422, -0.0007859, 0.0007757
9: 0.9979050, 1.0111693, 0.9990715, 1.0114216, -0.0094653, 0.0080557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050938, upper bound: 0.0053085
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050938, upper bound: 0.0053356
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0036457, -0.0004644, -0.0032717, -0.0004212, -0.0032245, 0.0028073
1: -0.0040642, 0.0036916, -0.0042401, 0.0027191, -0.0042534, 0.0053878
2: 0.0034161, 0.0096898, 0.0032216, 0.0090675, -0.0034383, 0.0042537
3: -0.0044612, -0.0034970, -0.0044028, -0.0034854, -0.0009758, 0.0009057
4: 0.0018963, 0.0074031, 0.0022166, 0.0074748, -0.0048587, 0.0045669
5: -0.0026217, 0.0029152, -0.0027010, 0.0025135, -0.0042496, 0.0046056
6: -0.0065499, -0.0034002, -0.0065513, -0.0035298, -0.0028823, 0.0029813
7: -0.0021012, 0.0033011, -0.0018149, 0.0034146, -0.0051584, 0.0047632
8: -0.0008434, 0.0000007, -0.0008594, -0.0000410, -0.0007900, 0.0008391
9: 0.9979852, 1.0111523, 0.9992105, 1.0114871, -0.0094220, 0.0080307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053085
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053356
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0036313, -0.0005710, -0.0028507, 0.0028507
1: -0.0038272, 0.0034772, -0.0038272, 0.0034772, -0.0040651, 0.0040651
2: 0.0036027, 0.0094329, 0.0036027, 0.0094329, -0.0033538, 0.0033538
3: -0.0044598, -0.0035728, -0.0044598, -0.0035728, -0.0008870, 0.0008870
4: 0.0021356, 0.0072439, 0.0021356, 0.0072439, -0.0045466, 0.0045466
5: -0.0024147, 0.0027050, -0.0024147, 0.0027050, -0.0042519, 0.0042519
6: -0.0064802, -0.0035495, -0.0064802, -0.0035495, -0.0028385, 0.0028385
7: -0.0018203, 0.0032344, -0.0018203, 0.0032344, -0.0047398, 0.0047398
8: -0.0007762, -0.0000005, -0.0007762, -0.0000005, -0.0007521, 0.0007521
9: 0.9985415, 1.0106769, 0.9985415, 1.0106769, -0.0079962, 0.0079962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052836, upper bound: 0.0049916
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052509, upper bound: 0.0050427
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0036387, -0.0005363, -0.0028990, 0.0028781
1: -0.0038272, 0.0034772, -0.0038154, 0.0035825, -0.0041413, 0.0040410
2: 0.0036027, 0.0094329, 0.0036428, 0.0095646, -0.0035396, 0.0033838
3: -0.0044598, -0.0035728, -0.0044676, -0.0035181, -0.0009417, 0.0008947
4: 0.0021356, 0.0072439, 0.0019980, 0.0072638, -0.0045813, 0.0046860
5: -0.0024147, 0.0027050, -0.0024372, 0.0027910, -0.0043409, 0.0042871
6: -0.0064802, -0.0035495, -0.0064982, -0.0034548, -0.0029339, 0.0028630
7: -0.0018203, 0.0032344, -0.0020380, 0.0032337, -0.0047788, 0.0050102
8: -0.0007762, -0.0000005, -0.0008161, -0.0000061, -0.0007610, 0.0008115
9: 0.9985415, 1.0106769, 0.9983037, 1.0106512, -0.0080306, 0.0082651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052836, upper bound: 0.0049916
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052509, upper bound: 0.0050427
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036314, -0.0005377, -0.0035705, -0.0005767, -0.0028618, 0.0028646
1: -0.0038139, 0.0035724, -0.0038394, 0.0033845, -0.0039729, 0.0041313
2: 0.0036437, 0.0095603, 0.0035640, 0.0093964, -0.0033476, 0.0035791
3: -0.0044638, -0.0035186, -0.0044440, -0.0035743, -0.0008895, 0.0009255
4: 0.0020007, 0.0072610, 0.0021587, 0.0072936, -0.0046920, 0.0045428
5: -0.0024348, 0.0027862, -0.0024499, 0.0026615, -0.0042250, 0.0043322
6: -0.0064949, -0.0034557, -0.0064974, -0.0035568, -0.0028474, 0.0029395
7: -0.0020364, 0.0032274, -0.0018110, 0.0032697, -0.0050338, 0.0047668
8: -0.0008156, -0.0000092, -0.0007755, -0.0000208, -0.0007949, 0.0007509
9: 0.9983147, 1.0106479, 0.9986401, 1.0107657, -0.0082929, 0.0079357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052514, upper bound: 0.0050643
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052514, upper bound: 0.0051268
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0035965, -0.0005449, -0.0035327, -0.0004941, -0.0029662, 0.0028546
1: -0.0038056, 0.0035105, -0.0039247, 0.0032870, -0.0039501, 0.0042475
2: 0.0036486, 0.0095299, 0.0035427, 0.0093514, -0.0033431, 0.0035395
3: -0.0044536, -0.0035206, -0.0044472, -0.0035057, -0.0009479, 0.0009266
4: 0.0020201, 0.0072449, 0.0021699, 0.0072936, -0.0046559, 0.0045597
5: -0.0024205, 0.0027468, -0.0024665, 0.0025930, -0.0042340, 0.0043059
6: -0.0064789, -0.0034600, -0.0064901, -0.0035363, -0.0028603, 0.0029193
7: -0.0020297, 0.0032048, -0.0019014, 0.0032620, -0.0050158, 0.0048243
8: -0.0008135, -0.0000178, -0.0008356, -0.0000212, -0.0007922, 0.0008128
9: 0.9983972, 1.0106291, 0.9987696, 1.0108156, -0.0082185, 0.0079294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052738, upper bound: 0.0050643
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052738, upper bound: 0.0051268
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0036792, -0.0004866, -0.0029164, 0.0028800
1: -0.0038272, 0.0034772, -0.0040906, 0.0036655, -0.0043671, 0.0044337
2: 0.0036027, 0.0094329, 0.0033639, 0.0095964, -0.0036420, 0.0036470
3: -0.0044598, -0.0035728, -0.0044669, -0.0035499, -0.0009099, 0.0008940
4: 0.0021356, 0.0072439, 0.0020077, 0.0074119, -0.0047036, 0.0046886
5: -0.0024147, 0.0027050, -0.0026266, 0.0028738, -0.0044573, 0.0044673
6: -0.0064802, -0.0035495, -0.0065589, -0.0034874, -0.0028950, 0.0029052
7: -0.0018203, 0.0032344, -0.0019106, 0.0033605, -0.0048422, 0.0048320
8: -0.0007762, -0.0000005, -0.0008055, 0.0000177, -0.0007838, 0.0007908
9: 0.9985415, 1.0106769, 0.9981174, 1.0112350, -0.0085887, 0.0085880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055592, upper bound: 0.0048748
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055315, upper bound: 0.0049368
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036313, -0.0005710, -0.0036869, -0.0004558, -0.0029682, 0.0029060
1: -0.0038272, 0.0034772, -0.0040733, 0.0037617, -0.0044241, 0.0043914
2: 0.0036027, 0.0094329, 0.0034106, 0.0097233, -0.0037921, 0.0036335
3: -0.0044598, -0.0035728, -0.0044745, -0.0034944, -0.0009654, 0.0009016
4: 0.0021356, 0.0072439, 0.0018748, 0.0074206, -0.0047217, 0.0048159
5: -0.0024147, 0.0027050, -0.0026369, 0.0029579, -0.0045385, 0.0044809
6: -0.0064802, -0.0035495, -0.0065684, -0.0033950, -0.0029829, 0.0029196
7: -0.0018203, 0.0032344, -0.0021094, 0.0033303, -0.0048513, 0.0050605
8: -0.0007762, -0.0000005, -0.0008459, 0.0000125, -0.0007887, 0.0008454
9: 0.9985415, 1.0106769, 0.9978935, 1.0111725, -0.0085720, 0.0088206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055592, upper bound: 0.0048748
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055315, upper bound: 0.0049368
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036314, -0.0005377, -0.0036152, -0.0004921, -0.0029268, 0.0028902
1: -0.0038139, 0.0035724, -0.0041026, 0.0035671, -0.0042665, 0.0044924
2: 0.0036437, 0.0095603, 0.0033249, 0.0095576, -0.0036189, 0.0038615
3: -0.0044638, -0.0035186, -0.0044507, -0.0035512, -0.0009126, 0.0009321
4: 0.0020007, 0.0072610, 0.0020335, 0.0074607, -0.0048471, 0.0046792
5: -0.0024348, 0.0027862, -0.0026573, 0.0028254, -0.0044210, 0.0045456
6: -0.0064949, -0.0034557, -0.0065760, -0.0034958, -0.0029021, 0.0030058
7: -0.0020364, 0.0032274, -0.0019001, 0.0033862, -0.0051323, 0.0048559
8: -0.0008156, -0.0000092, -0.0008048, -0.0000030, -0.0008126, 0.0007903
9: 0.9983147, 1.0106479, 0.9982234, 1.0113095, -0.0088709, 0.0084996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055381, upper bound: 0.0049391
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055381, upper bound: 0.0050184
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0035965, -0.0005449, -0.0035823, -0.0004043, -0.0030392, 0.0028850
1: -0.0038056, 0.0035105, -0.0041958, 0.0034754, -0.0042505, 0.0046262
2: 0.0036486, 0.0095299, 0.0032977, 0.0095160, -0.0036102, 0.0038566
3: -0.0044536, -0.0035206, -0.0044564, -0.0034786, -0.0009750, 0.0009358
4: 0.0020201, 0.0072449, 0.0020420, 0.0074682, -0.0048190, 0.0046898
5: -0.0024205, 0.0027468, -0.0026906, 0.0027678, -0.0044224, 0.0045390
6: -0.0064789, -0.0034600, -0.0065714, -0.0034767, -0.0029124, 0.0029889
7: -0.0020297, 0.0032048, -0.0019830, 0.0033746, -0.0051132, 0.0049030
8: -0.0008135, -0.0000178, -0.0008673, -0.0000032, -0.0008103, 0.0008495
9: 0.9983972, 1.0106291, 0.9983398, 1.0113953, -0.0088545, 0.0084790

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055490, upper bound: 0.0049391
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055490, upper bound: 0.0050184
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0036313, -0.0005710, -0.0028800, 0.0029164
1: -0.0040906, 0.0036655, -0.0038272, 0.0034772, -0.0044337, 0.0043671
2: 0.0033639, 0.0095964, 0.0036027, 0.0094329, -0.0036470, 0.0036420
3: -0.0044669, -0.0035499, -0.0044598, -0.0035728, -0.0008940, 0.0009099
4: 0.0020077, 0.0074119, 0.0021356, 0.0072439, -0.0046886, 0.0047036
5: -0.0026266, 0.0028738, -0.0024147, 0.0027050, -0.0044673, 0.0044573
6: -0.0065589, -0.0034874, -0.0064802, -0.0035495, -0.0029052, 0.0028950
7: -0.0019106, 0.0033605, -0.0018203, 0.0032344, -0.0048320, 0.0048422
8: -0.0008055, 0.0000177, -0.0007762, -0.0000005, -0.0007908, 0.0007838
9: 0.9981174, 1.0112350, 0.9985415, 1.0106769, -0.0085880, 0.0085887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051917, upper bound: 0.0052921
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051422, upper bound: 0.0053190
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0036387, -0.0005363, -0.0029283, 0.0029438
1: -0.0040906, 0.0036655, -0.0038154, 0.0035825, -0.0045099, 0.0043429
2: 0.0033639, 0.0095964, 0.0036428, 0.0095646, -0.0038328, 0.0036720
3: -0.0044669, -0.0035499, -0.0044676, -0.0035181, -0.0009488, 0.0009177
4: 0.0020077, 0.0074119, 0.0019980, 0.0072638, -0.0047232, 0.0048429
5: -0.0026266, 0.0028738, -0.0024372, 0.0027910, -0.0045563, 0.0044924
6: -0.0065589, -0.0034874, -0.0064982, -0.0034548, -0.0030007, 0.0029195
7: -0.0019106, 0.0033605, -0.0020380, 0.0032337, -0.0048710, 0.0051125
8: -0.0008055, 0.0000177, -0.0008161, -0.0000061, -0.0007995, 0.0008338
9: 0.9981174, 1.0112350, 0.9983037, 1.0106512, -0.0086224, 0.0088576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051917, upper bound: 0.0052921
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051422, upper bound: 0.0053190
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036793, -0.0004573, -0.0035705, -0.0005767, -0.0028893, 0.0029348
1: -0.0040719, 0.0037512, -0.0038394, 0.0033845, -0.0043386, 0.0044133
2: 0.0034115, 0.0097190, 0.0035640, 0.0093964, -0.0036166, 0.0038294
3: -0.0044706, -0.0034949, -0.0044440, -0.0035743, -0.0008963, 0.0009491
4: 0.0018776, 0.0074179, 0.0021587, 0.0072936, -0.0048213, 0.0046836
5: -0.0026345, 0.0029528, -0.0024499, 0.0026615, -0.0044197, 0.0045290
6: -0.0065652, -0.0033960, -0.0064974, -0.0035568, -0.0029043, 0.0029883
7: -0.0021077, 0.0033243, -0.0018110, 0.0032697, -0.0050838, 0.0048463
8: -0.0008454, 0.0000094, -0.0007755, -0.0000208, -0.0008246, 0.0007828
9: 0.9979050, 1.0111693, 0.9986401, 1.0107657, -0.0088453, 0.0084851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0053561
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0053850
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0036457, -0.0004644, -0.0035327, -0.0004941, -0.0029963, 0.0029246
1: -0.0040642, 0.0036916, -0.0039247, 0.0032870, -0.0043162, 0.0045333
2: 0.0034161, 0.0096898, 0.0035427, 0.0093514, -0.0036116, 0.0038022
3: -0.0044612, -0.0034970, -0.0044472, -0.0035057, -0.0009555, 0.0009501
4: 0.0018963, 0.0074031, 0.0021699, 0.0072936, -0.0047881, 0.0047015
5: -0.0026217, 0.0029152, -0.0024665, 0.0025930, -0.0044292, 0.0045064
6: -0.0065499, -0.0034002, -0.0064901, -0.0035363, -0.0029180, 0.0029688
7: -0.0021012, 0.0033011, -0.0019014, 0.0032620, -0.0050658, 0.0049023
8: -0.0008434, 0.0000007, -0.0008356, -0.0000212, -0.0008222, 0.0008363
9: 0.9979852, 1.0111523, 0.9987696, 1.0108156, -0.0087877, 0.0084781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053561
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053850
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0036792, -0.0004866, -0.0029604, 0.0029604
1: -0.0040906, 0.0036655, -0.0040906, 0.0036655, -0.0042400, 0.0042400
2: 0.0033639, 0.0095964, 0.0033639, 0.0095964, -0.0035166, 0.0035166
3: -0.0044669, -0.0035499, -0.0044669, -0.0035499, -0.0009170, 0.0009170
4: 0.0020077, 0.0074119, 0.0020077, 0.0074119, -0.0047256, 0.0047256
5: -0.0026266, 0.0028738, -0.0026266, 0.0028738, -0.0044388, 0.0044388
6: -0.0065589, -0.0034874, -0.0065589, -0.0034874, -0.0029321, 0.0029321
7: -0.0019106, 0.0033605, -0.0019106, 0.0033605, -0.0048727, 0.0048727
8: -0.0008055, 0.0000177, -0.0008055, 0.0000177, -0.0007672, 0.0007672
9: 0.9981174, 1.0112350, 0.9981174, 1.0112350, -0.0084081, 0.0084081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051917, upper bound: 0.0052921
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051422, upper bound: 0.0053190
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0036792, -0.0004866, -0.0036869, -0.0004558, -0.0030061, 0.0029789
1: -0.0040906, 0.0036655, -0.0040733, 0.0037617, -0.0043103, 0.0042178
2: 0.0033639, 0.0095964, 0.0034106, 0.0097233, -0.0036861, 0.0035314
3: -0.0044669, -0.0035499, -0.0044745, -0.0034944, -0.0009724, 0.0009246
4: 0.0020077, 0.0074119, 0.0018748, 0.0074206, -0.0047462, 0.0048582
5: -0.0026266, 0.0028738, -0.0026369, 0.0029579, -0.0045279, 0.0044623
6: -0.0065589, -0.0034874, -0.0065684, -0.0033950, -0.0030238, 0.0029481
7: -0.0019106, 0.0033605, -0.0021094, 0.0033303, -0.0048888, 0.0051146
8: -0.0008055, 0.0000177, -0.0008459, 0.0000125, -0.0007751, 0.0008244
9: 0.9981174, 1.0112350, 0.9978935, 1.0111725, -0.0084165, 0.0086597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051917, upper bound: 0.0052921
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051422, upper bound: 0.0053190
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036793, -0.0004573, -0.0036152, -0.0004921, -0.0029621, 0.0029695
1: -0.0040719, 0.0037512, -0.0041026, 0.0035671, -0.0041465, 0.0043009
2: 0.0034115, 0.0097190, 0.0033249, 0.0095576, -0.0034889, 0.0037194
3: -0.0044706, -0.0034949, -0.0044507, -0.0035512, -0.0009195, 0.0009558
4: 0.0018776, 0.0074179, 0.0020335, 0.0074607, -0.0048616, 0.0047024
5: -0.0026345, 0.0029528, -0.0026573, 0.0028254, -0.0043934, 0.0045183
6: -0.0065652, -0.0033960, -0.0065760, -0.0034958, -0.0029310, 0.0030285
7: -0.0021077, 0.0033243, -0.0019001, 0.0033862, -0.0051371, 0.0048811
8: -0.0008454, 0.0000094, -0.0008048, -0.0000030, -0.0008074, 0.0007647
9: 0.9979050, 1.0111693, 0.9982234, 1.0113095, -0.0086763, 0.0083033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0053561
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0053850
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0036457, -0.0004644, -0.0035823, -0.0004043, -0.0030580, 0.0029617
1: -0.0040642, 0.0036916, -0.0041958, 0.0034754, -0.0041225, 0.0044087
2: 0.0034161, 0.0096898, 0.0032977, 0.0095160, -0.0034855, 0.0036924
3: -0.0044612, -0.0034970, -0.0044564, -0.0034786, -0.0009827, 0.0009594
4: 0.0018963, 0.0074031, 0.0020420, 0.0074682, -0.0048348, 0.0047178
5: -0.0026217, 0.0029152, -0.0026906, 0.0027678, -0.0044049, 0.0045030
6: -0.0065499, -0.0034002, -0.0065714, -0.0034767, -0.0029446, 0.0030120
7: -0.0021012, 0.0033011, -0.0019830, 0.0033746, -0.0051183, 0.0049310
8: -0.0008434, 0.0000007, -0.0008673, -0.0000032, -0.0008117, 0.0008271
9: 0.9979852, 1.0111523, 0.9983398, 1.0113953, -0.0086373, 0.0082931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 104

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053561
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053850
time: 0.84 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.27 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051120, upper bound: 0.0049500
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050547, upper bound: 0.0049968
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051120, upper bound: 0.0049500
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050547, upper bound: 0.0049968
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0050194
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0050774
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0039866, upper bound: 0.0034043
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050455, upper bound: 0.0050455
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051120, upper bound: 0.0051781
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050547, upper bound: 0.0051884
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051120, upper bound: 0.0051781
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050547, upper bound: 0.0051884
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0052287
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0052458
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0039866, upper bound: 0.0037923
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050455, upper bound: 0.0052126
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050197, upper bound: 0.0052440
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0052690
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050197, upper bound: 0.0052440
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0052691
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053086
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053356
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0036646, upper bound: 0.0032867
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0053061
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050197, upper bound: 0.0054678
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0054692
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050197, upper bound: 0.0054678
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0054692
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055087
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055205
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0036654, upper bound: 0.0036584
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0054900
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053667, upper bound: 0.0048355
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053142, upper bound: 0.0048925
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053667, upper bound: 0.0048355
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053142, upper bound: 0.0048925
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053086, upper bound: 0.0048953
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053086, upper bound: 0.0049693
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0039981, upper bound: 0.0032163
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053060, upper bound: 0.0049377
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053667, upper bound: 0.0050451
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053142, upper bound: 0.0050661
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053667, upper bound: 0.0050451
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053142, upper bound: 0.0050661
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0050937
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0051407
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0039981, upper bound: 0.0035810
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0053060, upper bound: 0.0051086
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050198, upper bound: 0.0052440
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0052690
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050198, upper bound: 0.0052440
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0052691
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053086
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0053356
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0036646, upper bound: 0.0032084
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0053061
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050198, upper bound: 0.0054678
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0054692
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050198, upper bound: 0.0054678
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049442, upper bound: 0.0054692
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055087
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0055206
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0036654, upper bound: 0.0035831
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0054901
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052587, upper bound: 0.0049500
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052227, upper bound: 0.0049968
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052587, upper bound: 0.0049500
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052227, upper bound: 0.0049968
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052287, upper bound: 0.0050194
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052287, upper bound: 0.0050774
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052458, upper bound: 0.0050194
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052458, upper bound: 0.0050774
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055314, upper bound: 0.0048330
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055010, upper bound: 0.0048925
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055314, upper bound: 0.0048330
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055010, upper bound: 0.0048925
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055087, upper bound: 0.0048952
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055087, upper bound: 0.0049693
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0048952
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055205, upper bound: 0.0049693
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0052437
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051144, upper bound: 0.0052690
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0052437
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051144, upper bound: 0.0052690
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050937, upper bound: 0.0053085
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050937, upper bound: 0.0053356
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053085
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053356
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0052437
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051144, upper bound: 0.0052690
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0052437
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051144, upper bound: 0.0052690
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050938, upper bound: 0.0053085
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0050938, upper bound: 0.0053356
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053085
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051407, upper bound: 0.0053356
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052836, upper bound: 0.0049916
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052509, upper bound: 0.0050427
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052836, upper bound: 0.0049916
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052509, upper bound: 0.0050427
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052514, upper bound: 0.0050643
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052514, upper bound: 0.0051268
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052738, upper bound: 0.0050643
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0052738, upper bound: 0.0051268
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055592, upper bound: 0.0048748
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055315, upper bound: 0.0049368
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055592, upper bound: 0.0048748
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055315, upper bound: 0.0049368
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055381, upper bound: 0.0049391
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055381, upper bound: 0.0050184
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055490, upper bound: 0.0049391
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0055490, upper bound: 0.0050184
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051917, upper bound: 0.0052921
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051422, upper bound: 0.0053190
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051917, upper bound: 0.0052921
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051422, upper bound: 0.0053190
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0053561
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0053850
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053561
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053850
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051917, upper bound: 0.0052921
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051422, upper bound: 0.0053190
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051917, upper bound: 0.0052921
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051422, upper bound: 0.0053190
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0053561
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051166, upper bound: 0.0053850
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053561
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.27
Output dim: 9, lower bound: -0.0051683, upper bound: 0.0053850

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032511, -0.0005903, -0.0033167, -0.0005763, -0.0026362, 0.0027120
1: -0.0038966, 0.0026619, -0.0039123, 0.0027462, -0.0038153, 0.0037047
2: 0.0035287, 0.0089449, 0.0035197, 0.0089807, -0.0031425, 0.0031275
3: -0.0043674, -0.0035828, -0.0044032, -0.0035779, -0.0007895, 0.0008204
4: 0.0023372, 0.0072153, 0.0023121, 0.0072464, -0.0043485, 0.0043244
5: -0.0024009, 0.0024132, -0.0024287, 0.0024589, -0.0040168, 0.0040113
6: -0.0064198, -0.0036092, -0.0064506, -0.0036001, -0.0027065, 0.0027321
7: -0.0016456, 0.0032135, -0.0016613, 0.0032739, -0.0045303, 0.0044822
8: -0.0007589, -0.0000731, -0.0007641, -0.0000422, -0.0007053, 0.0006772
9: 0.9994834, 1.0107650, 0.9993888, 1.0108017, -0.0074982, 0.0075297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050109, upper bound: 0.0050109
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050109, upper bound: 0.0050109
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032511, -0.0005903, -0.0033266, -0.0005351, -0.0026856, 0.0027363
1: -0.0038966, 0.0026619, -0.0039036, 0.0028462, -0.0038916, 0.0036803
2: 0.0035287, 0.0089449, 0.0035445, 0.0091288, -0.0033327, 0.0031632
3: -0.0043674, -0.0035828, -0.0044085, -0.0035199, -0.0008475, 0.0008257
4: 0.0023372, 0.0072153, 0.0021667, 0.0072778, -0.0043890, 0.0044649
5: -0.0024009, 0.0024132, -0.0024611, 0.0025500, -0.0041074, 0.0040530
6: -0.0064198, -0.0036092, -0.0064781, -0.0035023, -0.0028030, 0.0027610
7: -0.0016456, 0.0032135, -0.0018995, 0.0032829, -0.0045791, 0.0047562
8: -0.0007589, -0.0000731, -0.0008058, -0.0000483, -0.0007106, 0.0007327
9: 0.9994834, 1.0107650, 0.9991314, 1.0107950, -0.0075418, 0.0077969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0049500
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0049500
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032122, -0.0004761, -0.0032741, -0.0005805, -0.0026317, 0.0027980
1: -0.0039313, 0.0026174, -0.0039274, 0.0026765, -0.0039126, 0.0037793
2: 0.0035478, 0.0090338, 0.0034727, 0.0089552, -0.0031497, 0.0033020
3: -0.0043747, -0.0034565, -0.0043905, -0.0035790, -0.0007957, 0.0009340
4: 0.0022042, 0.0072411, 0.0023307, 0.0072982, -0.0044990, 0.0043291
5: -0.0024413, 0.0024346, -0.0024637, 0.0024213, -0.0040239, 0.0040580
6: -0.0064378, -0.0034906, -0.0064743, -0.0036065, -0.0027148, 0.0028644
7: -0.0019763, 0.0032133, -0.0016544, 0.0033157, -0.0049430, 0.0045198
8: -0.0008618, -0.0000790, -0.0007635, -0.0000603, -0.0008016, 0.0006845
9: 0.9993894, 1.0107945, 0.9994619, 1.0108936, -0.0076679, 0.0075213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049500, upper bound: 0.0051420
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049500, upper bound: 0.0051420
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032511, -0.0005903, -0.0036241, -0.0005724, -0.0026787, 0.0030338
1: -0.0038966, 0.0026619, -0.0038258, 0.0034671, -0.0050179, 0.0040753
2: 0.0035287, 0.0089449, 0.0036035, 0.0094287, -0.0038963, 0.0032931
3: -0.0043674, -0.0035828, -0.0044561, -0.0035733, -0.0007941, 0.0008733
4: 0.0023372, 0.0072153, 0.0021384, 0.0072412, -0.0043655, 0.0044997
5: -0.0024009, 0.0024132, -0.0024123, 0.0027003, -0.0042821, 0.0040387
6: -0.0064198, -0.0036092, -0.0064769, -0.0035504, -0.0027538, 0.0027604
7: -0.0016456, 0.0032135, -0.0018186, 0.0032283, -0.0045490, 0.0047140
8: -0.0007589, -0.0000731, -0.0007758, -0.0000035, -0.0007523, 0.0006942
9: 0.9994834, 1.0107650, 0.9985527, 1.0106736, -0.0076570, 0.0087002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050109, upper bound: 0.0052146
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050109, upper bound: 0.0052146
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032014, -0.0005200, -0.0035891, -0.0005797, -0.0026217, 0.0030691
1: -0.0039413, 0.0025211, -0.0038173, 0.0034063, -0.0051400, 0.0040256
2: 0.0035219, 0.0088827, 0.0036085, 0.0093979, -0.0038671, 0.0032781
3: -0.0043672, -0.0035158, -0.0044456, -0.0035754, -0.0007918, 0.0009298
4: 0.0023499, 0.0072098, 0.0021579, 0.0072251, -0.0043724, 0.0044617
5: -0.0024138, 0.0023418, -0.0023980, 0.0026600, -0.0042564, 0.0040390
6: -0.0064091, -0.0035887, -0.0064608, -0.0035549, -0.0027326, 0.0027702
7: -0.0017354, 0.0032004, -0.0018117, 0.0032056, -0.0046079, 0.0046892
8: -0.0008200, -0.0000727, -0.0007736, -0.0000123, -0.0008077, 0.0006955
9: 0.9996502, 1.0107989, 0.9986357, 1.0106546, -0.0076197, 0.0086384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039881, upper bound: 0.0048215
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050338, upper bound: 0.0052000
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032511, -0.0005903, -0.0036314, -0.0005377, -0.0027135, 0.0030411
1: -0.0038966, 0.0026619, -0.0038139, 0.0035724, -0.0051174, 0.0040634
2: 0.0035287, 0.0089449, 0.0036437, 0.0095603, -0.0040677, 0.0032829
3: -0.0043674, -0.0035828, -0.0044638, -0.0035186, -0.0008488, 0.0008810
4: 0.0023372, 0.0072153, 0.0020007, 0.0072610, -0.0043920, 0.0046339
5: -0.0024009, 0.0024132, -0.0024348, 0.0027862, -0.0043737, 0.0040712
6: -0.0064198, -0.0036092, -0.0064949, -0.0034557, -0.0028503, 0.0027818
7: -0.0016456, 0.0032135, -0.0020364, 0.0032274, -0.0045781, 0.0049570
8: -0.0007589, -0.0000731, -0.0008156, -0.0000092, -0.0007498, 0.0007425
9: 0.9994834, 1.0107650, 0.9983147, 1.0106479, -0.0076816, 0.0089776

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0051781
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0051781
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032014, -0.0005200, -0.0035965, -0.0005449, -0.0026565, 0.0030766
1: -0.0039413, 0.0025211, -0.0038056, 0.0035105, -0.0052352, 0.0040137
2: 0.0035219, 0.0088827, 0.0036486, 0.0095299, -0.0040375, 0.0032676
3: -0.0043672, -0.0035158, -0.0044536, -0.0035206, -0.0008466, 0.0009378
4: 0.0023499, 0.0072098, 0.0020201, 0.0072449, -0.0043998, 0.0045964
5: -0.0024138, 0.0023418, -0.0024205, 0.0027468, -0.0043479, 0.0040716
6: -0.0064091, -0.0035887, -0.0064789, -0.0034600, -0.0028291, 0.0027917
7: -0.0017354, 0.0032004, -0.0020297, 0.0032048, -0.0046376, 0.0049321
8: -0.0008200, -0.0000727, -0.0008135, -0.0000178, -0.0008022, 0.0007408
9: 0.9996502, 1.0107989, 0.9983972, 1.0106291, -0.0076440, 0.0089120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0051884
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0050194, upper bound: 0.0051884
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032614, -0.0005487, -0.0035705, -0.0005767, -0.0026847, 0.0030218
1: -0.0038879, 0.0027628, -0.0038394, 0.0033845, -0.0049436, 0.0041694
2: 0.0035538, 0.0090925, 0.0035640, 0.0093964, -0.0038954, 0.0035281
3: -0.0043723, -0.0035247, -0.0044440, -0.0035743, -0.0007979, 0.0009193
4: 0.0021912, 0.0072481, 0.0021587, 0.0072936, -0.0045263, 0.0045121
5: -0.0024342, 0.0025053, -0.0024499, 0.0026615, -0.0042713, 0.0041396
6: -0.0064477, -0.0035112, -0.0064974, -0.0035568, -0.0027732, 0.0028696
7: -0.0018841, 0.0032217, -0.0018110, 0.0032697, -0.0048636, 0.0047619
8: -0.0008009, -0.0000795, -0.0007755, -0.0000208, -0.0007801, 0.0006961
9: 0.9992245, 1.0107591, 0.9986401, 1.0107657, -0.0079899, 0.0086610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049500, upper bound: 0.0052843
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049500, upper bound: 0.0052843
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032122, -0.0004761, -0.0035705, -0.0005767, -0.0026355, 0.0030945
1: -0.0039313, 0.0026174, -0.0038394, 0.0033845, -0.0051123, 0.0041591
2: 0.0035478, 0.0090338, 0.0035640, 0.0093964, -0.0038937, 0.0034565
3: -0.0043747, -0.0034565, -0.0044440, -0.0035743, -0.0008004, 0.0009875
4: 0.0022042, 0.0072411, 0.0021587, 0.0072936, -0.0045141, 0.0045036
5: -0.0024413, 0.0024346, -0.0024499, 0.0026615, -0.0042869, 0.0040856
6: -0.0064378, -0.0034906, -0.0064974, -0.0035568, -0.0027622, 0.0028901
7: -0.0019763, 0.0032133, -0.0018110, 0.0032697, -0.0049589, 0.0047509
8: -0.0008618, -0.0000790, -0.0007755, -0.0000208, -0.0008411, 0.0006966
9: 0.9993894, 1.0107945, 0.9986401, 1.0107657, -0.0078267, 0.0086820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049500, upper bound: 0.0052843
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049500, upper bound: 0.0052843
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032691, -0.0005476, -0.0035327, -0.0004941, -0.0027750, 0.0029852
1: -0.0038887, 0.0027333, -0.0039247, 0.0032870, -0.0048976, 0.0041724
2: 0.0035507, 0.0090913, 0.0035427, 0.0093514, -0.0038724, 0.0036323
3: -0.0043946, -0.0035242, -0.0044472, -0.0035057, -0.0008889, 0.0009230
4: 0.0021898, 0.0072243, 0.0021699, 0.0072936, -0.0045273, 0.0045438
5: -0.0024121, 0.0025060, -0.0024665, 0.0025930, -0.0042179, 0.0041761
6: -0.0064387, -0.0035089, -0.0064901, -0.0035363, -0.0028002, 0.0028727
7: -0.0018875, 0.0032388, -0.0019014, 0.0032620, -0.0051060, 0.0048329
8: -0.0007981, -0.0000571, -0.0008356, -0.0000212, -0.0007769, 0.0007785
9: 0.9992295, 1.0107375, 0.9987696, 1.0108156, -0.0081302, 0.0086466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049621, upper bound: 0.0052126
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049621, upper bound: 0.0052126
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032953, -0.0005048, -0.0033167, -0.0005763, -0.0026616, 0.0027771
1: -0.0041514, 0.0028352, -0.0039123, 0.0027462, -0.0041820, 0.0039969
2: 0.0032893, 0.0091091, 0.0035197, 0.0089807, -0.0034509, 0.0033896
3: -0.0043733, -0.0035593, -0.0044032, -0.0035779, -0.0007954, 0.0008439
4: 0.0022123, 0.0073908, 0.0023121, 0.0072464, -0.0044815, 0.0045007
5: -0.0026146, 0.0025679, -0.0024287, 0.0024589, -0.0042541, 0.0042005
6: -0.0065059, -0.0035509, -0.0064506, -0.0036001, -0.0027832, 0.0027869
7: -0.0017212, 0.0033423, -0.0016613, 0.0032739, -0.0046201, 0.0046014
8: -0.0007904, -0.0000553, -0.0007641, -0.0000422, -0.0007481, 0.0007066
9: 0.9990917, 1.0113105, 0.9993888, 1.0108017, -0.0080387, 0.0081578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048856, upper bound: 0.0053051
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048856, upper bound: 0.0053051
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032483, -0.0004310, -0.0032805, -0.0005837, -0.0026410, 0.0028495
1: -0.0042087, 0.0027037, -0.0039033, 0.0026753, -0.0043267, 0.0039523
2: 0.0032714, 0.0090578, 0.0035249, 0.0089465, -0.0034546, 0.0033668
3: -0.0043753, -0.0034894, -0.0043927, -0.0035801, -0.0007952, 0.0009033
4: 0.0022230, 0.0073920, 0.0023311, 0.0072283, -0.0044832, 0.0044711
5: -0.0026406, 0.0025054, -0.0024126, 0.0024187, -0.0042540, 0.0041916
6: -0.0064976, -0.0035325, -0.0064345, -0.0036046, -0.0027642, 0.0027948
7: -0.0018067, 0.0033265, -0.0016544, 0.0032507, -0.0046664, 0.0045796
8: -0.0008551, -0.0000546, -0.0007616, -0.0000506, -0.0008045, 0.0007070
9: 0.9992326, 1.0113724, 0.9994775, 1.0107807, -0.0079809, 0.0081534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0038699, upper bound: 0.0046112
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049238, upper bound: 0.0052988
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032953, -0.0005048, -0.0033266, -0.0005351, -0.0027111, 0.0028074
1: -0.0041514, 0.0028352, -0.0039036, 0.0028462, -0.0042583, 0.0039726
2: 0.0032893, 0.0091091, 0.0035445, 0.0091288, -0.0036411, 0.0034253
3: -0.0043733, -0.0035593, -0.0044085, -0.0035199, -0.0008534, 0.0008492
4: 0.0022123, 0.0073908, 0.0021667, 0.0072778, -0.0045220, 0.0046411
5: -0.0026146, 0.0025679, -0.0024611, 0.0025500, -0.0043447, 0.0042422
6: -0.0065059, -0.0035509, -0.0064781, -0.0035023, -0.0028797, 0.0028157
7: -0.0017212, 0.0033423, -0.0018995, 0.0032829, -0.0046689, 0.0048754
8: -0.0007904, -0.0000553, -0.0008058, -0.0000483, -0.0007420, 0.0007505
9: 0.9990917, 1.0113105, 0.9991314, 1.0107950, -0.0080824, 0.0084251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0052440
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0052440
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032483, -0.0004310, -0.0032904, -0.0005422, -0.0026905, 0.0028594
1: -0.0042087, 0.0027037, -0.0038946, 0.0027733, -0.0043978, 0.0039277
2: 0.0032714, 0.0090578, 0.0035496, 0.0090957, -0.0036405, 0.0034028
3: -0.0043753, -0.0034894, -0.0043986, -0.0035221, -0.0008532, 0.0009092
4: 0.0022230, 0.0073920, 0.0021858, 0.0072596, -0.0045239, 0.0046102
5: -0.0026406, 0.0025054, -0.0024446, 0.0025100, -0.0043433, 0.0042331
6: -0.0064976, -0.0035325, -0.0064623, -0.0035068, -0.0028603, 0.0028239
7: -0.0018067, 0.0033265, -0.0018928, 0.0032608, -0.0047179, 0.0048531
8: -0.0008551, -0.0000546, -0.0008034, -0.0000569, -0.0007982, 0.0007488
9: 0.9992326, 1.0113724, 0.9992200, 1.0107740, -0.0080251, 0.0084159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0033405, upper bound: 0.0040298
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0052381
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033030, -0.0004667, -0.0032741, -0.0005805, -0.0026805, 0.0028074
1: -0.0041405, 0.0029335, -0.0039274, 0.0026765, -0.0041081, 0.0040616
2: 0.0033329, 0.0092383, 0.0034727, 0.0089552, -0.0034292, 0.0035942
3: -0.0043792, -0.0035008, -0.0043905, -0.0035790, -0.0008003, 0.0008898
4: 0.0020784, 0.0074069, 0.0023307, 0.0072982, -0.0046309, 0.0044883
5: -0.0026286, 0.0026553, -0.0024637, 0.0024213, -0.0042184, 0.0042923
6: -0.0065165, -0.0034572, -0.0064743, -0.0036065, -0.0027885, 0.0028916
7: -0.0019377, 0.0033118, -0.0016544, 0.0033157, -0.0048955, 0.0046195
8: -0.0008327, -0.0000607, -0.0007635, -0.0000603, -0.0007724, 0.0007028
9: 0.9988671, 1.0112519, 0.9994619, 1.0108936, -0.0083347, 0.0080743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048355, upper bound: 0.0053887
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048355, upper bound: 0.0053887
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032562, -0.0003895, -0.0032741, -0.0005805, -0.0026756, 0.0028845
1: -0.0041975, 0.0027993, -0.0039274, 0.0026765, -0.0042901, 0.0040562
2: 0.0033144, 0.0091859, 0.0034727, 0.0089552, -0.0034664, 0.0035629
3: -0.0043838, -0.0034306, -0.0043905, -0.0035790, -0.0008049, 0.0009600
4: 0.0020888, 0.0074093, 0.0023307, 0.0072982, -0.0046233, 0.0044901
5: -0.0026567, 0.0025938, -0.0024637, 0.0024213, -0.0042588, 0.0042486
6: -0.0065101, -0.0034382, -0.0064743, -0.0036065, -0.0027815, 0.0029096
7: -0.0020236, 0.0032964, -0.0016544, 0.0033157, -0.0049787, 0.0046036
8: -0.0008982, -0.0000594, -0.0007635, -0.0000603, -0.0008379, 0.0007040
9: 0.9990049, 1.0113225, 0.9994619, 1.0108936, -0.0082324, 0.0081584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048355, upper bound: 0.0053887
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048355, upper bound: 0.0053887
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0033122, -0.0004655, -0.0032259, -0.0005107, -0.0028015, 0.0027604
1: -0.0041417, 0.0029099, -0.0039718, 0.0025366, -0.0040690, 0.0040509
2: 0.0033299, 0.0092411, 0.0034661, 0.0088936, -0.0034127, 0.0037153
3: -0.0044029, -0.0035004, -0.0043937, -0.0035120, -0.0008909, 0.0008934
4: 0.0020742, 0.0073865, 0.0023433, 0.0072933, -0.0046340, 0.0045213
5: -0.0026092, 0.0026619, -0.0024757, 0.0023506, -0.0041675, 0.0043290
6: -0.0065088, -0.0034544, -0.0064659, -0.0035860, -0.0028151, 0.0028950
7: -0.0019414, 0.0033244, -0.0017444, 0.0033059, -0.0051447, 0.0047026
8: -0.0008302, -0.0000377, -0.0008245, -0.0000594, -0.0007708, 0.0007868
9: 0.9988534, 1.0112327, 0.9996260, 1.0109274, -0.0084991, 0.0080676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048593, upper bound: 0.0053061
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048593, upper bound: 0.0053061
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032953, -0.0005048, -0.0036241, -0.0005724, -0.0027128, 0.0031193
1: -0.0041514, 0.0028352, -0.0038258, 0.0034671, -0.0053846, 0.0043675
2: 0.0032893, 0.0091091, 0.0036035, 0.0094287, -0.0042047, 0.0035552
3: -0.0043733, -0.0035593, -0.0044561, -0.0035733, -0.0008000, 0.0008968
4: 0.0022123, 0.0073908, 0.0021384, 0.0072412, -0.0044985, 0.0046759
5: -0.0026146, 0.0025679, -0.0024123, 0.0027003, -0.0045194, 0.0042279
6: -0.0065059, -0.0035509, -0.0064769, -0.0035504, -0.0028305, 0.0028151
7: -0.0017212, 0.0033423, -0.0018186, 0.0032283, -0.0046388, 0.0048333
8: -0.0007904, -0.0000553, -0.0007758, -0.0000035, -0.0007868, 0.0007204
9: 0.9990917, 1.0113105, 0.9985527, 1.0106736, -0.0081976, 0.0093284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048848, upper bound: 0.0055035
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048848, upper bound: 0.0055035
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032483, -0.0004310, -0.0035891, -0.0005797, -0.0026686, 0.0031581
1: -0.0042087, 0.0027037, -0.0038173, 0.0034063, -0.0055192, 0.0043221
2: 0.0032714, 0.0090578, 0.0036085, 0.0093979, -0.0042114, 0.0035332
3: -0.0043753, -0.0034894, -0.0044456, -0.0035754, -0.0007998, 0.0009562
4: 0.0022230, 0.0073920, 0.0021579, 0.0072251, -0.0045003, 0.0046462
5: -0.0026406, 0.0025054, -0.0023980, 0.0026600, -0.0045194, 0.0042194
6: -0.0064976, -0.0035325, -0.0064608, -0.0035549, -0.0028116, 0.0028228
7: -0.0018067, 0.0033265, -0.0018117, 0.0032056, -0.0046854, 0.0048108
8: -0.0008551, -0.0000546, -0.0007736, -0.0000123, -0.0008428, 0.0007190
9: 0.9992326, 1.0113724, 0.9986357, 1.0106546, -0.0081417, 0.0093282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0038699, upper bound: 0.0049620
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049238, upper bound: 0.0054819
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032953, -0.0005048, -0.0036314, -0.0005377, -0.0027576, 0.0031266
1: -0.0041514, 0.0028352, -0.0038139, 0.0035724, -0.0054841, 0.0043556
2: 0.0032893, 0.0091091, 0.0036437, 0.0095603, -0.0043761, 0.0035450
3: -0.0043733, -0.0035593, -0.0044638, -0.0035186, -0.0008547, 0.0009045
4: 0.0022123, 0.0073908, 0.0020007, 0.0072610, -0.0045250, 0.0048101
5: -0.0026146, 0.0025679, -0.0024348, 0.0027862, -0.0046110, 0.0042604
6: -0.0065059, -0.0035509, -0.0064949, -0.0034557, -0.0029270, 0.0028365
7: -0.0017212, 0.0033423, -0.0020364, 0.0032274, -0.0046679, 0.0050763
8: -0.0007904, -0.0000553, -0.0008156, -0.0000092, -0.0007812, 0.0007603
9: 0.9990917, 1.0113105, 0.9983147, 1.0106479, -0.0082221, 0.0096057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0054678
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0054678
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032483, -0.0004310, -0.0035965, -0.0005449, -0.0027034, 0.0031656
1: -0.0042087, 0.0027037, -0.0038056, 0.0035105, -0.0056145, 0.0043101
2: 0.0032714, 0.0090578, 0.0036486, 0.0095299, -0.0043818, 0.0035226
3: -0.0043753, -0.0034894, -0.0044536, -0.0035206, -0.0008547, 0.0009641
4: 0.0022230, 0.0073920, 0.0020201, 0.0072449, -0.0045276, 0.0047809
5: -0.0026406, 0.0025054, -0.0024205, 0.0027468, -0.0046109, 0.0042520
6: -0.0064976, -0.0035325, -0.0064789, -0.0034600, -0.0029080, 0.0028443
7: -0.0018067, 0.0033265, -0.0020297, 0.0032048, -0.0047152, 0.0050538
8: -0.0008551, -0.0000546, -0.0008135, -0.0000178, -0.0008373, 0.0007589
9: 0.9992326, 1.0113724, 0.9983972, 1.0106291, -0.0081660, 0.0096018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048951, upper bound: 0.0054692
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048951, upper bound: 0.0054692
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033030, -0.0004667, -0.0035705, -0.0005767, -0.0027262, 0.0031038
1: -0.0041405, 0.0029335, -0.0038394, 0.0033845, -0.0053078, 0.0044414
2: 0.0033329, 0.0092383, 0.0035640, 0.0093964, -0.0041731, 0.0037486
3: -0.0043792, -0.0035008, -0.0044440, -0.0035743, -0.0008049, 0.0009433
4: 0.0020784, 0.0074069, 0.0021587, 0.0072936, -0.0046460, 0.0046628
5: -0.0026286, 0.0026553, -0.0024499, 0.0026615, -0.0044814, 0.0043199
6: -0.0065165, -0.0034572, -0.0064974, -0.0035568, -0.0028360, 0.0029173
7: -0.0019377, 0.0033118, -0.0018110, 0.0032697, -0.0049114, 0.0048506
8: -0.0008327, -0.0000607, -0.0007755, -0.0000208, -0.0008119, 0.0007149
9: 0.9988671, 1.0112519, 0.9986401, 1.0107657, -0.0084934, 0.0092349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048330, upper bound: 0.0055551
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048330, upper bound: 0.0055551
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032562, -0.0003895, -0.0035705, -0.0005767, -0.0026794, 0.0031810
1: -0.0041975, 0.0027993, -0.0038394, 0.0033845, -0.0054897, 0.0044360
2: 0.0033144, 0.0091859, 0.0035640, 0.0093964, -0.0042104, 0.0037173
3: -0.0043838, -0.0034306, -0.0044440, -0.0035743, -0.0008095, 0.0010135
4: 0.0020888, 0.0074093, 0.0021587, 0.0072936, -0.0046385, 0.0046646
5: -0.0026567, 0.0025938, -0.0024499, 0.0026615, -0.0045218, 0.0042763
6: -0.0065101, -0.0034382, -0.0064974, -0.0035568, -0.0028290, 0.0029353
7: -0.0020236, 0.0032964, -0.0018110, 0.0032697, -0.0049946, 0.0048347
8: -0.0008982, -0.0000594, -0.0007755, -0.0000208, -0.0008774, 0.0007161
9: 0.9990049, 1.0113225, 0.9986401, 1.0107657, -0.0083912, 0.0093191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048330, upper bound: 0.0055551
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048330, upper bound: 0.0055551
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0033122, -0.0004655, -0.0035327, -0.0004941, -0.0028181, 0.0030672
1: -0.0041417, 0.0029099, -0.0039247, 0.0032870, -0.0052618, 0.0045072
2: 0.0033299, 0.0092411, 0.0035427, 0.0093514, -0.0041473, 0.0038844
3: -0.0044029, -0.0035004, -0.0044472, -0.0035057, -0.0008972, 0.0009468
4: 0.0020742, 0.0073865, 0.0021699, 0.0072936, -0.0046555, 0.0046948
5: -0.0026092, 0.0026619, -0.0024665, 0.0025930, -0.0044239, 0.0043684
6: -0.0065088, -0.0034544, -0.0064901, -0.0035363, -0.0028633, 0.0029222
7: -0.0019414, 0.0033244, -0.0019014, 0.0032620, -0.0051677, 0.0049194
8: -0.0008302, -0.0000377, -0.0008356, -0.0000212, -0.0008090, 0.0007980
9: 0.9988534, 1.0112327, 0.9987696, 1.0108156, -0.0086896, 0.0092185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048593, upper bound: 0.0054900
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048593, upper bound: 0.0054900
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032511, -0.0005903, -0.0033639, -0.0004904, -0.0027022, 0.0027423
1: -0.0038966, 0.0026619, -0.0041670, 0.0029255, -0.0041149, 0.0040722
2: 0.0035287, 0.0089449, 0.0032806, 0.0091473, -0.0034192, 0.0034320
3: -0.0043674, -0.0035828, -0.0044101, -0.0035543, -0.0008131, 0.0008274
4: 0.0023372, 0.0072153, 0.0021853, 0.0074202, -0.0045212, 0.0044621
5: -0.0024009, 0.0024132, -0.0026401, 0.0026165, -0.0042145, 0.0042443
6: -0.0064198, -0.0036092, -0.0065349, -0.0035410, -0.0027625, 0.0028071
7: -0.0016456, 0.0032135, -0.0017375, 0.0034001, -0.0046464, 0.0045744
8: -0.0007589, -0.0000731, -0.0007954, -0.0000243, -0.0007346, 0.0007199
9: 0.9994834, 1.0107650, 0.9989867, 1.0113446, -0.0081175, 0.0080988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053051, upper bound: 0.0048856
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053051, upper bound: 0.0048856
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032014, -0.0005200, -0.0033280, -0.0004977, -0.0026775, 0.0028080
1: -0.0039413, 0.0025211, -0.0041584, 0.0028568, -0.0042460, 0.0040235
2: 0.0035219, 0.0088827, 0.0032856, 0.0091165, -0.0033959, 0.0034161
3: -0.0043672, -0.0035158, -0.0044002, -0.0035566, -0.0008106, 0.0008844
4: 0.0023499, 0.0072098, 0.0022036, 0.0074036, -0.0045288, 0.0044259
5: -0.0024138, 0.0023418, -0.0026257, 0.0025782, -0.0041921, 0.0042447
6: -0.0064091, -0.0035887, -0.0065194, -0.0035457, -0.0027412, 0.0028176
7: -0.0017354, 0.0032004, -0.0017306, 0.0033760, -0.0047046, 0.0045496
8: -0.0008200, -0.0000727, -0.0007931, -0.0000327, -0.0007873, 0.0007204
9: 0.9996502, 1.0107989, 0.9990699, 1.0113254, -0.0080771, 0.0080451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0040105, upper bound: 0.0043163
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052988, upper bound: 0.0049238
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032511, -0.0005903, -0.0033709, -0.0004526, -0.0027546, 0.0027698
1: -0.0038966, 0.0026619, -0.0041560, 0.0030233, -0.0041735, 0.0040297
2: 0.0035287, 0.0089449, 0.0033238, 0.0092772, -0.0035719, 0.0034193
3: -0.0043674, -0.0035828, -0.0044165, -0.0034958, -0.0008716, 0.0008337
4: 0.0023372, 0.0072153, 0.0020514, 0.0074356, -0.0045379, 0.0045908
5: -0.0024009, 0.0024132, -0.0026542, 0.0027041, -0.0042972, 0.0042592
6: -0.0064198, -0.0036092, -0.0065462, -0.0034476, -0.0028523, 0.0028232
7: -0.0016456, 0.0032135, -0.0019537, 0.0033690, -0.0046582, 0.0048062
8: -0.0007589, -0.0000731, -0.0008375, -0.0000292, -0.0007298, 0.0007644
9: 0.9994834, 1.0107650, 0.9987615, 1.0112870, -0.0081018, 0.0083361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053086, upper bound: 0.0048355
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053086, upper bound: 0.0048355
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032014, -0.0005200, -0.0033348, -0.0004596, -0.0027299, 0.0028148
1: -0.0039413, 0.0025211, -0.0041477, 0.0029533, -0.0043019, 0.0039808
2: 0.0035219, 0.0088827, 0.0033287, 0.0092457, -0.0035474, 0.0034037
3: -0.0043672, -0.0035158, -0.0044070, -0.0034980, -0.0008692, 0.0008912
4: 0.0023499, 0.0072098, 0.0020698, 0.0074194, -0.0045457, 0.0045545
5: -0.0024138, 0.0023418, -0.0026399, 0.0026661, -0.0042733, 0.0042598
6: -0.0064091, -0.0035887, -0.0065310, -0.0034520, -0.0028311, 0.0028339
7: -0.0017354, 0.0032004, -0.0019470, 0.0033445, -0.0047173, 0.0047815
8: -0.0008200, -0.0000727, -0.0008352, -0.0000374, -0.0007826, 0.0007626
9: 0.9996502, 1.0107989, 0.9988433, 1.0112675, -0.0080621, 0.0082801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0033973, upper bound: 0.0036911
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053060, upper bound: 0.0048593
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032614, -0.0005487, -0.0033174, -0.0004942, -0.0027235, 0.0027682
1: -0.0038879, 0.0027628, -0.0041829, 0.0028498, -0.0040379, 0.0041502
2: 0.0035538, 0.0090925, 0.0032395, 0.0091180, -0.0034144, 0.0036633
3: -0.0043723, -0.0035247, -0.0043977, -0.0035553, -0.0008170, 0.0008730
4: 0.0021912, 0.0072481, 0.0022062, 0.0074727, -0.0046807, 0.0044707
5: -0.0024342, 0.0025053, -0.0026734, 0.0025751, -0.0041982, 0.0043405
6: -0.0064477, -0.0035112, -0.0065582, -0.0035483, -0.0027804, 0.0029193
7: -0.0018841, 0.0032217, -0.0017292, 0.0034301, -0.0049570, 0.0046202
8: -0.0008009, -0.0000795, -0.0007948, -0.0000422, -0.0007587, 0.0007153
9: 0.9992245, 1.0107591, 0.9990715, 1.0114216, -0.0084323, 0.0080443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052440, upper bound: 0.0050529
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052440, upper bound: 0.0050529
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032122, -0.0004761, -0.0033174, -0.0004942, -0.0027180, 0.0028413
1: -0.0039313, 0.0026174, -0.0041829, 0.0028498, -0.0042066, 0.0041399
2: 0.0035478, 0.0090338, 0.0032395, 0.0091180, -0.0034127, 0.0035916
3: -0.0043747, -0.0034565, -0.0043977, -0.0035553, -0.0008194, 0.0009412
4: 0.0022042, 0.0072411, 0.0022062, 0.0074727, -0.0046685, 0.0044621
5: -0.0024413, 0.0024346, -0.0026734, 0.0025751, -0.0042138, 0.0042865
6: -0.0064378, -0.0034906, -0.0065582, -0.0035483, -0.0027695, 0.0029398
7: -0.0019763, 0.0032133, -0.0017292, 0.0034301, -0.0050523, 0.0046093
8: -0.0008618, -0.0000790, -0.0007948, -0.0000422, -0.0008196, 0.0007158
9: 0.9993894, 1.0107945, 0.9990715, 1.0114216, -0.0082691, 0.0080653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052440, upper bound: 0.0050529
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052440, upper bound: 0.0050529
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032691, -0.0005476, -0.0032717, -0.0004212, -0.0028479, 0.0027241
1: -0.0038887, 0.0027333, -0.0042401, 0.0027191, -0.0040023, 0.0041597
2: 0.0035507, 0.0090913, 0.0032216, 0.0090675, -0.0033942, 0.0037932
3: -0.0043946, -0.0035242, -0.0044028, -0.0034854, -0.0009092, 0.0008786
4: 0.0021898, 0.0072243, 0.0022166, 0.0074748, -0.0046830, 0.0044982
5: -0.0024121, 0.0025060, -0.0027010, 0.0025135, -0.0041381, 0.0043883
6: -0.0064387, -0.0035089, -0.0065513, -0.0035298, -0.0028047, 0.0029234
7: -0.0018875, 0.0032388, -0.0018149, 0.0034146, -0.0052120, 0.0046940
8: -0.0007981, -0.0000571, -0.0008594, -0.0000410, -0.0007572, 0.0008023
9: 0.9992295, 1.0107375, 0.9992105, 1.0114871, -0.0086212, 0.0080217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052381, upper bound: 0.0049377
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052381, upper bound: 0.0049377
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032511, -0.0005903, -0.0036716, -0.0004880, -0.0027522, 0.0030812
1: -0.0038966, 0.0026619, -0.0040891, 0.0036551, -0.0052079, 0.0043392
2: 0.0035287, 0.0089449, 0.0033648, 0.0095920, -0.0040849, 0.0035204
3: -0.0043674, -0.0035828, -0.0044630, -0.0035504, -0.0008170, 0.0008802
4: 0.0023372, 0.0072153, 0.0020107, 0.0074093, -0.0045242, 0.0046288
5: -0.0024009, 0.0024132, -0.0026243, 0.0028686, -0.0044663, 0.0042464
6: -0.0064198, -0.0036092, -0.0065558, -0.0034884, -0.0028099, 0.0028296
7: -0.0016456, 0.0032135, -0.0019088, 0.0033546, -0.0046436, 0.0047802
8: -0.0007589, -0.0000731, -0.0008051, 0.0000146, -0.0007735, 0.0007320
9: 0.9994834, 1.0107650, 0.9981290, 1.0112319, -0.0081649, 0.0091550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053051, upper bound: 0.0050839
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053051, upper bound: 0.0050839
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032014, -0.0005200, -0.0036384, -0.0004955, -0.0027059, 0.0031185
1: -0.0039413, 0.0025211, -0.0040812, 0.0035948, -0.0053372, 0.0042901
2: 0.0035219, 0.0088827, 0.0033695, 0.0095627, -0.0040619, 0.0035044
3: -0.0043672, -0.0035158, -0.0044541, -0.0035526, -0.0008146, 0.0009383
4: 0.0023499, 0.0072098, 0.0020296, 0.0073942, -0.0045310, 0.0045936
5: -0.0024138, 0.0023418, -0.0026113, 0.0028308, -0.0044449, 0.0042466
6: -0.0064091, -0.0035887, -0.0065401, -0.0034928, -0.0027890, 0.0028393
7: -0.0017354, 0.0032004, -0.0019018, 0.0033310, -0.0047009, 0.0047554
8: -0.0008200, -0.0000727, -0.0008030, 0.0000058, -0.0008259, 0.0007304
9: 0.9996502, 1.0107989, 0.9982082, 1.0112145, -0.0081251, 0.0091024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053051, upper bound: 0.0051285
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053051, upper bound: 0.0051285
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032511, -0.0005903, -0.0036793, -0.0004573, -0.0027939, 0.0030890
1: -0.0038966, 0.0026619, -0.0040719, 0.0037512, -0.0052864, 0.0043047
2: 0.0035287, 0.0089449, 0.0034115, 0.0097190, -0.0042304, 0.0034951
3: -0.0043674, -0.0035828, -0.0044706, -0.0034949, -0.0008725, 0.0008879
4: 0.0023372, 0.0072153, 0.0018776, 0.0074179, -0.0045326, 0.0047534
5: -0.0024009, 0.0024132, -0.0026345, 0.0029528, -0.0045451, 0.0042543
6: -0.0064198, -0.0036092, -0.0065652, -0.0033960, -0.0028976, 0.0028424
7: -0.0016456, 0.0032135, -0.0021077, 0.0033243, -0.0046428, 0.0050000
8: -0.0007589, -0.0000731, -0.0008454, 0.0000094, -0.0007683, 0.0007723
9: 0.9994834, 1.0107650, 0.9979050, 1.0111693, -0.0081410, 0.0093977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0050451
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0050451
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032014, -0.0005200, -0.0036457, -0.0004644, -0.0027370, 0.0031257
1: -0.0039413, 0.0025211, -0.0040642, 0.0036916, -0.0054137, 0.0042558
2: 0.0035219, 0.0088827, 0.0034161, 0.0096898, -0.0042054, 0.0034797
3: -0.0043672, -0.0035158, -0.0044612, -0.0034970, -0.0008702, 0.0009454
4: 0.0023499, 0.0072098, 0.0018963, 0.0074031, -0.0045400, 0.0047176
5: -0.0024138, 0.0023418, -0.0026217, 0.0029152, -0.0045225, 0.0042549
6: -0.0064091, -0.0035887, -0.0065499, -0.0034002, -0.0028768, 0.0028528
7: -0.0017354, 0.0032004, -0.0021012, 0.0033011, -0.0047009, 0.0049754
8: -0.0008200, -0.0000727, -0.0008434, 0.0000007, -0.0008207, 0.0007707
9: 0.9996502, 1.0107989, 0.9979852, 1.0111523, -0.0081019, 0.0093454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0050661
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0053085, upper bound: 0.0050661
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032614, -0.0005487, -0.0036152, -0.0004921, -0.0027693, 0.0030665
1: -0.0038879, 0.0027628, -0.0041026, 0.0035671, -0.0051225, 0.0044239
2: 0.0035538, 0.0090925, 0.0033249, 0.0095576, -0.0040737, 0.0037506
3: -0.0043723, -0.0035247, -0.0044507, -0.0035512, -0.0008211, 0.0009260
4: 0.0021912, 0.0072481, 0.0020335, 0.0074607, -0.0046789, 0.0046358
5: -0.0024342, 0.0025053, -0.0026573, 0.0028254, -0.0044481, 0.0043404
6: -0.0064477, -0.0035112, -0.0065760, -0.0034958, -0.0028279, 0.0029370
7: -0.0018841, 0.0032217, -0.0019001, 0.0033862, -0.0049457, 0.0048254
8: -0.0008009, -0.0000795, -0.0008048, -0.0000030, -0.0007979, 0.0007254
9: 0.9992245, 1.0107591, 0.9982234, 1.0113095, -0.0084863, 0.0090950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052437, upper bound: 0.0052011
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052437, upper bound: 0.0052011
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032122, -0.0004761, -0.0036152, -0.0004921, -0.0027202, 0.0031391
1: -0.0039313, 0.0026174, -0.0041026, 0.0035671, -0.0052913, 0.0044136
2: 0.0035478, 0.0090338, 0.0033249, 0.0095576, -0.0040720, 0.0036790
3: -0.0043747, -0.0034565, -0.0044507, -0.0035512, -0.0008235, 0.0009942
4: 0.0022042, 0.0072411, 0.0020335, 0.0074607, -0.0046667, 0.0046273
5: -0.0024413, 0.0024346, -0.0026573, 0.0028254, -0.0044638, 0.0042864
6: -0.0064378, -0.0034906, -0.0065760, -0.0034958, -0.0028169, 0.0029575
7: -0.0019763, 0.0032133, -0.0019001, 0.0033862, -0.0050410, 0.0048145
8: -0.0008618, -0.0000790, -0.0008048, -0.0000030, -0.0008588, 0.0007259
9: 0.9993894, 1.0107945, 0.9982234, 1.0113095, -0.0083231, 0.0091160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052437, upper bound: 0.0052011
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052437, upper bound: 0.0052011
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032691, -0.0005476, -0.0035823, -0.0004043, -0.0028648, 0.0030348
1: -0.0038887, 0.0027333, -0.0041958, 0.0034754, -0.0050858, 0.0044744
2: 0.0035507, 0.0090913, 0.0032977, 0.0095160, -0.0040490, 0.0038776
3: -0.0043946, -0.0035242, -0.0044564, -0.0034786, -0.0009160, 0.0009323
4: 0.0021898, 0.0072243, 0.0020420, 0.0074682, -0.0046870, 0.0046610
5: -0.0024121, 0.0025060, -0.0026906, 0.0027678, -0.0043874, 0.0043926
6: -0.0064387, -0.0035089, -0.0065714, -0.0034767, -0.0028529, 0.0029430
7: -0.0018875, 0.0032388, -0.0019830, 0.0033746, -0.0052063, 0.0048904
8: -0.0007981, -0.0000571, -0.0008673, -0.0000032, -0.0007949, 0.0008102
9: 0.9992295, 1.0107375, 0.9983398, 1.0113953, -0.0086922, 0.0090713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052380, upper bound: 0.0051086
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052380, upper bound: 0.0051086
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032953, -0.0005048, -0.0033639, -0.0004904, -0.0027502, 0.0028233
1: -0.0041514, 0.0028352, -0.0041670, 0.0029255, -0.0039791, 0.0038741
2: 0.0032893, 0.0091091, 0.0032806, 0.0091473, -0.0032977, 0.0032790
3: -0.0043733, -0.0035593, -0.0044101, -0.0035543, -0.0008189, 0.0008508
4: 0.0022123, 0.0073908, 0.0021853, 0.0074202, -0.0045305, 0.0045125
5: -0.0026146, 0.0025679, -0.0026401, 0.0026165, -0.0042178, 0.0042067
6: -0.0065059, -0.0035509, -0.0065349, -0.0035410, -0.0028078, 0.0028309
7: -0.0017212, 0.0033423, -0.0017375, 0.0034001, -0.0046740, 0.0046290
8: -0.0007904, -0.0000553, -0.0007954, -0.0000243, -0.0007210, 0.0006929
9: 0.9990917, 1.0113105, 0.9989867, 1.0113446, -0.0078877, 0.0079342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048856, upper bound: 0.0053051
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048856, upper bound: 0.0053051
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032483, -0.0004310, -0.0033280, -0.0004977, -0.0027306, 0.0028970
1: -0.0042087, 0.0027037, -0.0041584, 0.0028568, -0.0041015, 0.0038361
2: 0.0032714, 0.0090578, 0.0032856, 0.0091165, -0.0032791, 0.0032682
3: -0.0043753, -0.0034894, -0.0044002, -0.0035566, -0.0008187, 0.0009107
4: 0.0022230, 0.0073920, 0.0022036, 0.0074036, -0.0045385, 0.0044823
5: -0.0026406, 0.0025054, -0.0026257, 0.0025782, -0.0042045, 0.0042069
6: -0.0064976, -0.0035325, -0.0065194, -0.0035457, -0.0027883, 0.0028424
7: -0.0018067, 0.0033265, -0.0017306, 0.0033760, -0.0047243, 0.0046065
8: -0.0008551, -0.0000546, -0.0007931, -0.0000327, -0.0007842, 0.0006972
9: 0.9992326, 1.0113724, 0.9990699, 1.0113254, -0.0078586, 0.0078984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0038699, upper bound: 0.0046080
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049238, upper bound: 0.0052988
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032953, -0.0005048, -0.0033709, -0.0004526, -0.0027971, 0.0028428
1: -0.0041514, 0.0028352, -0.0041560, 0.0030233, -0.0040481, 0.0038514
2: 0.0032893, 0.0091091, 0.0033238, 0.0092772, -0.0034741, 0.0032954
3: -0.0043733, -0.0035593, -0.0044165, -0.0034958, -0.0008775, 0.0008572
4: 0.0022123, 0.0073908, 0.0020514, 0.0074356, -0.0045544, 0.0046475
5: -0.0026146, 0.0025679, -0.0026542, 0.0027041, -0.0043077, 0.0042315
6: -0.0065059, -0.0035509, -0.0065462, -0.0034476, -0.0029005, 0.0028488
7: -0.0017212, 0.0033423, -0.0019537, 0.0033690, -0.0046929, 0.0048731
8: -0.0007904, -0.0000553, -0.0008375, -0.0000292, -0.0007289, 0.0007497
9: 0.9990917, 1.0113105, 0.9987615, 1.0112870, -0.0078949, 0.0081918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0052440
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048953, upper bound: 0.0052440
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032483, -0.0004310, -0.0033348, -0.0004596, -0.0027775, 0.0029038
1: -0.0042087, 0.0027037, -0.0041477, 0.0029533, -0.0041645, 0.0038131
2: 0.0032714, 0.0090578, 0.0033287, 0.0092457, -0.0034522, 0.0032851
3: -0.0043753, -0.0034894, -0.0044070, -0.0034980, -0.0008773, 0.0009175
4: 0.0022230, 0.0073920, 0.0020698, 0.0074194, -0.0045631, 0.0046173
5: -0.0026406, 0.0025054, -0.0026399, 0.0026661, -0.0042921, 0.0042324
6: -0.0064976, -0.0035325, -0.0065310, -0.0034520, -0.0028810, 0.0028608
7: -0.0018067, 0.0033265, -0.0019470, 0.0033445, -0.0047440, 0.0048507
8: -0.0008551, -0.0000546, -0.0008352, -0.0000374, -0.0007928, 0.0007538
9: 0.9992326, 1.0113724, 0.9988433, 1.0112675, -0.0078674, 0.0081490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0033246, upper bound: 0.0039477
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0049377, upper bound: 0.0052381
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033030, -0.0004667, -0.0033174, -0.0004942, -0.0027610, 0.0028494
1: -0.0041405, 0.0029335, -0.0041829, 0.0028498, -0.0039074, 0.0039493
2: 0.0033329, 0.0092383, 0.0032395, 0.0091180, -0.0032828, 0.0035034
3: -0.0043792, -0.0035008, -0.0043977, -0.0035553, -0.0008240, 0.0008969
4: 0.0020784, 0.0074069, 0.0022062, 0.0074727, -0.0046871, 0.0045029
5: -0.0026286, 0.0026553, -0.0026734, 0.0025751, -0.0041874, 0.0043055
6: -0.0065165, -0.0034572, -0.0065582, -0.0035483, -0.0028138, 0.0029395
7: -0.0019377, 0.0033118, -0.0017292, 0.0034301, -0.0049574, 0.0046503
8: -0.0008327, -0.0000607, -0.0007948, -0.0000422, -0.0007655, 0.0006948
9: 0.9988671, 1.0112519, 0.9990715, 1.0114216, -0.0081989, 0.0078554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048355, upper bound: 0.0053887
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048355, upper bound: 0.0053887
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032562, -0.0003895, -0.0033174, -0.0004942, -0.0027619, 0.0029278
1: -0.0041975, 0.0027993, -0.0041829, 0.0028498, -0.0040688, 0.0039312
2: 0.0033144, 0.0091859, 0.0032395, 0.0091180, -0.0032923, 0.0034387
3: -0.0043838, -0.0034306, -0.0043977, -0.0035553, -0.0008286, 0.0009671
4: 0.0020888, 0.0074093, 0.0022062, 0.0074727, -0.0046787, 0.0045018
5: -0.0026567, 0.0025938, -0.0026734, 0.0025751, -0.0042128, 0.0042558
6: -0.0065101, -0.0034382, -0.0065582, -0.0035483, -0.0028063, 0.0029588
7: -0.0020236, 0.0032964, -0.0017292, 0.0034301, -0.0050450, 0.0046376
8: -0.0008982, -0.0000594, -0.0007948, -0.0000422, -0.0008366, 0.0007019
9: 0.9990049, 1.0113225, 0.9990715, 1.0114216, -0.0080566, 0.0079067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048355, upper bound: 0.0053887
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048355, upper bound: 0.0053887
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0033122, -0.0004655, -0.0032717, -0.0004212, -0.0028910, 0.0028062
1: -0.0041417, 0.0029099, -0.0042401, 0.0027191, -0.0038815, 0.0038531
2: 0.0033299, 0.0092411, 0.0032216, 0.0090675, -0.0032763, 0.0036180
3: -0.0044029, -0.0035004, -0.0044028, -0.0034854, -0.0009175, 0.0009024
4: 0.0020742, 0.0073865, 0.0022166, 0.0074748, -0.0046941, 0.0045364
5: -0.0026092, 0.0026619, -0.0027010, 0.0025135, -0.0041413, 0.0043426
6: -0.0065088, -0.0034544, -0.0065513, -0.0035298, -0.0028410, 0.0029447
7: -0.0019414, 0.0033244, -0.0018149, 0.0034146, -0.0052211, 0.0047279
8: -0.0008302, -0.0000377, -0.0008594, -0.0000410, -0.0007893, 0.0007837
9: 0.9988534, 1.0112327, 0.9992105, 1.0114871, -0.0083509, 0.0078606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048593, upper bound: 0.0053061
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048593, upper bound: 0.0053061
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0032953, -0.0005048, -0.0036716, -0.0004880, -0.0027961, 0.0031626
1: -0.0041514, 0.0028352, -0.0040891, 0.0036551, -0.0051746, 0.0042435
2: 0.0032893, 0.0091091, 0.0033648, 0.0095920, -0.0040442, 0.0034304
3: -0.0043733, -0.0035593, -0.0044630, -0.0035504, -0.0008229, 0.0009037
4: 0.0022123, 0.0073908, 0.0020107, 0.0074093, -0.0045336, 0.0046754
5: -0.0026146, 0.0025679, -0.0026243, 0.0028686, -0.0044781, 0.0042201
6: -0.0065059, -0.0035509, -0.0065558, -0.0034884, -0.0028562, 0.0028539
7: -0.0017212, 0.0033423, -0.0019088, 0.0033546, -0.0046775, 0.0048410
8: -0.0007904, -0.0000553, -0.0008051, 0.0000146, -0.0007685, 0.0007080
9: 0.9990917, 1.0113105, 0.9981290, 1.0112319, -0.0080138, 0.0090761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048848, upper bound: 0.0055035
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048848, upper bound: 0.0055035
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032483, -0.0004310, -0.0036384, -0.0004955, -0.0027528, 0.0032074
1: -0.0042087, 0.0027037, -0.0040812, 0.0035948, -0.0052871, 0.0042049
2: 0.0032714, 0.0090578, 0.0033695, 0.0095627, -0.0040288, 0.0034199
3: -0.0043753, -0.0034894, -0.0044541, -0.0035526, -0.0008227, 0.0009646
4: 0.0022230, 0.0073920, 0.0020296, 0.0073942, -0.0045412, 0.0046456
5: -0.0026406, 0.0025054, -0.0026113, 0.0028308, -0.0044629, 0.0042207
6: -0.0064976, -0.0035325, -0.0065401, -0.0034928, -0.0028371, 0.0028649
7: -0.0018067, 0.0033265, -0.0019018, 0.0033310, -0.0047273, 0.0048184
8: -0.0008551, -0.0000546, -0.0008030, 0.0000058, -0.0008314, 0.0007121
9: 0.9992326, 1.0113724, 0.9982082, 1.0112145, -0.0079875, 0.0090397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048848, upper bound: 0.0055183
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048848, upper bound: 0.0055183
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0032953, -0.0005048, -0.0036793, -0.0004573, -0.0028380, 0.0031745
1: -0.0041514, 0.0028352, -0.0040719, 0.0037512, -0.0052703, 0.0042317
2: 0.0032893, 0.0091091, 0.0034115, 0.0097190, -0.0042082, 0.0034155
3: -0.0043733, -0.0035593, -0.0044706, -0.0034949, -0.0008783, 0.0009113
4: 0.0022123, 0.0073908, 0.0018776, 0.0074179, -0.0045504, 0.0048092
5: -0.0026146, 0.0025679, -0.0026345, 0.0029528, -0.0045669, 0.0042418
6: -0.0065059, -0.0035509, -0.0065652, -0.0033960, -0.0029447, 0.0028663
7: -0.0017212, 0.0033423, -0.0021077, 0.0033243, -0.0046837, 0.0050697
8: -0.0007904, -0.0000553, -0.0008454, 0.0000094, -0.0007757, 0.0007639
9: 0.9990917, 1.0113105, 0.9979050, 1.0111693, -0.0080148, 0.0093411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048951, upper bound: 0.0054678
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048951, upper bound: 0.0054678
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032483, -0.0004310, -0.0036457, -0.0004644, -0.0027839, 0.0032147
1: -0.0042087, 0.0027037, -0.0040642, 0.0036916, -0.0053800, 0.0041932
2: 0.0032714, 0.0090578, 0.0034161, 0.0096898, -0.0041916, 0.0034050
3: -0.0043753, -0.0034894, -0.0044612, -0.0034970, -0.0008783, 0.0009718
4: 0.0022230, 0.0073920, 0.0018963, 0.0074031, -0.0045589, 0.0047795
5: -0.0026406, 0.0025054, -0.0026217, 0.0029152, -0.0045513, 0.0042432
6: -0.0064976, -0.0035325, -0.0065499, -0.0034002, -0.0029257, 0.0028780
7: -0.0018067, 0.0033265, -0.0021012, 0.0033011, -0.0047344, 0.0050474
8: -0.0008551, -0.0000546, -0.0008434, 0.0000007, -0.0008389, 0.0007679
9: 0.9992326, 1.0113724, 0.9979852, 1.0111523, -0.0079886, 0.0093010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0054692
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048952, upper bound: 0.0054692
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0033030, -0.0004667, -0.0036152, -0.0004921, -0.0028084, 0.0031485
1: -0.0041405, 0.0029335, -0.0041026, 0.0035671, -0.0050954, 0.0043277
2: 0.0033329, 0.0092383, 0.0033249, 0.0095576, -0.0040193, 0.0036511
3: -0.0043792, -0.0035008, -0.0044507, -0.0035512, -0.0008280, 0.0009499
4: 0.0020784, 0.0074069, 0.0020335, 0.0074607, -0.0046868, 0.0046664
5: -0.0026286, 0.0026553, -0.0026573, 0.0028254, -0.0044478, 0.0043190
6: -0.0065165, -0.0034572, -0.0065760, -0.0034958, -0.0028625, 0.0029586
7: -0.0019377, 0.0033118, -0.0019001, 0.0033862, -0.0049542, 0.0048618
8: -0.0008327, -0.0000607, -0.0008048, -0.0000030, -0.0008120, 0.0007102
9: 0.9988671, 1.0112519, 0.9982234, 1.0113095, -0.0083287, 0.0089980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048331, upper bound: 0.0055551
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048331, upper bound: 0.0055551
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032562, -0.0003895, -0.0036152, -0.0004921, -0.0027641, 0.0032256
1: -0.0041975, 0.0027993, -0.0041026, 0.0035671, -0.0052568, 0.0043096
2: 0.0033144, 0.0091859, 0.0033249, 0.0095576, -0.0040288, 0.0035864
3: -0.0043838, -0.0034306, -0.0044507, -0.0035512, -0.0008326, 0.0010201
4: 0.0020888, 0.0074093, 0.0020335, 0.0074607, -0.0046784, 0.0046652
5: -0.0026567, 0.0025938, -0.0026573, 0.0028254, -0.0044732, 0.0042693
6: -0.0065101, -0.0034382, -0.0065760, -0.0034958, -0.0028550, 0.0029778
7: -0.0020236, 0.0032964, -0.0019001, 0.0033862, -0.0050418, 0.0048492
8: -0.0008982, -0.0000594, -0.0008048, -0.0000030, -0.0008832, 0.0007174
9: 0.9990049, 1.0113225, 0.9982234, 1.0113095, -0.0081864, 0.0090492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048331, upper bound: 0.0055551
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048331, upper bound: 0.0055551
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0033122, -0.0004655, -0.0035823, -0.0004043, -0.0029079, 0.0031168
1: -0.0041417, 0.0029099, -0.0041958, 0.0034754, -0.0050656, 0.0043108
2: 0.0033299, 0.0092411, 0.0032977, 0.0095160, -0.0040051, 0.0037749
3: -0.0044029, -0.0035004, -0.0044564, -0.0034786, -0.0009244, 0.0009561
4: 0.0020742, 0.0073865, 0.0020420, 0.0074682, -0.0046996, 0.0047003
5: -0.0026092, 0.0026619, -0.0026906, 0.0027678, -0.0044002, 0.0043661
6: -0.0065088, -0.0034544, -0.0065714, -0.0034767, -0.0028891, 0.0029652
7: -0.0019414, 0.0033244, -0.0019830, 0.0033746, -0.0052201, 0.0049310
8: -0.0008302, -0.0000377, -0.0008673, -0.0000032, -0.0008271, 0.0007991
9: 0.9988534, 1.0112327, 0.9983398, 1.0113953, -0.0085082, 0.0090004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048593, upper bound: 0.0054900
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0048593, upper bound: 0.0054901
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035519, -0.0005865, -0.0033167, -0.0005763, -0.0029755, 0.0027302
1: -0.0038110, 0.0033667, -0.0039123, 0.0027462, -0.0041842, 0.0049078
2: 0.0036121, 0.0093863, 0.0035197, 0.0089807, -0.0033102, 0.0038675
3: -0.0044198, -0.0035781, -0.0044032, -0.0035779, -0.0008419, 0.0008251
4: 0.0021656, 0.0072134, 0.0023121, 0.0072464, -0.0045217, 0.0043415
5: -0.0023877, 0.0026535, -0.0024287, 0.0024589, -0.0040449, 0.0042723
6: -0.0064448, -0.0035595, -0.0064506, -0.0036001, -0.0027340, 0.0027795
7: -0.0018019, 0.0031677, -0.0016613, 0.0032739, -0.0047602, 0.0045011
8: -0.0007711, -0.0000339, -0.0007641, -0.0000422, -0.0007226, 0.0007243
9: 0.9986633, 1.0106405, 0.9993888, 1.0108017, -0.0086515, 0.0076913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052146, upper bound: 0.0050109
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052146, upper bound: 0.0050109
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0035120, -0.0005037, -0.0032805, -0.0005837, -0.0029283, 0.0027768
1: -0.0038949, 0.0032696, -0.0039033, 0.0026753, -0.0042889, 0.0048531
2: 0.0035907, 0.0093407, 0.0035249, 0.0089465, -0.0032960, 0.0038423
3: -0.0044214, -0.0035095, -0.0043927, -0.0035801, -0.0008413, 0.0008832
4: 0.0021770, 0.0072133, 0.0023311, 0.0072283, -0.0045277, 0.0043102
5: -0.0024071, 0.0025843, -0.0024126, 0.0024187, -0.0040300, 0.0042688
6: -0.0064363, -0.0035390, -0.0064345, -0.0036046, -0.0027149, 0.0027902
7: -0.0018921, 0.0031579, -0.0016544, 0.0032507, -0.0048038, 0.0044833
8: -0.0008314, -0.0000344, -0.0007616, -0.0000506, -0.0007808, 0.0007260
9: 0.9987947, 1.0106918, 0.9994775, 1.0107807, -0.0086024, 0.0076441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042702, upper bound: 0.0045047
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051999, upper bound: 0.0050338
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035519, -0.0005865, -0.0033266, -0.0005351, -0.0030168, 0.0027402
1: -0.0038110, 0.0033667, -0.0039036, 0.0028462, -0.0042605, 0.0048834
2: 0.0036121, 0.0093863, 0.0035445, 0.0091288, -0.0035004, 0.0039032
3: -0.0044198, -0.0035781, -0.0044085, -0.0035199, -0.0008999, 0.0008304
4: 0.0021656, 0.0072134, 0.0021667, 0.0072778, -0.0045623, 0.0044819
5: -0.0023877, 0.0026535, -0.0024611, 0.0025500, -0.0041356, 0.0043139
6: -0.0064448, -0.0035595, -0.0064781, -0.0035023, -0.0028305, 0.0028084
7: -0.0018019, 0.0031677, -0.0018995, 0.0032829, -0.0048090, 0.0047751
8: -0.0007711, -0.0000339, -0.0008058, -0.0000483, -0.0007228, 0.0007719
9: 0.9986633, 1.0106405, 0.9991314, 1.0107950, -0.0086952, 0.0079586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052287, upper bound: 0.0049500
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052287, upper bound: 0.0049500
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0035120, -0.0005037, -0.0032904, -0.0005422, -0.0029698, 0.0027867
1: -0.0038949, 0.0032696, -0.0038946, 0.0027733, -0.0043600, 0.0048284
2: 0.0035907, 0.0093407, 0.0035496, 0.0090957, -0.0034819, 0.0038783
3: -0.0044214, -0.0035095, -0.0043986, -0.0035221, -0.0008993, 0.0008891
4: 0.0021770, 0.0072133, 0.0021858, 0.0072596, -0.0045684, 0.0044492
5: -0.0024071, 0.0025843, -0.0024446, 0.0025100, -0.0041193, 0.0043104
6: -0.0064363, -0.0035390, -0.0064623, -0.0035068, -0.0028110, 0.0028193
7: -0.0018921, 0.0031579, -0.0018928, 0.0032608, -0.0048553, 0.0047569
8: -0.0008314, -0.0000344, -0.0008034, -0.0000569, -0.0007745, 0.0007690
9: 0.9987947, 1.0106918, 0.9992200, 1.0107740, -0.0086466, 0.0079066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0039158, upper bound: 0.0040061
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0052126, upper bound: 0.0049621
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035587, -0.0005516, -0.0032741, -0.0005805, -0.0029781, 0.0027224
1: -0.0037992, 0.0034713, -0.0039274, 0.0026765, -0.0041130, 0.0050141
2: 0.0036529, 0.0095179, 0.0034727, 0.0089552, -0.0032948, 0.0040909
3: -0.0044261, -0.0035232, -0.0043905, -0.0035790, -0.0008472, 0.0008674
4: 0.0020275, 0.0072330, 0.0023307, 0.0072982, -0.0046770, 0.0043401
5: -0.0024099, 0.0027390, -0.0024637, 0.0024213, -0.0040269, 0.0043724
6: -0.0064627, -0.0034646, -0.0064743, -0.0036065, -0.0027453, 0.0028909
7: -0.0020203, 0.0031648, -0.0016544, 0.0033157, -0.0050467, 0.0045332
8: -0.0008111, -0.0000402, -0.0007635, -0.0000603, -0.0007508, 0.0007232
9: 0.9984236, 1.0106139, 0.9994619, 1.0108936, -0.0089897, 0.0076388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051781, upper bound: 0.0051420
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051781, upper bound: 0.0051420
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0035200, -0.0004660, -0.0032741, -0.0005805, -0.0029394, 0.0028081
1: -0.0038839, 0.0033729, -0.0039274, 0.0026765, -0.0042548, 0.0049730
2: 0.0036323, 0.0094723, 0.0034727, 0.0089552, -0.0033158, 0.0040422
3: -0.0044287, -0.0034538, -0.0043905, -0.0035790, -0.0008497, 0.0009368
4: 0.0020383, 0.0072340, 0.0023307, 0.0072982, -0.0046677, 0.0043406
5: -0.0024278, 0.0026741, -0.0024637, 0.0024213, -0.0040566, 0.0043231
6: -0.0064556, -0.0034437, -0.0064743, -0.0036065, -0.0027365, 0.0029110
7: -0.0021130, 0.0031570, -0.0016544, 0.0033157, -0.0051276, 0.0045231
8: -0.0008725, -0.0000410, -0.0007635, -0.0000603, -0.0008122, 0.0007225
9: 0.9985563, 1.0106697, 0.9994619, 1.0108936, -0.0088485, 0.0076870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051781, upper bound: 0.0051420
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051781, upper bound: 0.0051420
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035587, -0.0005516, -0.0032259, -0.0005107, -0.0030480, 0.0026742
1: -0.0037992, 0.0034713, -0.0039718, 0.0025366, -0.0041260, 0.0051736
2: 0.0036529, 0.0095179, 0.0034661, 0.0088936, -0.0032366, 0.0040901
3: -0.0044261, -0.0035232, -0.0043937, -0.0035120, -0.0009141, 0.0008706
4: 0.0020275, 0.0072330, 0.0023433, 0.0072933, -0.0046708, 0.0043302
5: -0.0024099, 0.0027390, -0.0024757, 0.0023506, -0.0039772, 0.0043906
6: -0.0064627, -0.0034646, -0.0064659, -0.0035860, -0.0027639, 0.0028812
7: -0.0020203, 0.0031648, -0.0017444, 0.0033059, -0.0050369, 0.0046223
8: -0.0008111, -0.0000402, -0.0008245, -0.0000594, -0.0007517, 0.0007842
9: 0.9984236, 1.0106139, 0.9996260, 1.0109274, -0.0090122, 0.0075014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051781, upper bound: 0.0050194
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051781, upper bound: 0.0050194
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0035200, -0.0004660, -0.0032259, -0.0005107, -0.0030093, 0.0027599
1: -0.0038839, 0.0033729, -0.0039718, 0.0025366, -0.0040827, 0.0049734
2: 0.0036323, 0.0094723, 0.0034661, 0.0088936, -0.0032840, 0.0040685
3: -0.0044287, -0.0034538, -0.0043937, -0.0035120, -0.0009167, 0.0009400
4: 0.0020383, 0.0072340, 0.0023433, 0.0072933, -0.0046982, 0.0043658
5: -0.0024278, 0.0026741, -0.0024757, 0.0023506, -0.0040474, 0.0043867
6: -0.0064556, -0.0034437, -0.0064659, -0.0035860, -0.0027644, 0.0029112
7: -0.0021130, 0.0031570, -0.0017444, 0.0033059, -0.0050920, 0.0045867
8: -0.0008725, -0.0000410, -0.0008245, -0.0000594, -0.0007800, 0.0007387
9: 0.9985563, 1.0106697, 0.9996260, 1.0109274, -0.0089605, 0.0076287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051781, upper bound: 0.0050774
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0051781, upper bound: 0.0050774
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035519, -0.0005865, -0.0033639, -0.0004904, -0.0030545, 0.0027774
1: -0.0038110, 0.0033667, -0.0041670, 0.0029255, -0.0044838, 0.0052753
2: 0.0036121, 0.0093863, 0.0032806, 0.0091473, -0.0035869, 0.0041720
3: -0.0044198, -0.0035781, -0.0044101, -0.0035543, -0.0008654, 0.0008321
4: 0.0021656, 0.0072134, 0.0021853, 0.0074202, -0.0046945, 0.0044792
5: -0.0023877, 0.0026535, -0.0026401, 0.0026165, -0.0042426, 0.0045053
6: -0.0064448, -0.0035595, -0.0065349, -0.0035410, -0.0027901, 0.0028545
7: -0.0018019, 0.0031677, -0.0017375, 0.0034001, -0.0048763, 0.0045933
8: -0.0007711, -0.0000339, -0.0007954, -0.0000243, -0.0007468, 0.0007615
9: 0.9986633, 1.0106405, 0.9989867, 1.0113446, -0.0092709, 0.0082605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055035, upper bound: 0.0048848
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055035, upper bound: 0.0048848
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0035120, -0.0005037, -0.0033280, -0.0004977, -0.0030143, 0.0028243
1: -0.0038949, 0.0032696, -0.0041584, 0.0028568, -0.0045875, 0.0052207
2: 0.0035907, 0.0093407, 0.0032856, 0.0091165, -0.0035816, 0.0041467
3: -0.0044214, -0.0035095, -0.0044002, -0.0035566, -0.0008648, 0.0008907
4: 0.0021770, 0.0072133, 0.0022036, 0.0074036, -0.0047011, 0.0044495
5: -0.0024071, 0.0025843, -0.0026257, 0.0025782, -0.0042311, 0.0045023
6: -0.0064363, -0.0035390, -0.0065194, -0.0035457, -0.0027709, 0.0028657
7: -0.0018921, 0.0031579, -0.0017306, 0.0033760, -0.0049196, 0.0045750
8: -0.0008314, -0.0000344, -0.0007931, -0.0000327, -0.0007987, 0.0007587
9: 0.9987947, 1.0106918, 0.9990699, 1.0113254, -0.0092206, 0.0082256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0042976, upper bound: 0.0043163
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054819, upper bound: 0.0049238
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035519, -0.0005865, -0.0033709, -0.0004526, -0.0030992, 0.0027845
1: -0.0038110, 0.0033667, -0.0041560, 0.0030233, -0.0045424, 0.0052328
2: 0.0036121, 0.0093863, 0.0033238, 0.0092772, -0.0037396, 0.0041593
3: -0.0044198, -0.0035781, -0.0044165, -0.0034958, -0.0009239, 0.0008384
4: 0.0021656, 0.0072134, 0.0020514, 0.0074356, -0.0047112, 0.0046079
5: -0.0023877, 0.0026535, -0.0026542, 0.0027041, -0.0043253, 0.0045201
6: -0.0064448, -0.0035595, -0.0065462, -0.0034476, -0.0028798, 0.0028705
7: -0.0018019, 0.0031677, -0.0019537, 0.0033690, -0.0048881, 0.0048251
8: -0.0007711, -0.0000339, -0.0008375, -0.0000292, -0.0007420, 0.0008036
9: 0.9986633, 1.0106405, 0.9987615, 1.0112870, -0.0092552, 0.0084978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 104

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055087, upper bound: 0.0048330
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0055087, upper bound: 0.0048330
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0035120, -0.0005037, -0.0033348, -0.0004596, -0.0030524, 0.0028311
1: -0.0038949, 0.0032696, -0.0041477, 0.0029533, -0.0046433, 0.0051780
2: 0.0035907, 0.0093407, 0.0033287, 0.0092457, -0.0037331, 0.0041342
3: -0.0044214, -0.0035095, -0.0044070, -0.0034980, -0.0009234, 0.0008975
4: 0.0021770, 0.0072133, 0.0020698, 0.0074194, -0.0047181, 0.0045781
5: -0.0024071, 0.0025843, -0.0026399, 0.0026661, -0.0043124, 0.0045174
6: -0.0064363, -0.0035390, -0.0065310, -0.0034520, -0.0028607, 0.0028820
7: -0.0018921, 0.0031579, -0.0019470, 0.0033445, -0.0049322, 0.0048069
8: -0.0008314, -0.0000344, -0.0008352, -0.0000374, -0.0007940, 0.0008008
9: 0.9987947, 1.0106918, 0.9988433, 1.0112675, -0.0092056, 0.0084605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0038101, upper bound: 0.0036911
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054901, upper bound: 0.0048593
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035587, -0.0005516, -0.0033174, -0.0004942, -0.0030644, 0.0027658
1: -0.0037992, 0.0034713, -0.0041829, 0.0028498, -0.0044070, 0.0053747
2: 0.0036529, 0.0095179, 0.0032395, 0.0091180, -0.0035578, 0.0043805
3: -0.0044261, -0.0035232, -0.0043977, -0.0035553, -0.0008709, 0.0008745
4: 0.0020275, 0.0072330, 0.0022062, 0.0074727, -0.0048466, 0.0044730
5: -0.0024099, 0.0027390, -0.0026734, 0.0025751, -0.0042168, 0.0046010
6: -0.0064627, -0.0034646, -0.0065582, -0.0035483, -0.0027999, 0.0029664
7: -0.0020203, 0.0031648, -0.0017292, 0.0034301, -0.0051559, 0.0046227
8: -0.0008111, -0.0000402, -0.0007948, -0.0000422, -0.0007688, 0.0007546
9: 0.9984236, 1.0106139, 0.9990715, 1.0114216, -0.0095909, 0.0081827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054678, upper bound: 0.0050529
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054678, upper bound: 0.0050529
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0035200, -0.0004660, -0.0033174, -0.0004942, -0.0030257, 0.0028514
1: -0.0038839, 0.0033729, -0.0041829, 0.0028498, -0.0045487, 0.0053336
2: 0.0036323, 0.0094723, 0.0032395, 0.0091180, -0.0035788, 0.0043318
3: -0.0044287, -0.0034538, -0.0043977, -0.0035553, -0.0008734, 0.0009439
4: 0.0020383, 0.0072340, 0.0022062, 0.0074727, -0.0048373, 0.0044736
5: -0.0024278, 0.0026741, -0.0026734, 0.0025751, -0.0042465, 0.0045517
6: -0.0064556, -0.0034437, -0.0065582, -0.0035483, -0.0027912, 0.0029864
7: -0.0021130, 0.0031570, -0.0017292, 0.0034301, -0.0052368, 0.0046125
8: -0.0008725, -0.0000410, -0.0007948, -0.0000422, -0.0008303, 0.0007538
9: 0.9985563, 1.0106697, 0.9990715, 1.0114216, -0.0094496, 0.0082309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054678, upper bound: 0.0050529
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054678, upper bound: 0.0050529
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035587, -0.0005516, -0.0032717, -0.0004212, -0.0031375, 0.0027201
1: -0.0037992, 0.0034713, -0.0042401, 0.0027191, -0.0044116, 0.0055511
2: 0.0036529, 0.0095179, 0.0032216, 0.0090675, -0.0035335, 0.0044177
3: -0.0044261, -0.0035232, -0.0044028, -0.0034854, -0.0009407, 0.0008796
4: 0.0020275, 0.0072330, 0.0022166, 0.0074748, -0.0048480, 0.0044661
5: -0.0024099, 0.0027390, -0.0027010, 0.0025135, -0.0041783, 0.0046423
6: -0.0064627, -0.0034646, -0.0065513, -0.0035298, -0.0028174, 0.0029587
7: -0.0020203, 0.0031648, -0.0018149, 0.0034146, -0.0051394, 0.0047001
8: -0.0008111, -0.0000402, -0.0008594, -0.0000410, -0.0007701, 0.0008192
9: 0.9984236, 1.0106139, 0.9992105, 1.0114871, -0.0096762, 0.0080922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054678, upper bound: 0.0048952
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0054678, upper bound: 0.0048952
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0035200, -0.0004660, -0.0032717, -0.0004212, -0.0030839, 0.0028057
1: -0.0038839, 0.0033729, -0.0042401, 0.0027191, -0.0043802, 0.0053390
2: 0.0036323, 0.0094723, 0.0032216, 0.0090675, -0.0035404, 0.0043560
3: -0.0044287, -0.0034538, -0.0044028, -0.0034854, -0.0009433, 0.0009490
4: 0.0020383, 0.0072340, 0.0022166, 0.0074748, -0.0048686, 0.0044937
5: -0.0024278, 0.0026741, -0.0027010, 0.0025135, -0.0042288, 0.0046148
6: -0.0064556, -0.0034437, -0.0065513, -0.0035298, -0.0028170, 0.0029870
7: -0.0021130, 0.0031570, -0.0018149, 0.0034146, -0.0051985, 0.0046697
8: -0.0008725, -0.0000410, -0.0008594, -0.0000410, -0.0008093, 0.0007844
9: 0.9985563, 1.0106697, 0.9992105, 1.0114871, -0.0095578, 0.0081547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.95 + 598.01 = 600.96 seconds
