## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00018056


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0068597, 0.0071577, 0.0068597, 0.0071577, -0.0001554, 0.0001554)
1: (0.0013686, 0.0019458, 0.0013686, 0.0019458, -0.0003009, 0.0003009)
2: (0.0012330, 0.0058885, 0.0012330, 0.0058885, -0.0024271, 0.0024271)
3: (-0.0030385, -0.0026227, -0.0030385, -0.0026227, -0.0002168, 0.0002168)
4: (0.0065939, 0.0086113, 0.0065939, 0.0086113, -0.0010518, 0.0010518)
5: (-0.0018052, -0.0015041, -0.0018052, -0.0015041, -0.0001570, 0.0001570)
6: (0.9930117, 0.9935640, 0.9930117, 0.9935640, -0.0002880, 0.0002880)
7: (-0.0014468, 0.0022051, -0.0014468, 0.0022051, -0.0019038, 0.0019038)
8: (0.0005351, 0.0016792, 0.0005351, 0.0016792, -0.0005965, 0.0005965)
9: (-0.0106806, -0.0083971, -0.0106806, -0.0083971, -0.0011905, 0.0011905)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 1.46 = 3.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0002139, upper bound: 0.0002140

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002067, upper bound: 0.0001803
time: 0.59 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0002067, upper bound: 0.0002067
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.34 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 6, lower bound: -0.0002067, upper bound: 0.0001803
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.34
Output dim: 6, lower bound: -0.0002067, upper bound: 0.0002067

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0068599, 0.0071403, 0.0068597, 0.0071539, -0.0001516, 0.0001401
1: 0.0013692, 0.0019122, 0.0013688, 0.0019386, -0.0002936, 0.0002714
2: 0.0015043, 0.0058843, 0.0012917, 0.0058876, -0.0021890, 0.0023683
3: -0.0030381, -0.0026469, -0.0030384, -0.0026279, -0.0002115, 0.0001955
4: 0.0065957, 0.0084937, 0.0065943, 0.0085859, -0.0010263, 0.0009486
5: -0.0017877, -0.0015043, -0.0018014, -0.0015041, -0.0001416, 0.0001532
6: 0.9930121, 0.9935318, 0.9930118, 0.9935570, -0.0002810, 0.0002597
7: -0.0014434, 0.0019923, -0.0014461, 0.0021591, -0.0018577, 0.0017171
8: 0.0005361, 0.0016125, 0.0005353, 0.0016648, -0.0005820, 0.0005380
9: -0.0105475, -0.0083992, -0.0106518, -0.0083975, -0.0010737, 0.0011616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001974, upper bound: 0.0001641
time: 0.62 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001974, upper bound: 0.0001713
time: 0.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0068417, 0.0071503, 0.0068598, 0.0071554, -0.0001745, 0.0001479
1: 0.0013339, 0.0019316, 0.0013689, 0.0019414, -0.0003380, 0.0002865
2: 0.0013475, 0.0061688, 0.0012684, 0.0058864, -0.0023105, 0.0027266
3: -0.0030635, -0.0026329, -0.0030383, -0.0026259, -0.0002435, 0.0002064
4: 0.0064724, 0.0085617, 0.0065948, 0.0085960, -0.0011816, 0.0010012
5: -0.0017978, -0.0014859, -0.0018029, -0.0015042, -0.0001495, 0.0001764
6: 0.9929785, 0.9935504, 0.9930119, 0.9935598, -0.0003235, 0.0002741
7: -0.0016666, 0.0021154, -0.0014451, 0.0021773, -0.0021388, 0.0018124
8: 0.0004662, 0.0016511, 0.0005356, 0.0016705, -0.0006701, 0.0005678
9: -0.0106245, -0.0082596, -0.0106632, -0.0083982, -0.0011333, 0.0013374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001974, upper bound: 0.0001895
time: 0.59 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001974, upper bound: 0.0001974
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.73 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 6, lower bound: -0.0001974, upper bound: 0.0001641
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 6, lower bound: -0.0001974, upper bound: 0.0001713
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 6, lower bound: -0.0001974, upper bound: 0.0001895
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 6, lower bound: -0.0001974, upper bound: 0.0001974

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0068604, 0.0071315, 0.0068598, 0.0071525, -0.0001497, 0.0001312
1: 0.0013701, 0.0018951, 0.0013689, 0.0019359, -0.0002899, 0.0002541
2: 0.0016421, 0.0058766, 0.0013132, 0.0058864, -0.0020496, 0.0023383
3: -0.0030374, -0.0026592, -0.0030383, -0.0026299, -0.0002088, 0.0001831
4: 0.0065991, 0.0084340, 0.0065948, 0.0085766, -0.0010133, 0.0008882
5: -0.0017788, -0.0015048, -0.0018000, -0.0015042, -0.0001326, 0.0001513
6: 0.9930131, 0.9935155, 0.9930119, 0.9935545, -0.0002774, 0.0002432
7: -0.0014374, 0.0018842, -0.0014451, 0.0021422, -0.0018342, 0.0016078
8: 0.0005380, 0.0015787, 0.0005356, 0.0016595, -0.0005746, 0.0005037
9: -0.0104799, -0.0084030, -0.0106412, -0.0083981, -0.0010053, 0.0011469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001827, upper bound: 0.0001641
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001827, upper bound: 0.0001641
time: 0.61 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0068506, 0.0071298, 0.0068599, 0.0071505, -0.0001645, 0.0001339
1: 0.0013511, 0.0018919, 0.0013691, 0.0019319, -0.0003187, 0.0002594
2: 0.0016680, 0.0060300, 0.0013451, 0.0058849, -0.0020925, 0.0025702
3: -0.0030511, -0.0026615, -0.0030382, -0.0026327, -0.0002296, 0.0001869
4: 0.0065326, 0.0084228, 0.0065955, 0.0085627, -0.0011138, 0.0009068
5: -0.0017771, -0.0014949, -0.0017980, -0.0015043, -0.0001354, 0.0001663
6: 0.9929949, 0.9935124, 0.9930121, 0.9935507, -0.0003049, 0.0002483
7: -0.0015578, 0.0018639, -0.0014439, 0.0021172, -0.0020161, 0.0016414
8: 0.0005003, 0.0015723, 0.0005360, 0.0016516, -0.0006316, 0.0005142
9: -0.0104672, -0.0083277, -0.0106256, -0.0083989, -0.0010264, 0.0012607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001827, upper bound: 0.0001713
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001827, upper bound: 0.0001713
time: 0.72 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.0068422, 0.0071416, 0.0068599, 0.0071540, -0.0001726, 0.0001389
1: 0.0013349, 0.0019146, 0.0013691, 0.0019387, -0.0003344, 0.0002691
2: 0.0014846, 0.0061610, 0.0012904, 0.0058851, -0.0021707, 0.0026971
3: -0.0030628, -0.0026452, -0.0030382, -0.0026278, -0.0002409, 0.0001939
4: 0.0064758, 0.0085023, 0.0065954, 0.0085864, -0.0011688, 0.0009406
5: -0.0017890, -0.0014864, -0.0018015, -0.0015043, -0.0001404, 0.0001745
6: 0.9929793, 0.9935342, 0.9930120, 0.9935573, -0.0003200, 0.0002575
7: -0.0016605, 0.0020078, -0.0014441, 0.0021601, -0.0021157, 0.0017027
8: 0.0004681, 0.0016174, 0.0005359, 0.0016651, -0.0006628, 0.0005335
9: -0.0105572, -0.0082634, -0.0106524, -0.0083988, -0.0010647, 0.0013229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001713, upper bound: 0.0001895
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001713, upper bound: 0.0001895
time: 0.62 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068600, 0.0071521, -0.0001825, 0.0001417
1: 0.0013156, 0.0019112, 0.0013693, 0.0019350, -0.0003535, 0.0002745
2: 0.0015121, 0.0063164, 0.0013205, 0.0058836, -0.0022138, 0.0028512
3: -0.0030767, -0.0026476, -0.0030381, -0.0026305, -0.0002547, 0.0001977
4: 0.0064085, 0.0084904, 0.0065960, 0.0085734, -0.0012355, 0.0009593
5: -0.0017872, -0.0014764, -0.0017996, -0.0015044, -0.0001432, 0.0001844
6: 0.9929609, 0.9935309, 0.9930122, 0.9935536, -0.0003383, 0.0002626
7: -0.0017824, 0.0019862, -0.0014429, 0.0021365, -0.0022365, 0.0017365
8: 0.0004299, 0.0016106, 0.0005363, 0.0016577, -0.0007007, 0.0005440
9: -0.0105437, -0.0081872, -0.0106377, -0.0083995, -0.0010858, 0.0013985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001713, upper bound: 0.0001974
time: 0.63 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001713, upper bound: 0.0001974
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.82 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001827, upper bound: 0.0001641
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001827, upper bound: 0.0001641
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001827, upper bound: 0.0001713
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001827, upper bound: 0.0001713
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001713, upper bound: 0.0001895
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001713, upper bound: 0.0001895
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001713, upper bound: 0.0001974
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001713, upper bound: 0.0001974

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0068604, 0.0071315, 0.0068600, 0.0071389, -0.0001380, 0.0001310
1: 0.0013701, 0.0018951, 0.0013693, 0.0019094, -0.0002673, 0.0002538
2: 0.0016421, 0.0058766, 0.0015265, 0.0058830, -0.0020474, 0.0021559
3: -0.0030374, -0.0026592, -0.0030380, -0.0026489, -0.0001926, 0.0001829
4: 0.0065991, 0.0084340, 0.0065963, 0.0084841, -0.0009342, 0.0008872
5: -0.0017788, -0.0015048, -0.0017862, -0.0015044, -0.0001324, 0.0001395
6: 0.9930131, 0.9935155, 0.9930123, 0.9935292, -0.0002558, 0.0002429
7: -0.0014374, 0.0018842, -0.0014424, 0.0019749, -0.0016911, 0.0016060
8: 0.0005380, 0.0015787, 0.0005364, 0.0016071, -0.0005298, 0.0005031
9: -0.0104799, -0.0084030, -0.0105366, -0.0083998, -0.0010042, 0.0010574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001779, upper bound: 0.0001641
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001779, upper bound: 0.0001641
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0068604, 0.0071315, 0.0068418, 0.0071489, -0.0001485, 0.0001526
1: 0.0013701, 0.0018951, 0.0013340, 0.0019289, -0.0002877, 0.0002955
2: 0.0016421, 0.0058766, 0.0013693, 0.0061676, -0.0023834, 0.0023202
3: -0.0030374, -0.0026592, -0.0030634, -0.0026349, -0.0002072, 0.0002129
4: 0.0065991, 0.0084340, 0.0064730, 0.0085523, -0.0010054, 0.0010328
5: -0.0017788, -0.0015048, -0.0017964, -0.0014860, -0.0001542, 0.0001501
6: 0.9930131, 0.9935155, 0.9929785, 0.9935479, -0.0002753, 0.0002828
7: -0.0014374, 0.0018842, -0.0016657, 0.0020982, -0.0018200, 0.0018696
8: 0.0005380, 0.0015787, 0.0004665, 0.0016457, -0.0005702, 0.0005857
9: -0.0104799, -0.0084030, -0.0106137, -0.0082602, -0.0011690, 0.0011380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001779, upper bound: 0.0001641
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001779, upper bound: 0.0001641
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068506, 0.0071298, 0.0068601, 0.0071370, -0.0001531, 0.0001338
1: 0.0013511, 0.0018919, 0.0013695, 0.0019058, -0.0002965, 0.0002591
2: 0.0016680, 0.0060300, 0.0015557, 0.0058815, -0.0020903, 0.0023913
3: -0.0030511, -0.0026615, -0.0030379, -0.0026515, -0.0002136, 0.0001867
4: 0.0065326, 0.0084228, 0.0065969, 0.0084715, -0.0010362, 0.0009058
5: -0.0017771, -0.0014949, -0.0017843, -0.0015045, -0.0001352, 0.0001547
6: 0.9929949, 0.9935124, 0.9930125, 0.9935257, -0.0002837, 0.0002480
7: -0.0015578, 0.0018639, -0.0014412, 0.0019520, -0.0018758, 0.0016396
8: 0.0005003, 0.0015723, 0.0005368, 0.0015999, -0.0005877, 0.0005137
9: -0.0104672, -0.0083277, -0.0105223, -0.0084005, -0.0010252, 0.0011729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001713
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001713
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0068506, 0.0071298, 0.0068419, 0.0071470, -0.0001625, 0.0001553
1: 0.0013511, 0.0018919, 0.0013342, 0.0019252, -0.0003148, 0.0003008
2: 0.0016680, 0.0060300, 0.0013991, 0.0061660, -0.0024260, 0.0025394
3: -0.0030511, -0.0026615, -0.0030633, -0.0026375, -0.0002268, 0.0002167
4: 0.0065326, 0.0084228, 0.0064736, 0.0085394, -0.0011004, 0.0010513
5: -0.0017771, -0.0014949, -0.0017945, -0.0014861, -0.0001569, 0.0001643
6: 0.9929949, 0.9935124, 0.9929788, 0.9935443, -0.0003013, 0.0002878
7: -0.0015578, 0.0018639, -0.0016644, 0.0020749, -0.0019920, 0.0019030
8: 0.0005003, 0.0015723, 0.0004669, 0.0016384, -0.0006241, 0.0005962
9: -0.0104672, -0.0083277, -0.0105991, -0.0082610, -0.0011899, 0.0012456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001713
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001713
time: 0.61 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0068422, 0.0071416, 0.0068600, 0.0071389, -0.0001594, 0.0001424
1: 0.0013349, 0.0019146, 0.0013693, 0.0019094, -0.0003088, 0.0002758
2: 0.0014846, 0.0061610, 0.0015265, 0.0058830, -0.0022248, 0.0024908
3: -0.0030628, -0.0026452, -0.0030380, -0.0026489, -0.0002225, 0.0001987
4: 0.0064758, 0.0085023, 0.0065963, 0.0084841, -0.0010794, 0.0009641
5: -0.0017890, -0.0014864, -0.0017862, -0.0015044, -0.0001439, 0.0001611
6: 0.9929793, 0.9935342, 0.9930123, 0.9935292, -0.0002955, 0.0002640
7: -0.0016605, 0.0020078, -0.0014424, 0.0019749, -0.0019538, 0.0017452
8: 0.0004681, 0.0016174, 0.0005364, 0.0016071, -0.0006121, 0.0005468
9: -0.0105572, -0.0082634, -0.0105366, -0.0083998, -0.0010912, 0.0012217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001895
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001895
time: 0.71 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0068422, 0.0071416, 0.0068418, 0.0071489, -0.0001459, 0.0001389
1: 0.0013349, 0.0019146, 0.0013340, 0.0019289, -0.0002825, 0.0002691
2: 0.0014846, 0.0061610, 0.0013693, 0.0061676, -0.0021702, 0.0022790
3: -0.0030628, -0.0026452, -0.0030634, -0.0026349, -0.0002035, 0.0001938
4: 0.0064758, 0.0085023, 0.0064730, 0.0085523, -0.0009876, 0.0009404
5: -0.0017890, -0.0014864, -0.0017964, -0.0014860, -0.0001404, 0.0001474
6: 0.9929793, 0.9935342, 0.9929785, 0.9935479, -0.0002704, 0.0002575
7: -0.0016605, 0.0020078, -0.0016657, 0.0020982, -0.0017877, 0.0017023
8: 0.0004681, 0.0016174, 0.0004665, 0.0016457, -0.0005601, 0.0005333
9: -0.0105572, -0.0082634, -0.0106137, -0.0082602, -0.0010645, 0.0011178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001895
time: 0.67 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001895
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068601, 0.0071370, -0.0001699, 0.0001451
1: 0.0013156, 0.0019112, 0.0013695, 0.0019058, -0.0003291, 0.0002811
2: 0.0015121, 0.0063164, 0.0015557, 0.0058815, -0.0022673, 0.0026547
3: -0.0030767, -0.0026476, -0.0030379, -0.0026515, -0.0002371, 0.0002025
4: 0.0064085, 0.0084904, 0.0065969, 0.0084715, -0.0011504, 0.0009825
5: -0.0017872, -0.0014764, -0.0017843, -0.0015045, -0.0001467, 0.0001717
6: 0.9929609, 0.9935309, 0.9930125, 0.9935257, -0.0003150, 0.0002690
7: -0.0017824, 0.0019862, -0.0014412, 0.0019520, -0.0020824, 0.0017785
8: 0.0004299, 0.0016106, 0.0005368, 0.0015999, -0.0006524, 0.0005572
9: -0.0105437, -0.0081872, -0.0105223, -0.0084005, -0.0011121, 0.0013021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001641, upper bound: 0.0001974
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001641, upper bound: 0.0001974
time: 0.64 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068419, 0.0071470, -0.0001609, 0.0001417
1: 0.0013156, 0.0019112, 0.0013342, 0.0019252, -0.0003117, 0.0002744
2: 0.0015121, 0.0063164, 0.0013991, 0.0061660, -0.0022132, 0.0025138
3: -0.0030767, -0.0026476, -0.0030633, -0.0026375, -0.0002245, 0.0001977
4: 0.0064085, 0.0084904, 0.0064736, 0.0085394, -0.0010893, 0.0009591
5: -0.0017872, -0.0014764, -0.0017945, -0.0014861, -0.0001432, 0.0001626
6: 0.9929609, 0.9935309, 0.9929788, 0.9935443, -0.0002982, 0.0002626
7: -0.0017824, 0.0019862, -0.0016644, 0.0020749, -0.0019719, 0.0017361
8: 0.0004299, 0.0016106, 0.0004669, 0.0016384, -0.0006178, 0.0005439
9: -0.0105437, -0.0081872, -0.0105991, -0.0082610, -0.0010856, 0.0012330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001641, upper bound: 0.0001973
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001641, upper bound: 0.0001974
time: 0.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.77 seconds
IS_A1_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001779, upper bound: 0.0001641
IS_A1_A1_B1_B2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001779, upper bound: 0.0001641
IS_A1_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001779, upper bound: 0.0001641
IS_A1_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001779, upper bound: 0.0001641
IS_A1_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001713
IS_A1_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001713
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001713
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001751, upper bound: 0.0001713
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001895
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001895
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001895
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001664, upper bound: 0.0001895
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001641, upper bound: 0.0001974
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001641, upper bound: 0.0001974
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001641, upper bound: 0.0001973
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 6, lower bound: -0.0001641, upper bound: 0.0001974

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0068422, 0.0071416, 0.0068604, 0.0071315, -0.0001522, 0.0001421
1: 0.0013349, 0.0019146, 0.0013701, 0.0018951, -0.0002947, 0.0002752
2: 0.0014846, 0.0061610, 0.0016421, 0.0058766, -0.0022197, 0.0023772
3: -0.0030628, -0.0026452, -0.0030374, -0.0026592, -0.0002123, 0.0001983
4: 0.0064758, 0.0085023, 0.0065991, 0.0084340, -0.0010301, 0.0009619
5: -0.0017890, -0.0014864, -0.0017788, -0.0015048, -0.0001436, 0.0001538
6: 0.9929793, 0.9935342, 0.9930131, 0.9935155, -0.0002820, 0.0002634
7: -0.0016605, 0.0020078, -0.0014374, 0.0018842, -0.0018647, 0.0017412
8: 0.0004681, 0.0016174, 0.0005380, 0.0015787, -0.0005842, 0.0005455
9: -0.0105572, -0.0082634, -0.0104799, -0.0084030, -0.0010888, 0.0011660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001745
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001555, upper bound: 0.0001744
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0068422, 0.0071416, 0.0068506, 0.0071298, -0.0001548, 0.0001581
1: 0.0013349, 0.0019146, 0.0013511, 0.0018919, -0.0002999, 0.0003062
2: 0.0014846, 0.0061610, 0.0016680, 0.0060300, -0.0024698, 0.0024187
3: -0.0030628, -0.0026452, -0.0030511, -0.0026615, -0.0002160, 0.0002206
4: 0.0064758, 0.0085023, 0.0065326, 0.0084228, -0.0010481, 0.0010703
5: -0.0017890, -0.0014864, -0.0017771, -0.0014949, -0.0001598, 0.0001565
6: 0.9929793, 0.9935342, 0.9929949, 0.9935124, -0.0002870, 0.0002930
7: -0.0016605, 0.0020078, -0.0015578, 0.0018639, -0.0018973, 0.0019374
8: 0.0004681, 0.0016174, 0.0005003, 0.0015723, -0.0005944, 0.0006070
9: -0.0105572, -0.0082634, -0.0104672, -0.0083277, -0.0012114, 0.0011863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001745
time: 0.61 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001555, upper bound: 0.0001744
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0068422, 0.0071416, 0.0068422, 0.0071416, -0.0001386, 0.0001386
1: 0.0013349, 0.0019146, 0.0013349, 0.0019146, -0.0002684, 0.0002684
2: 0.0014846, 0.0061610, 0.0014846, 0.0061610, -0.0021650, 0.0021650
3: -0.0030628, -0.0026452, -0.0030628, -0.0026452, -0.0001934, 0.0001934
4: 0.0064758, 0.0085023, 0.0064758, 0.0085023, -0.0009382, 0.0009382
5: -0.0017890, -0.0014864, -0.0017890, -0.0014864, -0.0001401, 0.0001401
6: 0.9929793, 0.9935342, 0.9929793, 0.9935342, -0.0002569, 0.0002569
7: -0.0016605, 0.0020078, -0.0016605, 0.0020078, -0.0016983, 0.0016983
8: 0.0004681, 0.0016174, 0.0004681, 0.0016174, -0.0005321, 0.0005321
9: -0.0105572, -0.0082634, -0.0105572, -0.0082634, -0.0010619, 0.0010619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001743
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001555, upper bound: 0.0001742
time: 0.63 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0068422, 0.0071416, 0.0068323, 0.0071398, -0.0001412, 0.0001545
1: 0.0013349, 0.0019146, 0.0013156, 0.0019112, -0.0002735, 0.0002993
2: 0.0014846, 0.0061610, 0.0015121, 0.0063164, -0.0024142, 0.0022060
3: -0.0030628, -0.0026452, -0.0030767, -0.0026476, -0.0001970, 0.0002156
4: 0.0064758, 0.0085023, 0.0064085, 0.0084904, -0.0009560, 0.0010462
5: -0.0017890, -0.0014864, -0.0017872, -0.0014764, -0.0001562, 0.0001427
6: 0.9929793, 0.9935342, 0.9929609, 0.9935309, -0.0002617, 0.0002864
7: -0.0016605, 0.0020078, -0.0017824, 0.0019862, -0.0017304, 0.0018938
8: 0.0004681, 0.0016174, 0.0004299, 0.0016106, -0.0005421, 0.0005933
9: -0.0105572, -0.0082634, -0.0105437, -0.0081872, -0.0011842, 0.0010820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001743
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001555, upper bound: 0.0001742
time: 0.64 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068604, 0.0071315, -0.0001636, 0.0001408
1: 0.0013156, 0.0019112, 0.0013701, 0.0018951, -0.0003169, 0.0002727
2: 0.0015121, 0.0063164, 0.0016421, 0.0058766, -0.0021995, 0.0025558
3: -0.0030767, -0.0026476, -0.0030374, -0.0026592, -0.0002283, 0.0001964
4: 0.0064085, 0.0084904, 0.0065991, 0.0084340, -0.0011075, 0.0009531
5: -0.0017872, -0.0014764, -0.0017788, -0.0015048, -0.0001423, 0.0001653
6: 0.9929609, 0.9935309, 0.9930131, 0.9935155, -0.0003032, 0.0002610
7: -0.0017824, 0.0019862, -0.0014374, 0.0018842, -0.0020048, 0.0017253
8: 0.0004299, 0.0016106, 0.0005380, 0.0015787, -0.0006281, 0.0005405
9: -0.0105437, -0.0081872, -0.0104799, -0.0084030, -0.0010788, 0.0012536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001838
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001787
time: 0.65 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068506, 0.0071298, -0.0001558, 0.0001458
1: 0.0013156, 0.0019112, 0.0013511, 0.0018919, -0.0003017, 0.0002823
2: 0.0015121, 0.0063164, 0.0016680, 0.0060300, -0.0022770, 0.0024338
3: -0.0030767, -0.0026476, -0.0030511, -0.0026615, -0.0002174, 0.0002034
4: 0.0064085, 0.0084904, 0.0065326, 0.0084228, -0.0010547, 0.0009867
5: -0.0017872, -0.0014764, -0.0017771, -0.0014949, -0.0001473, 0.0001574
6: 0.9929609, 0.9935309, 0.9929949, 0.9935124, -0.0002888, 0.0002702
7: -0.0017824, 0.0019862, -0.0015578, 0.0018639, -0.0019091, 0.0017861
8: 0.0004299, 0.0016106, 0.0005003, 0.0015723, -0.0005981, 0.0005596
9: -0.0105437, -0.0081872, -0.0104672, -0.0083277, -0.0011169, 0.0011937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001548, upper bound: 0.0001791
time: 0.61 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001787
time: 0.71 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068422, 0.0071416, -0.0001545, 0.0001412
1: 0.0013156, 0.0019112, 0.0013349, 0.0019146, -0.0002993, 0.0002735
2: 0.0015121, 0.0063164, 0.0014846, 0.0061610, -0.0022060, 0.0024142
3: -0.0030767, -0.0026476, -0.0030628, -0.0026452, -0.0002156, 0.0001970
4: 0.0064085, 0.0084904, 0.0064758, 0.0085023, -0.0010462, 0.0009560
5: -0.0017872, -0.0014764, -0.0017890, -0.0014864, -0.0001427, 0.0001562
6: 0.9929609, 0.9935309, 0.9929793, 0.9935342, -0.0002864, 0.0002617
7: -0.0017824, 0.0019862, -0.0016605, 0.0020078, -0.0018938, 0.0017304
8: 0.0004299, 0.0016106, 0.0004681, 0.0016174, -0.0005933, 0.0005421
9: -0.0105437, -0.0081872, -0.0105572, -0.0082634, -0.0010820, 0.0011842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001838
time: 0.63 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001780
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068323, 0.0071398, -0.0001423, 0.0001423
1: 0.0013156, 0.0019112, 0.0013156, 0.0019112, -0.0002756, 0.0002756
2: 0.0015121, 0.0063164, 0.0015121, 0.0063164, -0.0022227, 0.0022227
3: -0.0030767, -0.0026476, -0.0030767, -0.0026476, -0.0001985, 0.0001985
4: 0.0064085, 0.0084904, 0.0064085, 0.0084904, -0.0009632, 0.0009632
5: -0.0017872, -0.0014764, -0.0017872, -0.0014764, -0.0001438, 0.0001438
6: 0.9929609, 0.9935309, 0.9929609, 0.9935309, -0.0002637, 0.0002637
7: -0.0017824, 0.0019862, -0.0017824, 0.0019862, -0.0017436, 0.0017436
8: 0.0004299, 0.0016106, 0.0004299, 0.0016106, -0.0005462, 0.0005462
9: -0.0105437, -0.0081872, -0.0105437, -0.0081872, -0.0010902, 0.0010902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001548, upper bound: 0.0001786
time: 0.62 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001780
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.80 seconds
IS_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001745
IS_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001555, upper bound: 0.0001744
IS_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001745
IS_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001555, upper bound: 0.0001744
IS_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001743
IS_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001555, upper bound: 0.0001742
IS_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001573, upper bound: 0.0001743
IS_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001555, upper bound: 0.0001742
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001838
IS_A2_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001787
IS_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001548, upper bound: 0.0001791
IS_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001787
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001838
IS_A2_A2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001780
IS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001548, upper bound: 0.0001786
IS_A2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001780

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068632, 0.0071315, -0.0001636, 0.0001380
1: 0.0013156, 0.0019112, 0.0013755, 0.0018951, -0.0003169, 0.0002673
2: 0.0015121, 0.0063164, 0.0016421, 0.0058330, -0.0021557, 0.0025558
3: -0.0030767, -0.0026476, -0.0030335, -0.0026592, -0.0002283, 0.0001925
4: 0.0064085, 0.0084904, 0.0066180, 0.0084340, -0.0011075, 0.0009342
5: -0.0017872, -0.0014764, -0.0017788, -0.0015077, -0.0001395, 0.0001653
6: 0.9929609, 0.9935309, 0.9930182, 0.9935155, -0.0003032, 0.0002558
7: -0.0017824, 0.0019862, -0.0014032, 0.0018842, -0.0020048, 0.0016910
8: 0.0004299, 0.0016106, 0.0005487, 0.0015787, -0.0006281, 0.0005298
9: -0.0105437, -0.0081872, -0.0104799, -0.0084243, -0.0010574, 0.0012536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001789
time: 0.69 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001789
time: 0.78 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0068323, 0.0071398, 0.0068452, 0.0071416, -0.0001545, 0.0001384
1: 0.0013156, 0.0019112, 0.0013406, 0.0019146, -0.0002993, 0.0002680
2: 0.0015121, 0.0063164, 0.0014846, 0.0061152, -0.0021616, 0.0024142
3: -0.0030767, -0.0026476, -0.0030587, -0.0026452, -0.0002156, 0.0001931
4: 0.0064085, 0.0084904, 0.0064957, 0.0085023, -0.0010462, 0.0009367
5: -0.0017872, -0.0014764, -0.0017890, -0.0014894, -0.0001398, 0.0001562
6: 0.9929609, 0.9935309, 0.9929848, 0.9935342, -0.0002864, 0.0002565
7: -0.0017824, 0.0019862, -0.0016245, 0.0020078, -0.0018938, 0.0016956
8: 0.0004299, 0.0016106, 0.0004794, 0.0016174, -0.0005933, 0.0005312
9: -0.0105437, -0.0081872, -0.0105572, -0.0082859, -0.0010603, 0.0011842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: B, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001787
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001787
time: 0.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.21 seconds
IS_A2_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.21
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001789
IS_A2_A2_B1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.21
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001789
IS_A2_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.21
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001787
IS_A2_A2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.21
Output dim: 6, lower bound: -0.0001533, upper bound: 0.0001787

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.04 + 67.80 = 70.84 seconds
