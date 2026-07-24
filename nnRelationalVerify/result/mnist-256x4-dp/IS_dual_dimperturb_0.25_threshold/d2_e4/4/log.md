## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00164889


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0029248, 0.0052543, 0.0029248, 0.0052543, -0.0019256, 0.0019256)
1: (0.0016296, 0.0020607, 0.0016296, 0.0020607, -0.0003701, 0.0003701)
2: (0.0114284, 0.0127428, 0.0114284, 0.0127428, -0.0011092, 0.0011092)
3: (-0.0027513, -0.0013945, -0.0027513, -0.0013945, -0.0010660, 0.0010660)
4: (-0.0024226, -0.0010585, -0.0024226, -0.0010585, -0.0010439, 0.0010439)
5: (0.0050558, 0.0063957, 0.0050558, 0.0063957, -0.0010868, 0.0010868)
6: (-0.0020698, 0.0030759, -0.0020698, 0.0030759, -0.0039986, 0.0039987)
7: (-0.0068697, 0.0001744, -0.0068697, 0.0001744, -0.0054057, 0.0054057)
8: (0.9844620, 0.9893581, 0.9844620, 0.9893581, -0.0037611, 0.0037611)
9: (-0.0062079, -0.0017424, -0.0062079, -0.0017424, -0.0034184, 0.0034184)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.86 + 1.72 = 3.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0020554, upper bound: 0.0020555

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018200, upper bound: 0.0019588
time: 0.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019593, upper bound: 0.0019593
time: 0.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 8, lower bound: -0.0018200, upper bound: 0.0019588
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.82
Output dim: 8, lower bound: -0.0019593, upper bound: 0.0019593

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0031193, 0.0050939, 0.0029982, 0.0051050, -0.0014612, 0.0015758
1: 0.0017729, 0.0020582, 0.0017555, 0.0020598, -0.0002111, 0.0002277
2: 0.0115436, 0.0126353, 0.0115375, 0.0127022, -0.0008712, 0.0008078
3: -0.0027416, -0.0016125, -0.0027479, -0.0015432, -0.0009011, 0.0008355
4: -0.0022914, -0.0010690, -0.0023663, -0.0010622, -0.0009045, 0.0009754
5: 0.0051250, 0.0062818, 0.0051186, 0.0063527, -0.0009231, 0.0008559
6: -0.0019657, 0.0026239, -0.0019913, 0.0029053, -0.0036626, 0.0033961
7: -0.0061302, 0.0001204, -0.0065135, 0.0001553, -0.0046252, 0.0049882
8: 0.9848956, 0.9892987, 0.9846256, 0.9893233, -0.0032581, 0.0035138
9: -0.0061734, -0.0021765, -0.0061957, -0.0019315, -0.0031896, 0.0029575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017506, upper bound: 0.0018098
time: 0.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017506, upper bound: 0.0018968
time: 0.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0030383, 0.0053219, 0.0029773, 0.0051054, -0.0015155, 0.0019013
1: 0.0017612, 0.0020912, 0.0017524, 0.0020599, -0.0002189, 0.0002747
2: 0.0114175, 0.0126801, 0.0115372, 0.0127138, -0.0010512, 0.0008379
3: -0.0028719, -0.0015661, -0.0027481, -0.0015313, -0.0010872, 0.0008666
4: -0.0023415, -0.0009279, -0.0023793, -0.0010620, -0.0009381, 0.0011769
5: 0.0049915, 0.0063292, 0.0051183, 0.0063649, -0.0011138, 0.0008878
6: -0.0024956, 0.0028122, -0.0019923, 0.0029539, -0.0044191, 0.0035225
7: -0.0063866, 0.0008421, -0.0065797, 0.0001566, -0.0047973, 0.0060185
8: 0.9847150, 0.9898070, 0.9845790, 0.9893242, -0.0033793, 0.0042395
9: -0.0066348, -0.0020126, -0.0061965, -0.0018891, -0.0038484, 0.0030675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018976, upper bound: 0.0018102
time: 0.85 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018976, upper bound: 0.0018976
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 8, lower bound: -0.0017506, upper bound: 0.0018098
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 8, lower bound: -0.0017506, upper bound: 0.0018968
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 8, lower bound: -0.0018976, upper bound: 0.0018102
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.31
Output dim: 8, lower bound: -0.0018976, upper bound: 0.0018976

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0031817, 0.0050915, 0.0031422, 0.0050995, -0.0013874, 0.0014214
1: 0.0017820, 0.0020579, 0.0017763, 0.0020590, -0.0002004, 0.0002054
2: 0.0115449, 0.0126008, 0.0115405, 0.0126226, -0.0007859, 0.0007671
3: -0.0027402, -0.0016481, -0.0027447, -0.0016256, -0.0008128, 0.0007933
4: -0.0022528, -0.0010705, -0.0022772, -0.0010656, -0.0008588, 0.0008799
5: 0.0051264, 0.0062452, 0.0051218, 0.0062683, -0.0008327, 0.0008128
6: -0.0019602, 0.0024789, -0.0019785, 0.0025706, -0.0033038, 0.0032248
7: -0.0059328, 0.0001129, -0.0060577, 0.0001379, -0.0043918, 0.0044995
8: 0.9850346, 0.9892934, 0.9849467, 0.9893110, -0.0030937, 0.0031696
9: -0.0061685, -0.0023027, -0.0061845, -0.0022229, -0.0028771, 0.0028083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0017419
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0017419
time: 0.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0031546, 0.0050915, 0.0030648, 0.0051949, -0.0015684, 0.0014570
1: 0.0017781, 0.0020579, 0.0017651, 0.0020728, -0.0002266, 0.0002105
2: 0.0115449, 0.0126158, 0.0114877, 0.0126654, -0.0008055, 0.0008672
3: -0.0027402, -0.0016326, -0.0027993, -0.0015813, -0.0008331, 0.0008969
4: -0.0022695, -0.0010706, -0.0023251, -0.0010066, -0.0009709, 0.0009019
5: 0.0051265, 0.0062611, 0.0050659, 0.0063137, -0.0008535, 0.0009188
6: -0.0019600, 0.0025418, -0.0022003, 0.0027505, -0.0033865, 0.0036455
7: -0.0060185, 0.0001126, -0.0063027, 0.0004400, -0.0049649, 0.0046121
8: 0.9849744, 0.9892931, 0.9847741, 0.9895238, -0.0034973, 0.0032489
9: -0.0061683, -0.0022480, -0.0063777, -0.0020662, -0.0029491, 0.0031747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0018314
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0018314
time: 0.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0031025, 0.0053193, 0.0031241, 0.0050997, -0.0014421, 0.0017384
1: 0.0017705, 0.0020908, 0.0017736, 0.0020591, -0.0002083, 0.0002511
2: 0.0114189, 0.0126446, 0.0115404, 0.0126326, -0.0009611, 0.0007973
3: -0.0028705, -0.0016029, -0.0027449, -0.0016152, -0.0009940, 0.0008246
4: -0.0023018, -0.0009295, -0.0022884, -0.0010655, -0.0008927, 0.0010761
5: 0.0049930, 0.0062916, 0.0051216, 0.0062790, -0.0010184, 0.0008448
6: -0.0024896, 0.0026629, -0.0019792, 0.0026128, -0.0040405, 0.0033518
7: -0.0061834, 0.0008339, -0.0061151, 0.0001387, -0.0045648, 0.0055029
8: 0.9848582, 0.9898013, 0.9849063, 0.9893116, -0.0032155, 0.0038763
9: -0.0066296, -0.0021425, -0.0061851, -0.0021862, -0.0035187, 0.0029189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018031, upper bound: 0.0017421
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018315, upper bound: 0.0017421
time: 0.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0030717, 0.0053194, 0.0030433, 0.0051949, -0.0016225, 0.0017758
1: 0.0017661, 0.0020908, 0.0017620, 0.0020728, -0.0002344, 0.0002565
2: 0.0114189, 0.0126616, 0.0114878, 0.0126773, -0.0009818, 0.0008970
3: -0.0028705, -0.0015852, -0.0027993, -0.0015690, -0.0010154, 0.0009277
4: -0.0023208, -0.0009295, -0.0023384, -0.0010066, -0.0010043, 0.0010992
5: 0.0049929, 0.0063096, 0.0050659, 0.0063263, -0.0010402, 0.0009504
6: -0.0024898, 0.0027345, -0.0022003, 0.0028005, -0.0041274, 0.0037711
7: -0.0062808, 0.0008342, -0.0063707, 0.0004399, -0.0051359, 0.0056211
8: 0.9847895, 0.9898015, 0.9847263, 0.9895238, -0.0036178, 0.0039596
9: -0.0066298, -0.0020802, -0.0063776, -0.0020228, -0.0035943, 0.0032840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018031, upper bound: 0.0018315
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018316, upper bound: 0.0018316
time: 0.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0017419
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0017419
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0018314
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0018314
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 8, lower bound: -0.0018031, upper bound: 0.0017421
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 8, lower bound: -0.0018315, upper bound: 0.0017421
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 8, lower bound: -0.0018031, upper bound: 0.0018315
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 8, lower bound: -0.0018316, upper bound: 0.0018316

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032643, 0.0050840, 0.0031786, 0.0050967, -0.0012891, 0.0013736
1: 0.0017939, 0.0020568, 0.0017815, 0.0020586, -0.0001862, 0.0001984
2: 0.0115490, 0.0125551, 0.0115421, 0.0126025, -0.0007594, 0.0007127
3: -0.0027359, -0.0016954, -0.0027431, -0.0016464, -0.0007854, 0.0007371
4: -0.0022016, -0.0010752, -0.0022547, -0.0010674, -0.0007979, 0.0008503
5: 0.0051308, 0.0061968, 0.0051234, 0.0062470, -0.0008047, 0.0007551
6: -0.0019427, 0.0022869, -0.0019721, 0.0024860, -0.0031927, 0.0029961
7: -0.0056713, 0.0000891, -0.0059425, 0.0001290, -0.0040805, 0.0043481
8: 0.9852189, 0.9892766, 0.9850280, 0.9893048, -0.0028744, 0.0030629
9: -0.0061533, -0.0024700, -0.0061789, -0.0022966, -0.0027803, 0.0026092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0016270
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0017419
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0032609, 0.0051444, 0.0031777, 0.0050962, -0.0013024, 0.0014946
1: 0.0017934, 0.0020655, 0.0017814, 0.0020585, -0.0001882, 0.0002159
2: 0.0115157, 0.0125570, 0.0115423, 0.0126030, -0.0008263, 0.0007200
3: -0.0027704, -0.0016934, -0.0027428, -0.0016459, -0.0008546, 0.0007447
4: -0.0022037, -0.0010378, -0.0022552, -0.0010677, -0.0008062, 0.0009252
5: 0.0050955, 0.0061988, 0.0051237, 0.0062476, -0.0008755, 0.0007629
6: -0.0020829, 0.0022948, -0.0019709, 0.0024881, -0.0034738, 0.0030271
7: -0.0056820, 0.0002800, -0.0059453, 0.0001275, -0.0041226, 0.0047311
8: 0.9852114, 0.9894111, 0.9850259, 0.9893036, -0.0029041, 0.0033327
9: -0.0062754, -0.0024631, -0.0061779, -0.0022948, -0.0030252, 0.0026361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0016270
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0017419
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032397, 0.0050840, 0.0031011, 0.0051921, -0.0014698, 0.0014091
1: 0.0017903, 0.0020568, 0.0017703, 0.0020724, -0.0002123, 0.0002036
2: 0.0115491, 0.0125687, 0.0114893, 0.0126454, -0.0007791, 0.0008126
3: -0.0027359, -0.0016813, -0.0027977, -0.0016020, -0.0008057, 0.0008405
4: -0.0022169, -0.0010752, -0.0023027, -0.0010083, -0.0009099, 0.0008723
5: 0.0051309, 0.0062112, 0.0050675, 0.0062924, -0.0008255, 0.0008610
6: -0.0019425, 0.0023441, -0.0021939, 0.0026663, -0.0032752, 0.0034163
7: -0.0057491, 0.0000888, -0.0061879, 0.0004312, -0.0046527, 0.0044605
8: 0.9851641, 0.9892765, 0.9848550, 0.9895177, -0.0032775, 0.0031421
9: -0.0061531, -0.0024202, -0.0063721, -0.0021396, -0.0028522, 0.0029751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0016949
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0018314
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032336, 0.0051443, 0.0031004, 0.0051915, -0.0014832, 0.0015301
1: 0.0017895, 0.0020655, 0.0017702, 0.0020723, -0.0002143, 0.0002211
2: 0.0115157, 0.0125721, 0.0114896, 0.0126457, -0.0008459, 0.0008200
3: -0.0027704, -0.0016778, -0.0027974, -0.0016017, -0.0008749, 0.0008481
4: -0.0022207, -0.0010379, -0.0023031, -0.0010086, -0.0009181, 0.0009471
5: 0.0050955, 0.0062148, 0.0050679, 0.0062928, -0.0008963, 0.0008688
6: -0.0020828, 0.0023583, -0.0021925, 0.0026677, -0.0035563, 0.0034473
7: -0.0057685, 0.0002799, -0.0061899, 0.0004293, -0.0046949, 0.0048434
8: 0.9851504, 0.9894110, 0.9848536, 0.9895163, -0.0033072, 0.0034118
9: -0.0062753, -0.0024078, -0.0063709, -0.0021383, -0.0030970, 0.0030021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0016949
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0018314
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0031927, 0.0053126, 0.0031607, 0.0050969, -0.0013435, 0.0016943
1: 0.0017836, 0.0020898, 0.0017789, 0.0020587, -0.0001941, 0.0002448
2: 0.0114227, 0.0125947, 0.0115419, 0.0126124, -0.0009368, 0.0007428
3: -0.0028666, -0.0016544, -0.0027433, -0.0016362, -0.0009688, 0.0007682
4: -0.0022459, -0.0009337, -0.0022657, -0.0010672, -0.0008316, 0.0010488
5: 0.0049969, 0.0062388, 0.0051233, 0.0062575, -0.0009925, 0.0007870
6: -0.0024739, 0.0024533, -0.0019726, 0.0025276, -0.0039381, 0.0031226
7: -0.0058979, 0.0008126, -0.0059990, 0.0001298, -0.0042527, 0.0053634
8: 0.9850593, 0.9897863, 0.9849880, 0.9893053, -0.0029957, 0.0037781
9: -0.0066159, -0.0023251, -0.0061794, -0.0022604, -0.0034295, 0.0027193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018026, upper bound: 0.0016230
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018026, upper bound: 0.0016262
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0031801, 0.0053794, 0.0031597, 0.0050965, -0.0013630, 0.0017780
1: 0.0017817, 0.0020995, 0.0017788, 0.0020586, -0.0001969, 0.0002569
2: 0.0113857, 0.0126017, 0.0115422, 0.0126130, -0.0009830, 0.0007536
3: -0.0029048, -0.0016472, -0.0027430, -0.0016355, -0.0010167, 0.0007794
4: -0.0022537, -0.0008923, -0.0022664, -0.0010675, -0.0008437, 0.0011006
5: 0.0049578, 0.0062461, 0.0051236, 0.0062581, -0.0010415, 0.0007984
6: -0.0026292, 0.0024825, -0.0019716, 0.0025300, -0.0041325, 0.0031679
7: -0.0059377, 0.0010240, -0.0060024, 0.0001284, -0.0043144, 0.0056282
8: 0.9850312, 0.9899353, 0.9849856, 0.9893044, -0.0030392, 0.0039646
9: -0.0067511, -0.0022996, -0.0061785, -0.0022583, -0.0035988, 0.0027588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018314, upper bound: 0.0016230
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018314, upper bound: 0.0016262
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0031633, 0.0053127, 0.0030811, 0.0051921, -0.0015247, 0.0017316
1: 0.0017793, 0.0020898, 0.0017674, 0.0020724, -0.0002203, 0.0002502
2: 0.0114226, 0.0126110, 0.0114893, 0.0126564, -0.0009573, 0.0008430
3: -0.0028666, -0.0016376, -0.0027977, -0.0015906, -0.0009901, 0.0008719
4: -0.0022641, -0.0009336, -0.0023150, -0.0010083, -0.0009438, 0.0010719
5: 0.0049969, 0.0062560, 0.0050675, 0.0063042, -0.0010143, 0.0008932
6: -0.0024741, 0.0025216, -0.0021938, 0.0027127, -0.0040246, 0.0035439
7: -0.0059909, 0.0008128, -0.0062512, 0.0004311, -0.0048265, 0.0054812
8: 0.9849937, 0.9897864, 0.9848104, 0.9895175, -0.0033999, 0.0038611
9: -0.0066161, -0.0022656, -0.0063720, -0.0020992, -0.0035048, 0.0030862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018026, upper bound: 0.0016771
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018026, upper bound: 0.0016814
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0031495, 0.0053795, 0.0030785, 0.0051915, -0.0015439, 0.0018157
1: 0.0017773, 0.0020995, 0.0017671, 0.0020723, -0.0002230, 0.0002623
2: 0.0113857, 0.0126186, 0.0114896, 0.0126578, -0.0010039, 0.0008536
3: -0.0029049, -0.0016297, -0.0027974, -0.0015892, -0.0010382, 0.0008828
4: -0.0022727, -0.0008923, -0.0023166, -0.0010086, -0.0009557, 0.0011240
5: 0.0049577, 0.0062641, 0.0050679, 0.0063056, -0.0010636, 0.0009044
6: -0.0026295, 0.0025538, -0.0021926, 0.0027186, -0.0042202, 0.0035884
7: -0.0060347, 0.0010244, -0.0062592, 0.0004294, -0.0048871, 0.0057475
8: 0.9849629, 0.9899355, 0.9848047, 0.9895163, -0.0034426, 0.0040487
9: -0.0067514, -0.0022376, -0.0063709, -0.0020940, -0.0036751, 0.0031249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018314, upper bound: 0.0016771
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018314, upper bound: 0.0016814
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.30 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0016270
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0017419
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0016270
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0017419
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0016949
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0016654, upper bound: 0.0018314
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0016949
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0016771, upper bound: 0.0018314
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0018026, upper bound: 0.0016230
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0018026, upper bound: 0.0016262
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0018314, upper bound: 0.0016230
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0018314, upper bound: 0.0016262
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0018026, upper bound: 0.0016771
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0018026, upper bound: 0.0016814
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0018314, upper bound: 0.0016771
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 8, lower bound: -0.0018314, upper bound: 0.0016814

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032643, 0.0050840, 0.0032938, 0.0050858, -0.0012800, 0.0012541
1: 0.0017939, 0.0020568, 0.0017982, 0.0020570, -0.0001849, 0.0001812
2: 0.0115490, 0.0125551, 0.0115481, 0.0125388, -0.0006934, 0.0007077
3: -0.0027359, -0.0016954, -0.0027369, -0.0017122, -0.0007171, 0.0007319
4: -0.0022016, -0.0010752, -0.0021833, -0.0010741, -0.0007923, 0.0007763
5: 0.0051308, 0.0061968, 0.0051298, 0.0061795, -0.0007347, 0.0007498
6: -0.0019427, 0.0022869, -0.0019468, 0.0022183, -0.0029150, 0.0029751
7: -0.0056713, 0.0000891, -0.0055778, 0.0000946, -0.0040518, 0.0039699
8: 0.9852189, 0.9892766, 0.9852848, 0.9892805, -0.0028542, 0.0027965
9: -0.0061533, -0.0024700, -0.0061568, -0.0025298, -0.0025385, 0.0025908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016194, upper bound: 0.0016270
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016194, upper bound: 0.0016270
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032643, 0.0050840, 0.0032204, 0.0053136, -0.0015590, 0.0013808
1: 0.0017939, 0.0020568, 0.0017876, 0.0020900, -0.0002252, 0.0001995
2: 0.0115490, 0.0125551, 0.0114221, 0.0125794, -0.0007634, 0.0008619
3: -0.0027359, -0.0016954, -0.0028672, -0.0016703, -0.0007895, 0.0008915
4: -0.0022016, -0.0010752, -0.0022288, -0.0009330, -0.0009651, 0.0008547
5: 0.0051308, 0.0061968, 0.0049963, 0.0062225, -0.0008088, 0.0009133
6: -0.0019427, 0.0022869, -0.0024764, 0.0023889, -0.0032093, 0.0036236
7: -0.0056713, 0.0000891, -0.0058101, 0.0008159, -0.0049350, 0.0043708
8: 0.9852189, 0.9892766, 0.9851211, 0.9897885, -0.0034763, 0.0030789
9: -0.0061533, -0.0024700, -0.0066180, -0.0023812, -0.0027948, 0.0031556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016194, upper bound: 0.0017419
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016194, upper bound: 0.0017419
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032609, 0.0051444, 0.0032980, 0.0050852, -0.0012932, 0.0013764
1: 0.0017934, 0.0020655, 0.0017988, 0.0020570, -0.0001868, 0.0001988
2: 0.0115157, 0.0125570, 0.0115484, 0.0125365, -0.0007610, 0.0007150
3: -0.0027704, -0.0016934, -0.0027366, -0.0017147, -0.0007870, 0.0007395
4: -0.0022037, -0.0010378, -0.0021807, -0.0010745, -0.0008005, 0.0008520
5: 0.0050955, 0.0061988, 0.0051302, 0.0061771, -0.0008063, 0.0007576
6: -0.0020829, 0.0022948, -0.0019453, 0.0022085, -0.0031990, 0.0030058
7: -0.0056820, 0.0002800, -0.0055644, 0.0000927, -0.0040937, 0.0043568
8: 0.9852114, 0.9894111, 0.9852942, 0.9892791, -0.0028837, 0.0030690
9: -0.0062754, -0.0024631, -0.0061556, -0.0025383, -0.0027859, 0.0026176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016323, upper bound: 0.0016270
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016323, upper bound: 0.0016269
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032609, 0.0051444, 0.0032205, 0.0053131, -0.0015723, 0.0014890
1: 0.0017934, 0.0020655, 0.0017876, 0.0020899, -0.0002272, 0.0002151
2: 0.0115157, 0.0125570, 0.0114224, 0.0125794, -0.0008232, 0.0008693
3: -0.0027704, -0.0016934, -0.0028669, -0.0016703, -0.0008514, 0.0008991
4: -0.0022037, -0.0010378, -0.0022288, -0.0009334, -0.0009733, 0.0009217
5: 0.0050955, 0.0061988, 0.0049967, 0.0062225, -0.0008722, 0.0009210
6: -0.0020829, 0.0022948, -0.0024750, 0.0023888, -0.0034608, 0.0036544
7: -0.0056820, 0.0002800, -0.0058100, 0.0008141, -0.0049770, 0.0047133
8: 0.9852114, 0.9894111, 0.9851211, 0.9897873, -0.0035059, 0.0033202
9: -0.0062754, -0.0024631, -0.0066169, -0.0023813, -0.0030138, 0.0031825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016323, upper bound: 0.0017419
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016323, upper bound: 0.0017419
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032397, 0.0050840, 0.0032239, 0.0051815, -0.0014610, 0.0012879
1: 0.0017903, 0.0020568, 0.0017881, 0.0020709, -0.0002111, 0.0001861
2: 0.0115491, 0.0125687, 0.0114952, 0.0125774, -0.0007120, 0.0008077
3: -0.0027359, -0.0016813, -0.0027916, -0.0016723, -0.0007364, 0.0008354
4: -0.0022169, -0.0010752, -0.0022266, -0.0010149, -0.0009044, 0.0007972
5: 0.0051309, 0.0062112, 0.0050737, 0.0062205, -0.0007544, 0.0008558
6: -0.0019425, 0.0023441, -0.0021692, 0.0023807, -0.0029933, 0.0033957
7: -0.0057491, 0.0000888, -0.0057990, 0.0003976, -0.0046246, 0.0040767
8: 0.9851641, 0.9892765, 0.9851289, 0.9894938, -0.0032577, 0.0028717
9: -0.0061531, -0.0024202, -0.0063506, -0.0023883, -0.0026067, 0.0029571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016095, upper bound: 0.0016949
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016095, upper bound: 0.0016270
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032397, 0.0050840, 0.0031406, 0.0054208, -0.0017485, 0.0014190
1: 0.0017903, 0.0020568, 0.0017760, 0.0021054, -0.0002526, 0.0002050
2: 0.0115491, 0.0125687, 0.0113628, 0.0126235, -0.0007845, 0.0009667
3: -0.0027359, -0.0016813, -0.0029285, -0.0016246, -0.0008114, 0.0009998
4: -0.0022169, -0.0010752, -0.0022782, -0.0008667, -0.0010823, 0.0008784
5: 0.0051309, 0.0062112, 0.0049336, 0.0062693, -0.0008312, 0.0010242
6: -0.0019425, 0.0023441, -0.0027254, 0.0025743, -0.0032981, 0.0040639
7: -0.0057491, 0.0000888, -0.0060627, 0.0011551, -0.0055347, 0.0044918
8: 0.9851641, 0.9892765, 0.9849433, 0.9900275, -0.0038987, 0.0031641
9: -0.0061531, -0.0024202, -0.0068349, -0.0022197, -0.0028722, 0.0035390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016095, upper bound: 0.0018314
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016095, upper bound: 0.0017419
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032336, 0.0051443, 0.0032273, 0.0051808, -0.0014742, 0.0014105
1: 0.0017895, 0.0020655, 0.0017886, 0.0020708, -0.0002130, 0.0002038
2: 0.0115157, 0.0125721, 0.0114955, 0.0125755, -0.0007798, 0.0008151
3: -0.0027704, -0.0016778, -0.0027912, -0.0016742, -0.0008065, 0.0008430
4: -0.0022207, -0.0010379, -0.0022245, -0.0010153, -0.0009126, 0.0008731
5: 0.0050955, 0.0062148, 0.0050742, 0.0062185, -0.0008262, 0.0008636
6: -0.0020828, 0.0023583, -0.0021676, 0.0023728, -0.0032783, 0.0034265
7: -0.0057685, 0.0002799, -0.0057882, 0.0003953, -0.0046666, 0.0044648
8: 0.9851504, 0.9894110, 0.9851366, 0.9894923, -0.0032872, 0.0031451
9: -0.0062753, -0.0024078, -0.0063491, -0.0023952, -0.0028549, 0.0029839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016230, upper bound: 0.0016949
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016230, upper bound: 0.0016270
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032336, 0.0051443, 0.0031403, 0.0054202, -0.0017618, 0.0015266
1: 0.0017895, 0.0020655, 0.0017760, 0.0021054, -0.0002545, 0.0002205
2: 0.0115157, 0.0125721, 0.0113632, 0.0126237, -0.0008440, 0.0009741
3: -0.0027704, -0.0016778, -0.0029281, -0.0016244, -0.0008729, 0.0010074
4: -0.0022207, -0.0010379, -0.0022784, -0.0008671, -0.0010906, 0.0009450
5: 0.0050955, 0.0062148, 0.0049339, 0.0062695, -0.0008943, 0.0010321
6: -0.0020828, 0.0023583, -0.0027240, 0.0025752, -0.0035482, 0.0040949
7: -0.0057685, 0.0002799, -0.0060639, 0.0011531, -0.0055769, 0.0048323
8: 0.9851504, 0.9894110, 0.9849424, 0.9900261, -0.0039285, 0.0034040
9: -0.0062753, -0.0024078, -0.0068337, -0.0022189, -0.0030899, 0.0035660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016230, upper bound: 0.0018314
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016230, upper bound: 0.0017419
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0031927, 0.0053126, 0.0032938, 0.0050858, -0.0014191, 0.0015340
1: 0.0017836, 0.0020898, 0.0017982, 0.0020570, -0.0002050, 0.0002216
2: 0.0114227, 0.0125947, 0.0115481, 0.0125388, -0.0008481, 0.0007846
3: -0.0028666, -0.0016544, -0.0027369, -0.0017122, -0.0008771, 0.0008114
4: -0.0022459, -0.0009337, -0.0021833, -0.0010741, -0.0008784, 0.0009496
5: 0.0049969, 0.0062388, 0.0051298, 0.0061795, -0.0008986, 0.0008313
6: -0.0024739, 0.0024533, -0.0019468, 0.0022183, -0.0035654, 0.0032983
7: -0.0058979, 0.0008126, -0.0055778, 0.0000946, -0.0044920, 0.0048558
8: 0.9850593, 0.9897863, 0.9852848, 0.9892805, -0.0031642, 0.0034205
9: -0.0066159, -0.0023251, -0.0061568, -0.0025298, -0.0031049, 0.0028723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017436, upper bound: 0.0016230
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017436, upper bound: 0.0016230
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0031927, 0.0053126, 0.0032204, 0.0053136, -0.0013928, 0.0013625
1: 0.0017836, 0.0020898, 0.0017876, 0.0020900, -0.0002012, 0.0001968
2: 0.0114227, 0.0125947, 0.0114221, 0.0125794, -0.0007533, 0.0007700
3: -0.0028666, -0.0016544, -0.0028672, -0.0016703, -0.0007791, 0.0007964
4: -0.0022459, -0.0009337, -0.0022288, -0.0009330, -0.0008622, 0.0008434
5: 0.0049969, 0.0062388, 0.0049963, 0.0062225, -0.0007982, 0.0008159
6: -0.0024739, 0.0024533, -0.0024764, 0.0023889, -0.0031669, 0.0032372
7: -0.0058979, 0.0008126, -0.0058101, 0.0008159, -0.0044088, 0.0043130
8: 0.9850593, 0.9897863, 0.9851211, 0.9897885, -0.0031056, 0.0030382
9: -0.0066159, -0.0023251, -0.0066180, -0.0023812, -0.0027579, 0.0028191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017436, upper bound: 0.0016262
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017436, upper bound: 0.0016262
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0031801, 0.0053794, 0.0032980, 0.0050852, -0.0014319, 0.0016273
1: 0.0017817, 0.0020995, 0.0017988, 0.0020570, -0.0002069, 0.0002351
2: 0.0113857, 0.0126017, 0.0115484, 0.0125365, -0.0008997, 0.0007917
3: -0.0029048, -0.0016472, -0.0027366, -0.0017147, -0.0009305, 0.0008188
4: -0.0022537, -0.0008923, -0.0021807, -0.0010745, -0.0008864, 0.0010073
5: 0.0049578, 0.0062461, 0.0051302, 0.0061771, -0.0009533, 0.0008388
6: -0.0026292, 0.0024825, -0.0019453, 0.0022085, -0.0037824, 0.0033282
7: -0.0059377, 0.0010240, -0.0055644, 0.0000927, -0.0045327, 0.0051513
8: 0.9850312, 0.9899353, 0.9852942, 0.9892791, -0.0031929, 0.0036287
9: -0.0067511, -0.0022996, -0.0061556, -0.0025383, -0.0032939, 0.0028983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0016230
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0016230
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0031801, 0.0053794, 0.0032205, 0.0053131, -0.0014122, 0.0014811
1: 0.0017817, 0.0020995, 0.0017876, 0.0020899, -0.0002040, 0.0002140
2: 0.0113857, 0.0126017, 0.0114224, 0.0125794, -0.0008189, 0.0007808
3: -0.0029048, -0.0016472, -0.0028669, -0.0016703, -0.0008469, 0.0008075
4: -0.0022537, -0.0008923, -0.0022288, -0.0009334, -0.0008742, 0.0009168
5: 0.0049578, 0.0062461, 0.0049967, 0.0062225, -0.0008676, 0.0008273
6: -0.0026292, 0.0024825, -0.0024750, 0.0023888, -0.0034425, 0.0032823
7: -0.0059377, 0.0010240, -0.0058100, 0.0008141, -0.0044702, 0.0046883
8: 0.9850312, 0.9899353, 0.9851211, 0.9897873, -0.0031489, 0.0033026
9: -0.0067511, -0.0022996, -0.0066169, -0.0023813, -0.0029979, 0.0028584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0016262
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0016262
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0031633, 0.0053127, 0.0032239, 0.0051815, -0.0016049, 0.0015679
1: 0.0017793, 0.0020898, 0.0017881, 0.0020709, -0.0002319, 0.0002265
2: 0.0114226, 0.0126110, 0.0114952, 0.0125774, -0.0008668, 0.0008873
3: -0.0028666, -0.0016376, -0.0027916, -0.0016723, -0.0008965, 0.0009177
4: -0.0022641, -0.0009336, -0.0022266, -0.0010149, -0.0009935, 0.0009705
5: 0.0049969, 0.0062560, 0.0050737, 0.0062205, -0.0009185, 0.0009402
6: -0.0024741, 0.0025216, -0.0021692, 0.0023807, -0.0036442, 0.0037303
7: -0.0059909, 0.0008128, -0.0057990, 0.0003976, -0.0050804, 0.0049630
8: 0.9849937, 0.9897864, 0.9851289, 0.9894938, -0.0035787, 0.0034961
9: -0.0066161, -0.0022656, -0.0063506, -0.0023883, -0.0031735, 0.0032485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0016771
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0016230
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0031633, 0.0053127, 0.0031406, 0.0054208, -0.0015833, 0.0014026
1: 0.0017793, 0.0020898, 0.0017760, 0.0021054, -0.0002287, 0.0002026
2: 0.0114226, 0.0126110, 0.0113628, 0.0126235, -0.0007754, 0.0008754
3: -0.0028666, -0.0016376, -0.0029285, -0.0016246, -0.0008020, 0.0009054
4: -0.0022641, -0.0009336, -0.0022782, -0.0008667, -0.0009801, 0.0008682
5: 0.0049969, 0.0062560, 0.0049336, 0.0062693, -0.0008216, 0.0009275
6: -0.0024741, 0.0025216, -0.0027254, 0.0025743, -0.0032600, 0.0036801
7: -0.0059909, 0.0008128, -0.0060627, 0.0011551, -0.0050120, 0.0044398
8: 0.9849937, 0.9897864, 0.9849433, 0.9900275, -0.0035305, 0.0031275
9: -0.0066161, -0.0022656, -0.0068349, -0.0022197, -0.0028389, 0.0032048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0016814
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0016262
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0031495, 0.0053795, 0.0032273, 0.0051808, -0.0016177, 0.0016616
1: 0.0017773, 0.0020995, 0.0017886, 0.0020708, -0.0002337, 0.0002400
2: 0.0113857, 0.0126186, 0.0114955, 0.0125755, -0.0009186, 0.0008944
3: -0.0029049, -0.0016297, -0.0027912, -0.0016742, -0.0009501, 0.0009250
4: -0.0022727, -0.0008923, -0.0022245, -0.0010153, -0.0010014, 0.0010285
5: 0.0049577, 0.0062641, 0.0050742, 0.0062185, -0.0009733, 0.0009476
6: -0.0026295, 0.0025538, -0.0021676, 0.0023728, -0.0038619, 0.0037599
7: -0.0060347, 0.0010244, -0.0057882, 0.0003953, -0.0051206, 0.0052596
8: 0.9849629, 0.9899355, 0.9851366, 0.9894923, -0.0036071, 0.0037050
9: -0.0067514, -0.0022376, -0.0063491, -0.0023952, -0.0033631, 0.0032743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017419, upper bound: 0.0016771
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017419, upper bound: 0.0016230
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0031495, 0.0053795, 0.0031403, 0.0054202, -0.0016024, 0.0015210
1: 0.0017773, 0.0020995, 0.0017760, 0.0021054, -0.0002315, 0.0002197
2: 0.0113857, 0.0126186, 0.0113632, 0.0126237, -0.0008409, 0.0008859
3: -0.0029049, -0.0016297, -0.0029281, -0.0016244, -0.0008697, 0.0009163
4: -0.0022727, -0.0008923, -0.0022784, -0.0008671, -0.0009919, 0.0009415
5: 0.0049577, 0.0062641, 0.0049339, 0.0062695, -0.0008910, 0.0009387
6: -0.0026295, 0.0025538, -0.0027240, 0.0025752, -0.0035353, 0.0037244
7: -0.0060347, 0.0010244, -0.0060639, 0.0011531, -0.0050723, 0.0048147
8: 0.9849629, 0.9899355, 0.9849424, 0.9900261, -0.0035730, 0.0033916
9: -0.0067514, -0.0022376, -0.0068337, -0.0022189, -0.0030787, 0.0032434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017419, upper bound: 0.0016814
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017419, upper bound: 0.0016262
time: 0.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.45 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016194, upper bound: 0.0016270
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016194, upper bound: 0.0016270
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016194, upper bound: 0.0017419
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016194, upper bound: 0.0017419
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016323, upper bound: 0.0016270
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016323, upper bound: 0.0016269
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016323, upper bound: 0.0017419
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016323, upper bound: 0.0017419
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016095, upper bound: 0.0016949
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016095, upper bound: 0.0016270
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016095, upper bound: 0.0018314
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016095, upper bound: 0.0017419
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016230, upper bound: 0.0016949
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016230, upper bound: 0.0016270
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016230, upper bound: 0.0018314
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0016230, upper bound: 0.0017419
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017436, upper bound: 0.0016230
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017436, upper bound: 0.0016230
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017436, upper bound: 0.0016262
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017436, upper bound: 0.0016262
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0016230
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0016230
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0016262
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017689, upper bound: 0.0016262
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0016771
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0016230
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0016814
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017244, upper bound: 0.0016262
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017419, upper bound: 0.0016771
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017419, upper bound: 0.0016230
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017419, upper bound: 0.0016814
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.45
Output dim: 8, lower bound: -0.0017419, upper bound: 0.0016262

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0050813, 0.0032204, 0.0053136, -0.0014768, 0.0013779
1: 0.0018054, 0.0020564, 0.0017876, 0.0020900, -0.0002134, 0.0001991
2: 0.0115506, 0.0125110, 0.0114221, 0.0125794, -0.0007618, 0.0008165
3: -0.0027343, -0.0017410, -0.0028672, -0.0016703, -0.0007879, 0.0008444
4: -0.0021523, -0.0010769, -0.0022288, -0.0009330, -0.0009142, 0.0008529
5: 0.0051324, 0.0061501, 0.0049963, 0.0062225, -0.0008072, 0.0008651
6: -0.0019363, 0.0021016, -0.0024764, 0.0023889, -0.0032026, 0.0034325
7: -0.0054189, 0.0000803, -0.0058101, 0.0008159, -0.0046748, 0.0043616
8: 0.9853967, 0.9892704, 0.9851211, 0.9897885, -0.0032930, 0.0030724
9: -0.0061477, -0.0026314, -0.0066180, -0.0023812, -0.0027889, 0.0029892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015418, upper bound: 0.0016729
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015418, upper bound: 0.0016644
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032764, 0.0051771, 0.0032204, 0.0053136, -0.0015760, 0.0015191
1: 0.0017957, 0.0020702, 0.0017876, 0.0020900, -0.0002277, 0.0002195
2: 0.0114976, 0.0125484, 0.0114221, 0.0125794, -0.0008399, 0.0008713
3: -0.0027891, -0.0017023, -0.0028672, -0.0016703, -0.0008686, 0.0009012
4: -0.0021941, -0.0010176, -0.0022288, -0.0009330, -0.0009756, 0.0009404
5: 0.0050763, 0.0061897, 0.0049963, 0.0062225, -0.0008899, 0.0009232
6: -0.0021589, 0.0022586, -0.0024764, 0.0023889, -0.0035309, 0.0036630
7: -0.0056328, 0.0003835, -0.0058101, 0.0008159, -0.0049887, 0.0048087
8: 0.9852459, 0.9894841, 0.9851211, 0.9897885, -0.0035142, 0.0033874
9: -0.0063416, -0.0024946, -0.0066180, -0.0023812, -0.0030748, 0.0031899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015418, upper bound: 0.0016728
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015418, upper bound: 0.0016644
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033398, 0.0051412, 0.0032205, 0.0053131, -0.0014899, 0.0014855
1: 0.0018048, 0.0020651, 0.0017876, 0.0020899, -0.0002152, 0.0002146
2: 0.0115174, 0.0125134, 0.0114224, 0.0125794, -0.0008213, 0.0008237
3: -0.0027686, -0.0017386, -0.0028669, -0.0016703, -0.0008494, 0.0008519
4: -0.0021549, -0.0010398, -0.0022288, -0.0009334, -0.0009223, 0.0009196
5: 0.0050974, 0.0061526, 0.0049967, 0.0062225, -0.0008702, 0.0008728
6: -0.0020755, 0.0021113, -0.0024750, 0.0023888, -0.0034528, 0.0034629
7: -0.0054321, 0.0002700, -0.0058100, 0.0008141, -0.0047162, 0.0047024
8: 0.9853873, 0.9894040, 0.9851211, 0.9897873, -0.0033222, 0.0033125
9: -0.0062690, -0.0026229, -0.0066169, -0.0023813, -0.0030069, 0.0030157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015558, upper bound: 0.0016729
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015558, upper bound: 0.0016644
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032677, 0.0052398, 0.0032205, 0.0053131, -0.0015895, 0.0016085
1: 0.0017944, 0.0020793, 0.0017876, 0.0020899, -0.0002296, 0.0002324
2: 0.0114629, 0.0125532, 0.0114224, 0.0125794, -0.0008893, 0.0008788
3: -0.0028250, -0.0016973, -0.0028669, -0.0016703, -0.0009197, 0.0009089
4: -0.0021995, -0.0009788, -0.0022288, -0.0009334, -0.0009839, 0.0009957
5: 0.0050396, 0.0061948, 0.0049967, 0.0062225, -0.0009422, 0.0009311
6: -0.0023046, 0.0022790, -0.0024750, 0.0023888, -0.0037385, 0.0036943
7: -0.0056605, 0.0005820, -0.0058100, 0.0008141, -0.0050314, 0.0050915
8: 0.9852265, 0.9896238, 0.9851211, 0.9897873, -0.0035442, 0.0035866
9: -0.0064685, -0.0024769, -0.0066169, -0.0023813, -0.0032557, 0.0032172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015558, upper bound: 0.0016729
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015558, upper bound: 0.0016644
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0050813, 0.0032239, 0.0051815, -0.0013391, 0.0013482
1: 0.0018054, 0.0020564, 0.0017881, 0.0020709, -0.0001935, 0.0001948
2: 0.0115506, 0.0125110, 0.0114952, 0.0125774, -0.0007454, 0.0007404
3: -0.0027343, -0.0017410, -0.0027916, -0.0016723, -0.0007709, 0.0007657
4: -0.0021523, -0.0010769, -0.0022266, -0.0010149, -0.0008289, 0.0008346
5: 0.0051324, 0.0061501, 0.0050737, 0.0062205, -0.0007898, 0.0007845
6: -0.0019363, 0.0021016, -0.0021692, 0.0023807, -0.0031336, 0.0031125
7: -0.0054189, 0.0000803, -0.0057990, 0.0003976, -0.0042390, 0.0042677
8: 0.9853967, 0.9892704, 0.9851289, 0.9894938, -0.0029860, 0.0030063
9: -0.0061477, -0.0026314, -0.0063506, -0.0023883, -0.0027289, 0.0027105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015381, upper bound: 0.0016314
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015383, upper bound: 0.0016157
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033440, 0.0050813, 0.0031406, 0.0054208, -0.0016266, 0.0014927
1: 0.0018054, 0.0020564, 0.0017760, 0.0021054, -0.0002350, 0.0002156
2: 0.0115506, 0.0125110, 0.0113628, 0.0126235, -0.0008253, 0.0008993
3: -0.0027343, -0.0017410, -0.0029285, -0.0016246, -0.0008535, 0.0009301
4: -0.0021523, -0.0010769, -0.0022782, -0.0008667, -0.0010069, 0.0009240
5: 0.0051324, 0.0061501, 0.0049336, 0.0062693, -0.0008744, 0.0009529
6: -0.0019363, 0.0021016, -0.0027254, 0.0025743, -0.0034694, 0.0037807
7: -0.0054189, 0.0000803, -0.0060627, 0.0011551, -0.0051490, 0.0047250
8: 0.9853967, 0.9892704, 0.9849433, 0.9900275, -0.0036271, 0.0033284
9: -0.0061477, -0.0026314, -0.0068349, -0.0022197, -0.0030213, 0.0032924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017613
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017499
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032764, 0.0051771, 0.0031406, 0.0054208, -0.0015298, 0.0014308
1: 0.0017957, 0.0020702, 0.0017760, 0.0021054, -0.0002210, 0.0002067
2: 0.0114976, 0.0125484, 0.0113628, 0.0126235, -0.0007911, 0.0008458
3: -0.0027891, -0.0017023, -0.0029285, -0.0016246, -0.0008182, 0.0008747
4: -0.0021941, -0.0010176, -0.0022782, -0.0008667, -0.0009470, 0.0008857
5: 0.0050763, 0.0061897, 0.0049336, 0.0062693, -0.0008382, 0.0008961
6: -0.0021589, 0.0022586, -0.0027254, 0.0025743, -0.0033257, 0.0035556
7: -0.0056328, 0.0003835, -0.0060627, 0.0011551, -0.0048424, 0.0045293
8: 0.9852459, 0.9894841, 0.9849433, 0.9900275, -0.0034111, 0.0031905
9: -0.0063416, -0.0024946, -0.0068349, -0.0022197, -0.0028961, 0.0030964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0016728
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0016644
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033398, 0.0051412, 0.0032273, 0.0051808, -0.0013521, 0.0014670
1: 0.0018048, 0.0020651, 0.0017886, 0.0020708, -0.0001953, 0.0002119
2: 0.0115174, 0.0125134, 0.0114955, 0.0125755, -0.0008111, 0.0007475
3: -0.0027686, -0.0017386, -0.0027912, -0.0016742, -0.0008389, 0.0007731
4: -0.0021549, -0.0010398, -0.0022245, -0.0010153, -0.0008370, 0.0009081
5: 0.0050974, 0.0061526, 0.0050742, 0.0062185, -0.0008594, 0.0007920
6: -0.0020755, 0.0021113, -0.0021676, 0.0023728, -0.0034098, 0.0031426
7: -0.0054321, 0.0002700, -0.0057882, 0.0003953, -0.0042799, 0.0046438
8: 0.9853873, 0.9894040, 0.9851366, 0.9894923, -0.0030149, 0.0032712
9: -0.0062690, -0.0026229, -0.0063491, -0.0023952, -0.0029694, 0.0027367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015511, upper bound: 0.0016314
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015515, upper bound: 0.0016157
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033398, 0.0051412, 0.0031403, 0.0054202, -0.0016397, 0.0016019
1: 0.0018048, 0.0020651, 0.0017760, 0.0021054, -0.0002369, 0.0002314
2: 0.0115174, 0.0125134, 0.0113632, 0.0126237, -0.0008856, 0.0009065
3: -0.0027686, -0.0017386, -0.0029281, -0.0016244, -0.0009160, 0.0009376
4: -0.0021549, -0.0010398, -0.0022784, -0.0008671, -0.0010150, 0.0009916
5: 0.0050974, 0.0061526, 0.0049339, 0.0062695, -0.0009384, 0.0009605
6: -0.0020755, 0.0021113, -0.0027240, 0.0025752, -0.0037232, 0.0038110
7: -0.0054321, 0.0002700, -0.0060639, 0.0011531, -0.0051903, 0.0050707
8: 0.9853873, 0.9894040, 0.9849424, 0.9900261, -0.0036562, 0.0035719
9: -0.0062690, -0.0026229, -0.0068337, -0.0022189, -0.0032423, 0.0033188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015433, upper bound: 0.0017613
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0017499
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0032677, 0.0052398, 0.0031403, 0.0054202, -0.0015448, 0.0015364
1: 0.0017944, 0.0020793, 0.0017760, 0.0021054, -0.0002232, 0.0002220
2: 0.0114629, 0.0125532, 0.0113632, 0.0126237, -0.0008494, 0.0008541
3: -0.0028250, -0.0016973, -0.0029281, -0.0016244, -0.0008785, 0.0008833
4: -0.0021995, -0.0009788, -0.0022784, -0.0008671, -0.0009563, 0.0009511
5: 0.0050396, 0.0061948, 0.0049339, 0.0062695, -0.0009000, 0.0009050
6: -0.0023046, 0.0022790, -0.0027240, 0.0025752, -0.0035710, 0.0035906
7: -0.0056605, 0.0005820, -0.0060639, 0.0011531, -0.0048901, 0.0048634
8: 0.9852265, 0.9896238, 0.9849424, 0.9900261, -0.0034447, 0.0034259
9: -0.0064685, -0.0024769, -0.0068337, -0.0022189, -0.0031098, 0.0031268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015433, upper bound: 0.0016728
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016644
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032750, 0.0053096, 0.0032938, 0.0050858, -0.0013289, 0.0015309
1: 0.0017954, 0.0020894, 0.0017982, 0.0020570, -0.0001920, 0.0002212
2: 0.0114243, 0.0125492, 0.0115481, 0.0125388, -0.0008464, 0.0007347
3: -0.0028649, -0.0017015, -0.0027369, -0.0017122, -0.0008754, 0.0007599
4: -0.0021950, -0.0009356, -0.0021833, -0.0010741, -0.0008226, 0.0009476
5: 0.0049987, 0.0061905, 0.0051298, 0.0061795, -0.0008968, 0.0007784
6: -0.0024670, 0.0022619, -0.0019468, 0.0022183, -0.0035582, 0.0030886
7: -0.0056373, 0.0008031, -0.0055778, 0.0000946, -0.0042064, 0.0048459
8: 0.9852429, 0.9897796, 0.9852848, 0.9892805, -0.0029631, 0.0034136
9: -0.0066099, -0.0024917, -0.0061568, -0.0025298, -0.0030986, 0.0026897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016760, upper bound: 0.0015433
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016658, upper bound: 0.0015454
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0031954, 0.0054167, 0.0032938, 0.0050858, -0.0014392, 0.0016806
1: 0.0017840, 0.0021049, 0.0017982, 0.0020570, -0.0002079, 0.0002428
2: 0.0113651, 0.0125932, 0.0115481, 0.0125388, -0.0009292, 0.0007957
3: -0.0029262, -0.0016560, -0.0027369, -0.0017122, -0.0009610, 0.0008229
4: -0.0022442, -0.0008692, -0.0021833, -0.0010741, -0.0008909, 0.0010403
5: 0.0049359, 0.0062372, 0.0051298, 0.0061795, -0.0009845, 0.0008431
6: -0.0027160, 0.0024469, -0.0019468, 0.0022183, -0.0039063, 0.0033450
7: -0.0058892, 0.0011422, -0.0055778, 0.0000946, -0.0045556, 0.0053200
8: 0.9850654, 0.9900184, 0.9852848, 0.9892805, -0.0032091, 0.0037475
9: -0.0068267, -0.0023307, -0.0061568, -0.0025298, -0.0034018, 0.0029130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016760, upper bound: 0.0015433
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016658, upper bound: 0.0015454
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032750, 0.0053096, 0.0032204, 0.0053136, -0.0013082, 0.0013595
1: 0.0017954, 0.0020894, 0.0017876, 0.0020900, -0.0001890, 0.0001964
2: 0.0114243, 0.0125492, 0.0114221, 0.0125794, -0.0007516, 0.0007233
3: -0.0028649, -0.0017015, -0.0028672, -0.0016703, -0.0007774, 0.0007481
4: -0.0021950, -0.0009356, -0.0022288, -0.0009330, -0.0008098, 0.0008415
5: 0.0049987, 0.0061905, 0.0049963, 0.0062225, -0.0007964, 0.0007664
6: -0.0024670, 0.0022619, -0.0024764, 0.0023889, -0.0031598, 0.0030407
7: -0.0056373, 0.0008031, -0.0058101, 0.0008159, -0.0041411, 0.0043033
8: 0.9852429, 0.9897796, 0.9851211, 0.9897885, -0.0029171, 0.0030314
9: -0.0066099, -0.0024917, -0.0066180, -0.0023812, -0.0027517, 0.0026480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016656, upper bound: 0.0015628
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016658, upper bound: 0.0015458
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0031954, 0.0054167, 0.0032204, 0.0053136, -0.0014101, 0.0015093
1: 0.0017840, 0.0021049, 0.0017876, 0.0020900, -0.0002037, 0.0002180
2: 0.0113651, 0.0125932, 0.0114221, 0.0125794, -0.0008344, 0.0007796
3: -0.0029262, -0.0016560, -0.0028672, -0.0016703, -0.0008630, 0.0008063
4: -0.0022442, -0.0008692, -0.0022288, -0.0009330, -0.0008729, 0.0009343
5: 0.0049359, 0.0062372, 0.0049963, 0.0062225, -0.0008841, 0.0008260
6: -0.0027160, 0.0024469, -0.0024764, 0.0023889, -0.0035080, 0.0032774
7: -0.0058892, 0.0011422, -0.0058101, 0.0008159, -0.0044635, 0.0047776
8: 0.9850654, 0.9900184, 0.9851211, 0.9897885, -0.0031442, 0.0033654
9: -0.0068267, -0.0023307, -0.0066180, -0.0023812, -0.0030549, 0.0028541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016656, upper bound: 0.0015628
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016658, upper bound: 0.0015458
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032625, 0.0053762, 0.0032980, 0.0050852, -0.0013418, 0.0016238
1: 0.0017936, 0.0020990, 0.0017988, 0.0020570, -0.0001939, 0.0002346
2: 0.0113875, 0.0125561, 0.0115484, 0.0125365, -0.0008978, 0.0007419
3: -0.0029030, -0.0016943, -0.0027366, -0.0017147, -0.0009285, 0.0007673
4: -0.0022027, -0.0008943, -0.0021807, -0.0010745, -0.0008306, 0.0010052
5: 0.0049597, 0.0061979, 0.0051302, 0.0061771, -0.0009512, 0.0007860
6: -0.0026217, 0.0022911, -0.0019453, 0.0022085, -0.0037741, 0.0031188
7: -0.0056769, 0.0010138, -0.0055644, 0.0000927, -0.0042475, 0.0051400
8: 0.9852149, 0.9899280, 0.9852942, 0.9892791, -0.0029920, 0.0036208
9: -0.0067446, -0.0024664, -0.0061556, -0.0025383, -0.0032867, 0.0027160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017011, upper bound: 0.0015433
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0015454
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0031820, 0.0054850, 0.0032980, 0.0050852, -0.0014519, 0.0017727
1: 0.0017820, 0.0021147, 0.0017988, 0.0020570, -0.0002098, 0.0002561
2: 0.0113273, 0.0126006, 0.0115484, 0.0125365, -0.0009801, 0.0008027
3: -0.0029652, -0.0016483, -0.0027366, -0.0017147, -0.0010137, 0.0008302
4: -0.0022526, -0.0008270, -0.0021807, -0.0010745, -0.0008988, 0.0010974
5: 0.0048959, 0.0062451, 0.0051302, 0.0061771, -0.0010385, 0.0008505
6: -0.0028747, 0.0024782, -0.0019453, 0.0022085, -0.0041203, 0.0033747
7: -0.0059319, 0.0013584, -0.0055644, 0.0000927, -0.0045960, 0.0056115
8: 0.9850354, 0.9901708, 0.9852942, 0.9892791, -0.0032375, 0.0039529
9: -0.0069650, -0.0023034, -0.0061556, -0.0025383, -0.0035882, 0.0029388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017011, upper bound: 0.0015433
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0015454
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032625, 0.0053762, 0.0032205, 0.0053131, -0.0013276, 0.0014775
1: 0.0017936, 0.0020990, 0.0017876, 0.0020899, -0.0001918, 0.0002135
2: 0.0113875, 0.0125561, 0.0114224, 0.0125794, -0.0008169, 0.0007340
3: -0.0029030, -0.0016943, -0.0028669, -0.0016703, -0.0008448, 0.0007591
4: -0.0022027, -0.0008943, -0.0022288, -0.0009334, -0.0008218, 0.0009146
5: 0.0049597, 0.0061979, 0.0049967, 0.0062225, -0.0008655, 0.0007777
6: -0.0026217, 0.0022911, -0.0024750, 0.0023888, -0.0034341, 0.0030856
7: -0.0056769, 0.0010138, -0.0058100, 0.0008141, -0.0042024, 0.0046769
8: 0.9852149, 0.9899280, 0.9851211, 0.9897873, -0.0029602, 0.0032945
9: -0.0067446, -0.0024664, -0.0066169, -0.0023813, -0.0029906, 0.0026871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016918, upper bound: 0.0015628
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0015458
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0031820, 0.0054850, 0.0032205, 0.0053131, -0.0014296, 0.0016091
1: 0.0017820, 0.0021147, 0.0017876, 0.0020899, -0.0002065, 0.0002325
2: 0.0113273, 0.0126006, 0.0114224, 0.0125794, -0.0008896, 0.0007904
3: -0.0029652, -0.0016483, -0.0028669, -0.0016703, -0.0009201, 0.0008175
4: -0.0022526, -0.0008270, -0.0022288, -0.0009334, -0.0008850, 0.0009961
5: 0.0048959, 0.0062451, 0.0049967, 0.0062225, -0.0009426, 0.0008375
6: -0.0028747, 0.0024782, -0.0024750, 0.0023888, -0.0037401, 0.0033228
7: -0.0059319, 0.0013584, -0.0058100, 0.0008141, -0.0045254, 0.0050936
8: 0.9850354, 0.9901708, 0.9851211, 0.9897873, -0.0031878, 0.0035881
9: -0.0069650, -0.0023034, -0.0066169, -0.0023813, -0.0032570, 0.0028937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016918, upper bound: 0.0015628
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0015458
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032750, 0.0053096, 0.0032239, 0.0051815, -0.0014702, 0.0016278
1: 0.0017954, 0.0020894, 0.0017881, 0.0020709, -0.0002124, 0.0002352
2: 0.0114243, 0.0125492, 0.0114952, 0.0125774, -0.0009000, 0.0008128
3: -0.0028649, -0.0017015, -0.0027916, -0.0016723, -0.0009308, 0.0008407
4: -0.0021950, -0.0009356, -0.0022266, -0.0010149, -0.0009101, 0.0010077
5: 0.0049987, 0.0061905, 0.0050737, 0.0062205, -0.0009536, 0.0008612
6: -0.0024670, 0.0022619, -0.0021692, 0.0023807, -0.0037835, 0.0034171
7: -0.0056373, 0.0008031, -0.0057990, 0.0003976, -0.0046539, 0.0051529
8: 0.9852429, 0.9897796, 0.9851289, 0.9894938, -0.0032783, 0.0036298
9: -0.0066099, -0.0024917, -0.0063506, -0.0023883, -0.0032949, 0.0029758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016592, upper bound: 0.0015973
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016455, upper bound: 0.0015974
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0031954, 0.0054167, 0.0032239, 0.0051815, -0.0013811, 0.0015833
1: 0.0017840, 0.0021049, 0.0017881, 0.0020709, -0.0001995, 0.0002287
2: 0.0113651, 0.0125932, 0.0114952, 0.0125774, -0.0008754, 0.0007636
3: -0.0029262, -0.0016560, -0.0027916, -0.0016723, -0.0009053, 0.0007897
4: -0.0022442, -0.0008692, -0.0022266, -0.0010149, -0.0008549, 0.0009801
5: 0.0049359, 0.0062372, 0.0050737, 0.0062205, -0.0009275, 0.0008091
6: -0.0027160, 0.0024469, -0.0021692, 0.0023807, -0.0036800, 0.0032101
7: -0.0058892, 0.0011422, -0.0057990, 0.0003976, -0.0043719, 0.0050119
8: 0.9850654, 0.9900184, 0.9851289, 0.9894938, -0.0030797, 0.0035305
9: -0.0068267, -0.0023307, -0.0063506, -0.0023883, -0.0032047, 0.0027955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016592, upper bound: 0.0015433
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016455, upper bound: 0.0015454
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032750, 0.0053096, 0.0031406, 0.0054208, -0.0014584, 0.0014629
1: 0.0017954, 0.0020894, 0.0017760, 0.0021054, -0.0002107, 0.0002114
2: 0.0114243, 0.0125492, 0.0113628, 0.0126235, -0.0008088, 0.0008063
3: -0.0028649, -0.0017015, -0.0029285, -0.0016246, -0.0008365, 0.0008339
4: -0.0021950, -0.0009356, -0.0022782, -0.0008667, -0.0009028, 0.0009056
5: 0.0049987, 0.0061905, 0.0049336, 0.0062693, -0.0008570, 0.0008543
6: -0.0024670, 0.0022619, -0.0027254, 0.0025743, -0.0034003, 0.0033898
7: -0.0056373, 0.0008031, -0.0060627, 0.0011551, -0.0046166, 0.0046309
8: 0.9852429, 0.9897796, 0.9849433, 0.9900275, -0.0032520, 0.0032621
9: -0.0066099, -0.0024917, -0.0068349, -0.0022197, -0.0029611, 0.0029520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016450, upper bound: 0.0016172
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016455, upper bound: 0.0015982
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0031954, 0.0054167, 0.0031406, 0.0054208, -0.0013659, 0.0014168
1: 0.0017840, 0.0021049, 0.0017760, 0.0021054, -0.0001973, 0.0002047
2: 0.0113651, 0.0125932, 0.0113628, 0.0126235, -0.0007833, 0.0007552
3: -0.0029262, -0.0016560, -0.0029285, -0.0016246, -0.0008102, 0.0007810
4: -0.0022442, -0.0008692, -0.0022782, -0.0008667, -0.0008455, 0.0008771
5: 0.0049359, 0.0062372, 0.0049336, 0.0062693, -0.0008300, 0.0008001
6: -0.0027160, 0.0024469, -0.0027254, 0.0025743, -0.0032931, 0.0031747
7: -0.0058892, 0.0011422, -0.0060627, 0.0011551, -0.0043236, 0.0044850
8: 0.9850654, 0.9900184, 0.9849433, 0.9900275, -0.0030456, 0.0031593
9: -0.0068267, -0.0023307, -0.0068349, -0.0022197, -0.0028678, 0.0027646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016450, upper bound: 0.0015628
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016455, upper bound: 0.0015458
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0032625, 0.0053762, 0.0032273, 0.0051808, -0.0014831, 0.0017179
1: 0.0017936, 0.0020990, 0.0017886, 0.0020708, -0.0002143, 0.0002482
2: 0.0113875, 0.0125561, 0.0114955, 0.0125755, -0.0009498, 0.0008200
3: -0.0029030, -0.0016943, -0.0027912, -0.0016742, -0.0009823, 0.0008480
4: -0.0022027, -0.0008943, -0.0022245, -0.0010153, -0.0009180, 0.0010634
5: 0.0049597, 0.0061979, 0.0050742, 0.0062185, -0.0010063, 0.0008688
6: -0.0026217, 0.0022911, -0.0021676, 0.0023728, -0.0039929, 0.0034471
7: -0.0056769, 0.0010138, -0.0057882, 0.0003953, -0.0046946, 0.0054379
8: 0.9852149, 0.9899280, 0.9851366, 0.9894923, -0.0033070, 0.0038306
9: -0.0067446, -0.0024664, -0.0063491, -0.0023952, -0.0034772, 0.0030019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016728, upper bound: 0.0015973
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015974
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0031820, 0.0054850, 0.0032273, 0.0051808, -0.0013961, 0.0016771
1: 0.0017820, 0.0021147, 0.0017886, 0.0020708, -0.0002017, 0.0002423
2: 0.0113273, 0.0126006, 0.0114955, 0.0125755, -0.0009272, 0.0007719
3: -0.0029652, -0.0016483, -0.0027912, -0.0016742, -0.0009590, 0.0007983
4: -0.0022526, -0.0008270, -0.0022245, -0.0010153, -0.0008642, 0.0010381
5: 0.0048959, 0.0062451, 0.0050742, 0.0062185, -0.0009824, 0.0008178
6: -0.0028747, 0.0024782, -0.0021676, 0.0023728, -0.0038980, 0.0032449
7: -0.0059319, 0.0013584, -0.0057882, 0.0003953, -0.0044193, 0.0053087
8: 0.9850354, 0.9901708, 0.9851366, 0.9894923, -0.0031131, 0.0037396
9: -0.0069650, -0.0023034, -0.0063491, -0.0023952, -0.0033945, 0.0028258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016728, upper bound: 0.0015433
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015454
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0032625, 0.0053762, 0.0031403, 0.0054202, -0.0014777, 0.0015764
1: 0.0017936, 0.0020990, 0.0017760, 0.0021054, -0.0002135, 0.0002278
2: 0.0113875, 0.0125561, 0.0113632, 0.0126237, -0.0008716, 0.0008170
3: -0.0029030, -0.0016943, -0.0029281, -0.0016244, -0.0009014, 0.0008450
4: -0.0022027, -0.0008943, -0.0022784, -0.0008671, -0.0009147, 0.0009758
5: 0.0049597, 0.0061979, 0.0049339, 0.0062695, -0.0009235, 0.0008656
6: -0.0026217, 0.0022911, -0.0027240, 0.0025752, -0.0036641, 0.0034345
7: -0.0056769, 0.0010138, -0.0060639, 0.0011531, -0.0046776, 0.0049902
8: 0.9852149, 0.9899280, 0.9849424, 0.9900261, -0.0032950, 0.0035152
9: -0.0067446, -0.0024664, -0.0068337, -0.0022189, -0.0031909, 0.0029910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0016171
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015982
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0031820, 0.0054850, 0.0031403, 0.0054202, -0.0013826, 0.0015336
1: 0.0017820, 0.0021147, 0.0017760, 0.0021054, -0.0001997, 0.0002216
2: 0.0113273, 0.0126006, 0.0113632, 0.0126237, -0.0008479, 0.0007644
3: -0.0029652, -0.0016483, -0.0029281, -0.0016244, -0.0008769, 0.0007906
4: -0.0022526, -0.0008270, -0.0022784, -0.0008671, -0.0008559, 0.0009493
5: 0.0048959, 0.0062451, 0.0049339, 0.0062695, -0.0008984, 0.0008099
6: -0.0028747, 0.0024782, -0.0027240, 0.0025752, -0.0035645, 0.0032136
7: -0.0059319, 0.0013584, -0.0060639, 0.0011531, -0.0043766, 0.0048545
8: 0.9850354, 0.9901708, 0.9849424, 0.9900261, -0.0030830, 0.0034196
9: -0.0069650, -0.0023034, -0.0068337, -0.0022189, -0.0031041, 0.0027985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015628
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015458
time: 0.77 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.58 seconds
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015418, upper bound: 0.0016729
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015418, upper bound: 0.0016644
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015418, upper bound: 0.0016728
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015418, upper bound: 0.0016644
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015558, upper bound: 0.0016729
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015558, upper bound: 0.0016644
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015558, upper bound: 0.0016729
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015558, upper bound: 0.0016644
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015381, upper bound: 0.0016314
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015383, upper bound: 0.0016157
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017613
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017499
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0016728
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0016644
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015511, upper bound: 0.0016314
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015515, upper bound: 0.0016157
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015433, upper bound: 0.0017613
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0017499
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015433, upper bound: 0.0016728
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016644
IS_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016760, upper bound: 0.0015433
IS_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016658, upper bound: 0.0015454
IS_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016760, upper bound: 0.0015433
IS_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016658, upper bound: 0.0015454
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016656, upper bound: 0.0015628
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016658, upper bound: 0.0015458
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016656, upper bound: 0.0015628
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016658, upper bound: 0.0015458
IS_A2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0017011, upper bound: 0.0015433
IS_A2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0015454
IS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0017011, upper bound: 0.0015433
IS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0015454
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016918, upper bound: 0.0015628
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0015458
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016918, upper bound: 0.0015628
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016920, upper bound: 0.0015458
IS_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016592, upper bound: 0.0015973
IS_A2_B2_A1_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016455, upper bound: 0.0015974
IS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016592, upper bound: 0.0015433
IS_A2_B2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016455, upper bound: 0.0015454
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016450, upper bound: 0.0016172
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016455, upper bound: 0.0015982
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016450, upper bound: 0.0015628
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016455, upper bound: 0.0015458
IS_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016728, upper bound: 0.0015973
IS_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015974
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016728, upper bound: 0.0015433
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015454
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0016171
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015982
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015628
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.58
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015458

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033496, 0.0050810, 0.0032503, 0.0053118, -0.0014538, 0.0013287
1: 0.0018062, 0.0020564, 0.0017919, 0.0020897, -0.0002100, 0.0001920
2: 0.0115507, 0.0125079, 0.0114231, 0.0125629, -0.0007346, 0.0008037
3: -0.0027342, -0.0017442, -0.0028661, -0.0016873, -0.0007597, 0.0008313
4: -0.0021488, -0.0010771, -0.0022103, -0.0009342, -0.0008999, 0.0008225
5: 0.0051326, 0.0061469, 0.0049974, 0.0062050, -0.0007783, 0.0008516
6: -0.0019356, 0.0020886, -0.0024721, 0.0023195, -0.0030882, 0.0033789
7: -0.0054012, 0.0000795, -0.0057156, 0.0008101, -0.0046018, 0.0042058
8: 0.9854091, 0.9892699, 0.9851876, 0.9897845, -0.0032416, 0.0029627
9: -0.0061472, -0.0026427, -0.0066143, -0.0024416, -0.0026893, 0.0029425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015438, upper bound: 0.0016763
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015438, upper bound: 0.0017017
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033776, 0.0050800, 0.0032962, 0.0053713, -0.0014940, 0.0013515
1: 0.0018103, 0.0020562, 0.0017985, 0.0020983, -0.0002158, 0.0001952
2: 0.0115512, 0.0124925, 0.0113902, 0.0125375, -0.0007472, 0.0008260
3: -0.0027336, -0.0017601, -0.0029002, -0.0017136, -0.0007728, 0.0008543
4: -0.0021315, -0.0010777, -0.0021819, -0.0008974, -0.0009248, 0.0008366
5: 0.0051332, 0.0061305, 0.0049626, 0.0061782, -0.0007917, 0.0008752
6: -0.0019334, 0.0020236, -0.0026104, 0.0022128, -0.0031412, 0.0034724
7: -0.0053127, 0.0000764, -0.0055704, 0.0009984, -0.0047291, 0.0042780
8: 0.9854715, 0.9892677, 0.9852900, 0.9899172, -0.0033313, 0.0030135
9: -0.0061452, -0.0026993, -0.0067347, -0.0025345, -0.0027355, 0.0030239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015438, upper bound: 0.0016665
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015438, upper bound: 0.0016925
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032816, 0.0051768, 0.0032503, 0.0053118, -0.0015529, 0.0014699
1: 0.0017964, 0.0020702, 0.0017919, 0.0020897, -0.0002243, 0.0002124
2: 0.0114978, 0.0125455, 0.0114231, 0.0125629, -0.0008127, 0.0008585
3: -0.0027889, -0.0017053, -0.0028661, -0.0016873, -0.0008405, 0.0008879
4: -0.0021909, -0.0010178, -0.0022103, -0.0009342, -0.0009612, 0.0009099
5: 0.0050765, 0.0061867, 0.0049974, 0.0062050, -0.0008611, 0.0009097
6: -0.0021582, 0.0022466, -0.0024721, 0.0023195, -0.0034164, 0.0036093
7: -0.0056164, 0.0003826, -0.0057156, 0.0008101, -0.0049155, 0.0046529
8: 0.9852576, 0.9894834, 0.9851876, 0.9897845, -0.0034626, 0.0032776
9: -0.0063410, -0.0025051, -0.0066143, -0.0024416, -0.0029752, 0.0031431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015826, upper bound: 0.0016568
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015826, upper bound: 0.0016729
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033118, 0.0051758, 0.0032962, 0.0053713, -0.0015915, 0.0014928
1: 0.0018008, 0.0020701, 0.0017985, 0.0020983, -0.0002299, 0.0002157
2: 0.0114983, 0.0125288, 0.0113902, 0.0125375, -0.0008254, 0.0008799
3: -0.0027884, -0.0017226, -0.0029002, -0.0017136, -0.0008536, 0.0009100
4: -0.0021722, -0.0010184, -0.0021819, -0.0008974, -0.0009852, 0.0009241
5: 0.0050771, 0.0061690, 0.0049626, 0.0061782, -0.0008745, 0.0009323
6: -0.0021560, 0.0021764, -0.0026104, 0.0022128, -0.0034698, 0.0036991
7: -0.0055207, 0.0003795, -0.0055704, 0.0009984, -0.0050379, 0.0047256
8: 0.9853249, 0.9894812, 0.9852900, 0.9899172, -0.0035488, 0.0033288
9: -0.0063390, -0.0025662, -0.0067347, -0.0025345, -0.0030216, 0.0032214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015828, upper bound: 0.0016451
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015828, upper bound: 0.0016644
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033455, 0.0051409, 0.0032501, 0.0053112, -0.0014669, 0.0014362
1: 0.0018056, 0.0020650, 0.0017918, 0.0020896, -0.0002119, 0.0002075
2: 0.0115176, 0.0125102, 0.0114234, 0.0125630, -0.0007940, 0.0008110
3: -0.0027684, -0.0017418, -0.0028658, -0.0016873, -0.0008212, 0.0008388
4: -0.0021513, -0.0010400, -0.0022104, -0.0009345, -0.0009080, 0.0008890
5: 0.0050975, 0.0061492, 0.0049977, 0.0062051, -0.0008413, 0.0008593
6: -0.0020748, 0.0020981, -0.0024708, 0.0023199, -0.0033381, 0.0034094
7: -0.0054141, 0.0002690, -0.0057162, 0.0008083, -0.0046433, 0.0045462
8: 0.9854000, 0.9894034, 0.9851873, 0.9897832, -0.0032708, 0.0032024
9: -0.0062684, -0.0026344, -0.0066132, -0.0024413, -0.0029070, 0.0029690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015577, upper bound: 0.0016763
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015577, upper bound: 0.0016763
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033729, 0.0051398, 0.0032965, 0.0053706, -0.0015069, 0.0014597
1: 0.0018096, 0.0020649, 0.0017985, 0.0020982, -0.0002177, 0.0002109
2: 0.0115182, 0.0124951, 0.0113906, 0.0125373, -0.0008070, 0.0008331
3: -0.0027678, -0.0017575, -0.0028997, -0.0017138, -0.0008347, 0.0008616
4: -0.0021344, -0.0010406, -0.0021817, -0.0008978, -0.0009328, 0.0009036
5: 0.0050981, 0.0061332, 0.0049630, 0.0061780, -0.0008551, 0.0008827
6: -0.0020724, 0.0020344, -0.0026087, 0.0022121, -0.0033927, 0.0035024
7: -0.0053274, 0.0002657, -0.0055694, 0.0009961, -0.0047700, 0.0046206
8: 0.9854611, 0.9894011, 0.9852907, 0.9899155, -0.0033601, 0.0032548
9: -0.0062663, -0.0026899, -0.0067333, -0.0025351, -0.0029545, 0.0030501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015578, upper bound: 0.0016665
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015578, upper bound: 0.0016665
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032731, 0.0052395, 0.0032501, 0.0053112, -0.0015663, 0.0015593
1: 0.0017952, 0.0020793, 0.0017918, 0.0020896, -0.0002263, 0.0002253
2: 0.0114631, 0.0125503, 0.0114234, 0.0125630, -0.0008621, 0.0008659
3: -0.0028248, -0.0017004, -0.0028658, -0.0016873, -0.0008916, 0.0008956
4: -0.0021962, -0.0009790, -0.0022104, -0.0009345, -0.0009695, 0.0009652
5: 0.0050398, 0.0061917, 0.0049977, 0.0062051, -0.0009135, 0.0009175
6: -0.0023040, 0.0022665, -0.0024708, 0.0023199, -0.0036243, 0.0036404
7: -0.0056434, 0.0005811, -0.0057162, 0.0008083, -0.0049579, 0.0049360
8: 0.9852385, 0.9896232, 0.9851873, 0.9897832, -0.0034925, 0.0034770
9: -0.0064679, -0.0024878, -0.0066132, -0.0024413, -0.0031562, 0.0031702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015973, upper bound: 0.0016568
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015973, upper bound: 0.0016568
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033023, 0.0052385, 0.0032965, 0.0053706, -0.0016048, 0.0015815
1: 0.0017994, 0.0020791, 0.0017985, 0.0020982, -0.0002318, 0.0002285
2: 0.0114637, 0.0125341, 0.0113906, 0.0125373, -0.0008744, 0.0008872
3: -0.0028242, -0.0017171, -0.0028997, -0.0017138, -0.0009043, 0.0009176
4: -0.0021781, -0.0009796, -0.0021817, -0.0008978, -0.0009934, 0.0009790
5: 0.0050404, 0.0061746, 0.0049630, 0.0061780, -0.0009264, 0.0009401
6: -0.0023016, 0.0021986, -0.0026087, 0.0022121, -0.0036758, 0.0037299
7: -0.0055510, 0.0005779, -0.0055694, 0.0009961, -0.0050798, 0.0050061
8: 0.9853037, 0.9896209, 0.9852907, 0.9899155, -0.0035783, 0.0035264
9: -0.0064659, -0.0025469, -0.0067333, -0.0025351, -0.0032011, 0.0032482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015974, upper bound: 0.0016451
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015974, upper bound: 0.0016451
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033496, 0.0050810, 0.0031694, 0.0054190, -0.0016032, 0.0014467
1: 0.0018062, 0.0020564, 0.0017802, 0.0021052, -0.0002316, 0.0002090
2: 0.0115507, 0.0125079, 0.0113638, 0.0126076, -0.0007999, 0.0008864
3: -0.0027342, -0.0017442, -0.0029275, -0.0016411, -0.0008273, 0.0009167
4: -0.0021488, -0.0010771, -0.0022604, -0.0008678, -0.0009924, 0.0008956
5: 0.0051326, 0.0061469, 0.0049346, 0.0062524, -0.0008475, 0.0009392
6: -0.0019356, 0.0020886, -0.0027213, 0.0025075, -0.0033626, 0.0037263
7: -0.0054012, 0.0000795, -0.0059717, 0.0011494, -0.0050750, 0.0045796
8: 0.9854091, 0.9892699, 0.9850073, 0.9900236, -0.0035749, 0.0032260
9: -0.0061472, -0.0026427, -0.0068313, -0.0022779, -0.0029283, 0.0032451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017322
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017613
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033776, 0.0050800, 0.0032162, 0.0054795, -0.0016294, 0.0014669
1: 0.0018103, 0.0020562, 0.0017869, 0.0021139, -0.0002354, 0.0002119
2: 0.0115512, 0.0124925, 0.0113304, 0.0125817, -0.0008110, 0.0009008
3: -0.0027336, -0.0017601, -0.0029620, -0.0016678, -0.0008388, 0.0009317
4: -0.0021315, -0.0010777, -0.0022314, -0.0008304, -0.0010086, 0.0009080
5: 0.0051332, 0.0061305, 0.0048992, 0.0062250, -0.0008593, 0.0009545
6: -0.0019334, 0.0020236, -0.0028619, 0.0023987, -0.0034095, 0.0037872
7: -0.0053127, 0.0000764, -0.0058236, 0.0013410, -0.0051578, 0.0046434
8: 0.9854715, 0.9892677, 0.9851117, 0.9901584, -0.0036333, 0.0032709
9: -0.0061452, -0.0026993, -0.0069538, -0.0023726, -0.0029691, 0.0032980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017219
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017499
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032816, 0.0051768, 0.0031694, 0.0054190, -0.0015075, 0.0013809
1: 0.0017964, 0.0020702, 0.0017802, 0.0021052, -0.0002178, 0.0001995
2: 0.0114978, 0.0125455, 0.0113638, 0.0126076, -0.0007634, 0.0008335
3: -0.0027889, -0.0017053, -0.0029275, -0.0016411, -0.0007896, 0.0008620
4: -0.0021909, -0.0010178, -0.0022604, -0.0008678, -0.0009332, 0.0008548
5: 0.0050765, 0.0061867, 0.0049346, 0.0062524, -0.0008089, 0.0008831
6: -0.0021582, 0.0022466, -0.0027213, 0.0025075, -0.0032095, 0.0035039
7: -0.0056164, 0.0003826, -0.0059717, 0.0011494, -0.0047720, 0.0043711
8: 0.9852576, 0.9894834, 0.9850073, 0.9900236, -0.0033615, 0.0030791
9: -0.0063410, -0.0025051, -0.0068313, -0.0022779, -0.0027950, 0.0030513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015826, upper bound: 0.0016568
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015826, upper bound: 0.0016729
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033118, 0.0051758, 0.0032162, 0.0054795, -0.0015472, 0.0014008
1: 0.0018008, 0.0020701, 0.0017869, 0.0021139, -0.0002235, 0.0002024
2: 0.0114983, 0.0125288, 0.0113304, 0.0125817, -0.0007745, 0.0008554
3: -0.0027884, -0.0017226, -0.0029620, -0.0016678, -0.0008010, 0.0008847
4: -0.0021722, -0.0010184, -0.0022314, -0.0008304, -0.0009577, 0.0008671
5: 0.0050771, 0.0061690, 0.0048992, 0.0062250, -0.0008206, 0.0009063
6: -0.0021560, 0.0021764, -0.0028619, 0.0023987, -0.0032560, 0.0035961
7: -0.0055207, 0.0003795, -0.0058236, 0.0013410, -0.0048976, 0.0044343
8: 0.9853249, 0.9894812, 0.9851117, 0.9901584, -0.0034500, 0.0031236
9: -0.0063390, -0.0025662, -0.0069538, -0.0023726, -0.0028354, 0.0031317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015828, upper bound: 0.0016451
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015828, upper bound: 0.0016644
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033455, 0.0051409, 0.0031684, 0.0054184, -0.0016163, 0.0015561
1: 0.0018056, 0.0020650, 0.0017800, 0.0021051, -0.0002335, 0.0002248
2: 0.0115176, 0.0125102, 0.0113642, 0.0126081, -0.0008603, 0.0008936
3: -0.0027684, -0.0017418, -0.0029271, -0.0016405, -0.0008898, 0.0009242
4: -0.0021513, -0.0010400, -0.0022610, -0.0008682, -0.0010005, 0.0009633
5: 0.0050975, 0.0061492, 0.0049350, 0.0062530, -0.0009116, 0.0009468
6: -0.0020748, 0.0020981, -0.0027199, 0.0025098, -0.0036169, 0.0037567
7: -0.0054141, 0.0002690, -0.0059748, 0.0011475, -0.0051163, 0.0049259
8: 0.9854000, 0.9894034, 0.9850051, 0.9900222, -0.0036040, 0.0034699
9: -0.0062684, -0.0026344, -0.0068301, -0.0022759, -0.0031497, 0.0032715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015433, upper bound: 0.0017319
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015433, upper bound: 0.0017319
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033729, 0.0051398, 0.0032168, 0.0054788, -0.0016423, 0.0015740
1: 0.0018096, 0.0020649, 0.0017870, 0.0021138, -0.0002373, 0.0002274
2: 0.0115182, 0.0124951, 0.0113308, 0.0125814, -0.0008702, 0.0009080
3: -0.0027678, -0.0017575, -0.0029616, -0.0016682, -0.0009001, 0.0009391
4: -0.0021344, -0.0010406, -0.0022310, -0.0008308, -0.0010166, 0.0009744
5: 0.0050981, 0.0061332, 0.0048996, 0.0062247, -0.0009221, 0.0009620
6: -0.0020724, 0.0020344, -0.0028601, 0.0023973, -0.0036585, 0.0038171
7: -0.0053274, 0.0002657, -0.0058217, 0.0013386, -0.0051985, 0.0049826
8: 0.9854611, 0.9894011, 0.9851130, 0.9901567, -0.0036620, 0.0035098
9: -0.0062663, -0.0026899, -0.0069523, -0.0023738, -0.0031860, 0.0033241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0017217
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0017216
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032731, 0.0052395, 0.0031684, 0.0054184, -0.0015227, 0.0014871
1: 0.0017952, 0.0020793, 0.0017800, 0.0021051, -0.0002200, 0.0002149
2: 0.0114631, 0.0125503, 0.0113642, 0.0126081, -0.0008222, 0.0008418
3: -0.0028248, -0.0017004, -0.0029271, -0.0016405, -0.0008504, 0.0008707
4: -0.0021962, -0.0009790, -0.0022610, -0.0008682, -0.0009425, 0.0009206
5: 0.0050398, 0.0061917, 0.0049350, 0.0062530, -0.0008712, 0.0008920
6: -0.0023040, 0.0022665, -0.0027199, 0.0025098, -0.0034565, 0.0035391
7: -0.0056434, 0.0005811, -0.0059748, 0.0011475, -0.0048199, 0.0047075
8: 0.9852385, 0.9896232, 0.9850051, 0.9900222, -0.0033952, 0.0033161
9: -0.0064679, -0.0024878, -0.0068301, -0.0022759, -0.0030101, 0.0030820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015973, upper bound: 0.0016568
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0015973, upper bound: 0.0016568
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033023, 0.0052385, 0.0032168, 0.0054788, -0.0015620, 0.0015079
1: 0.0017994, 0.0020791, 0.0017870, 0.0021138, -0.0002257, 0.0002179
2: 0.0114637, 0.0125341, 0.0113308, 0.0125814, -0.0008337, 0.0008636
3: -0.0028242, -0.0017171, -0.0029616, -0.0016682, -0.0008622, 0.0008932
4: -0.0021781, -0.0009796, -0.0022310, -0.0008308, -0.0009669, 0.0009334
5: 0.0050404, 0.0061746, 0.0048996, 0.0062247, -0.0008833, 0.0009150
6: -0.0023016, 0.0021986, -0.0028601, 0.0023973, -0.0035048, 0.0036306
7: -0.0055510, 0.0005779, -0.0058217, 0.0013386, -0.0049446, 0.0047733
8: 0.9853037, 0.9896209, 0.9851130, 0.9901567, -0.0034831, 0.0033624
9: -0.0064659, -0.0025469, -0.0069523, -0.0023738, -0.0030522, 0.0031617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015974, upper bound: 0.0016451
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015974, upper bound: 0.0016451
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033029, 0.0053079, 0.0032992, 0.0050855, -0.0012801, 0.0015074
1: 0.0017995, 0.0020891, 0.0017989, 0.0020570, -0.0001849, 0.0002178
2: 0.0114252, 0.0125337, 0.0115482, 0.0125358, -0.0008334, 0.0007077
3: -0.0028639, -0.0017175, -0.0027367, -0.0017153, -0.0008619, 0.0007320
4: -0.0021777, -0.0009366, -0.0021800, -0.0010743, -0.0007924, 0.0009331
5: 0.0049997, 0.0061742, 0.0051300, 0.0061764, -0.0008830, 0.0007499
6: -0.0024631, 0.0021970, -0.0019461, 0.0022057, -0.0035036, 0.0029753
7: -0.0055489, 0.0007978, -0.0055607, 0.0000936, -0.0040521, 0.0047716
8: 0.9853051, 0.9897758, 0.9852968, 0.9892798, -0.0028544, 0.0033612
9: -0.0066065, -0.0025482, -0.0061562, -0.0025407, -0.0030511, 0.0025910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016766, upper bound: 0.0015419
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016766, upper bound: 0.0015577
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033502, 0.0053675, 0.0033282, 0.0050845, -0.0013051, 0.0015495
1: 0.0018063, 0.0020977, 0.0018031, 0.0020569, -0.0001885, 0.0002239
2: 0.0113923, 0.0125076, 0.0115488, 0.0125198, -0.0008567, 0.0007216
3: -0.0028980, -0.0017445, -0.0027362, -0.0017319, -0.0008860, 0.0007463
4: -0.0021484, -0.0008997, -0.0021621, -0.0010749, -0.0008079, 0.0009592
5: 0.0049648, 0.0061465, 0.0051306, 0.0061594, -0.0009077, 0.0007645
6: -0.0026015, 0.0020871, -0.0019437, 0.0021384, -0.0036015, 0.0030334
7: -0.0053992, 0.0009863, -0.0054691, 0.0000905, -0.0041312, 0.0049049
8: 0.9854105, 0.9899086, 0.9853613, 0.9892776, -0.0029101, 0.0034551
9: -0.0067270, -0.0026440, -0.0061542, -0.0025993, -0.0031364, 0.0026416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016664, upper bound: 0.0015419
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016664, upper bound: 0.0015578
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0032246, 0.0054151, 0.0032992, 0.0050855, -0.0013925, 0.0016568
1: 0.0017882, 0.0021046, 0.0017989, 0.0020570, -0.0002012, 0.0002394
2: 0.0113660, 0.0125771, 0.0115482, 0.0125358, -0.0009160, 0.0007699
3: -0.0029252, -0.0016727, -0.0027367, -0.0017153, -0.0009474, 0.0007962
4: -0.0022262, -0.0008702, -0.0021800, -0.0010743, -0.0008620, 0.0010256
5: 0.0049369, 0.0062201, 0.0051300, 0.0061764, -0.0009706, 0.0008157
6: -0.0027123, 0.0023792, -0.0019461, 0.0022057, -0.0038509, 0.0032364
7: -0.0057970, 0.0011372, -0.0055607, 0.0000936, -0.0044078, 0.0052446
8: 0.9851304, 0.9900149, 0.9852968, 0.9892798, -0.0031049, 0.0036944
9: -0.0068235, -0.0023896, -0.0061562, -0.0025407, -0.0033536, 0.0028184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017319, upper bound: 0.0015308
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017319, upper bound: 0.0015433
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0032707, 0.0054757, 0.0033282, 0.0050845, -0.0014138, 0.0016848
1: 0.0017948, 0.0021134, 0.0018031, 0.0020569, -0.0002043, 0.0002434
2: 0.0113325, 0.0125516, 0.0115488, 0.0125198, -0.0009315, 0.0007817
3: -0.0029599, -0.0016991, -0.0027362, -0.0017319, -0.0009634, 0.0008084
4: -0.0021976, -0.0008327, -0.0021621, -0.0010749, -0.0008752, 0.0010429
5: 0.0049014, 0.0061931, 0.0051306, 0.0061594, -0.0009870, 0.0008282
6: -0.0028531, 0.0022719, -0.0019437, 0.0021384, -0.0039160, 0.0032861
7: -0.0056508, 0.0013290, -0.0054691, 0.0000905, -0.0044754, 0.0053333
8: 0.9852333, 0.9901500, 0.9853613, 0.9892776, -0.0031526, 0.0037569
9: -0.0069462, -0.0024831, -0.0061542, -0.0025993, -0.0034103, 0.0028617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015308
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015454
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032808, 0.0053093, 0.0032503, 0.0053118, -0.0012857, 0.0013111
1: 0.0017963, 0.0020893, 0.0017919, 0.0020897, -0.0001857, 0.0001894
2: 0.0114245, 0.0125460, 0.0114231, 0.0125629, -0.0007249, 0.0007108
3: -0.0028647, -0.0017048, -0.0028661, -0.0016873, -0.0007497, 0.0007352
4: -0.0021914, -0.0009358, -0.0022103, -0.0009342, -0.0007959, 0.0008116
5: 0.0049989, 0.0061871, 0.0049974, 0.0062050, -0.0007680, 0.0007531
6: -0.0024662, 0.0022484, -0.0024721, 0.0023195, -0.0030473, 0.0029883
7: -0.0056189, 0.0008020, -0.0057156, 0.0008101, -0.0040698, 0.0041501
8: 0.9852557, 0.9897788, 0.9851876, 0.9897845, -0.0028668, 0.0029234
9: -0.0066092, -0.0025035, -0.0066143, -0.0024416, -0.0026537, 0.0026023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016663, upper bound: 0.0015610
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016663, upper bound: 0.0015759
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033073, 0.0053082, 0.0032962, 0.0053713, -0.0013335, 0.0013384
1: 0.0018001, 0.0020892, 0.0017985, 0.0020983, -0.0001927, 0.0001934
2: 0.0114251, 0.0125314, 0.0113902, 0.0125375, -0.0007400, 0.0007373
3: -0.0028641, -0.0017199, -0.0029002, -0.0017136, -0.0007653, 0.0007625
4: -0.0021750, -0.0009364, -0.0021819, -0.0008974, -0.0008255, 0.0008285
5: 0.0049995, 0.0061717, 0.0049626, 0.0061782, -0.0007840, 0.0007812
6: -0.0024638, 0.0021870, -0.0026104, 0.0022128, -0.0031109, 0.0030994
7: -0.0055352, 0.0007987, -0.0055704, 0.0009984, -0.0042211, 0.0042367
8: 0.9853148, 0.9897765, 0.9852900, 0.9899172, -0.0029735, 0.0029844
9: -0.0066071, -0.0025570, -0.0067347, -0.0025345, -0.0027091, 0.0026991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016664, upper bound: 0.0015420
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016664, upper bound: 0.0015579
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032012, 0.0054164, 0.0032503, 0.0053118, -0.0013880, 0.0014605
1: 0.0017848, 0.0021048, 0.0017919, 0.0020897, -0.0002005, 0.0002110
2: 0.0113653, 0.0125900, 0.0114231, 0.0125629, -0.0008075, 0.0007674
3: -0.0029260, -0.0016593, -0.0028661, -0.0016873, -0.0008351, 0.0007937
4: -0.0022407, -0.0008694, -0.0022103, -0.0009342, -0.0008592, 0.0009041
5: 0.0049361, 0.0062338, 0.0049974, 0.0062050, -0.0008556, 0.0008131
6: -0.0027152, 0.0024336, -0.0024721, 0.0023195, -0.0033947, 0.0032261
7: -0.0058711, 0.0011412, -0.0057156, 0.0008101, -0.0043937, 0.0046233
8: 0.9850782, 0.9900177, 0.9851876, 0.9897845, -0.0030950, 0.0032567
9: -0.0068261, -0.0023422, -0.0066143, -0.0024416, -0.0029562, 0.0028095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015495
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015627
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032280, 0.0054154, 0.0032962, 0.0053713, -0.0014319, 0.0014893
1: 0.0017887, 0.0021047, 0.0017985, 0.0020983, -0.0002069, 0.0002152
2: 0.0113658, 0.0125752, 0.0113902, 0.0125375, -0.0008234, 0.0007917
3: -0.0029254, -0.0016746, -0.0029002, -0.0017136, -0.0008516, 0.0008188
4: -0.0022241, -0.0008701, -0.0021819, -0.0008974, -0.0008864, 0.0009219
5: 0.0049367, 0.0062181, 0.0049626, 0.0061782, -0.0008724, 0.0008388
6: -0.0027128, 0.0023712, -0.0026104, 0.0022128, -0.0034616, 0.0033281
7: -0.0057860, 0.0011380, -0.0055704, 0.0009984, -0.0045326, 0.0047143
8: 0.9851381, 0.9900154, 0.9852900, 0.9899172, -0.0031929, 0.0033209
9: -0.0068240, -0.0023966, -0.0067347, -0.0025345, -0.0030145, 0.0028983

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015314
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015458
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0032911, 0.0053744, 0.0033034, 0.0050849, -0.0012934, 0.0015999
1: 0.0017978, 0.0020987, 0.0017996, 0.0020569, -0.0001869, 0.0002311
2: 0.0113885, 0.0125403, 0.0115486, 0.0125335, -0.0008845, 0.0007151
3: -0.0029019, -0.0017107, -0.0027364, -0.0017177, -0.0009148, 0.0007396
4: -0.0021850, -0.0008955, -0.0021774, -0.0010747, -0.0008006, 0.0009903
5: 0.0049607, 0.0061811, 0.0051303, 0.0061739, -0.0009372, 0.0007577
6: -0.0026175, 0.0022245, -0.0019446, 0.0021959, -0.0037185, 0.0030061
7: -0.0055863, 0.0010081, -0.0055474, 0.0000917, -0.0040941, 0.0050643
8: 0.9852787, 0.9899241, 0.9853063, 0.9892784, -0.0028840, 0.0035674
9: -0.0067410, -0.0025243, -0.0061550, -0.0025492, -0.0032383, 0.0026179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015419
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015419
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033390, 0.0054324, 0.0033315, 0.0050839, -0.0013195, 0.0016416
1: 0.0018047, 0.0021071, 0.0018036, 0.0020568, -0.0001906, 0.0002372
2: 0.0113564, 0.0125138, 0.0115491, 0.0125180, -0.0009076, 0.0007295
3: -0.0029351, -0.0017381, -0.0027358, -0.0017338, -0.0009387, 0.0007545
4: -0.0021554, -0.0008595, -0.0021600, -0.0010753, -0.0008168, 0.0010162
5: 0.0049268, 0.0061531, 0.0051309, 0.0061575, -0.0009616, 0.0007730
6: -0.0027524, 0.0021132, -0.0019423, 0.0021308, -0.0038155, 0.0030670
7: -0.0054348, 0.0011918, -0.0054586, 0.0000885, -0.0041770, 0.0051963
8: 0.9853855, 0.9900534, 0.9853687, 0.9892762, -0.0029424, 0.0036604
9: -0.0068584, -0.0026212, -0.0061530, -0.0026060, -0.0033227, 0.0026709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016925, upper bound: 0.0015419
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016925, upper bound: 0.0015419
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0032098, 0.0054834, 0.0033034, 0.0050849, -0.0014051, 0.0017487
1: 0.0017860, 0.0021145, 0.0017996, 0.0020569, -0.0002030, 0.0002526
2: 0.0113282, 0.0125852, 0.0115486, 0.0125335, -0.0009668, 0.0007768
3: -0.0029643, -0.0016642, -0.0027364, -0.0017177, -0.0009999, 0.0008034
4: -0.0022354, -0.0008280, -0.0021774, -0.0010747, -0.0008698, 0.0010825
5: 0.0048969, 0.0062287, 0.0051303, 0.0061739, -0.0010244, 0.0008231
6: -0.0028710, 0.0024135, -0.0019446, 0.0021959, -0.0040645, 0.0032658
7: -0.0058437, 0.0013533, -0.0055474, 0.0000917, -0.0044477, 0.0055355
8: 0.9850973, 0.9901671, 0.9853063, 0.9892784, -0.0031331, 0.0038993
9: -0.0069617, -0.0023597, -0.0061550, -0.0025492, -0.0035396, 0.0028440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017613, upper bound: 0.0015308
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017613, upper bound: 0.0015307
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0032605, 0.0055373, 0.0033315, 0.0050839, -0.0014277, 0.0017795
1: 0.0017933, 0.0021223, 0.0018036, 0.0020568, -0.0002063, 0.0002571
2: 0.0112984, 0.0125572, 0.0115491, 0.0125180, -0.0009838, 0.0007893
3: -0.0029951, -0.0016932, -0.0027358, -0.0017338, -0.0010175, 0.0008164
4: -0.0022040, -0.0007946, -0.0021600, -0.0010753, -0.0008838, 0.0011015
5: 0.0048653, 0.0061991, 0.0051309, 0.0061575, -0.0010424, 0.0008363
6: -0.0029961, 0.0022957, -0.0019423, 0.0021308, -0.0041359, 0.0033183
7: -0.0056833, 0.0015238, -0.0054586, 0.0000885, -0.0045193, 0.0056328
8: 0.9852105, 0.9902872, 0.9853687, 0.9892762, -0.0031835, 0.0039679
9: -0.0070707, -0.0024623, -0.0061530, -0.0026060, -0.0036018, 0.0028898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015308
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015308
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032683, 0.0053758, 0.0032501, 0.0053112, -0.0013051, 0.0014269
1: 0.0017945, 0.0020990, 0.0017918, 0.0020896, -0.0001885, 0.0002061
2: 0.0113877, 0.0125529, 0.0114234, 0.0125630, -0.0007889, 0.0007216
3: -0.0029028, -0.0016976, -0.0028658, -0.0016873, -0.0008159, 0.0007463
4: -0.0021992, -0.0008946, -0.0022104, -0.0009345, -0.0008079, 0.0008833
5: 0.0049599, 0.0061945, 0.0049977, 0.0062051, -0.0008359, 0.0007645
6: -0.0026209, 0.0022776, -0.0024708, 0.0023199, -0.0033165, 0.0030334
7: -0.0056587, 0.0010127, -0.0057162, 0.0008083, -0.0041312, 0.0045167
8: 0.9852278, 0.9899272, 0.9851873, 0.9897832, -0.0029101, 0.0031817
9: -0.0067439, -0.0024780, -0.0066132, -0.0024413, -0.0028881, 0.0026416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016923, upper bound: 0.0015611
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016923, upper bound: 0.0015610
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032953, 0.0053747, 0.0032965, 0.0053706, -0.0013523, 0.0014533
1: 0.0017984, 0.0020988, 0.0017985, 0.0020982, -0.0001954, 0.0002100
2: 0.0113883, 0.0125380, 0.0113906, 0.0125373, -0.0008035, 0.0007477
3: -0.0029021, -0.0017131, -0.0028997, -0.0017138, -0.0008310, 0.0007733
4: -0.0021825, -0.0008953, -0.0021817, -0.0008978, -0.0008371, 0.0008996
5: 0.0049606, 0.0061787, 0.0049630, 0.0061780, -0.0008514, 0.0007922
6: -0.0026182, 0.0022149, -0.0026087, 0.0022121, -0.0033779, 0.0031432
7: -0.0055732, 0.0010091, -0.0055694, 0.0009961, -0.0042807, 0.0046005
8: 0.9852880, 0.9899247, 0.9852907, 0.9899155, -0.0030154, 0.0032407
9: -0.0067416, -0.0025327, -0.0067333, -0.0025351, -0.0029417, 0.0027372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016925, upper bound: 0.0015420
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016925, upper bound: 0.0015420
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0031874, 0.0054847, 0.0032501, 0.0053112, -0.0014076, 0.0015585
1: 0.0017828, 0.0021147, 0.0017918, 0.0020896, -0.0002034, 0.0002252
2: 0.0113275, 0.0125976, 0.0114234, 0.0125630, -0.0008616, 0.0007782
3: -0.0029650, -0.0016514, -0.0028658, -0.0016873, -0.0008912, 0.0008049
4: -0.0022492, -0.0008272, -0.0022104, -0.0009345, -0.0008713, 0.0009647
5: 0.0048961, 0.0062419, 0.0049977, 0.0062051, -0.0009130, 0.0008246
6: -0.0028740, 0.0024655, -0.0024708, 0.0023199, -0.0036224, 0.0032716
7: -0.0059146, 0.0013574, -0.0057162, 0.0008083, -0.0044557, 0.0049334
8: 0.9850476, 0.9901701, 0.9851873, 0.9897832, -0.0031387, 0.0034752
9: -0.0069643, -0.0023144, -0.0066132, -0.0024413, -0.0031545, 0.0028491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015495
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015495
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032159, 0.0054836, 0.0032965, 0.0053706, -0.0014511, 0.0015849
1: 0.0017869, 0.0021145, 0.0017985, 0.0020982, -0.0002096, 0.0002290
2: 0.0113281, 0.0125819, 0.0113906, 0.0125373, -0.0008763, 0.0008023
3: -0.0029644, -0.0016677, -0.0028997, -0.0017138, -0.0009063, 0.0008298
4: -0.0022316, -0.0008278, -0.0021817, -0.0008978, -0.0008983, 0.0009811
5: 0.0048968, 0.0062252, 0.0049630, 0.0061780, -0.0009284, 0.0008501
6: -0.0028715, 0.0023993, -0.0026087, 0.0022121, -0.0036838, 0.0033728
7: -0.0058244, 0.0013540, -0.0055694, 0.0009961, -0.0045934, 0.0050169
8: 0.9851111, 0.9901676, 0.9852907, 0.9899155, -0.0032357, 0.0035340
9: -0.0069621, -0.0023721, -0.0067333, -0.0025351, -0.0032080, 0.0029371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015314
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015314
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0033029, 0.0053079, 0.0032293, 0.0051812, -0.0014214, 0.0016043
1: 0.0017995, 0.0020891, 0.0017888, 0.0020708, -0.0002054, 0.0002318
2: 0.0114252, 0.0125337, 0.0114953, 0.0125745, -0.0008870, 0.0007859
3: -0.0028639, -0.0017175, -0.0027915, -0.0016753, -0.0009174, 0.0008128
4: -0.0021777, -0.0009366, -0.0022233, -0.0010150, -0.0008799, 0.0009931
5: 0.0049997, 0.0061742, 0.0050739, 0.0062173, -0.0009398, 0.0008327
6: -0.0024631, 0.0021970, -0.0021685, 0.0023683, -0.0037289, 0.0033038
7: -0.0055489, 0.0007978, -0.0057821, 0.0003966, -0.0044995, 0.0050785
8: 0.9853051, 0.9897758, 0.9851409, 0.9894931, -0.0031695, 0.0035774
9: -0.0066065, -0.0025482, -0.0063499, -0.0023991, -0.0032473, 0.0028771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016592, upper bound: 0.0015825
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016592, upper bound: 0.0015973
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0032246, 0.0054151, 0.0032293, 0.0051812, -0.0013322, 0.0015606
1: 0.0017882, 0.0021046, 0.0017888, 0.0020708, -0.0001925, 0.0002255
2: 0.0113660, 0.0125771, 0.0114953, 0.0125745, -0.0008628, 0.0007365
3: -0.0029252, -0.0016727, -0.0027915, -0.0016753, -0.0008924, 0.0007617
4: -0.0022262, -0.0008702, -0.0022233, -0.0010150, -0.0008246, 0.0009660
5: 0.0049369, 0.0062201, 0.0050739, 0.0062173, -0.0009142, 0.0007804
6: -0.0027123, 0.0023792, -0.0021685, 0.0023683, -0.0036273, 0.0030963
7: -0.0057970, 0.0011372, -0.0057821, 0.0003966, -0.0042169, 0.0049400
8: 0.9851304, 0.9900149, 0.9851409, 0.9894931, -0.0029705, 0.0034799
9: -0.0068235, -0.0023896, -0.0063499, -0.0023991, -0.0031588, 0.0026964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017318, upper bound: 0.0015308
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017318, upper bound: 0.0015433
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0032911, 0.0053744, 0.0032324, 0.0051805, -0.0014346, 0.0016947
1: 0.0017978, 0.0020987, 0.0017893, 0.0020707, -0.0002073, 0.0002448
2: 0.0113885, 0.0125403, 0.0114957, 0.0125727, -0.0009370, 0.0007932
3: -0.0029019, -0.0017107, -0.0027911, -0.0016772, -0.0009691, 0.0008203
4: -0.0021850, -0.0008955, -0.0022213, -0.0010155, -0.0008880, 0.0010491
5: 0.0049607, 0.0061811, 0.0050743, 0.0062155, -0.0009928, 0.0008404
6: -0.0026175, 0.0022245, -0.0021668, 0.0023609, -0.0039390, 0.0033344
7: -0.0055863, 0.0010081, -0.0057721, 0.0003943, -0.0045412, 0.0053646
8: 0.9852787, 0.9899241, 0.9851479, 0.9894916, -0.0031989, 0.0037789
9: -0.0067410, -0.0025243, -0.0063485, -0.0024055, -0.0034303, 0.0029037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016728, upper bound: 0.0015825
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016728, upper bound: 0.0015825
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0033390, 0.0054324, 0.0032627, 0.0051794, -0.0014610, 0.0017326
1: 0.0018047, 0.0021071, 0.0017937, 0.0020706, -0.0002111, 0.0002503
2: 0.0113564, 0.0125138, 0.0114963, 0.0125560, -0.0009579, 0.0008077
3: -0.0029351, -0.0017381, -0.0027905, -0.0016944, -0.0009907, 0.0008354
4: -0.0021554, -0.0008595, -0.0022026, -0.0010161, -0.0009044, 0.0010725
5: 0.0049268, 0.0061531, 0.0050749, 0.0061978, -0.0010149, 0.0008558
6: -0.0027524, 0.0021132, -0.0021644, 0.0022906, -0.0040270, 0.0033957
7: -0.0054348, 0.0011918, -0.0056763, 0.0003911, -0.0046247, 0.0054844
8: 0.9853855, 0.9900534, 0.9852154, 0.9894893, -0.0032577, 0.0038633
9: -0.0068584, -0.0026212, -0.0063464, -0.0024668, -0.0035069, 0.0029571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015827
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015827
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0032098, 0.0054834, 0.0032324, 0.0051805, -0.0013473, 0.0016538
1: 0.0017860, 0.0021145, 0.0017893, 0.0020707, -0.0001947, 0.0002389
2: 0.0113282, 0.0125852, 0.0114957, 0.0125727, -0.0009143, 0.0007449
3: -0.0029643, -0.0016642, -0.0027911, -0.0016772, -0.0009457, 0.0007704
4: -0.0022354, -0.0008280, -0.0022213, -0.0010155, -0.0008340, 0.0010237
5: 0.0048969, 0.0062287, 0.0050743, 0.0062155, -0.0009688, 0.0007893
6: -0.0028710, 0.0024135, -0.0021668, 0.0023609, -0.0038439, 0.0031316
7: -0.0058437, 0.0013533, -0.0057721, 0.0003943, -0.0042649, 0.0052351
8: 0.9850973, 0.9901671, 0.9851479, 0.9894916, -0.0030043, 0.0036877
9: -0.0069617, -0.0023597, -0.0063485, -0.0024055, -0.0033474, 0.0027271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017613, upper bound: 0.0015308
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017613, upper bound: 0.0015308
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0032605, 0.0055373, 0.0032627, 0.0051794, -0.0013705, 0.0016930
1: 0.0017933, 0.0021223, 0.0017937, 0.0020706, -0.0001980, 0.0002446
2: 0.0112984, 0.0125572, 0.0114963, 0.0125560, -0.0009360, 0.0007577
3: -0.0029951, -0.0016932, -0.0027905, -0.0016944, -0.0009681, 0.0007837
4: -0.0022040, -0.0007946, -0.0022026, -0.0010161, -0.0008484, 0.0010480
5: 0.0048653, 0.0061991, 0.0050749, 0.0061978, -0.0009917, 0.0008028
6: -0.0029961, 0.0022957, -0.0021644, 0.0022906, -0.0039349, 0.0031855
7: -0.0056833, 0.0015238, -0.0056763, 0.0003911, -0.0043383, 0.0053590
8: 0.9852105, 0.9902872, 0.9852154, 0.9894893, -0.0030560, 0.0037750
9: -0.0070707, -0.0024623, -0.0063464, -0.0024668, -0.0034267, 0.0027740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015308
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015308
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032683, 0.0053758, 0.0031684, 0.0054184, -0.0014538, 0.0015313
1: 0.0017945, 0.0020990, 0.0017800, 0.0021051, -0.0002100, 0.0002212
2: 0.0113877, 0.0125529, 0.0113642, 0.0126081, -0.0008466, 0.0008038
3: -0.0029028, -0.0016976, -0.0029271, -0.0016405, -0.0008756, 0.0008313
4: -0.0021992, -0.0008946, -0.0022610, -0.0008682, -0.0008999, 0.0009479
5: 0.0049599, 0.0061945, 0.0049350, 0.0062530, -0.0008970, 0.0008516
6: -0.0026209, 0.0022776, -0.0027199, 0.0025098, -0.0035591, 0.0033791
7: -0.0056587, 0.0010127, -0.0059748, 0.0011475, -0.0046020, 0.0048472
8: 0.9852278, 0.9899272, 0.9850051, 0.9900222, -0.0032418, 0.0034145
9: -0.0067439, -0.0024780, -0.0068301, -0.0022759, -0.0030994, 0.0029427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0016057
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0016058
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032953, 0.0053747, 0.0032168, 0.0054788, -0.0014845, 0.0015551
1: 0.0017984, 0.0020988, 0.0017870, 0.0021138, -0.0002145, 0.0002247
2: 0.0113883, 0.0125380, 0.0113308, 0.0125814, -0.0008598, 0.0008208
3: -0.0029021, -0.0017131, -0.0029616, -0.0016682, -0.0008892, 0.0008489
4: -0.0021825, -0.0008953, -0.0022310, -0.0008308, -0.0009189, 0.0009626
5: 0.0049606, 0.0061787, 0.0048996, 0.0062247, -0.0009110, 0.0008696
6: -0.0026182, 0.0022149, -0.0028601, 0.0023973, -0.0036144, 0.0034504
7: -0.0055732, 0.0010091, -0.0058217, 0.0013386, -0.0046992, 0.0049225
8: 0.9852880, 0.9899247, 0.9851130, 0.9901567, -0.0033102, 0.0034675
9: -0.0067416, -0.0025327, -0.0069523, -0.0023738, -0.0031476, 0.0030048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015834
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015834
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0031874, 0.0054847, 0.0031684, 0.0054184, -0.0013598, 0.0014832
1: 0.0017828, 0.0021147, 0.0017800, 0.0021051, -0.0001965, 0.0002143
2: 0.0113275, 0.0125976, 0.0113642, 0.0126081, -0.0008200, 0.0007518
3: -0.0029650, -0.0016514, -0.0029271, -0.0016405, -0.0008481, 0.0007775
4: -0.0022492, -0.0008272, -0.0022610, -0.0008682, -0.0008417, 0.0009181
5: 0.0048961, 0.0062419, 0.0049350, 0.0062530, -0.0008689, 0.0007966
6: -0.0028740, 0.0024655, -0.0027199, 0.0025098, -0.0034474, 0.0031606
7: -0.0059146, 0.0013574, -0.0059748, 0.0011475, -0.0043044, 0.0046951
8: 0.9850476, 0.9901701, 0.9850051, 0.9900222, -0.0030321, 0.0033073
9: -0.0069643, -0.0023144, -0.0068301, -0.0022759, -0.0030022, 0.0027523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015495
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015495
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032159, 0.0054836, 0.0032168, 0.0054788, -0.0014048, 0.0015056
1: 0.0017869, 0.0021145, 0.0017870, 0.0021138, -0.0002030, 0.0002175
2: 0.0113281, 0.0125819, 0.0113308, 0.0125814, -0.0008324, 0.0007767
3: -0.0029644, -0.0016677, -0.0029616, -0.0016682, -0.0008609, 0.0008033
4: -0.0022316, -0.0008278, -0.0022310, -0.0008308, -0.0008696, 0.0009320
5: 0.0048968, 0.0062252, 0.0048996, 0.0062247, -0.0008820, 0.0008229
6: -0.0028715, 0.0023993, -0.0028601, 0.0023973, -0.0034995, 0.0032651
7: -0.0058244, 0.0013540, -0.0058217, 0.0013386, -0.0044468, 0.0047660
8: 0.9851111, 0.9901676, 0.9851130, 0.9901567, -0.0031324, 0.0033573
9: -0.0069621, -0.0023721, -0.0069523, -0.0023738, -0.0030475, 0.0028434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015314
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015314
time: 0.88 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.41 seconds
IS_A1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015438, upper bound: 0.0016763
IS_A1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015438, upper bound: 0.0017017
IS_A1_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015438, upper bound: 0.0016665
IS_A1_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015438, upper bound: 0.0016925
IS_A1_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015826, upper bound: 0.0016568
IS_A1_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015826, upper bound: 0.0016729
IS_A1_B1_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015828, upper bound: 0.0016451
IS_A1_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015828, upper bound: 0.0016644
IS_A1_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015577, upper bound: 0.0016763
IS_A1_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015577, upper bound: 0.0016763
IS_A1_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015578, upper bound: 0.0016665
IS_A1_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015578, upper bound: 0.0016665
IS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015973, upper bound: 0.0016568
IS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015973, upper bound: 0.0016568
IS_A1_B1_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015974, upper bound: 0.0016451
IS_A1_B1_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015974, upper bound: 0.0016451
IS_A1_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017322
IS_A1_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017613
IS_A1_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017219
IS_A1_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015322, upper bound: 0.0017499
IS_A1_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015826, upper bound: 0.0016568
IS_A1_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015826, upper bound: 0.0016729
IS_A1_B2_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015828, upper bound: 0.0016451
IS_A1_B2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015828, upper bound: 0.0016644
IS_A1_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015433, upper bound: 0.0017319
IS_A1_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015433, upper bound: 0.0017319
IS_A1_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0017217
IS_A1_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0017216
IS_A1_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015973, upper bound: 0.0016568
IS_A1_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015973, upper bound: 0.0016568
IS_A1_B2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015974, upper bound: 0.0016451
IS_A1_B2_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0015974, upper bound: 0.0016451
IS_A2_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016766, upper bound: 0.0015419
IS_A2_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016766, upper bound: 0.0015577
IS_A2_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016664, upper bound: 0.0015419
IS_A2_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016664, upper bound: 0.0015578
IS_A2_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017319, upper bound: 0.0015308
IS_A2_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017319, upper bound: 0.0015433
IS_A2_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015308
IS_A2_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015454
IS_A2_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016663, upper bound: 0.0015610
IS_A2_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016663, upper bound: 0.0015759
IS_A2_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016664, upper bound: 0.0015420
IS_A2_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016664, upper bound: 0.0015579
IS_A2_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015495
IS_A2_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015627
IS_A2_B1_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015314
IS_A2_B1_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017217, upper bound: 0.0015458
IS_A2_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015419
IS_A2_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017017, upper bound: 0.0015419
IS_A2_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016925, upper bound: 0.0015419
IS_A2_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016925, upper bound: 0.0015419
IS_A2_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017613, upper bound: 0.0015308
IS_A2_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017613, upper bound: 0.0015307
IS_A2_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015308
IS_A2_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015308
IS_A2_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016923, upper bound: 0.0015611
IS_A2_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016923, upper bound: 0.0015610
IS_A2_B1_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016925, upper bound: 0.0015420
IS_A2_B1_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016925, upper bound: 0.0015420
IS_A2_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015495
IS_A2_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015495
IS_A2_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015314
IS_A2_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015314
IS_A2_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016592, upper bound: 0.0015825
IS_A2_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016592, upper bound: 0.0015973
IS_A2_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017318, upper bound: 0.0015308
IS_A2_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017318, upper bound: 0.0015433
IS_A2_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016728, upper bound: 0.0015825
IS_A2_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016728, upper bound: 0.0015825
IS_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015827
IS_A2_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015827
IS_A2_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017613, upper bound: 0.0015308
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017613, upper bound: 0.0015308
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015308
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015308
IS_A2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0016057
IS_A2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0016058
IS_A2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015834
IS_A2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0016644, upper bound: 0.0015834
IS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015495
IS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015495
IS_A2_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015314
IS_A2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.41
Output dim: 8, lower bound: -0.0017499, upper bound: 0.0015314

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033496, 0.0050810, 0.0033029, 0.0053079, -0.0014502, 0.0012760
1: 0.0018062, 0.0020564, 0.0017995, 0.0020891, -0.0002095, 0.0001843
2: 0.0115507, 0.0125079, 0.0114252, 0.0125337, -0.0007055, 0.0008018
3: -0.0027342, -0.0017442, -0.0028639, -0.0017175, -0.0007296, 0.0008293
4: -0.0021488, -0.0010771, -0.0021777, -0.0009366, -0.0008977, 0.0007899
5: 0.0051326, 0.0061469, 0.0049997, 0.0061742, -0.0007475, 0.0008495
6: -0.0019356, 0.0020886, -0.0024631, 0.0021970, -0.0029658, 0.0033707
7: -0.0054012, 0.0000795, -0.0055489, 0.0007978, -0.0045907, 0.0040391
8: 0.9854091, 0.9892699, 0.9853051, 0.9897758, -0.0032338, 0.0028453
9: -0.0061472, -0.0026427, -0.0066065, -0.0025482, -0.0025827, 0.0029354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014841, upper bound: 0.0016275
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016275
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033496, 0.0050810, 0.0032911, 0.0053743, -0.0015291, 0.0012885
1: 0.0018062, 0.0020564, 0.0017978, 0.0020987, -0.0002209, 0.0001862
2: 0.0115507, 0.0125079, 0.0113885, 0.0125403, -0.0007124, 0.0008454
3: -0.0027342, -0.0017442, -0.0029019, -0.0017107, -0.0007368, 0.0008744
4: -0.0021488, -0.0010771, -0.0021850, -0.0008955, -0.0009466, 0.0007976
5: 0.0051326, 0.0061469, 0.0049608, 0.0061811, -0.0007548, 0.0008958
6: -0.0019356, 0.0020886, -0.0026174, 0.0022245, -0.0029949, 0.0035541
7: -0.0054012, 0.0000795, -0.0055863, 0.0010080, -0.0048404, 0.0040788
8: 0.9854091, 0.9892699, 0.9852787, 0.9899240, -0.0034097, 0.0028732
9: -0.0061472, -0.0026427, -0.0067409, -0.0025243, -0.0026081, 0.0030951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014841, upper bound: 0.0016490
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016490
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033776, 0.0050800, 0.0033502, 0.0053675, -0.0014905, 0.0013009
1: 0.0018103, 0.0020562, 0.0018063, 0.0020977, -0.0002153, 0.0001879
2: 0.0115512, 0.0124925, 0.0113923, 0.0125076, -0.0007193, 0.0008240
3: -0.0027336, -0.0017601, -0.0028980, -0.0017445, -0.0007439, 0.0008523
4: -0.0021315, -0.0010777, -0.0021484, -0.0008997, -0.0009226, 0.0008053
5: 0.0051332, 0.0061305, 0.0049648, 0.0061465, -0.0007621, 0.0008731
6: -0.0019334, 0.0020236, -0.0026015, 0.0020871, -0.0030237, 0.0034642
7: -0.0053127, 0.0000764, -0.0053992, 0.0009863, -0.0047180, 0.0041181
8: 0.9854715, 0.9892677, 0.9854105, 0.9899086, -0.0033235, 0.0029008
9: -0.0061452, -0.0026993, -0.0067270, -0.0026440, -0.0026332, 0.0030168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014842, upper bound: 0.0016184
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016184
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033776, 0.0050800, 0.0033390, 0.0054320, -0.0015661, 0.0013142
1: 0.0018103, 0.0020562, 0.0018047, 0.0021071, -0.0002263, 0.0001899
2: 0.0115512, 0.0124925, 0.0113566, 0.0125138, -0.0007266, 0.0008659
3: -0.0027336, -0.0017601, -0.0029349, -0.0017381, -0.0007514, 0.0008955
4: -0.0021315, -0.0010777, -0.0021554, -0.0008597, -0.0009695, 0.0008135
5: 0.0051332, 0.0061305, 0.0049270, 0.0061531, -0.0007698, 0.0009174
6: -0.0019334, 0.0020236, -0.0027516, 0.0021132, -0.0030545, 0.0036401
7: -0.0053127, 0.0000764, -0.0054348, 0.0011907, -0.0049575, 0.0041599
8: 0.9854715, 0.9892677, 0.9853855, 0.9900526, -0.0034922, 0.0029303
9: -0.0061452, -0.0026993, -0.0068577, -0.0026212, -0.0026600, 0.0031700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014842, upper bound: 0.0016421
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016421
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0032816, 0.0051768, 0.0033029, 0.0053079, -0.0015493, 0.0014172
1: 0.0017964, 0.0020702, 0.0017995, 0.0020891, -0.0002238, 0.0002047
2: 0.0114978, 0.0125455, 0.0114252, 0.0125337, -0.0007836, 0.0008566
3: -0.0027889, -0.0017053, -0.0028639, -0.0017175, -0.0008104, 0.0008859
4: -0.0021909, -0.0010178, -0.0021777, -0.0009366, -0.0009591, 0.0008773
5: 0.0050765, 0.0061867, 0.0049997, 0.0061742, -0.0008302, 0.0009076
6: -0.0021582, 0.0022466, -0.0024631, 0.0021970, -0.0032941, 0.0036011
7: -0.0056164, 0.0003826, -0.0055489, 0.0007978, -0.0049044, 0.0044862
8: 0.9852576, 0.9894834, 0.9853051, 0.9897758, -0.0034547, 0.0031602
9: -0.0063410, -0.0025051, -0.0066065, -0.0025482, -0.0028686, 0.0031360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015257, upper bound: 0.0016067
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0016067
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0032816, 0.0051768, 0.0032911, 0.0053743, -0.0016282, 0.0014298
1: 0.0017964, 0.0020702, 0.0017978, 0.0020987, -0.0002352, 0.0002066
2: 0.0114978, 0.0125455, 0.0113885, 0.0125403, -0.0007905, 0.0009002
3: -0.0027889, -0.0017053, -0.0029019, -0.0017107, -0.0008176, 0.0009310
4: -0.0021909, -0.0010178, -0.0021850, -0.0008955, -0.0010079, 0.0008851
5: 0.0050765, 0.0061867, 0.0049608, 0.0061811, -0.0008376, 0.0009538
6: -0.0021582, 0.0022466, -0.0026174, 0.0022245, -0.0033232, 0.0037845
7: -0.0056164, 0.0003826, -0.0055863, 0.0010080, -0.0051541, 0.0045259
8: 0.9852576, 0.9894834, 0.9852787, 0.9899240, -0.0036307, 0.0031881
9: -0.0063410, -0.0025051, -0.0067409, -0.0025243, -0.0028940, 0.0032957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015257, upper bound: 0.0016202
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0016202
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033118, 0.0051758, 0.0033390, 0.0054320, -0.0016637, 0.0014555
1: 0.0018008, 0.0020701, 0.0018047, 0.0021071, -0.0002404, 0.0002103
2: 0.0114983, 0.0125288, 0.0113566, 0.0125138, -0.0008047, 0.0009198
3: -0.0027884, -0.0017226, -0.0029349, -0.0017381, -0.0008323, 0.0009513
4: -0.0021722, -0.0010184, -0.0021554, -0.0008597, -0.0010298, 0.0009010
5: 0.0050771, 0.0061690, 0.0049270, 0.0061531, -0.0008526, 0.0009746
6: -0.0021560, 0.0021764, -0.0027516, 0.0021132, -0.0033831, 0.0038668
7: -0.0055207, 0.0003795, -0.0054348, 0.0011907, -0.0052663, 0.0046074
8: 0.9853249, 0.9894812, 0.9853855, 0.9900526, -0.0037097, 0.0032456
9: -0.0063390, -0.0025662, -0.0068577, -0.0026212, -0.0029461, 0.0033674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015262, upper bound: 0.0016122
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015317, upper bound: 0.0016122
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033455, 0.0051409, 0.0033029, 0.0053079, -0.0014910, 0.0013836
1: 0.0018056, 0.0020650, 0.0017995, 0.0020891, -0.0002154, 0.0001999
2: 0.0115176, 0.0125102, 0.0114252, 0.0125337, -0.0007649, 0.0008243
3: -0.0027684, -0.0017418, -0.0028639, -0.0017175, -0.0007911, 0.0008526
4: -0.0021513, -0.0010400, -0.0021777, -0.0009366, -0.0009230, 0.0008564
5: 0.0050975, 0.0061492, 0.0049997, 0.0061742, -0.0008105, 0.0008734
6: -0.0020748, 0.0020981, -0.0024631, 0.0021970, -0.0032158, 0.0034655
7: -0.0054141, 0.0002690, -0.0055489, 0.0007978, -0.0047197, 0.0043796
8: 0.9854000, 0.9894034, 0.9853051, 0.9897758, -0.0033247, 0.0030851
9: -0.0062684, -0.0026344, -0.0066065, -0.0025482, -0.0028004, 0.0030179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014937, upper bound: 0.0016275
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015033, upper bound: 0.0016274
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033455, 0.0051409, 0.0032911, 0.0053743, -0.0014682, 0.0012952
1: 0.0018056, 0.0020650, 0.0017978, 0.0020987, -0.0002121, 0.0001871
2: 0.0115176, 0.0125102, 0.0113885, 0.0125403, -0.0007161, 0.0008117
3: -0.0027684, -0.0017418, -0.0029019, -0.0017107, -0.0007406, 0.0008395
4: -0.0021513, -0.0010400, -0.0021850, -0.0008955, -0.0009089, 0.0008017
5: 0.0050975, 0.0061492, 0.0049608, 0.0061811, -0.0007587, 0.0008601
6: -0.0020748, 0.0020981, -0.0026174, 0.0022245, -0.0030104, 0.0034126
7: -0.0054141, 0.0002690, -0.0055863, 0.0010080, -0.0046476, 0.0040999
8: 0.9854000, 0.9894034, 0.9852787, 0.9899240, -0.0032739, 0.0028880
9: -0.0062684, -0.0026344, -0.0067409, -0.0025243, -0.0026216, 0.0029718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014937, upper bound: 0.0016274
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015033, upper bound: 0.0016275
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033729, 0.0051398, 0.0033502, 0.0053675, -0.0015342, 0.0014071
1: 0.0018096, 0.0020649, 0.0018063, 0.0020977, -0.0002217, 0.0002033
2: 0.0115182, 0.0124951, 0.0113923, 0.0125076, -0.0007779, 0.0008482
3: -0.0027678, -0.0017575, -0.0028980, -0.0017445, -0.0008046, 0.0008773
4: -0.0021344, -0.0010406, -0.0021484, -0.0008997, -0.0009497, 0.0008710
5: 0.0050981, 0.0061332, 0.0049648, 0.0061465, -0.0008243, 0.0008987
6: -0.0020724, 0.0020344, -0.0026015, 0.0020871, -0.0032704, 0.0035659
7: -0.0053274, 0.0002657, -0.0053992, 0.0009863, -0.0048565, 0.0044540
8: 0.9854611, 0.9894011, 0.9854105, 0.9899086, -0.0034210, 0.0031375
9: -0.0062663, -0.0026899, -0.0067270, -0.0026440, -0.0028480, 0.0031054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014936, upper bound: 0.0016183
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015034, upper bound: 0.0016183
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033729, 0.0051398, 0.0033390, 0.0054320, -0.0015093, 0.0013213
1: 0.0018096, 0.0020649, 0.0018047, 0.0021071, -0.0002181, 0.0001909
2: 0.0115182, 0.0124951, 0.0113566, 0.0125138, -0.0007305, 0.0008345
3: -0.0027678, -0.0017575, -0.0029349, -0.0017381, -0.0007555, 0.0008630
4: -0.0021344, -0.0010406, -0.0021554, -0.0008597, -0.0009343, 0.0008179
5: 0.0050981, 0.0061332, 0.0049270, 0.0061531, -0.0007740, 0.0008842
6: -0.0020724, 0.0020344, -0.0027516, 0.0021132, -0.0030710, 0.0035081
7: -0.0053274, 0.0002657, -0.0054348, 0.0011907, -0.0047777, 0.0041824
8: 0.9854611, 0.9894011, 0.9853855, 0.9900526, -0.0033655, 0.0029462
9: -0.0062663, -0.0026899, -0.0068577, -0.0026212, -0.0026743, 0.0030550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014936, upper bound: 0.0016183
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015034, upper bound: 0.0016183
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0032731, 0.0052395, 0.0033029, 0.0053079, -0.0015846, 0.0015067
1: 0.0017952, 0.0020793, 0.0017995, 0.0020891, -0.0002289, 0.0002177
2: 0.0114631, 0.0125503, 0.0114252, 0.0125337, -0.0008330, 0.0008761
3: -0.0028248, -0.0017004, -0.0028639, -0.0017175, -0.0008615, 0.0009061
4: -0.0021962, -0.0009790, -0.0021777, -0.0009366, -0.0009809, 0.0009327
5: 0.0050398, 0.0061917, 0.0049997, 0.0061742, -0.0008826, 0.0009283
6: -0.0023040, 0.0022665, -0.0024631, 0.0021970, -0.0035020, 0.0036832
7: -0.0056434, 0.0005811, -0.0055489, 0.0007978, -0.0050161, 0.0047694
8: 0.9852385, 0.9896232, 0.9853051, 0.9897758, -0.0035335, 0.0033596
9: -0.0064679, -0.0024878, -0.0066065, -0.0025482, -0.0030497, 0.0032075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015387, upper bound: 0.0016067
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016067
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0032731, 0.0052395, 0.0032911, 0.0053743, -0.0015676, 0.0014352
1: 0.0017952, 0.0020793, 0.0017978, 0.0020987, -0.0002265, 0.0002073
2: 0.0114631, 0.0125503, 0.0113885, 0.0125403, -0.0007935, 0.0008667
3: -0.0028248, -0.0017004, -0.0029019, -0.0017107, -0.0008207, 0.0008964
4: -0.0021962, -0.0009790, -0.0021850, -0.0008955, -0.0009704, 0.0008884
5: 0.0050398, 0.0061917, 0.0049608, 0.0061811, -0.0008407, 0.0009183
6: -0.0023040, 0.0022665, -0.0026174, 0.0022245, -0.0033358, 0.0036436
7: -0.0056434, 0.0005811, -0.0055863, 0.0010080, -0.0049623, 0.0045431
8: 0.9852385, 0.9896232, 0.9852787, 0.9899240, -0.0034955, 0.0032002
9: -0.0064679, -0.0024878, -0.0067409, -0.0025243, -0.0029050, 0.0031730

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015387, upper bound: 0.0016067
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016067
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033496, 0.0050810, 0.0032246, 0.0054151, -0.0015997, 0.0013884
1: 0.0018062, 0.0020564, 0.0017882, 0.0021046, -0.0002311, 0.0002006
2: 0.0115507, 0.0125079, 0.0113660, 0.0125771, -0.0007676, 0.0008844
3: -0.0027342, -0.0017442, -0.0029252, -0.0016727, -0.0007939, 0.0009147
4: -0.0021488, -0.0010771, -0.0022262, -0.0008702, -0.0009902, 0.0008594
5: 0.0051326, 0.0061469, 0.0049369, 0.0062201, -0.0008133, 0.0009371
6: -0.0019356, 0.0020886, -0.0027123, 0.0023792, -0.0032269, 0.0037181
7: -0.0054012, 0.0000795, -0.0057970, 0.0011372, -0.0050637, 0.0043948
8: 0.9854091, 0.9892699, 0.9851304, 0.9900149, -0.0035670, 0.0030958
9: -0.0061472, -0.0026427, -0.0068235, -0.0023896, -0.0028101, 0.0032379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014703, upper bound: 0.0016846
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014801, upper bound: 0.0016846
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033496, 0.0050810, 0.0032098, 0.0054834, -0.0016780, 0.0014060
1: 0.0018062, 0.0020564, 0.0017860, 0.0021145, -0.0002424, 0.0002031
2: 0.0115507, 0.0125079, 0.0113282, 0.0125852, -0.0007773, 0.0009277
3: -0.0027342, -0.0017442, -0.0029643, -0.0016642, -0.0008040, 0.0009595
4: -0.0021488, -0.0010771, -0.0022354, -0.0008280, -0.0010387, 0.0008703
5: 0.0051326, 0.0061469, 0.0048969, 0.0062287, -0.0008236, 0.0009830
6: -0.0019356, 0.0020886, -0.0028710, 0.0024135, -0.0032679, 0.0039002
7: -0.0054012, 0.0000795, -0.0058437, 0.0013533, -0.0053118, 0.0044506
8: 0.9854091, 0.9892699, 0.9850973, 0.9901671, -0.0037417, 0.0031351
9: -0.0061472, -0.0026427, -0.0069617, -0.0023597, -0.0028458, 0.0033965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014703, upper bound: 0.0017101
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014801, upper bound: 0.0017101
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033776, 0.0050800, 0.0032707, 0.0054757, -0.0016258, 0.0014097
1: 0.0018103, 0.0020562, 0.0017948, 0.0021134, -0.0002349, 0.0002037
2: 0.0115512, 0.0124925, 0.0113325, 0.0125516, -0.0007794, 0.0008989
3: -0.0027336, -0.0017601, -0.0029599, -0.0016991, -0.0008061, 0.0009296
4: -0.0021315, -0.0010777, -0.0021976, -0.0008327, -0.0010064, 0.0008726
5: 0.0051332, 0.0061305, 0.0049014, 0.0061931, -0.0008258, 0.0009524
6: -0.0019334, 0.0020236, -0.0028531, 0.0022719, -0.0032765, 0.0037788
7: -0.0053127, 0.0000764, -0.0056508, 0.0013290, -0.0051464, 0.0044623
8: 0.9854715, 0.9892677, 0.9852333, 0.9901500, -0.0036252, 0.0031433
9: -0.0061452, -0.0026993, -0.0069462, -0.0024831, -0.0028533, 0.0032907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014704, upper bound: 0.0016759
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014802, upper bound: 0.0016759
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033776, 0.0050800, 0.0032605, 0.0055373, -0.0017045, 0.0014256
1: 0.0018103, 0.0020562, 0.0017933, 0.0021223, -0.0002463, 0.0002060
2: 0.0115512, 0.0124925, 0.0112984, 0.0125572, -0.0007882, 0.0009424
3: -0.0027336, -0.0017601, -0.0029951, -0.0016932, -0.0008152, 0.0009747
4: -0.0021315, -0.0010777, -0.0022040, -0.0007946, -0.0010551, 0.0008825
5: 0.0051332, 0.0061305, 0.0048653, 0.0061991, -0.0008351, 0.0009985
6: -0.0019334, 0.0020236, -0.0029961, 0.0022957, -0.0033136, 0.0039618
7: -0.0053127, 0.0000764, -0.0056833, 0.0015238, -0.0053956, 0.0045128
8: 0.9854715, 0.9892677, 0.9852105, 0.9902872, -0.0038008, 0.0031789
9: -0.0061452, -0.0026993, -0.0070707, -0.0024623, -0.0028856, 0.0034501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014704, upper bound: 0.0017019
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014802, upper bound: 0.0017019
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0032816, 0.0051768, 0.0032246, 0.0054151, -0.0015039, 0.0013281
1: 0.0017964, 0.0020702, 0.0017882, 0.0021046, -0.0002173, 0.0001919
2: 0.0114978, 0.0125455, 0.0113660, 0.0125771, -0.0007343, 0.0008315
3: -0.0027889, -0.0017053, -0.0029252, -0.0016727, -0.0007594, 0.0008599
4: -0.0021909, -0.0010178, -0.0022262, -0.0008702, -0.0009309, 0.0008221
5: 0.0050765, 0.0061867, 0.0049369, 0.0062201, -0.0007780, 0.0008810
6: -0.0021582, 0.0022466, -0.0027123, 0.0023792, -0.0030868, 0.0034955
7: -0.0056164, 0.0003826, -0.0057970, 0.0011372, -0.0047606, 0.0042040
8: 0.9852576, 0.9894834, 0.9851304, 0.9900149, -0.0033534, 0.0029614
9: -0.0063410, -0.0025051, -0.0068235, -0.0023896, -0.0026882, 0.0030440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015257, upper bound: 0.0016067
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0016067
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0032816, 0.0051768, 0.0032098, 0.0054834, -0.0015831, 0.0013413
1: 0.0017964, 0.0020702, 0.0017860, 0.0021145, -0.0002287, 0.0001938
2: 0.0114978, 0.0125455, 0.0113282, 0.0125852, -0.0007416, 0.0008753
3: -0.0027889, -0.0017053, -0.0029643, -0.0016642, -0.0007670, 0.0009053
4: -0.0021909, -0.0010178, -0.0022354, -0.0008280, -0.0009800, 0.0008303
5: 0.0050765, 0.0061867, 0.0048969, 0.0062287, -0.0007857, 0.0009274
6: -0.0021582, 0.0022466, -0.0028710, 0.0024135, -0.0031176, 0.0036797
7: -0.0056164, 0.0003826, -0.0058437, 0.0013533, -0.0050114, 0.0042459
8: 0.9852576, 0.9894834, 0.9850973, 0.9901671, -0.0035301, 0.0029909
9: -0.0063410, -0.0025051, -0.0069617, -0.0023597, -0.0027149, 0.0032044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015257, upper bound: 0.0016202
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0016201
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033118, 0.0051758, 0.0032605, 0.0055373, -0.0016174, 0.0013643
1: 0.0018008, 0.0020701, 0.0017933, 0.0021223, -0.0002337, 0.0001971
2: 0.0114983, 0.0125288, 0.0112984, 0.0125572, -0.0007543, 0.0008942
3: -0.0027884, -0.0017226, -0.0029951, -0.0016932, -0.0007801, 0.0009248
4: -0.0021722, -0.0010184, -0.0022040, -0.0007946, -0.0010012, 0.0008446
5: 0.0050771, 0.0061690, 0.0048653, 0.0061991, -0.0007992, 0.0009475
6: -0.0021560, 0.0021764, -0.0029961, 0.0022957, -0.0031711, 0.0037592
7: -0.0055207, 0.0003795, -0.0056833, 0.0015238, -0.0051198, 0.0043188
8: 0.9853249, 0.9894812, 0.9852105, 0.9902872, -0.0036065, 0.0030422
9: -0.0063390, -0.0025662, -0.0070707, -0.0024623, -0.0027616, 0.0032737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015262, upper bound: 0.0016122
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015317, upper bound: 0.0016122
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0033455, 0.0051409, 0.0032246, 0.0054151, -0.0016404, 0.0014959
1: 0.0018056, 0.0020650, 0.0017882, 0.0021046, -0.0002370, 0.0002161
2: 0.0115176, 0.0125102, 0.0113660, 0.0125771, -0.0008270, 0.0009070
3: -0.0027684, -0.0017418, -0.0029252, -0.0016727, -0.0008554, 0.0009380
4: -0.0021513, -0.0010400, -0.0022262, -0.0008702, -0.0010155, 0.0009260
5: 0.0050975, 0.0061492, 0.0049369, 0.0062201, -0.0008763, 0.0009610
6: -0.0020748, 0.0020981, -0.0027123, 0.0023792, -0.0034769, 0.0038128
7: -0.0054141, 0.0002690, -0.0057970, 0.0011372, -0.0051927, 0.0047352
8: 0.9854000, 0.9894034, 0.9851304, 0.9900149, -0.0036579, 0.0033356
9: -0.0062684, -0.0026344, -0.0068235, -0.0023896, -0.0030278, 0.0033204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014798, upper bound: 0.0016846
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014911, upper bound: 0.0016846
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0033455, 0.0051409, 0.0032098, 0.0054834, -0.0016192, 0.0014069
1: 0.0018056, 0.0020650, 0.0017860, 0.0021145, -0.0002339, 0.0002033
2: 0.0115176, 0.0125102, 0.0113282, 0.0125852, -0.0007778, 0.0008952
3: -0.0027684, -0.0017418, -0.0029643, -0.0016642, -0.0008045, 0.0009259
4: -0.0021513, -0.0010400, -0.0022354, -0.0008280, -0.0010023, 0.0008709
5: 0.0050975, 0.0061492, 0.0048969, 0.0062287, -0.0008242, 0.0009485
6: -0.0020748, 0.0020981, -0.0028710, 0.0024135, -0.0032700, 0.0037635
7: -0.0054141, 0.0002690, -0.0058437, 0.0013533, -0.0051256, 0.0044535
8: 0.9854000, 0.9894034, 0.9850973, 0.9901671, -0.0036106, 0.0031371
9: -0.0062684, -0.0026344, -0.0069617, -0.0023597, -0.0028477, 0.0032774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014798, upper bound: 0.0016846
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014911, upper bound: 0.0016845
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033729, 0.0051398, 0.0032707, 0.0054757, -0.0016695, 0.0015158
1: 0.0018096, 0.0020649, 0.0017948, 0.0021134, -0.0002412, 0.0002190
2: 0.0115182, 0.0124951, 0.0113325, 0.0125516, -0.0008380, 0.0009230
3: -0.0027678, -0.0017575, -0.0029599, -0.0016991, -0.0008667, 0.0009547
4: -0.0021344, -0.0010406, -0.0021976, -0.0008327, -0.0010335, 0.0009383
5: 0.0050981, 0.0061332, 0.0049014, 0.0061931, -0.0008880, 0.0009780
6: -0.0020724, 0.0020344, -0.0028531, 0.0022719, -0.0035231, 0.0038805
7: -0.0053274, 0.0002657, -0.0056508, 0.0013290, -0.0052849, 0.0047982
8: 0.9854611, 0.9894011, 0.9852333, 0.9901500, -0.0037228, 0.0033800
9: -0.0062663, -0.0026899, -0.0069462, -0.0024831, -0.0030681, 0.0033793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014803, upper bound: 0.0016759
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014927, upper bound: 0.0016759
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033729, 0.0051398, 0.0032605, 0.0055373, -0.0016441, 0.0014294
1: 0.0018096, 0.0020649, 0.0017933, 0.0021223, -0.0002375, 0.0002065
2: 0.0115182, 0.0124951, 0.0112984, 0.0125572, -0.0007903, 0.0009090
3: -0.0027678, -0.0017575, -0.0029951, -0.0016932, -0.0008173, 0.0009401
4: -0.0021344, -0.0010406, -0.0022040, -0.0007946, -0.0010177, 0.0008848
5: 0.0050981, 0.0061332, 0.0048653, 0.0061991, -0.0008373, 0.0009631
6: -0.0020724, 0.0020344, -0.0029961, 0.0022957, -0.0033223, 0.0038214
7: -0.0053274, 0.0002657, -0.0056833, 0.0015238, -0.0052044, 0.0045247
8: 0.9854611, 0.9894011, 0.9852105, 0.9902872, -0.0036661, 0.0031873
9: -0.0062663, -0.0026899, -0.0070707, -0.0024623, -0.0028932, 0.0033278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014803, upper bound: 0.0016759
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014928, upper bound: 0.0016758
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0032731, 0.0052395, 0.0032246, 0.0054151, -0.0015449, 0.0014343
1: 0.0017952, 0.0020793, 0.0017882, 0.0021046, -0.0002232, 0.0002072
2: 0.0114631, 0.0125503, 0.0113660, 0.0125771, -0.0007930, 0.0008541
3: -0.0028248, -0.0017004, -0.0029252, -0.0016727, -0.0008201, 0.0008834
4: -0.0021962, -0.0009790, -0.0022262, -0.0008702, -0.0009563, 0.0008879
5: 0.0050398, 0.0061917, 0.0049369, 0.0062201, -0.0008402, 0.0009050
6: -0.0023040, 0.0022665, -0.0027123, 0.0023792, -0.0033337, 0.0035907
7: -0.0056434, 0.0005811, -0.0057970, 0.0011372, -0.0048902, 0.0045402
8: 0.9852385, 0.9896232, 0.9851304, 0.9900149, -0.0034448, 0.0031982
9: -0.0064679, -0.0024878, -0.0068235, -0.0023896, -0.0029031, 0.0031269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015387, upper bound: 0.0016067
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016067
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0032731, 0.0052395, 0.0032098, 0.0054834, -0.0015248, 0.0013486
1: 0.0017952, 0.0020793, 0.0017860, 0.0021145, -0.0002203, 0.0001948
2: 0.0114631, 0.0125503, 0.0113282, 0.0125852, -0.0007456, 0.0008430
3: -0.0028248, -0.0017004, -0.0029643, -0.0016642, -0.0007711, 0.0008719
4: -0.0021962, -0.0009790, -0.0022354, -0.0008280, -0.0009439, 0.0008348
5: 0.0050398, 0.0061917, 0.0048969, 0.0062287, -0.0007900, 0.0008932
6: -0.0023040, 0.0022665, -0.0028710, 0.0024135, -0.0031345, 0.0035442
7: -0.0056434, 0.0005811, -0.0058437, 0.0013533, -0.0048268, 0.0042690
8: 0.9852385, 0.9896232, 0.9850973, 0.9901671, -0.0034001, 0.0030071
9: -0.0064679, -0.0024878, -0.0069617, -0.0023597, -0.0027297, 0.0030864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015387, upper bound: 0.0016067
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016067
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033029, 0.0053079, 0.0033496, 0.0050810, -0.0012760, 0.0014502
1: 0.0017995, 0.0020891, 0.0018062, 0.0020564, -0.0001843, 0.0002095
2: 0.0114252, 0.0125337, 0.0115507, 0.0125079, -0.0008018, 0.0007055
3: -0.0028639, -0.0017175, -0.0027342, -0.0017442, -0.0008293, 0.0007296
4: -0.0021777, -0.0009366, -0.0021488, -0.0010771, -0.0007899, 0.0008977
5: 0.0049997, 0.0061742, 0.0051326, 0.0061469, -0.0008495, 0.0007475
6: -0.0024631, 0.0021970, -0.0019356, 0.0020886, -0.0033707, 0.0029658
7: -0.0055489, 0.0007978, -0.0054012, 0.0000795, -0.0040391, 0.0045907
8: 0.9853051, 0.9897758, 0.9854091, 0.9892699, -0.0028453, 0.0032338
9: -0.0066065, -0.0025482, -0.0061472, -0.0026427, -0.0029354, 0.0025827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016277, upper bound: 0.0014816
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016277, upper bound: 0.0014877
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033029, 0.0053079, 0.0033455, 0.0051409, -0.0013836, 0.0014910
1: 0.0017995, 0.0020891, 0.0018056, 0.0020650, -0.0001999, 0.0002154
2: 0.0114252, 0.0125337, 0.0115176, 0.0125102, -0.0008243, 0.0007649
3: -0.0028639, -0.0017175, -0.0027684, -0.0017418, -0.0008526, 0.0007911
4: -0.0021777, -0.0009366, -0.0021513, -0.0010400, -0.0008564, 0.0009230
5: 0.0049997, 0.0061742, 0.0050975, 0.0061492, -0.0008734, 0.0008105
6: -0.0024631, 0.0021970, -0.0020748, 0.0020981, -0.0034655, 0.0032158
7: -0.0055489, 0.0007978, -0.0054141, 0.0002690, -0.0043796, 0.0047197
8: 0.9853051, 0.9897758, 0.9854000, 0.9894034, -0.0030851, 0.0033247
9: -0.0066065, -0.0025482, -0.0062684, -0.0026344, -0.0030179, 0.0028004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016277, upper bound: 0.0014937
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016277, upper bound: 0.0015033
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033502, 0.0053675, 0.0033776, 0.0050800, -0.0013009, 0.0014905
1: 0.0018063, 0.0020977, 0.0018103, 0.0020562, -0.0001879, 0.0002153
2: 0.0113923, 0.0125076, 0.0115512, 0.0124925, -0.0008240, 0.0007193
3: -0.0028980, -0.0017445, -0.0027336, -0.0017601, -0.0008523, 0.0007439
4: -0.0021484, -0.0008997, -0.0021315, -0.0010777, -0.0008053, 0.0009226
5: 0.0049648, 0.0061465, 0.0051332, 0.0061305, -0.0008731, 0.0007621
6: -0.0026015, 0.0020871, -0.0019334, 0.0020236, -0.0034642, 0.0030237
7: -0.0053992, 0.0009863, -0.0053127, 0.0000764, -0.0041181, 0.0047180
8: 0.9854105, 0.9899086, 0.9854715, 0.9892677, -0.0029008, 0.0033235
9: -0.0067270, -0.0026440, -0.0061452, -0.0026993, -0.0030168, 0.0026332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0014817
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0014878
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033502, 0.0053675, 0.0033729, 0.0051398, -0.0014071, 0.0015342
1: 0.0018063, 0.0020977, 0.0018096, 0.0020649, -0.0002033, 0.0002216
2: 0.0113923, 0.0125076, 0.0115182, 0.0124951, -0.0008482, 0.0007779
3: -0.0028980, -0.0017445, -0.0027678, -0.0017575, -0.0008773, 0.0008046
4: -0.0021484, -0.0008997, -0.0021344, -0.0010406, -0.0008710, 0.0009497
5: 0.0049648, 0.0061465, 0.0050981, 0.0061332, -0.0008987, 0.0008243
6: -0.0026015, 0.0020871, -0.0020724, 0.0020344, -0.0035659, 0.0032704
7: -0.0053992, 0.0009863, -0.0053274, 0.0002657, -0.0044540, 0.0048565
8: 0.9854105, 0.9899086, 0.9854611, 0.9894011, -0.0031375, 0.0034210
9: -0.0067270, -0.0026440, -0.0062663, -0.0026899, -0.0031054, 0.0028480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0014937
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0015034
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032246, 0.0054151, 0.0033496, 0.0050810, -0.0013884, 0.0015997
1: 0.0017882, 0.0021046, 0.0018062, 0.0020564, -0.0002006, 0.0002311
2: 0.0113660, 0.0125771, 0.0115507, 0.0125079, -0.0008844, 0.0007676
3: -0.0029252, -0.0016727, -0.0027342, -0.0017442, -0.0009147, 0.0007939
4: -0.0022262, -0.0008702, -0.0021488, -0.0010771, -0.0008594, 0.0009902
5: 0.0049369, 0.0062201, 0.0051326, 0.0061469, -0.0009371, 0.0008133
6: -0.0027123, 0.0023792, -0.0019356, 0.0020886, -0.0037181, 0.0032269
7: -0.0057970, 0.0011372, -0.0054012, 0.0000795, -0.0043948, 0.0050637
8: 0.9851304, 0.9900149, 0.9854091, 0.9892699, -0.0030958, 0.0035670
9: -0.0068235, -0.0023896, -0.0061472, -0.0026427, -0.0032379, 0.0028101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014670
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014782
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032246, 0.0054151, 0.0033455, 0.0051409, -0.0014959, 0.0016404
1: 0.0017882, 0.0021046, 0.0018056, 0.0020650, -0.0002161, 0.0002370
2: 0.0113660, 0.0125771, 0.0115176, 0.0125102, -0.0009070, 0.0008270
3: -0.0029252, -0.0016727, -0.0027684, -0.0017418, -0.0009380, 0.0008554
4: -0.0022262, -0.0008702, -0.0021513, -0.0010400, -0.0009260, 0.0010155
5: 0.0049369, 0.0062201, 0.0050975, 0.0061492, -0.0009610, 0.0008763
6: -0.0027123, 0.0023792, -0.0020748, 0.0020981, -0.0038128, 0.0034769
7: -0.0057970, 0.0011372, -0.0054141, 0.0002690, -0.0047352, 0.0051927
8: 0.9851304, 0.9900149, 0.9854000, 0.9894034, -0.0033356, 0.0036579
9: -0.0068235, -0.0023896, -0.0062684, -0.0026344, -0.0033204, 0.0030278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014798
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014911
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032707, 0.0054757, 0.0033776, 0.0050800, -0.0014097, 0.0016258
1: 0.0017948, 0.0021134, 0.0018103, 0.0020562, -0.0002037, 0.0002349
2: 0.0113325, 0.0125516, 0.0115512, 0.0124925, -0.0008989, 0.0007794
3: -0.0029599, -0.0016991, -0.0027336, -0.0017601, -0.0009296, 0.0008061
4: -0.0021976, -0.0008327, -0.0021315, -0.0010777, -0.0008726, 0.0010064
5: 0.0049014, 0.0061931, 0.0051332, 0.0061305, -0.0009524, 0.0008258
6: -0.0028531, 0.0022719, -0.0019334, 0.0020236, -0.0037788, 0.0032765
7: -0.0056508, 0.0013290, -0.0053127, 0.0000764, -0.0044623, 0.0051464
8: 0.9852333, 0.9901500, 0.9854715, 0.9892677, -0.0031433, 0.0036252
9: -0.0069462, -0.0024831, -0.0061452, -0.0026993, -0.0032907, 0.0028533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014672
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014782
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032707, 0.0054757, 0.0033729, 0.0051398, -0.0015158, 0.0016695
1: 0.0017948, 0.0021134, 0.0018096, 0.0020649, -0.0002190, 0.0002412
2: 0.0113325, 0.0125516, 0.0115182, 0.0124951, -0.0009230, 0.0008380
3: -0.0029599, -0.0016991, -0.0027678, -0.0017575, -0.0009547, 0.0008667
4: -0.0021976, -0.0008327, -0.0021344, -0.0010406, -0.0009383, 0.0010335
5: 0.0049014, 0.0061931, 0.0050981, 0.0061332, -0.0009780, 0.0008880
6: -0.0028531, 0.0022719, -0.0020724, 0.0020344, -0.0038805, 0.0035231
7: -0.0056508, 0.0013290, -0.0053274, 0.0002657, -0.0047982, 0.0052849
8: 0.9852333, 0.9901500, 0.9854611, 0.9894011, -0.0033800, 0.0037228
9: -0.0069462, -0.0024831, -0.0062663, -0.0026899, -0.0033793, 0.0030681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014803
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014927
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0032808, 0.0053093, 0.0033029, 0.0053079, -0.0012822, 0.0012579
1: 0.0017963, 0.0020893, 0.0017995, 0.0020891, -0.0001852, 0.0001817
2: 0.0114245, 0.0125460, 0.0114252, 0.0125337, -0.0006955, 0.0007089
3: -0.0028647, -0.0017048, -0.0028639, -0.0017175, -0.0007193, 0.0007332
4: -0.0021914, -0.0009358, -0.0021777, -0.0009366, -0.0007937, 0.0007787
5: 0.0049989, 0.0061871, 0.0049997, 0.0061742, -0.0007369, 0.0007511
6: -0.0024662, 0.0022484, -0.0024631, 0.0021970, -0.0029238, 0.0029802
7: -0.0056189, 0.0008020, -0.0055489, 0.0007978, -0.0040587, 0.0039819
8: 0.9852557, 0.9897788, 0.9853051, 0.9897758, -0.0028590, 0.0028049
9: -0.0066092, -0.0025035, -0.0066065, -0.0025482, -0.0025461, 0.0025952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016087, upper bound: 0.0015079
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0015079
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0032808, 0.0053093, 0.0032911, 0.0053743, -0.0013876, 0.0012915
1: 0.0017963, 0.0020893, 0.0017978, 0.0020987, -0.0002005, 0.0001866
2: 0.0114245, 0.0125460, 0.0113885, 0.0125403, -0.0007140, 0.0007672
3: -0.0028647, -0.0017048, -0.0029019, -0.0017107, -0.0007385, 0.0007935
4: -0.0021914, -0.0009358, -0.0021850, -0.0008955, -0.0008590, 0.0007994
5: 0.0049989, 0.0061871, 0.0049608, 0.0061811, -0.0007565, 0.0008129
6: -0.0024662, 0.0022484, -0.0026174, 0.0022245, -0.0030017, 0.0032252
7: -0.0056189, 0.0008020, -0.0055863, 0.0010080, -0.0043925, 0.0040880
8: 0.9852557, 0.9897788, 0.9852787, 0.9899240, -0.0030941, 0.0028797
9: -0.0066092, -0.0025035, -0.0067409, -0.0025243, -0.0026140, 0.0028087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016087, upper bound: 0.0015229
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0015230
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0033073, 0.0053082, 0.0033502, 0.0053675, -0.0013295, 0.0012841
1: 0.0018001, 0.0020892, 0.0018063, 0.0020977, -0.0001921, 0.0001855
2: 0.0114251, 0.0125314, 0.0113923, 0.0125076, -0.0007100, 0.0007351
3: -0.0028641, -0.0017199, -0.0028980, -0.0017445, -0.0007343, 0.0007602
4: -0.0021750, -0.0009364, -0.0021484, -0.0008997, -0.0008230, 0.0007949
5: 0.0049995, 0.0061717, 0.0049648, 0.0061465, -0.0007522, 0.0007788
6: -0.0024638, 0.0021870, -0.0026015, 0.0020871, -0.0029847, 0.0030902
7: -0.0055352, 0.0007987, -0.0053992, 0.0009863, -0.0042086, 0.0040648
8: 0.9853148, 0.9897765, 0.9854105, 0.9899086, -0.0029646, 0.0028634
9: -0.0066071, -0.0025570, -0.0067270, -0.0026440, -0.0025992, 0.0026911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016088, upper bound: 0.0014891
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0014891
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0033073, 0.0053082, 0.0033390, 0.0054320, -0.0014392, 0.0013166
1: 0.0018001, 0.0020892, 0.0018047, 0.0021071, -0.0002079, 0.0001902
2: 0.0114251, 0.0125314, 0.0113566, 0.0125138, -0.0007279, 0.0007957
3: -0.0028641, -0.0017199, -0.0029349, -0.0017381, -0.0007528, 0.0008229
4: -0.0021750, -0.0009364, -0.0021554, -0.0008597, -0.0008909, 0.0008150
5: 0.0049995, 0.0061717, 0.0049270, 0.0061531, -0.0007712, 0.0008431
6: -0.0024638, 0.0021870, -0.0027516, 0.0021132, -0.0030601, 0.0033450
7: -0.0055352, 0.0007987, -0.0054348, 0.0011907, -0.0045556, 0.0041676
8: 0.9853148, 0.9897765, 0.9853855, 0.9900526, -0.0032091, 0.0029357
9: -0.0066071, -0.0025570, -0.0068577, -0.0026212, -0.0026648, 0.0029130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016088, upper bound: 0.0015043
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0015043
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0032012, 0.0054164, 0.0033029, 0.0053079, -0.0013845, 0.0014074
1: 0.0017848, 0.0021048, 0.0017995, 0.0020891, -0.0002000, 0.0002033
2: 0.0113653, 0.0125900, 0.0114252, 0.0125337, -0.0007781, 0.0007655
3: -0.0029260, -0.0016593, -0.0028639, -0.0017175, -0.0008048, 0.0007917
4: -0.0022407, -0.0008694, -0.0021777, -0.0009366, -0.0008570, 0.0008712
5: 0.0049361, 0.0062338, 0.0049997, 0.0061742, -0.0008244, 0.0008111
6: -0.0027152, 0.0024336, -0.0024631, 0.0021970, -0.0032712, 0.0032180
7: -0.0058711, 0.0011412, -0.0055489, 0.0007978, -0.0043827, 0.0044550
8: 0.9850782, 0.9900177, 0.9853051, 0.9897758, -0.0030872, 0.0031382
9: -0.0068261, -0.0023422, -0.0066065, -0.0025482, -0.0028487, 0.0028024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014920
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014984
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0032012, 0.0054164, 0.0032911, 0.0053743, -0.0014900, 0.0014409
1: 0.0017848, 0.0021048, 0.0017978, 0.0020987, -0.0002153, 0.0002082
2: 0.0113653, 0.0125900, 0.0113885, 0.0125403, -0.0007966, 0.0008238
3: -0.0029260, -0.0016593, -0.0029019, -0.0017107, -0.0008239, 0.0008520
4: -0.0022407, -0.0008694, -0.0021850, -0.0008955, -0.0009223, 0.0008920
5: 0.0049361, 0.0062338, 0.0049608, 0.0061811, -0.0008441, 0.0008728
6: -0.0027152, 0.0024336, -0.0026174, 0.0022245, -0.0033491, 0.0034631
7: -0.0058711, 0.0011412, -0.0055863, 0.0010080, -0.0047164, 0.0045612
8: 0.9850782, 0.9900177, 0.9852787, 0.9899240, -0.0033223, 0.0032130
9: -0.0068261, -0.0023422, -0.0067409, -0.0025243, -0.0029165, 0.0030158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0015029
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0015111
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0032280, 0.0054154, 0.0033502, 0.0053675, -0.0014279, 0.0014350
1: 0.0017887, 0.0021047, 0.0018063, 0.0020977, -0.0002063, 0.0002073
2: 0.0113658, 0.0125752, 0.0113923, 0.0125076, -0.0007934, 0.0007895
3: -0.0029254, -0.0016746, -0.0028980, -0.0017445, -0.0008206, 0.0008165
4: -0.0022241, -0.0008701, -0.0021484, -0.0008997, -0.0008839, 0.0008883
5: 0.0049367, 0.0062181, 0.0049648, 0.0061465, -0.0008406, 0.0008365
6: -0.0027128, 0.0023712, -0.0026015, 0.0020871, -0.0033354, 0.0033189
7: -0.0057860, 0.0011380, -0.0053992, 0.0009863, -0.0045200, 0.0045425
8: 0.9851381, 0.9900154, 0.9854105, 0.9899086, -0.0031840, 0.0031998
9: -0.0068240, -0.0023966, -0.0067270, -0.0026440, -0.0029046, 0.0028902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016665, upper bound: 0.0014806
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014806
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0032280, 0.0054154, 0.0033390, 0.0054320, -0.0015376, 0.0014675
1: 0.0017887, 0.0021047, 0.0018047, 0.0021071, -0.0002221, 0.0002120
2: 0.0113658, 0.0125752, 0.0113566, 0.0125138, -0.0008113, 0.0008501
3: -0.0029254, -0.0016746, -0.0029349, -0.0017381, -0.0008391, 0.0008792
4: -0.0022241, -0.0008701, -0.0021554, -0.0008597, -0.0009518, 0.0009084
5: 0.0049367, 0.0062181, 0.0049270, 0.0061531, -0.0008596, 0.0009007
6: -0.0027128, 0.0023712, -0.0027516, 0.0021132, -0.0034108, 0.0035737
7: -0.0057860, 0.0011380, -0.0054348, 0.0011907, -0.0048671, 0.0046452
8: 0.9851381, 0.9900154, 0.9853855, 0.9900526, -0.0034285, 0.0032722
9: -0.0068240, -0.0023966, -0.0068577, -0.0026212, -0.0029703, 0.0031122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016665, upper bound: 0.0014943
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014943
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032911, 0.0053744, 0.0033496, 0.0050810, -0.0012885, 0.0015292
1: 0.0017978, 0.0020987, 0.0018062, 0.0020564, -0.0001862, 0.0002209
2: 0.0113885, 0.0125403, 0.0115507, 0.0125079, -0.0008454, 0.0007124
3: -0.0029019, -0.0017107, -0.0027342, -0.0017442, -0.0008744, 0.0007368
4: -0.0021850, -0.0008955, -0.0021488, -0.0010771, -0.0007976, 0.0009466
5: 0.0049607, 0.0061811, 0.0051326, 0.0061469, -0.0008958, 0.0007548
6: -0.0026175, 0.0022245, -0.0019356, 0.0020886, -0.0035543, 0.0029949
7: -0.0055863, 0.0010081, -0.0054012, 0.0000795, -0.0040788, 0.0048406
8: 0.9852787, 0.9899241, 0.9854091, 0.9892699, -0.0028732, 0.0034098
9: -0.0067410, -0.0025243, -0.0061472, -0.0026427, -0.0030952, 0.0026081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014816
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014877
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032911, 0.0053744, 0.0033455, 0.0051409, -0.0012952, 0.0014683
1: 0.0017978, 0.0020987, 0.0018056, 0.0020650, -0.0001871, 0.0002121
2: 0.0113885, 0.0125403, 0.0115176, 0.0125102, -0.0008118, 0.0007161
3: -0.0029019, -0.0017107, -0.0027684, -0.0017418, -0.0008396, 0.0007406
4: -0.0021850, -0.0008955, -0.0021513, -0.0010400, -0.0008017, 0.0009089
5: 0.0049607, 0.0061811, 0.0050975, 0.0061492, -0.0008601, 0.0007587
6: -0.0026175, 0.0022245, -0.0020748, 0.0020981, -0.0034127, 0.0030104
7: -0.0055863, 0.0010081, -0.0054141, 0.0002690, -0.0040999, 0.0046478
8: 0.9852787, 0.9899241, 0.9854000, 0.9894034, -0.0028880, 0.0032740
9: -0.0067410, -0.0025243, -0.0062684, -0.0026344, -0.0029719, 0.0026216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014816
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014878
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033390, 0.0054324, 0.0033776, 0.0050800, -0.0013142, 0.0015666
1: 0.0018047, 0.0021071, 0.0018103, 0.0020562, -0.0001899, 0.0002263
2: 0.0113564, 0.0125138, 0.0115512, 0.0124925, -0.0008661, 0.0007266
3: -0.0029351, -0.0017381, -0.0027336, -0.0017601, -0.0008958, 0.0007514
4: -0.0021554, -0.0008595, -0.0021315, -0.0010777, -0.0008135, 0.0009698
5: 0.0049268, 0.0061531, 0.0051332, 0.0061305, -0.0009177, 0.0007698
6: -0.0027524, 0.0021132, -0.0019334, 0.0020236, -0.0036413, 0.0030545
7: -0.0054348, 0.0011918, -0.0053127, 0.0000764, -0.0041599, 0.0049591
8: 0.9853855, 0.9900534, 0.9854715, 0.9892677, -0.0029303, 0.0034933
9: -0.0068584, -0.0026212, -0.0061452, -0.0026993, -0.0031710, 0.0026600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014817
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014878
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033390, 0.0054324, 0.0033729, 0.0051398, -0.0013213, 0.0015098
1: 0.0018047, 0.0021071, 0.0018096, 0.0020649, -0.0001909, 0.0002181
2: 0.0113564, 0.0125138, 0.0115182, 0.0124951, -0.0008347, 0.0007305
3: -0.0029351, -0.0017381, -0.0027678, -0.0017575, -0.0008633, 0.0007555
4: -0.0021554, -0.0008595, -0.0021344, -0.0010406, -0.0008179, 0.0009346
5: 0.0049268, 0.0061531, 0.0050981, 0.0061332, -0.0008845, 0.0007740
6: -0.0027524, 0.0021132, -0.0020724, 0.0020344, -0.0035093, 0.0030710
7: -0.0054348, 0.0011918, -0.0053274, 0.0002657, -0.0041824, 0.0047793
8: 0.9853855, 0.9900534, 0.9854611, 0.9894011, -0.0029462, 0.0033666
9: -0.0068584, -0.0026212, -0.0062663, -0.0026899, -0.0030560, 0.0026743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014817
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014878
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032098, 0.0054834, 0.0033496, 0.0050810, -0.0014060, 0.0016780
1: 0.0017860, 0.0021145, 0.0018062, 0.0020564, -0.0002031, 0.0002424
2: 0.0113282, 0.0125852, 0.0115507, 0.0125079, -0.0009277, 0.0007773
3: -0.0029643, -0.0016642, -0.0027342, -0.0017442, -0.0009595, 0.0008040
4: -0.0022354, -0.0008280, -0.0021488, -0.0010771, -0.0008703, 0.0010387
5: 0.0048969, 0.0062287, 0.0051326, 0.0061469, -0.0009830, 0.0008236
6: -0.0028710, 0.0024135, -0.0019356, 0.0020886, -0.0039002, 0.0032679
7: -0.0058437, 0.0013533, -0.0054012, 0.0000795, -0.0044506, 0.0053118
8: 0.9850973, 0.9901671, 0.9854091, 0.9892699, -0.0031351, 0.0037417
9: -0.0069617, -0.0023597, -0.0061472, -0.0026427, -0.0033965, 0.0028458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032098, 0.0054834, 0.0033455, 0.0051409, -0.0014069, 0.0016192
1: 0.0017860, 0.0021145, 0.0018056, 0.0020650, -0.0002033, 0.0002339
2: 0.0113282, 0.0125852, 0.0115176, 0.0125102, -0.0008952, 0.0007778
3: -0.0029643, -0.0016642, -0.0027684, -0.0017418, -0.0009259, 0.0008045
4: -0.0022354, -0.0008280, -0.0021513, -0.0010400, -0.0008709, 0.0010023
5: 0.0048969, 0.0062287, 0.0050975, 0.0061492, -0.0009485, 0.0008242
6: -0.0028710, 0.0024135, -0.0020748, 0.0020981, -0.0037635, 0.0032700
7: -0.0058437, 0.0013533, -0.0054141, 0.0002690, -0.0044535, 0.0051256
8: 0.9850973, 0.9901671, 0.9854000, 0.9894034, -0.0031371, 0.0036106
9: -0.0069617, -0.0023597, -0.0062684, -0.0026344, -0.0032774, 0.0028477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032605, 0.0055373, 0.0033776, 0.0050800, -0.0014256, 0.0017045
1: 0.0017933, 0.0021223, 0.0018103, 0.0020562, -0.0002060, 0.0002463
2: 0.0112984, 0.0125572, 0.0115512, 0.0124925, -0.0009424, 0.0007882
3: -0.0029951, -0.0016932, -0.0027336, -0.0017601, -0.0009747, 0.0008152
4: -0.0022040, -0.0007946, -0.0021315, -0.0010777, -0.0008825, 0.0010551
5: 0.0048653, 0.0061991, 0.0051332, 0.0061305, -0.0009985, 0.0008351
6: -0.0029961, 0.0022957, -0.0019334, 0.0020236, -0.0039618, 0.0033136
7: -0.0056833, 0.0015238, -0.0053127, 0.0000764, -0.0045128, 0.0053956
8: 0.9852105, 0.9902872, 0.9854715, 0.9892677, -0.0031789, 0.0038008
9: -0.0070707, -0.0024623, -0.0061452, -0.0026993, -0.0034501, 0.0028856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032605, 0.0055373, 0.0033729, 0.0051398, -0.0014294, 0.0016441
1: 0.0017933, 0.0021223, 0.0018096, 0.0020649, -0.0002065, 0.0002375
2: 0.0112984, 0.0125572, 0.0115182, 0.0124951, -0.0009090, 0.0007903
3: -0.0029951, -0.0016932, -0.0027678, -0.0017575, -0.0009401, 0.0008173
4: -0.0022040, -0.0007946, -0.0021344, -0.0010406, -0.0008848, 0.0010177
5: 0.0048653, 0.0061991, 0.0050981, 0.0061332, -0.0009631, 0.0008373
6: -0.0029961, 0.0022957, -0.0020724, 0.0020344, -0.0038214, 0.0033223
7: -0.0056833, 0.0015238, -0.0053274, 0.0002657, -0.0045247, 0.0052044
8: 0.9852105, 0.9902872, 0.9854611, 0.9894011, -0.0031873, 0.0036661
9: -0.0070707, -0.0024623, -0.0062663, -0.0026899, -0.0033278, 0.0028932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0032683, 0.0053758, 0.0033029, 0.0053079, -0.0013186, 0.0013632
1: 0.0017945, 0.0020990, 0.0017995, 0.0020891, -0.0001905, 0.0001969
2: 0.0113877, 0.0125529, 0.0114252, 0.0125337, -0.0007537, 0.0007290
3: -0.0029028, -0.0016976, -0.0028639, -0.0017175, -0.0007795, 0.0007540
4: -0.0021992, -0.0008946, -0.0021777, -0.0009366, -0.0008162, 0.0008438
5: 0.0049599, 0.0061945, 0.0049997, 0.0061742, -0.0007985, 0.0007724
6: -0.0026209, 0.0022776, -0.0024631, 0.0021970, -0.0031684, 0.0030648
7: -0.0056587, 0.0010127, -0.0055489, 0.0007978, -0.0041740, 0.0043150
8: 0.9852278, 0.9899272, 0.9853051, 0.9897758, -0.0029402, 0.0030396
9: -0.0067439, -0.0024780, -0.0066065, -0.0025482, -0.0027592, 0.0026690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0015019
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0015079
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0032683, 0.0053758, 0.0032911, 0.0053743, -0.0013068, 0.0012827
1: 0.0017945, 0.0020990, 0.0017978, 0.0020987, -0.0001888, 0.0001853
2: 0.0113877, 0.0125529, 0.0113885, 0.0125403, -0.0007091, 0.0007225
3: -0.0029028, -0.0016976, -0.0029019, -0.0017107, -0.0007334, 0.0007472
4: -0.0021992, -0.0008946, -0.0021850, -0.0008955, -0.0008089, 0.0007940
5: 0.0049599, 0.0061945, 0.0049608, 0.0061811, -0.0007514, 0.0007655
6: -0.0026209, 0.0022776, -0.0026174, 0.0022245, -0.0029812, 0.0030373
7: -0.0056587, 0.0010127, -0.0055863, 0.0010080, -0.0041366, 0.0040602
8: 0.9852278, 0.9899272, 0.9852787, 0.9899240, -0.0029139, 0.0028601
9: -0.0067439, -0.0024780, -0.0067409, -0.0025243, -0.0025962, 0.0026450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016298, upper bound: 0.0015079
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0015079
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0032953, 0.0053747, 0.0033502, 0.0053675, -0.0013691, 0.0013887
1: 0.0017984, 0.0020988, 0.0018063, 0.0020977, -0.0001978, 0.0002006
2: 0.0113883, 0.0125380, 0.0113923, 0.0125076, -0.0007678, 0.0007569
3: -0.0029021, -0.0017131, -0.0028980, -0.0017445, -0.0007941, 0.0007828
4: -0.0021825, -0.0008953, -0.0021484, -0.0008997, -0.0008475, 0.0008596
5: 0.0049606, 0.0061787, 0.0049648, 0.0061465, -0.0008135, 0.0008020
6: -0.0026182, 0.0022149, -0.0026015, 0.0020871, -0.0032277, 0.0031821
7: -0.0055732, 0.0010091, -0.0053992, 0.0009863, -0.0043337, 0.0043958
8: 0.9852880, 0.9899247, 0.9854105, 0.9899086, -0.0030528, 0.0030965
9: -0.0067416, -0.0025327, -0.0067270, -0.0026440, -0.0028108, 0.0027711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016306, upper bound: 0.0014891
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014891
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0032953, 0.0053747, 0.0033390, 0.0054320, -0.0013556, 0.0013087
1: 0.0017984, 0.0020988, 0.0018047, 0.0021071, -0.0001959, 0.0001891
2: 0.0113883, 0.0125380, 0.0113566, 0.0125138, -0.0007235, 0.0007495
3: -0.0029021, -0.0017131, -0.0029349, -0.0017381, -0.0007483, 0.0007752
4: -0.0021825, -0.0008953, -0.0021554, -0.0008597, -0.0008392, 0.0008101
5: 0.0049606, 0.0061787, 0.0049270, 0.0061531, -0.0007666, 0.0007941
6: -0.0026182, 0.0022149, -0.0027516, 0.0021132, -0.0030417, 0.0031509
7: -0.0055732, 0.0010091, -0.0054348, 0.0011907, -0.0042912, 0.0041426
8: 0.9852880, 0.9899247, 0.9853855, 0.9900526, -0.0030228, 0.0029181
9: -0.0067416, -0.0025327, -0.0068577, -0.0026212, -0.0026489, 0.0027439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016306, upper bound: 0.0014891
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014891
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0031874, 0.0054847, 0.0033029, 0.0053079, -0.0014185, 0.0014948
1: 0.0017828, 0.0021147, 0.0017995, 0.0020891, -0.0002049, 0.0002160
2: 0.0113275, 0.0125976, 0.0114252, 0.0125337, -0.0008264, 0.0007842
3: -0.0029650, -0.0016514, -0.0028639, -0.0017175, -0.0008547, 0.0008111
4: -0.0022492, -0.0008272, -0.0021777, -0.0009366, -0.0008781, 0.0009253
5: 0.0048961, 0.0062419, 0.0049997, 0.0061742, -0.0008756, 0.0008309
6: -0.0028740, 0.0024655, -0.0024631, 0.0021970, -0.0034743, 0.0032970
7: -0.0059146, 0.0013574, -0.0055489, 0.0007978, -0.0044902, 0.0047317
8: 0.9850476, 0.9901701, 0.9853051, 0.9897758, -0.0031630, 0.0033331
9: -0.0069643, -0.0023144, -0.0066065, -0.0025482, -0.0030256, 0.0028711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0031874, 0.0054847, 0.0032911, 0.0053743, -0.0014093, 0.0014320
1: 0.0017828, 0.0021147, 0.0017978, 0.0020987, -0.0002036, 0.0002069
2: 0.0113275, 0.0125976, 0.0113885, 0.0125403, -0.0007917, 0.0007791
3: -0.0029650, -0.0016514, -0.0029019, -0.0017107, -0.0008188, 0.0008058
4: -0.0022492, -0.0008272, -0.0021850, -0.0008955, -0.0008724, 0.0008864
5: 0.0048961, 0.0062419, 0.0049608, 0.0061811, -0.0008388, 0.0008255
6: -0.0028740, 0.0024655, -0.0026174, 0.0022245, -0.0033283, 0.0032755
7: -0.0059146, 0.0013574, -0.0055863, 0.0010080, -0.0044610, 0.0045328
8: 0.9850476, 0.9901701, 0.9852787, 0.9899240, -0.0031424, 0.0031930
9: -0.0069643, -0.0023144, -0.0067409, -0.0025243, -0.0028984, 0.0028525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0032159, 0.0054836, 0.0033502, 0.0053675, -0.0014622, 0.0015202
1: 0.0017869, 0.0021145, 0.0018063, 0.0020977, -0.0002112, 0.0002196
2: 0.0113281, 0.0125819, 0.0113923, 0.0125076, -0.0008405, 0.0008084
3: -0.0029644, -0.0016677, -0.0028980, -0.0017445, -0.0008693, 0.0008361
4: -0.0022316, -0.0008278, -0.0021484, -0.0008997, -0.0009051, 0.0009411
5: 0.0048968, 0.0062252, 0.0049648, 0.0061465, -0.0008906, 0.0008565
6: -0.0028715, 0.0023993, -0.0026015, 0.0020871, -0.0035335, 0.0033985
7: -0.0058244, 0.0013540, -0.0053992, 0.0009863, -0.0046284, 0.0048123
8: 0.9851111, 0.9901676, 0.9854105, 0.9899086, -0.0032604, 0.0033899
9: -0.0069621, -0.0023721, -0.0067270, -0.0026440, -0.0030771, 0.0029596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0014806
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0032159, 0.0054836, 0.0033390, 0.0054320, -0.0014544, 0.0014596
1: 0.0017869, 0.0021145, 0.0018047, 0.0021071, -0.0002101, 0.0002109
2: 0.0113281, 0.0125819, 0.0113566, 0.0125138, -0.0008070, 0.0008041
3: -0.0029644, -0.0016677, -0.0029349, -0.0017381, -0.0008346, 0.0008316
4: -0.0022316, -0.0008278, -0.0021554, -0.0008597, -0.0009003, 0.0009035
5: 0.0048968, 0.0062252, 0.0049270, 0.0061531, -0.0008550, 0.0008520
6: -0.0028715, 0.0023993, -0.0027516, 0.0021132, -0.0033925, 0.0033805
7: -0.0058244, 0.0013540, -0.0054348, 0.0011907, -0.0046039, 0.0046203
8: 0.9851111, 0.9901676, 0.9853855, 0.9900526, -0.0032431, 0.0032546
9: -0.0069621, -0.0023721, -0.0068577, -0.0026212, -0.0029543, 0.0029439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0014806
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0033029, 0.0053079, 0.0032816, 0.0051768, -0.0014172, 0.0015493
1: 0.0017995, 0.0020891, 0.0017964, 0.0020702, -0.0002047, 0.0002238
2: 0.0114252, 0.0125337, 0.0114978, 0.0125455, -0.0008566, 0.0007836
3: -0.0028639, -0.0017175, -0.0027889, -0.0017053, -0.0008859, 0.0008104
4: -0.0021777, -0.0009366, -0.0021909, -0.0010178, -0.0008773, 0.0009591
5: 0.0049997, 0.0061742, 0.0050765, 0.0061867, -0.0009076, 0.0008302
6: -0.0024631, 0.0021970, -0.0021582, 0.0022466, -0.0036011, 0.0032941
7: -0.0055489, 0.0007978, -0.0056164, 0.0003826, -0.0044862, 0.0049044
8: 0.9853051, 0.9897758, 0.9852576, 0.9894834, -0.0031602, 0.0034547
9: -0.0066065, -0.0025482, -0.0063410, -0.0025051, -0.0031360, 0.0028686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015248
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015312
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0033029, 0.0053079, 0.0032731, 0.0052395, -0.0015067, 0.0015846
1: 0.0017995, 0.0020891, 0.0017952, 0.0020793, -0.0002177, 0.0002289
2: 0.0114252, 0.0125337, 0.0114631, 0.0125503, -0.0008761, 0.0008330
3: -0.0028639, -0.0017175, -0.0028248, -0.0017004, -0.0009061, 0.0008615
4: -0.0021777, -0.0009366, -0.0021962, -0.0009790, -0.0009327, 0.0009809
5: 0.0049997, 0.0061742, 0.0050398, 0.0061917, -0.0009283, 0.0008826
6: -0.0024631, 0.0021970, -0.0023040, 0.0022665, -0.0036832, 0.0035020
7: -0.0055489, 0.0007978, -0.0056434, 0.0005811, -0.0047694, 0.0050161
8: 0.9853051, 0.9897758, 0.9852385, 0.9896232, -0.0033596, 0.0035335
9: -0.0066065, -0.0025482, -0.0064679, -0.0024878, -0.0032075, 0.0030497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015387
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015454
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032246, 0.0054151, 0.0032816, 0.0051768, -0.0013281, 0.0015039
1: 0.0017882, 0.0021046, 0.0017964, 0.0020702, -0.0001919, 0.0002173
2: 0.0113660, 0.0125771, 0.0114978, 0.0125455, -0.0008315, 0.0007343
3: -0.0029252, -0.0016727, -0.0027889, -0.0017053, -0.0008599, 0.0007594
4: -0.0022262, -0.0008702, -0.0021909, -0.0010178, -0.0008221, 0.0009309
5: 0.0049369, 0.0062201, 0.0050765, 0.0061867, -0.0008810, 0.0007780
6: -0.0027123, 0.0023792, -0.0021582, 0.0022466, -0.0034955, 0.0030868
7: -0.0057970, 0.0011372, -0.0056164, 0.0003826, -0.0042040, 0.0047606
8: 0.9851304, 0.9900149, 0.9852576, 0.9894834, -0.0029614, 0.0033534
9: -0.0068235, -0.0023896, -0.0063410, -0.0025051, -0.0030440, 0.0026882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014670
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014782
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032246, 0.0054151, 0.0032731, 0.0052395, -0.0014343, 0.0015449
1: 0.0017882, 0.0021046, 0.0017952, 0.0020793, -0.0002072, 0.0002232
2: 0.0113660, 0.0125771, 0.0114631, 0.0125503, -0.0008541, 0.0007930
3: -0.0029252, -0.0016727, -0.0028248, -0.0017004, -0.0008834, 0.0008201
4: -0.0022262, -0.0008702, -0.0021962, -0.0009790, -0.0008879, 0.0009563
5: 0.0049369, 0.0062201, 0.0050398, 0.0061917, -0.0009050, 0.0008402
6: -0.0027123, 0.0023792, -0.0023040, 0.0022665, -0.0035907, 0.0033337
7: -0.0057970, 0.0011372, -0.0056434, 0.0005811, -0.0045402, 0.0048902
8: 0.9851304, 0.9900149, 0.9852385, 0.9896232, -0.0031982, 0.0034448
9: -0.0068235, -0.0023896, -0.0064679, -0.0024878, -0.0031270, 0.0029031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014798
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014911
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032911, 0.0053744, 0.0032816, 0.0051768, -0.0014298, 0.0016283
1: 0.0017978, 0.0020987, 0.0017964, 0.0020702, -0.0002066, 0.0002352
2: 0.0113885, 0.0125403, 0.0114978, 0.0125455, -0.0009002, 0.0007905
3: -0.0029019, -0.0017107, -0.0027889, -0.0017053, -0.0009311, 0.0008176
4: -0.0021850, -0.0008955, -0.0021909, -0.0010178, -0.0008851, 0.0010079
5: 0.0049607, 0.0061811, 0.0050765, 0.0061867, -0.0009538, 0.0008376
6: -0.0026175, 0.0022245, -0.0021582, 0.0022466, -0.0037846, 0.0033232
7: -0.0055863, 0.0010081, -0.0056164, 0.0003826, -0.0045259, 0.0051543
8: 0.9852787, 0.9899241, 0.9852576, 0.9894834, -0.0031881, 0.0036308
9: -0.0067410, -0.0025243, -0.0063410, -0.0025051, -0.0032958, 0.0028940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0015248
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0015312
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032911, 0.0053744, 0.0032731, 0.0052395, -0.0014352, 0.0015677
1: 0.0017978, 0.0020987, 0.0017952, 0.0020793, -0.0002073, 0.0002265
2: 0.0113885, 0.0125403, 0.0114631, 0.0125503, -0.0008667, 0.0007935
3: -0.0029019, -0.0017107, -0.0028248, -0.0017004, -0.0008964, 0.0008207
4: -0.0021850, -0.0008955, -0.0021962, -0.0009790, -0.0008884, 0.0009704
5: 0.0049607, 0.0061811, 0.0050398, 0.0061917, -0.0009183, 0.0008407
6: -0.0026175, 0.0022245, -0.0023040, 0.0022665, -0.0036437, 0.0033358
7: -0.0055863, 0.0010081, -0.0056434, 0.0005811, -0.0045431, 0.0049624
8: 0.9852787, 0.9899241, 0.9852385, 0.9896232, -0.0032002, 0.0034956
9: -0.0067410, -0.0025243, -0.0064679, -0.0024878, -0.0031731, 0.0029050

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0015248
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0015312
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0033390, 0.0054324, 0.0033118, 0.0051758, -0.0014555, 0.0016642
1: 0.0018047, 0.0021071, 0.0018008, 0.0020701, -0.0002103, 0.0002404
2: 0.0113564, 0.0125138, 0.0114983, 0.0125288, -0.0009201, 0.0008047
3: -0.0029351, -0.0017381, -0.0027884, -0.0017226, -0.0009516, 0.0008323
4: -0.0021554, -0.0008595, -0.0021722, -0.0010184, -0.0009010, 0.0010301
5: 0.0049268, 0.0061531, 0.0050771, 0.0061690, -0.0009749, 0.0008526
6: -0.0027524, 0.0021132, -0.0021560, 0.0021764, -0.0038680, 0.0033831
7: -0.0054348, 0.0011918, -0.0055207, 0.0003795, -0.0046074, 0.0052678
8: 0.9853855, 0.9900534, 0.9853249, 0.9894812, -0.0032456, 0.0037108
9: -0.0068584, -0.0026212, -0.0063390, -0.0025662, -0.0033684, 0.0029461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015255
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015316
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033390, 0.0054324, 0.0033023, 0.0052385, -0.0014616, 0.0016077
1: 0.0018047, 0.0021071, 0.0017994, 0.0020791, -0.0002112, 0.0002323
2: 0.0113564, 0.0125138, 0.0114637, 0.0125341, -0.0008889, 0.0008081
3: -0.0029351, -0.0017381, -0.0028242, -0.0017171, -0.0009193, 0.0008358
4: -0.0021554, -0.0008595, -0.0021781, -0.0009796, -0.0009048, 0.0009952
5: 0.0049268, 0.0061531, 0.0050404, 0.0061746, -0.0009418, 0.0008562
6: -0.0027524, 0.0021132, -0.0023016, 0.0021986, -0.0037367, 0.0033972
7: -0.0054348, 0.0011918, -0.0055510, 0.0005779, -0.0046267, 0.0050891
8: 0.9853855, 0.9900534, 0.9853037, 0.9896209, -0.0032591, 0.0035849
9: -0.0068584, -0.0026212, -0.0064659, -0.0025469, -0.0032541, 0.0029584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015255
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015316
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0032098, 0.0054834, 0.0032816, 0.0051768, -0.0013413, 0.0015831
1: 0.0017860, 0.0021145, 0.0017964, 0.0020702, -0.0001938, 0.0002287
2: 0.0113282, 0.0125852, 0.0114978, 0.0125455, -0.0008753, 0.0007416
3: -0.0029643, -0.0016642, -0.0027889, -0.0017053, -0.0009053, 0.0007670
4: -0.0022354, -0.0008280, -0.0021909, -0.0010178, -0.0008303, 0.0009800
5: 0.0048969, 0.0062287, 0.0050765, 0.0061867, -0.0009274, 0.0007857
6: -0.0028710, 0.0024135, -0.0021582, 0.0022466, -0.0036797, 0.0031176
7: -0.0058437, 0.0013533, -0.0056164, 0.0003826, -0.0042459, 0.0050114
8: 0.9850973, 0.9901671, 0.9852576, 0.9894834, -0.0029909, 0.0035301
9: -0.0069617, -0.0023597, -0.0063410, -0.0025051, -0.0032044, 0.0027149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032098, 0.0054834, 0.0032731, 0.0052395, -0.0013486, 0.0015248
1: 0.0017860, 0.0021145, 0.0017952, 0.0020793, -0.0001948, 0.0002203
2: 0.0113282, 0.0125852, 0.0114631, 0.0125503, -0.0008430, 0.0007456
3: -0.0029643, -0.0016642, -0.0028248, -0.0017004, -0.0008719, 0.0007711
4: -0.0022354, -0.0008280, -0.0021962, -0.0009790, -0.0008348, 0.0009439
5: 0.0048969, 0.0062287, 0.0050398, 0.0061917, -0.0008932, 0.0007900
6: -0.0028710, 0.0024135, -0.0023040, 0.0022665, -0.0035442, 0.0031345
7: -0.0058437, 0.0013533, -0.0056434, 0.0005811, -0.0042690, 0.0048268
8: 0.9850973, 0.9901671, 0.9852385, 0.9896232, -0.0030071, 0.0034001
9: -0.0069617, -0.0023597, -0.0064679, -0.0024878, -0.0030864, 0.0027297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0032605, 0.0055373, 0.0033118, 0.0051758, -0.0013643, 0.0016174
1: 0.0017933, 0.0021223, 0.0018008, 0.0020701, -0.0001971, 0.0002337
2: 0.0112984, 0.0125572, 0.0114983, 0.0125288, -0.0008942, 0.0007543
3: -0.0029951, -0.0016932, -0.0027884, -0.0017226, -0.0009248, 0.0007801
4: -0.0022040, -0.0007946, -0.0021722, -0.0010184, -0.0008446, 0.0010012
5: 0.0048653, 0.0061991, 0.0050771, 0.0061690, -0.0009475, 0.0007992
6: -0.0029961, 0.0022957, -0.0021560, 0.0021764, -0.0037592, 0.0031711
7: -0.0056833, 0.0015238, -0.0055207, 0.0003795, -0.0043188, 0.0051198
8: 0.9852105, 0.9902872, 0.9853249, 0.9894812, -0.0030422, 0.0036065
9: -0.0070707, -0.0024623, -0.0063390, -0.0025662, -0.0032737, 0.0027616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0032605, 0.0055373, 0.0033023, 0.0052385, -0.0013711, 0.0015644
1: 0.0017933, 0.0021223, 0.0017994, 0.0020791, -0.0001981, 0.0002260
2: 0.0112984, 0.0125572, 0.0114637, 0.0125341, -0.0008649, 0.0007580
3: -0.0029951, -0.0016932, -0.0028242, -0.0017171, -0.0008945, 0.0007840
4: -0.0022040, -0.0007946, -0.0021781, -0.0009796, -0.0008487, 0.0009684
5: 0.0048653, 0.0061991, 0.0050404, 0.0061746, -0.0009164, 0.0008032
6: -0.0029961, 0.0022957, -0.0023016, 0.0021986, -0.0036361, 0.0031868
7: -0.0056833, 0.0015238, -0.0055510, 0.0005779, -0.0043401, 0.0049521
8: 0.9852105, 0.9902872, 0.9853037, 0.9896209, -0.0030573, 0.0034884
9: -0.0070707, -0.0024623, -0.0064659, -0.0025469, -0.0031665, 0.0027752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0032683, 0.0053758, 0.0032246, 0.0054151, -0.0014670, 0.0014660
1: 0.0017945, 0.0020990, 0.0017882, 0.0021046, -0.0002119, 0.0002118
2: 0.0113877, 0.0125529, 0.0113660, 0.0125771, -0.0008105, 0.0008111
3: -0.0029028, -0.0016976, -0.0029252, -0.0016727, -0.0008383, 0.0008389
4: -0.0021992, -0.0008946, -0.0022262, -0.0008702, -0.0009081, 0.0009075
5: 0.0049599, 0.0061945, 0.0049369, 0.0062201, -0.0008588, 0.0008594
6: -0.0026209, 0.0022776, -0.0027123, 0.0023792, -0.0034075, 0.0034097
7: -0.0056587, 0.0010127, -0.0057970, 0.0011372, -0.0046438, 0.0046407
8: 0.9852278, 0.9899272, 0.9851304, 0.9900149, -0.0032712, 0.0032690
9: -0.0067439, -0.0024780, -0.0068235, -0.0023896, -0.0029674, 0.0029694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016087, upper bound: 0.0015570
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015570
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0032683, 0.0053758, 0.0032098, 0.0054834, -0.0014551, 0.0013852
1: 0.0017945, 0.0020990, 0.0017860, 0.0021145, -0.0002102, 0.0002001
2: 0.0113877, 0.0125529, 0.0113282, 0.0125852, -0.0007659, 0.0008045
3: -0.0029028, -0.0016976, -0.0029643, -0.0016642, -0.0007921, 0.0008320
4: -0.0021992, -0.0008946, -0.0022354, -0.0008280, -0.0009007, 0.0008575
5: 0.0049599, 0.0061945, 0.0048969, 0.0062287, -0.0008115, 0.0008524
6: -0.0026209, 0.0022776, -0.0028710, 0.0024135, -0.0032197, 0.0033821
7: -0.0056587, 0.0010127, -0.0058437, 0.0013533, -0.0046061, 0.0043849
8: 0.9852278, 0.9899272, 0.9850973, 0.9901671, -0.0032446, 0.0030888
9: -0.0067439, -0.0024780, -0.0069617, -0.0023597, -0.0028038, 0.0029453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016087, upper bound: 0.0015570
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015570
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0032953, 0.0053747, 0.0032707, 0.0054757, -0.0015012, 0.0014953
1: 0.0017984, 0.0020988, 0.0017948, 0.0021134, -0.0002169, 0.0002160
2: 0.0113883, 0.0125380, 0.0113325, 0.0125516, -0.0008267, 0.0008300
3: -0.0029021, -0.0017131, -0.0029599, -0.0016991, -0.0008550, 0.0008584
4: -0.0021825, -0.0008953, -0.0021976, -0.0008327, -0.0009293, 0.0009256
5: 0.0049606, 0.0061787, 0.0049014, 0.0061931, -0.0008759, 0.0008794
6: -0.0026182, 0.0022149, -0.0028531, 0.0022719, -0.0034755, 0.0034893
7: -0.0055732, 0.0010091, -0.0056508, 0.0013290, -0.0047521, 0.0047333
8: 0.9852880, 0.9899247, 0.9852333, 0.9901500, -0.0033475, 0.0033343
9: -0.0067416, -0.0025327, -0.0069462, -0.0024831, -0.0030266, 0.0030386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016090, upper bound: 0.0015344
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015344
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0032953, 0.0053747, 0.0032605, 0.0055373, -0.0014861, 0.0014151
1: 0.0017984, 0.0020988, 0.0017933, 0.0021223, -0.0002147, 0.0002044
2: 0.0113883, 0.0125380, 0.0112984, 0.0125572, -0.0007824, 0.0008216
3: -0.0029021, -0.0017131, -0.0029951, -0.0016932, -0.0008092, 0.0008498
4: -0.0021825, -0.0008953, -0.0022040, -0.0007946, -0.0009199, 0.0008760
5: 0.0049606, 0.0061787, 0.0048653, 0.0061991, -0.0008290, 0.0008706
6: -0.0026182, 0.0022149, -0.0029961, 0.0022957, -0.0032891, 0.0034542
7: -0.0055732, 0.0010091, -0.0056833, 0.0015238, -0.0047043, 0.0044794
8: 0.9852880, 0.9899247, 0.9852105, 0.9902872, -0.0033138, 0.0031554
9: -0.0067416, -0.0025327, -0.0070707, -0.0024623, -0.0028643, 0.0030081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016090, upper bound: 0.0015344
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015344
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0031874, 0.0054847, 0.0032246, 0.0054151, -0.0013751, 0.0014187
1: 0.0017828, 0.0021147, 0.0017882, 0.0021046, -0.0001987, 0.0002050
2: 0.0113275, 0.0125976, 0.0113660, 0.0125771, -0.0007844, 0.0007603
3: -0.0029650, -0.0016514, -0.0029252, -0.0016727, -0.0008113, 0.0007863
4: -0.0022492, -0.0008272, -0.0022262, -0.0008702, -0.0008512, 0.0008782
5: 0.0048961, 0.0062419, 0.0049369, 0.0062201, -0.0008311, 0.0008056
6: -0.0028740, 0.0024655, -0.0027123, 0.0023792, -0.0032976, 0.0031962
7: -0.0059146, 0.0013574, -0.0057970, 0.0011372, -0.0043530, 0.0044910
8: 0.9850476, 0.9901701, 0.9851304, 0.9900149, -0.0030663, 0.0031635
9: -0.0069643, -0.0023144, -0.0068235, -0.0023896, -0.0028717, 0.0027834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0031874, 0.0054847, 0.0032098, 0.0054834, -0.0013613, 0.0013365
1: 0.0017828, 0.0021147, 0.0017860, 0.0021145, -0.0001967, 0.0001931
2: 0.0113275, 0.0125976, 0.0113282, 0.0125852, -0.0007389, 0.0007526
3: -0.0029650, -0.0016514, -0.0029643, -0.0016642, -0.0007642, 0.0007784
4: -0.0022492, -0.0008272, -0.0022354, -0.0008280, -0.0008427, 0.0008273
5: 0.0048961, 0.0062419, 0.0048969, 0.0062287, -0.0007829, 0.0007975
6: -0.0028740, 0.0024655, -0.0028710, 0.0024135, -0.0031063, 0.0031641
7: -0.0059146, 0.0013574, -0.0058437, 0.0013533, -0.0043092, 0.0042306
8: 0.9850476, 0.9901701, 0.9850973, 0.9901671, -0.0030355, 0.0029801
9: -0.0069643, -0.0023144, -0.0069617, -0.0023597, -0.0027051, 0.0027554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016860, upper bound: 0.0014984
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0032159, 0.0054836, 0.0032707, 0.0054757, -0.0014237, 0.0014418
1: 0.0017869, 0.0021145, 0.0017948, 0.0021134, -0.0002057, 0.0002083
2: 0.0113281, 0.0125819, 0.0113325, 0.0125516, -0.0007972, 0.0007872
3: -0.0029644, -0.0016677, -0.0029599, -0.0016991, -0.0008245, 0.0008141
4: -0.0022316, -0.0008278, -0.0021976, -0.0008327, -0.0008813, 0.0008925
5: 0.0048968, 0.0062252, 0.0049014, 0.0061931, -0.0008446, 0.0008340
6: -0.0028715, 0.0023993, -0.0028531, 0.0022719, -0.0033513, 0.0033092
7: -0.0058244, 0.0013540, -0.0056508, 0.0013290, -0.0045068, 0.0045641
8: 0.9851111, 0.9901676, 0.9852333, 0.9901500, -0.0031747, 0.0032151
9: -0.0069621, -0.0023721, -0.0069462, -0.0024831, -0.0029184, 0.0028818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016877, upper bound: 0.0014806
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0032159, 0.0054836, 0.0032605, 0.0055373, -0.0014083, 0.0013619
1: 0.0017869, 0.0021145, 0.0017933, 0.0021223, -0.0002035, 0.0001967
2: 0.0113281, 0.0125819, 0.0112984, 0.0125572, -0.0007529, 0.0007786
3: -0.0029644, -0.0016677, -0.0029951, -0.0016932, -0.0007787, 0.0008053
4: -0.0022316, -0.0008278, -0.0022040, -0.0007946, -0.0008718, 0.0008430
5: 0.0048968, 0.0062252, 0.0048653, 0.0061991, -0.0007978, 0.0008250
6: -0.0028715, 0.0023993, -0.0029961, 0.0022957, -0.0031653, 0.0032733
7: -0.0058244, 0.0013540, -0.0056833, 0.0015238, -0.0044579, 0.0043109
8: 0.9851111, 0.9901676, 0.9852105, 0.9902872, -0.0031403, 0.0030367
9: -0.0069621, -0.0023721, -0.0070707, -0.0024623, -0.0027565, 0.0028505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 112

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0016877, upper bound: 0.0014806
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
time: 0.84 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.48 seconds
IS_A1_B1_A1_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014841, upper bound: 0.0016275
IS_A1_B1_A1_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016275
IS_A1_B1_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014841, upper bound: 0.0016490
IS_A1_B1_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016490
IS_A1_B1_A1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014842, upper bound: 0.0016184
IS_A1_B1_A1_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016184
IS_A1_B1_A1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014842, upper bound: 0.0016421
IS_A1_B1_A1_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016421
IS_A1_B1_A1_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015257, upper bound: 0.0016067
IS_A1_B1_A1_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0016067
IS_A1_B1_A1_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015257, upper bound: 0.0016202
IS_A1_B1_A1_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0016202
IS_A1_B1_A1_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015262, upper bound: 0.0016122
IS_A1_B1_A1_B2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015317, upper bound: 0.0016122
IS_A1_B1_A2_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014937, upper bound: 0.0016275
IS_A1_B1_A2_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015033, upper bound: 0.0016274
IS_A1_B1_A2_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014937, upper bound: 0.0016274
IS_A1_B1_A2_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015033, upper bound: 0.0016275
IS_A1_B1_A2_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014936, upper bound: 0.0016183
IS_A1_B1_A2_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015034, upper bound: 0.0016183
IS_A1_B1_A2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014936, upper bound: 0.0016183
IS_A1_B1_A2_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015034, upper bound: 0.0016183
IS_A1_B1_A2_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015387, upper bound: 0.0016067
IS_A1_B1_A2_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016067
IS_A1_B1_A2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015387, upper bound: 0.0016067
IS_A1_B1_A2_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016067
IS_A1_B2_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014703, upper bound: 0.0016846
IS_A1_B2_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014801, upper bound: 0.0016846
IS_A1_B2_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014703, upper bound: 0.0017101
IS_A1_B2_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014801, upper bound: 0.0017101
IS_A1_B2_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014704, upper bound: 0.0016759
IS_A1_B2_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014802, upper bound: 0.0016759
IS_A1_B2_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014704, upper bound: 0.0017019
IS_A1_B2_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014802, upper bound: 0.0017019
IS_A1_B2_A1_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015257, upper bound: 0.0016067
IS_A1_B2_A1_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0016067
IS_A1_B2_A1_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015257, upper bound: 0.0016202
IS_A1_B2_A1_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015315, upper bound: 0.0016201
IS_A1_B2_A1_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015262, upper bound: 0.0016122
IS_A1_B2_A1_B2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015317, upper bound: 0.0016122
IS_A1_B2_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014798, upper bound: 0.0016846
IS_A1_B2_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014911, upper bound: 0.0016846
IS_A1_B2_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014798, upper bound: 0.0016846
IS_A1_B2_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014911, upper bound: 0.0016845
IS_A1_B2_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014803, upper bound: 0.0016759
IS_A1_B2_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014927, upper bound: 0.0016759
IS_A1_B2_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014803, upper bound: 0.0016759
IS_A1_B2_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0014928, upper bound: 0.0016758
IS_A1_B2_A2_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015387, upper bound: 0.0016067
IS_A1_B2_A2_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016067
IS_A1_B2_A2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015387, upper bound: 0.0016067
IS_A1_B2_A2_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0015454, upper bound: 0.0016067
IS_A2_B1_A1_B1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016277, upper bound: 0.0014816
IS_A2_B1_A1_B1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016277, upper bound: 0.0014877
IS_A2_B1_A1_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016277, upper bound: 0.0014937
IS_A2_B1_A1_B1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016277, upper bound: 0.0015033
IS_A2_B1_A1_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0014817
IS_A2_B1_A1_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0014878
IS_A2_B1_A1_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0014937
IS_A2_B1_A1_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0015034
IS_A2_B1_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014670
IS_A2_B1_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014782
IS_A2_B1_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014798
IS_A2_B1_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014911
IS_A2_B1_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014672
IS_A2_B1_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014782
IS_A2_B1_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014803
IS_A2_B1_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014927
IS_A2_B1_A1_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016087, upper bound: 0.0015079
IS_A2_B1_A1_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0015079
IS_A2_B1_A1_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016087, upper bound: 0.0015229
IS_A2_B1_A1_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0015230
IS_A2_B1_A1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016088, upper bound: 0.0014891
IS_A2_B1_A1_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0014891
IS_A2_B1_A1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016088, upper bound: 0.0015043
IS_A2_B1_A1_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016187, upper bound: 0.0015043
IS_A2_B1_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014920
IS_A2_B1_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014984
IS_A2_B1_A1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0015029
IS_A2_B1_A1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0015111
IS_A2_B1_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016665, upper bound: 0.0014806
IS_A2_B1_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014806
IS_A2_B1_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016665, upper bound: 0.0014943
IS_A2_B1_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014943
IS_A2_B1_A2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014816
IS_A2_B1_A2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014877
IS_A2_B1_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014816
IS_A2_B1_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014878
IS_A2_B1_A2_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014817
IS_A2_B1_A2_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014878
IS_A2_B1_A2_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014817
IS_A2_B1_A2_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014878
IS_A2_B1_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
IS_A2_B1_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
IS_A2_B1_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
IS_A2_B1_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
IS_A2_B1_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
IS_A2_B1_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
IS_A2_B1_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
IS_A2_B1_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
IS_A2_B1_A2_B2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0015019
IS_A2_B1_A2_B2_A1_B1_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0015079
IS_A2_B1_A2_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016298, upper bound: 0.0015079
IS_A2_B1_A2_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0015079
IS_A2_B1_A2_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016306, upper bound: 0.0014891
IS_A2_B1_A2_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014891
IS_A2_B1_A2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016306, upper bound: 0.0014891
IS_A2_B1_A2_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016421, upper bound: 0.0014891
IS_A2_B1_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
IS_A2_B1_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
IS_A2_B1_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
IS_A2_B1_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
IS_A2_B1_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0014806
IS_A2_B1_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
IS_A2_B1_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0014806
IS_A2_B1_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
IS_A2_B2_A1_B1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015248
IS_A2_B2_A1_B1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015312
IS_A2_B2_A1_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015387
IS_A2_B2_A1_B1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016102, upper bound: 0.0015454
IS_A2_B2_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014670
IS_A2_B2_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014782
IS_A2_B2_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014798
IS_A2_B2_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014911
IS_A2_B2_A2_B1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0015248
IS_A2_B2_A2_B1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0015312
IS_A2_B2_A2_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0015248
IS_A2_B2_A2_B1_A1_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016201, upper bound: 0.0015312
IS_A2_B2_A2_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015255
IS_A2_B2_A2_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015316
IS_A2_B2_A2_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015255
IS_A2_B2_A2_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015316
IS_A2_B2_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
IS_A2_B2_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
IS_A2_B2_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
IS_A2_B2_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
IS_A2_B2_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
IS_A2_B2_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
IS_A2_B2_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
IS_A2_B2_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
IS_A2_B2_A2_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016087, upper bound: 0.0015570
IS_A2_B2_A2_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015570
IS_A2_B2_A2_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016087, upper bound: 0.0015570
IS_A2_B2_A2_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015570
IS_A2_B2_A2_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016090, upper bound: 0.0015344
IS_A2_B2_A2_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015344
IS_A2_B2_A2_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016090, upper bound: 0.0015344
IS_A2_B2_A2_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016122, upper bound: 0.0015344
IS_A2_B2_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
IS_A2_B2_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
IS_A2_B2_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016860, upper bound: 0.0014984
IS_A2_B2_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
IS_A2_B2_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016877, upper bound: 0.0014806
IS_A2_B2_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
IS_A2_B2_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0016877, upper bound: 0.0014806
IS_A2_B2_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.48
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033783, 0.0050740, 0.0032983, 0.0053729, -0.0015003, 0.0012762
1: 0.0018104, 0.0020553, 0.0017988, 0.0020985, -0.0002168, 0.0001844
2: 0.0115546, 0.0124921, 0.0113893, 0.0125363, -0.0007056, 0.0008295
3: -0.0027302, -0.0017606, -0.0029011, -0.0017148, -0.0007298, 0.0008579
4: -0.0021310, -0.0010814, -0.0021806, -0.0008964, -0.0009287, 0.0007900
5: 0.0051367, 0.0061300, 0.0049616, 0.0061769, -0.0007476, 0.0008789
6: -0.0019194, 0.0020218, -0.0026140, 0.0022078, -0.0029663, 0.0034872
7: -0.0053103, 0.0000573, -0.0055635, 0.0010034, -0.0047492, 0.0040399
8: 0.9854732, 0.9892542, 0.9852948, 0.9899207, -0.0033455, 0.0028458
9: -0.0061330, -0.0027008, -0.0067379, -0.0025389, -0.0025832, 0.0030368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014842, upper bound: 0.0016490
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014842, upper bound: 0.0016490
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033967, 0.0051195, 0.0033134, 0.0053692, -0.0014895, 0.0013093
1: 0.0018130, 0.0020619, 0.0018010, 0.0020980, -0.0002152, 0.0001892
2: 0.0115294, 0.0124819, 0.0113914, 0.0125280, -0.0007239, 0.0008235
3: -0.0027562, -0.0017711, -0.0028990, -0.0017235, -0.0007487, 0.0008517
4: -0.0021196, -0.0010532, -0.0021712, -0.0008987, -0.0009220, 0.0008105
5: 0.0051101, 0.0061192, 0.0049638, 0.0061680, -0.0007670, 0.0008725
6: -0.0020251, 0.0019790, -0.0026054, 0.0021727, -0.0030433, 0.0034619
7: -0.0052520, 0.0002013, -0.0055157, 0.0009917, -0.0047148, 0.0041447
8: 0.9855142, 0.9893556, 0.9853284, 0.9899125, -0.0033212, 0.0029196
9: -0.0062250, -0.0027381, -0.0067305, -0.0025694, -0.0026502, 0.0030148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016490
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016490
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0033783, 0.0050740, 0.0032317, 0.0054135, -0.0015707, 0.0013760
1: 0.0018104, 0.0020553, 0.0017892, 0.0021044, -0.0002269, 0.0001988
2: 0.0115546, 0.0124921, 0.0113669, 0.0125732, -0.0007608, 0.0008684
3: -0.0027302, -0.0017606, -0.0029243, -0.0016767, -0.0007868, 0.0008982
4: -0.0021310, -0.0010814, -0.0022218, -0.0008712, -0.0009723, 0.0008518
5: 0.0051367, 0.0061300, 0.0049378, 0.0062159, -0.0008061, 0.0009201
6: -0.0019194, 0.0020218, -0.0027086, 0.0023627, -0.0031982, 0.0036508
7: -0.0053103, 0.0000573, -0.0057745, 0.0011321, -0.0049721, 0.0043557
8: 0.9854732, 0.9892542, 0.9851462, 0.9900114, -0.0035025, 0.0030682
9: -0.0061330, -0.0027008, -0.0068203, -0.0024040, -0.0027852, 0.0031793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014724, upper bound: 0.0016876
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014724, upper bound: 0.0016876
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033967, 0.0051195, 0.0032468, 0.0054095, -0.0015595, 0.0014080
1: 0.0018130, 0.0020619, 0.0017914, 0.0021038, -0.0002253, 0.0002034
2: 0.0115294, 0.0124819, 0.0113691, 0.0125648, -0.0007784, 0.0008622
3: -0.0027562, -0.0017711, -0.0029220, -0.0016853, -0.0008051, 0.0008918
4: -0.0021196, -0.0010532, -0.0022125, -0.0008737, -0.0009654, 0.0008716
5: 0.0051101, 0.0061192, 0.0049402, 0.0062071, -0.0008248, 0.0009136
6: -0.0020251, 0.0019790, -0.0026992, 0.0023276, -0.0032726, 0.0036248
7: -0.0052520, 0.0002013, -0.0057267, 0.0011194, -0.0049366, 0.0044570
8: 0.9855142, 0.9893556, 0.9851799, 0.9900024, -0.0034775, 0.0031396
9: -0.0062250, -0.0027381, -0.0068121, -0.0024345, -0.0028499, 0.0031566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014826, upper bound: 0.0016876
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014826, upper bound: 0.0016876
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0033783, 0.0050740, 0.0032168, 0.0054818, -0.0016491, 0.0013939
1: 0.0018104, 0.0020553, 0.0017870, 0.0021143, -0.0002382, 0.0002014
2: 0.0115546, 0.0124921, 0.0113291, 0.0125814, -0.0007706, 0.0009117
3: -0.0027302, -0.0017606, -0.0029633, -0.0016682, -0.0007970, 0.0009430
4: -0.0021310, -0.0010814, -0.0022311, -0.0008290, -0.0010208, 0.0008628
5: 0.0051367, 0.0061300, 0.0048978, 0.0062247, -0.0008165, 0.0009660
6: -0.0019194, 0.0020218, -0.0028671, 0.0023974, -0.0032397, 0.0038329
7: -0.0053103, 0.0000573, -0.0058217, 0.0013481, -0.0052201, 0.0044122
8: 0.9854732, 0.9892542, 0.9851129, 0.9901636, -0.0036771, 0.0031080
9: -0.0061330, -0.0027008, -0.0069584, -0.0023738, -0.0028213, 0.0033379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014703, upper bound: 0.0017101
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014703, upper bound: 0.0017101
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033967, 0.0051195, 0.0032328, 0.0054776, -0.0016378, 0.0014250
1: 0.0018130, 0.0020619, 0.0017894, 0.0021137, -0.0002366, 0.0002059
2: 0.0115294, 0.0124819, 0.0113314, 0.0125725, -0.0007879, 0.0009055
3: -0.0027562, -0.0017711, -0.0029610, -0.0016774, -0.0008149, 0.0009365
4: -0.0021196, -0.0010532, -0.0022211, -0.0008315, -0.0010138, 0.0008821
5: 0.0051101, 0.0061192, 0.0049003, 0.0062153, -0.0008348, 0.0009594
6: -0.0020251, 0.0019790, -0.0028575, 0.0023600, -0.0033122, 0.0038068
7: -0.0052520, 0.0002013, -0.0057708, 0.0013350, -0.0051845, 0.0045109
8: 0.9855142, 0.9893556, 0.9851488, 0.9901542, -0.0036520, 0.0031776
9: -0.0062250, -0.0027381, -0.0069500, -0.0024063, -0.0028844, 0.0033151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014801, upper bound: 0.0017101
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0014801, upper bound: 0.0017101
time: 0.80 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.50 seconds
IS_A1_B1_A1_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014842, upper bound: 0.0016490
IS_A1_B1_A1_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014842, upper bound: 0.0016490
IS_A1_B1_A1_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016490
IS_A1_B1_A1_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014899, upper bound: 0.0016490
IS_A1_B2_A1_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014724, upper bound: 0.0016876
IS_A1_B2_A1_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014724, upper bound: 0.0016876
IS_A1_B2_A1_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014826, upper bound: 0.0016876
IS_A1_B2_A1_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014826, upper bound: 0.0016876
IS_A1_B2_A1_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014703, upper bound: 0.0017101
IS_A1_B2_A1_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014703, upper bound: 0.0017101
IS_A1_B2_A1_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014801, upper bound: 0.0017101
IS_A1_B2_A1_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 3.50
Output dim: 8, lower bound: -0.0014801, upper bound: 0.0017101
IS_A1_B2_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014704, upper bound: 0.0016759
IS_A1_B2_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014802, upper bound: 0.0016759
IS_A1_B2_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014704, upper bound: 0.0017019
IS_A1_B2_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014802, upper bound: 0.0017019
IS_A1_B2_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014798, upper bound: 0.0016846
IS_A1_B2_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014911, upper bound: 0.0016846
IS_A1_B2_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014798, upper bound: 0.0016846
IS_A1_B2_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014911, upper bound: 0.0016845
IS_A1_B2_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014803, upper bound: 0.0016759
IS_A1_B2_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014927, upper bound: 0.0016759
IS_A1_B2_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014803, upper bound: 0.0016759
IS_A1_B2_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0014928, upper bound: 0.0016758
IS_A2_B1_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014670
IS_A2_B1_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014782
IS_A2_B1_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014798
IS_A2_B1_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014911
IS_A2_B1_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014672
IS_A2_B1_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014782
IS_A2_B1_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014803
IS_A2_B1_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016762, upper bound: 0.0014927
IS_A2_B1_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014920
IS_A2_B1_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014984
IS_A2_B1_A1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0015029
IS_A2_B1_A1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0015111
IS_A2_B1_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016665, upper bound: 0.0014806
IS_A2_B1_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014806
IS_A2_B1_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016665, upper bound: 0.0014943
IS_A2_B1_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016764, upper bound: 0.0014943
IS_A2_B1_A2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014816
IS_A2_B1_A2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014877
IS_A2_B1_A2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014816
IS_A2_B1_A2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016490, upper bound: 0.0014878
IS_A2_B1_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
IS_A2_B1_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
IS_A2_B1_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
IS_A2_B1_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
IS_A2_B1_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
IS_A2_B1_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
IS_A2_B1_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
IS_A2_B1_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
IS_A2_B1_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
IS_A2_B1_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
IS_A2_B1_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
IS_A2_B1_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
IS_A2_B1_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0014806
IS_A2_B1_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
IS_A2_B1_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016876, upper bound: 0.0014806
IS_A2_B1_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
IS_A2_B2_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014670
IS_A2_B2_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014782
IS_A2_B2_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014798
IS_A2_B2_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016847, upper bound: 0.0014911
IS_A2_B2_A2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
IS_A2_B2_A2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
IS_A2_B2_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014670
IS_A2_B2_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0014782
IS_A2_B2_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
IS_A2_B2_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
IS_A2_B2_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014672
IS_A2_B2_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014782
IS_A2_B2_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014920
IS_A2_B2_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
IS_A2_B2_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016860, upper bound: 0.0014984
IS_A2_B2_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014984
IS_A2_B2_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016877, upper bound: 0.0014806
IS_A2_B2_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806
IS_A2_B2_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0016877, upper bound: 0.0014806
IS_A2_B2_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 8, lower bound: -0.0017019, upper bound: 0.0014806

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.58 + 597.73 = 601.31 seconds
