## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01246608


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0078559, 0.0189545, 0.0078559, 0.0189545, -0.0110986, 0.0110986)
1: (-0.0040650, 0.0013262, -0.0040650, 0.0013262, -0.0053912, 0.0053912)
2: (0.0033398, 0.0093792, 0.0033398, 0.0093792, -0.0060394, 0.0060394)
3: (-0.0010515, 0.0039196, -0.0010515, 0.0039196, -0.0049711, 0.0049711)
4: (-0.0054231, -0.0011749, -0.0054231, -0.0011749, -0.0039527, 0.0039527)
5: (-0.0006676, 0.0041297, -0.0006676, 0.0041297, -0.0047973, 0.0047973)
6: (-0.0065855, 0.0016182, -0.0065855, 0.0016182, -0.0082037, 0.0082037)
7: (-0.0271360, -0.0026791, -0.0271360, -0.0026791, -0.0221395, 0.0221395)
8: (0.9708539, 0.9942262, 0.9708539, 0.9942262, -0.0233723, 0.0233723)
9: (-0.0061610, 0.0100897, -0.0061610, 0.0100897, -0.0153951, 0.0153951)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.19 + 2.42 = 3.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0138512, upper bound: 0.0138512

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132636, upper bound: 0.0135384
time: 1.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135950, upper bound: 0.0135950
time: 1.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.27 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.27
Output dim: 8, lower bound: -0.0132636, upper bound: 0.0135384
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.27
Output dim: 8, lower bound: -0.0135950, upper bound: 0.0135950

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0077550, 0.0186224, 0.0079721, 0.0188417, -0.0110868, 0.0106503
1: -0.0040875, 0.0011669, -0.0040317, 0.0012513, -0.0053388, 0.0051986
2: 0.0034511, 0.0094544, 0.0033775, 0.0092980, -0.0058469, 0.0060769
3: -0.0010872, 0.0038478, -0.0009994, 0.0038843, -0.0049715, 0.0048472
4: -0.0051469, -0.0011445, -0.0053317, -0.0011851, -0.0036345, 0.0038491
5: -0.0005563, 0.0041725, -0.0006290, 0.0040765, -0.0046328, 0.0048015
6: -0.0066295, 0.0013309, -0.0065812, 0.0014832, -0.0081126, 0.0079120
7: -0.0255020, -0.0025031, -0.0265932, -0.0027414, -0.0203622, 0.0215953
8: 0.9723470, 0.9943814, 0.9713595, 0.9941511, -0.0218041, 0.0230219
9: -0.0062749, 0.0090190, -0.0061175, 0.0097313, -0.0150160, 0.0141941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130232, upper bound: 0.0132989
time: 1.24 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130232, upper bound: 0.0133153
time: 1.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0079324, 0.0188500, 0.0078611, 0.0189473, -0.0110149, 0.0109889
1: -0.0040432, 0.0012711, -0.0040635, 0.0013223, -0.0053655, 0.0053345
2: 0.0033735, 0.0093257, 0.0033421, 0.0093756, -0.0060020, 0.0059835
3: -0.0010155, 0.0038963, -0.0010490, 0.0039180, -0.0049335, 0.0049453
4: -0.0053598, -0.0011820, -0.0054188, -0.0011754, -0.0037848, 0.0039409
5: -0.0006193, 0.0040946, -0.0006643, 0.0041274, -0.0047467, 0.0047589
6: -0.0065813, 0.0015237, -0.0065852, 0.0016116, -0.0081929, 0.0081089
7: -0.0267701, -0.0027223, -0.0271110, -0.0026820, -0.0211142, 0.0220798
8: 0.9712029, 0.9941757, 0.9708782, 0.9942228, -0.0230200, 0.0232975
9: -0.0061311, 0.0098465, -0.0061590, 0.0100730, -0.0153469, 0.0148006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135384, upper bound: 0.0132636
time: 1.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0135384, upper bound: 0.0135950
time: 1.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.49 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 8, lower bound: -0.0130232, upper bound: 0.0132989
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 8, lower bound: -0.0130232, upper bound: 0.0133153
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 8, lower bound: -0.0135384, upper bound: 0.0132636
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.49
Output dim: 8, lower bound: -0.0135384, upper bound: 0.0135950

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0078785, 0.0186157, 0.0080238, 0.0188209, -0.0109424, 0.0105920
1: -0.0040533, 0.0011382, -0.0039908, 0.0012203, -0.0052736, 0.0051290
2: 0.0034545, 0.0093678, 0.0033878, 0.0092739, -0.0058194, 0.0059799
3: -0.0010415, 0.0037955, -0.0010245, 0.0037412, -0.0047827, 0.0048200
4: -0.0050993, -0.0011542, -0.0051945, -0.0011491, -0.0035882, 0.0036883
5: -0.0005550, 0.0041164, -0.0006317, 0.0040425, -0.0045975, 0.0047481
6: -0.0066259, 0.0011734, -0.0066404, 0.0011065, -0.0077324, 0.0078138
7: -0.0252407, -0.0025639, -0.0258322, -0.0025426, -0.0201164, 0.0207437
8: 0.9725391, 0.9943007, 0.9719166, 0.9942814, -0.0217423, 0.0223842
9: -0.0062317, 0.0088537, -0.0062438, 0.0092531, -0.0144578, 0.0140462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130682
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130682
time: 1.19 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0078036, 0.0186193, 0.0081538, 0.0188298, -0.0110262, 0.0104655
1: -0.0040738, 0.0011539, -0.0039813, 0.0012043, -0.0052781, 0.0051353
2: 0.0034528, 0.0094203, 0.0033841, 0.0091707, -0.0057179, 0.0060363
3: -0.0010687, 0.0038230, -0.0009308, 0.0037883, -0.0048570, 0.0047539
4: -0.0051226, -0.0011483, -0.0052388, -0.0011993, -0.0035948, 0.0037145
5: -0.0005559, 0.0041503, -0.0006275, 0.0039943, -0.0045502, 0.0047777
6: -0.0066277, 0.0012608, -0.0065744, 0.0012183, -0.0078460, 0.0078352
7: -0.0253678, -0.0025270, -0.0260756, -0.0028303, -0.0201487, 0.0208778
8: 0.9724487, 0.9943504, 0.9717454, 0.9940353, -0.0215866, 0.0226051
9: -0.0062581, 0.0089341, -0.0060542, 0.0094042, -0.0145613, 0.0140401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130872
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130872
time: 1.16 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0079324, 0.0188500, 0.0077550, 0.0186224, -0.0106900, 0.0110951
1: -0.0040432, 0.0012711, -0.0040875, 0.0011669, -0.0052101, 0.0053585
2: 0.0033735, 0.0093257, 0.0034511, 0.0094544, -0.0060809, 0.0058746
3: -0.0010155, 0.0038963, -0.0010872, 0.0038478, -0.0048633, 0.0049835
4: -0.0053598, -0.0011820, -0.0051469, -0.0011445, -0.0039029, 0.0036378
5: -0.0006193, 0.0040946, -0.0005563, 0.0041725, -0.0047918, 0.0046509
6: -0.0065813, 0.0015237, -0.0066295, 0.0013309, -0.0079121, 0.0081532
7: -0.0267701, -0.0027223, -0.0255020, -0.0025031, -0.0219035, 0.0203782
8: 0.9712029, 0.9941757, 0.9723470, 0.9943814, -0.0231786, 0.0218287
9: -0.0061311, 0.0098465, -0.0062749, 0.0090190, -0.0142087, 0.0152085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132989, upper bound: 0.0130232
time: 1.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133153, upper bound: 0.0130232
time: 1.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0079324, 0.0188500, 0.0079324, 0.0188500, -0.0109176, 0.0109176
1: -0.0040432, 0.0012711, -0.0040432, 0.0012711, -0.0053142, 0.0053142
2: 0.0033735, 0.0093257, 0.0033735, 0.0093257, -0.0059522, 0.0059522
3: -0.0010155, 0.0038963, -0.0010155, 0.0038963, -0.0049118, 0.0049118
4: -0.0053598, -0.0011820, -0.0053598, -0.0011820, -0.0037772, 0.0037772
5: -0.0006193, 0.0040946, -0.0006193, 0.0040946, -0.0047139, 0.0047139
6: -0.0065813, 0.0015237, -0.0065813, 0.0015237, -0.0081050, 0.0081050
7: -0.0267701, -0.0027223, -0.0267701, -0.0027223, -0.0210762, 0.0210762
8: 0.9712029, 0.9941757, 0.9712029, 0.9941757, -0.0229729, 0.0229729
9: -0.0061311, 0.0098465, -0.0061311, 0.0098465, -0.0147694, 0.0147694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0132989, upper bound: 0.0131365
time: 1.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0133153, upper bound: 0.0131365
time: 1.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.73
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130682
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.73
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130682
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.73
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130872
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.73
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130872
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.73
Output dim: 8, lower bound: -0.0132989, upper bound: 0.0130232
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.73
Output dim: 8, lower bound: -0.0133153, upper bound: 0.0130232
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.73
Output dim: 8, lower bound: -0.0132989, upper bound: 0.0131365
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.73
Output dim: 8, lower bound: -0.0133153, upper bound: 0.0131365

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0082006, 0.0185661, 0.0081087, 0.0188077, -0.0106070, 0.0104575
1: -0.0039645, 0.0010567, -0.0039686, 0.0011990, -0.0051635, 0.0050253
2: 0.0034723, 0.0091412, 0.0033926, 0.0092140, -0.0057417, 0.0057486
3: -0.0009118, 0.0037130, -0.0009904, 0.0037184, -0.0046303, 0.0047034
4: -0.0050060, -0.0011893, -0.0051689, -0.0011581, -0.0034780, 0.0036223
5: -0.0005472, 0.0039708, -0.0006295, 0.0040046, -0.0045518, 0.0046003
6: -0.0066061, 0.0008964, -0.0066353, 0.0010330, -0.0076391, 0.0075317
7: -0.0247159, -0.0027709, -0.0256880, -0.0025970, -0.0194459, 0.0203547
8: 0.9729701, 0.9940659, 0.9720328, 0.9942193, -0.0212492, 0.0220332
9: -0.0060926, 0.0085146, -0.0062072, 0.0091601, -0.0141982, 0.0136433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127816
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130682
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0068974, 0.0185647, 0.0081367, 0.0188027, -0.0119053, 0.0104280
1: -0.0042330, 0.0012826, -0.0039622, 0.0011922, -0.0054252, 0.0052448
2: 0.0034733, 0.0100833, 0.0033944, 0.0091942, -0.0057209, 0.0066889
3: -0.0014806, 0.0037621, -0.0009798, 0.0037148, -0.0051954, 0.0047420
4: -0.0050100, -0.0009890, -0.0051618, -0.0011608, -0.0034941, 0.0038593
5: -0.0005496, 0.0045201, -0.0006288, 0.0039926, -0.0045422, 0.0051489
6: -0.0067770, 0.0012379, -0.0066340, 0.0010130, -0.0077900, 0.0078719
7: -0.0247173, -0.0015889, -0.0256506, -0.0026133, -0.0195370, 0.0217165
8: 0.9729388, 0.9952987, 0.9720643, 0.9941990, -0.0212602, 0.0232345
9: -0.0068935, 0.0085286, -0.0061956, 0.0091358, -0.0151193, 0.0136928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127816
time: 1.29 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130682
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0081232, 0.0185697, 0.0082367, 0.0188166, -0.0106934, 0.0103330
1: -0.0039856, 0.0010719, -0.0039593, 0.0011832, -0.0051687, 0.0050312
2: 0.0034706, 0.0091953, 0.0033888, 0.0091120, -0.0056414, 0.0058064
3: -0.0009399, 0.0037392, -0.0008974, 0.0037650, -0.0047049, 0.0046367
4: -0.0050280, -0.0011835, -0.0052127, -0.0012079, -0.0034852, 0.0036486
5: -0.0005482, 0.0040053, -0.0006253, 0.0039573, -0.0045055, 0.0046306
6: -0.0066081, 0.0009823, -0.0065693, 0.0011448, -0.0077528, 0.0075516
7: -0.0248342, -0.0027339, -0.0259294, -0.0028823, -0.0194769, 0.0204866
8: 0.9728831, 0.9941160, 0.9718627, 0.9939756, -0.0210925, 0.0222533
9: -0.0061191, 0.0085898, -0.0060186, 0.0093100, -0.0143034, 0.0136369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127847
time: 1.64 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130872
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068217, 0.0185682, 0.0082633, 0.0188113, -0.0119896, 0.0103049
1: -0.0042533, 0.0013018, -0.0039533, 0.0011765, -0.0054297, 0.0052552
2: 0.0034716, 0.0101364, 0.0033906, 0.0090932, -0.0056215, 0.0067458
3: -0.0015084, 0.0037888, -0.0008873, 0.0037615, -0.0052699, 0.0046760
4: -0.0050347, -0.0009833, -0.0052067, -0.0012105, -0.0035015, 0.0038855
5: -0.0005506, 0.0045540, -0.0006245, 0.0039458, -0.0044964, 0.0051785
6: -0.0067789, 0.0013238, -0.0065678, 0.0011245, -0.0079035, 0.0078916
7: -0.0248467, -0.0015529, -0.0258971, -0.0028986, -0.0195750, 0.0218533
8: 0.9728436, 0.9953477, 0.9718916, 0.9939551, -0.0211115, 0.0234562
9: -0.0069195, 0.0086104, -0.0060069, 0.0092886, -0.0152213, 0.0136902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127847
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130872
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0079843, 0.0188288, 0.0078785, 0.0186157, -0.0106314, 0.0109503
1: -0.0040020, 0.0012391, -0.0040533, 0.0011382, -0.0051403, 0.0052925
2: 0.0033841, 0.0093012, 0.0034545, 0.0093678, -0.0059836, 0.0058467
3: -0.0010403, 0.0037526, -0.0010415, 0.0037955, -0.0048358, 0.0047942
4: -0.0052220, -0.0011460, -0.0050993, -0.0011542, -0.0037364, 0.0035917
5: -0.0006211, 0.0040606, -0.0005550, 0.0041164, -0.0047375, 0.0046155
6: -0.0066404, 0.0011438, -0.0066259, 0.0011734, -0.0078139, 0.0077696
7: -0.0260004, -0.0025233, -0.0252407, -0.0025639, -0.0210082, 0.0201336
8: 0.9717661, 0.9943070, 0.9725391, 0.9943007, -0.0225346, 0.0217679
9: -0.0062574, 0.0093624, -0.0062317, 0.0088537, -0.0140613, 0.0146264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126931
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0127670
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081132, 0.0188382, 0.0078036, 0.0186193, -0.0105060, 0.0110345
1: -0.0039931, 0.0012240, -0.0040738, 0.0011539, -0.0051470, 0.0052978
2: 0.0033799, 0.0091990, 0.0034528, 0.0094203, -0.0060405, 0.0057462
3: -0.0009472, 0.0038002, -0.0010687, 0.0038230, -0.0047702, 0.0048689
4: -0.0052701, -0.0011960, -0.0051226, -0.0011483, -0.0037666, 0.0035983
5: -0.0006178, 0.0040126, -0.0005559, 0.0041503, -0.0047680, 0.0045685
6: -0.0065746, 0.0012590, -0.0066277, 0.0012608, -0.0078354, 0.0078867
7: -0.0262704, -0.0028103, -0.0253678, -0.0025270, -0.0211729, 0.0201653
8: 0.9715734, 0.9940606, 0.9724487, 0.9943504, -0.0227770, 0.0216119
9: -0.0060684, 0.0095303, -0.0062581, 0.0089341, -0.0140552, 0.0147472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126931
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0127670
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0079843, 0.0188288, 0.0080469, 0.0188428, -0.0108584, 0.0107820
1: -0.0040020, 0.0012391, -0.0040117, 0.0012438, -0.0052459, 0.0052509
2: 0.0033841, 0.0093012, 0.0033772, 0.0092456, -0.0058614, 0.0059241
3: -0.0010403, 0.0037526, -0.0009731, 0.0038421, -0.0048824, 0.0047257
4: -0.0052220, -0.0011460, -0.0053111, -0.0011907, -0.0036166, 0.0037348
5: -0.0006211, 0.0040606, -0.0006180, 0.0040429, -0.0046640, 0.0046786
6: -0.0066404, 0.0011438, -0.0065778, 0.0013648, -0.0080052, 0.0077215
7: -0.0260004, -0.0025233, -0.0264972, -0.0027772, -0.0202287, 0.0208644
8: 0.9717661, 0.9943070, 0.9714053, 0.9941028, -0.0223367, 0.0229017
9: -0.0062574, 0.0093624, -0.0060917, 0.0096739, -0.0146330, 0.0142085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128625
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0129170
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081132, 0.0188382, 0.0079788, 0.0188470, -0.0107337, 0.0108593
1: -0.0039931, 0.0012240, -0.0040303, 0.0012589, -0.0052520, 0.0052542
2: 0.0033799, 0.0091990, 0.0033752, 0.0092932, -0.0059133, 0.0058238
3: -0.0009472, 0.0038002, -0.0009980, 0.0038711, -0.0048183, 0.0047982
4: -0.0052701, -0.0011960, -0.0053364, -0.0011856, -0.0036448, 0.0037384
5: -0.0006178, 0.0040126, -0.0006189, 0.0040735, -0.0046913, 0.0046315
6: -0.0065746, 0.0012590, -0.0065796, 0.0014540, -0.0080286, 0.0078386
7: -0.0262704, -0.0028103, -0.0266397, -0.0027450, -0.0203622, 0.0208732
8: 0.9715734, 0.9940606, 0.9712989, 0.9941462, -0.0225728, 0.0227617
9: -0.0060684, 0.0095303, -0.0061149, 0.0097640, -0.0146213, 0.0143210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128625
time: 1.81 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0129170
time: 1.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.44 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127816
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130682
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127816
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130682
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0127847
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0126931, upper bound: 0.0130872
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0127847
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0127670, upper bound: 0.0130872
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0126931
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0130682, upper bound: 0.0127670
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0126931
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0130872, upper bound: 0.0127670
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0128625
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0131515, upper bound: 0.0129170
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0128625
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 8, lower bound: -0.0131682, upper bound: 0.0129170

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0082006, 0.0185661, 0.0078612, 0.0185899, -0.0103893, 0.0107049
1: -0.0039645, 0.0010567, -0.0040267, 0.0011182, -0.0050827, 0.0050834
2: 0.0034723, 0.0091412, 0.0034657, 0.0093932, -0.0059209, 0.0056754
3: -0.0009118, 0.0037130, -0.0010985, 0.0036865, -0.0045984, 0.0048115
4: -0.0050060, -0.0011893, -0.0049890, -0.0011051, -0.0035070, 0.0034283
5: -0.0005472, 0.0039708, -0.0005575, 0.0041116, -0.0046589, 0.0045282
6: -0.0066061, 0.0008964, -0.0066939, 0.0008959, -0.0075020, 0.0075903
7: -0.0247159, -0.0027709, -0.0246223, -0.0022947, -0.0196581, 0.0192613
8: 0.9729701, 0.9940659, 0.9730005, 0.9945157, -0.0215456, 0.0210654
9: -0.0060926, 0.0085146, -0.0064020, 0.0084639, -0.0134669, 0.0137673

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0127816
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0127816
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0082006, 0.0185661, 0.0080695, 0.0188152, -0.0106146, 0.0104966
1: -0.0039645, 0.0010567, -0.0039797, 0.0012173, -0.0051818, 0.0050364
2: 0.0034723, 0.0091412, 0.0033890, 0.0092414, -0.0057691, 0.0057521
3: -0.0009118, 0.0037130, -0.0010061, 0.0037298, -0.0046416, 0.0047192
4: -0.0050060, -0.0011893, -0.0051954, -0.0011550, -0.0034816, 0.0036697
5: -0.0005472, 0.0039708, -0.0006189, 0.0040227, -0.0045699, 0.0045896
6: -0.0066061, 0.0008964, -0.0066353, 0.0010696, -0.0076756, 0.0075317
7: -0.0247159, -0.0027709, -0.0258514, -0.0025773, -0.0194631, 0.0206163
8: 0.9729701, 0.9940659, 0.9718860, 0.9942452, -0.0212751, 0.0221799
9: -0.0060926, 0.0085146, -0.0062211, 0.0092663, -0.0143646, 0.0136587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130682
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130682
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068974, 0.0185647, 0.0078817, 0.0185843, -0.0116869, 0.0106829
1: -0.0042330, 0.0012826, -0.0040217, 0.0011110, -0.0053440, 0.0053043
2: 0.0034733, 0.0100833, 0.0034678, 0.0093788, -0.0059055, 0.0066156
3: -0.0014806, 0.0037621, -0.0010904, 0.0036834, -0.0051641, 0.0048525
4: -0.0050100, -0.0009890, -0.0049796, -0.0011070, -0.0035240, 0.0036626
5: -0.0005496, 0.0045201, -0.0005567, 0.0041026, -0.0046522, 0.0050768
6: -0.0067770, 0.0012379, -0.0066925, 0.0008779, -0.0076549, 0.0079305
7: -0.0247173, -0.0015889, -0.0245771, -0.0023076, -0.0197557, 0.0206072
8: 0.9729388, 0.9952987, 0.9730402, 0.9945000, -0.0215612, 0.0222585
9: -0.0068935, 0.0085286, -0.0063927, 0.0084343, -0.0143775, 0.0138212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0127816
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0127816
time: 1.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0068974, 0.0185647, 0.0080966, 0.0188115, -0.0119141, 0.0104681
1: -0.0042330, 0.0012826, -0.0039736, 0.0012120, -0.0054450, 0.0052562
2: 0.0034733, 0.0100833, 0.0033906, 0.0092222, -0.0057489, 0.0066928
3: -0.0014806, 0.0037621, -0.0009961, 0.0037263, -0.0052069, 0.0047582
4: -0.0050100, -0.0009890, -0.0051905, -0.0011575, -0.0034978, 0.0039078
5: -0.0005496, 0.0045201, -0.0006182, 0.0040111, -0.0045607, 0.0051383
6: -0.0067770, 0.0012379, -0.0066341, 0.0010511, -0.0078281, 0.0078720
7: -0.0247173, -0.0015889, -0.0258253, -0.0025931, -0.0195546, 0.0219851
8: 0.9729388, 0.9952987, 0.9719090, 0.9942254, -0.0212867, 0.0233898
9: -0.0068935, 0.0085286, -0.0062098, 0.0092495, -0.0152898, 0.0137086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0130682
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0130682
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0081232, 0.0185697, 0.0080239, 0.0185974, -0.0104742, 0.0105458
1: -0.0039856, 0.0010719, -0.0040117, 0.0010961, -0.0050817, 0.0050836
2: 0.0034706, 0.0091953, 0.0034624, 0.0092661, -0.0057955, 0.0057329
3: -0.0009399, 0.0037392, -0.0009836, 0.0037321, -0.0046719, 0.0047228
4: -0.0050280, -0.0011835, -0.0050278, -0.0011686, -0.0034972, 0.0034511
5: -0.0005482, 0.0040053, -0.0005526, 0.0040497, -0.0045980, 0.0045579
6: -0.0066081, 0.0009823, -0.0066172, 0.0009942, -0.0076023, 0.0075996
7: -0.0248342, -0.0027339, -0.0248398, -0.0026507, -0.0195795, 0.0193677
8: 0.9728831, 0.9941160, 0.9728556, 0.9941984, -0.0213153, 0.0212604
9: -0.0061191, 0.0085898, -0.0061724, 0.0085985, -0.0135594, 0.0136989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0127847
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0127710
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0081232, 0.0185697, 0.0081960, 0.0188245, -0.0107013, 0.0103737
1: -0.0039856, 0.0010719, -0.0039710, 0.0012024, -0.0051880, 0.0050429
2: 0.0034706, 0.0091953, 0.0033848, 0.0091405, -0.0056698, 0.0058105
3: -0.0009399, 0.0037392, -0.0009138, 0.0037767, -0.0047165, 0.0046530
4: -0.0050280, -0.0011835, -0.0052431, -0.0012046, -0.0034888, 0.0037007
5: -0.0005482, 0.0040053, -0.0006156, 0.0039755, -0.0045238, 0.0046209
6: -0.0066081, 0.0009823, -0.0065694, 0.0011851, -0.0077931, 0.0075517
7: -0.0248342, -0.0027339, -0.0261207, -0.0028622, -0.0194936, 0.0207834
8: 0.9728831, 0.9941160, 0.9716934, 0.9940014, -0.0211183, 0.0224226
9: -0.0061191, 0.0085898, -0.0060329, 0.0094339, -0.0144900, 0.0136522

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130872
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130642
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068217, 0.0185682, 0.0080430, 0.0185916, -0.0117699, 0.0105252
1: -0.0042533, 0.0013018, -0.0040073, 0.0010892, -0.0053425, 0.0053092
2: 0.0034716, 0.0101364, 0.0034645, 0.0092525, -0.0057809, 0.0066719
3: -0.0015084, 0.0037888, -0.0009760, 0.0037293, -0.0052377, 0.0047647
4: -0.0050347, -0.0009833, -0.0050205, -0.0011705, -0.0035144, 0.0036856
5: -0.0005506, 0.0045540, -0.0005517, 0.0040415, -0.0045921, 0.0051057
6: -0.0067789, 0.0013238, -0.0066158, 0.0009769, -0.0077559, 0.0079395
7: -0.0248467, -0.0015529, -0.0247996, -0.0026624, -0.0196853, 0.0207252
8: 0.9728436, 0.9953477, 0.9728891, 0.9941831, -0.0213395, 0.0224587
9: -0.0069195, 0.0086104, -0.0061640, 0.0085722, -0.0144692, 0.0137566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0127847
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0127710
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0068217, 0.0185682, 0.0082217, 0.0188207, -0.0119990, 0.0103465
1: -0.0042533, 0.0013018, -0.0039652, 0.0011971, -0.0054504, 0.0052670
2: 0.0034716, 0.0101364, 0.0033862, 0.0091222, -0.0056506, 0.0067502
3: -0.0015084, 0.0037888, -0.0009042, 0.0037733, -0.0052817, 0.0046930
4: -0.0050347, -0.0009833, -0.0052392, -0.0012072, -0.0035051, 0.0039387
5: -0.0005506, 0.0045540, -0.0006149, 0.0039644, -0.0045151, 0.0051689
6: -0.0067789, 0.0013238, -0.0065680, 0.0011661, -0.0079450, 0.0078918
7: -0.0248467, -0.0015529, -0.0260971, -0.0028778, -0.0195924, 0.0221483
8: 0.9728436, 0.9953477, 0.9717151, 0.9939815, -0.0211379, 0.0236326
9: -0.0069195, 0.0086104, -0.0060216, 0.0094185, -0.0154079, 0.0137059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0130872
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0130642
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0080695, 0.0188152, 0.0082006, 0.0185661, -0.0104966, 0.0106146
1: -0.0039797, 0.0012173, -0.0039645, 0.0010567, -0.0050364, 0.0051818
2: 0.0033890, 0.0092414, 0.0034723, 0.0091412, -0.0057521, 0.0057691
3: -0.0010061, 0.0037298, -0.0009118, 0.0037130, -0.0047192, 0.0046416
4: -0.0051954, -0.0011550, -0.0050060, -0.0011893, -0.0036697, 0.0034816
5: -0.0006189, 0.0040227, -0.0005472, 0.0039708, -0.0045896, 0.0045699
6: -0.0066353, 0.0010696, -0.0066061, 0.0008964, -0.0075317, 0.0076756
7: -0.0258514, -0.0025773, -0.0247159, -0.0027709, -0.0206163, 0.0194631
8: 0.9718860, 0.9942452, 0.9729701, 0.9940659, -0.0221799, 0.0212751
9: -0.0062211, 0.0092663, -0.0060926, 0.0085146, -0.0136587, 0.0143646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126931
time: 1.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126931
time: 1.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0080966, 0.0188115, 0.0068974, 0.0185647, -0.0104681, 0.0119141
1: -0.0039736, 0.0012120, -0.0042330, 0.0012826, -0.0052562, 0.0054450
2: 0.0033906, 0.0092222, 0.0034733, 0.0100833, -0.0066928, 0.0057489
3: -0.0009961, 0.0037263, -0.0014806, 0.0037621, -0.0047582, 0.0052069
4: -0.0051905, -0.0011575, -0.0050100, -0.0009890, -0.0039078, 0.0034978
5: -0.0006182, 0.0040111, -0.0005496, 0.0045201, -0.0051383, 0.0045607
6: -0.0066341, 0.0010511, -0.0067770, 0.0012379, -0.0078720, 0.0078281
7: -0.0258253, -0.0025931, -0.0247173, -0.0015889, -0.0219851, 0.0195546
8: 0.9719090, 0.9942254, 0.9729388, 0.9952987, -0.0233898, 0.0212867
9: -0.0062098, 0.0092495, -0.0068935, 0.0085286, -0.0137086, 0.0152898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
time: 1.74 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0081960, 0.0188245, 0.0081232, 0.0185697, -0.0103737, 0.0107013
1: -0.0039710, 0.0012024, -0.0039856, 0.0010719, -0.0050429, 0.0051880
2: 0.0033848, 0.0091405, 0.0034706, 0.0091953, -0.0058105, 0.0056698
3: -0.0009138, 0.0037767, -0.0009399, 0.0037392, -0.0046530, 0.0047165
4: -0.0052431, -0.0012046, -0.0050280, -0.0011835, -0.0037007, 0.0034888
5: -0.0006156, 0.0039755, -0.0005482, 0.0040053, -0.0046209, 0.0045238
6: -0.0065694, 0.0011851, -0.0066081, 0.0009823, -0.0075517, 0.0077931
7: -0.0261207, -0.0028622, -0.0248342, -0.0027339, -0.0207834, 0.0194936
8: 0.9716934, 0.9940014, 0.9728831, 0.9941160, -0.0224226, 0.0211183
9: -0.0060329, 0.0094339, -0.0061191, 0.0085898, -0.0136522, 0.0144900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126931
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126931
time: 1.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0082217, 0.0188207, 0.0068217, 0.0185682, -0.0103465, 0.0119990
1: -0.0039652, 0.0011971, -0.0042533, 0.0013018, -0.0052670, 0.0054504
2: 0.0033862, 0.0091222, 0.0034716, 0.0101364, -0.0067502, 0.0056506
3: -0.0009042, 0.0037733, -0.0015084, 0.0037888, -0.0046930, 0.0052817
4: -0.0052392, -0.0012072, -0.0050347, -0.0009833, -0.0039387, 0.0035051
5: -0.0006149, 0.0039644, -0.0005506, 0.0045540, -0.0051689, 0.0045151
6: -0.0065680, 0.0011661, -0.0067789, 0.0013238, -0.0078918, 0.0079450
7: -0.0260971, -0.0028778, -0.0248467, -0.0015529, -0.0221483, 0.0195924
8: 0.9717151, 0.9939815, 0.9728436, 0.9953477, -0.0236326, 0.0211379
9: -0.0060216, 0.0094185, -0.0069195, 0.0086104, -0.0137059, 0.0154079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127670
time: 1.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127670
time: 1.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0080695, 0.0188152, 0.0083568, 0.0187905, -0.0107210, 0.0104585
1: -0.0039797, 0.0012173, -0.0039289, 0.0011624, -0.0051421, 0.0051462
2: 0.0033890, 0.0092414, 0.0033961, 0.0090266, -0.0056375, 0.0058453
3: -0.0010061, 0.0037298, -0.0008474, 0.0037561, -0.0047622, 0.0045771
4: -0.0051954, -0.0011550, -0.0052092, -0.0012242, -0.0035537, 0.0036221
5: -0.0006189, 0.0040227, -0.0006098, 0.0039046, -0.0045235, 0.0046325
6: -0.0066353, 0.0010696, -0.0065582, 0.0010844, -0.0077197, 0.0076277
7: -0.0258514, -0.0025773, -0.0259289, -0.0029789, -0.0198536, 0.0201829
8: 0.9718860, 0.9942452, 0.9718639, 0.9938715, -0.0219854, 0.0223812
9: -0.0062211, 0.0092663, -0.0059534, 0.0093075, -0.0142228, 0.0139643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0128625
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0128625
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0080966, 0.0188115, 0.0069692, 0.0187955, -0.0106989, 0.0118423
1: -0.0039736, 0.0012120, -0.0042135, 0.0013943, -0.0053679, 0.0054255
2: 0.0033906, 0.0092222, 0.0033944, 0.0100326, -0.0066421, 0.0058278
3: -0.0009961, 0.0037263, -0.0014595, 0.0038055, -0.0048016, 0.0051858
4: -0.0051905, -0.0011575, -0.0052279, -0.0010157, -0.0038036, 0.0036450
5: -0.0006182, 0.0040111, -0.0006129, 0.0044893, -0.0051075, 0.0046240
6: -0.0066341, 0.0010511, -0.0067396, 0.0014358, -0.0080699, 0.0077907
7: -0.0258253, -0.0025931, -0.0260001, -0.0017402, -0.0212852, 0.0203113
8: 0.9719090, 0.9942254, 0.9717728, 0.9951893, -0.0232803, 0.0224526
9: -0.0062098, 0.0092495, -0.0068028, 0.0093666, -0.0143013, 0.0149405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0129170
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0129170
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0081960, 0.0188245, 0.0082872, 0.0187946, -0.0105986, 0.0105373
1: -0.0039710, 0.0012024, -0.0039479, 0.0011771, -0.0051481, 0.0051503
2: 0.0033848, 0.0091405, 0.0033941, 0.0090752, -0.0056904, 0.0057464
3: -0.0009138, 0.0037767, -0.0008729, 0.0037831, -0.0046969, 0.0046495
4: -0.0052431, -0.0012046, -0.0052345, -0.0012188, -0.0035820, 0.0036262
5: -0.0006156, 0.0039755, -0.0006108, 0.0039357, -0.0045513, 0.0045863
6: -0.0065694, 0.0011851, -0.0065601, 0.0011731, -0.0077426, 0.0077451
7: -0.0261207, -0.0028622, -0.0260690, -0.0029448, -0.0199867, 0.0201954
8: 0.9716934, 0.9940014, 0.9717584, 0.9939172, -0.0222238, 0.0222430
9: -0.0060329, 0.0094339, -0.0059781, 0.0093962, -0.0142127, 0.0140758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128625
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128625
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0082217, 0.0188207, 0.0069017, 0.0187996, -0.0105780, 0.0119190
1: -0.0039652, 0.0011971, -0.0042315, 0.0014111, -0.0053763, 0.0054286
2: 0.0033862, 0.0091222, 0.0033924, 0.0100798, -0.0066936, 0.0057299
3: -0.0009042, 0.0037733, -0.0014844, 0.0038328, -0.0047371, 0.0052577
4: -0.0052392, -0.0012072, -0.0052537, -0.0010103, -0.0038298, 0.0036491
5: -0.0006149, 0.0039644, -0.0006138, 0.0045196, -0.0051345, 0.0045783
6: -0.0065680, 0.0011661, -0.0067415, 0.0015231, -0.0080911, 0.0079075
7: -0.0260971, -0.0028778, -0.0261429, -0.0017066, -0.0214256, 0.0203228
8: 0.9717151, 0.9939815, 0.9716690, 0.9952324, -0.0235173, 0.0223125
9: -0.0060216, 0.0094185, -0.0068265, 0.0094569, -0.0142924, 0.0150418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129170
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129169
time: 1.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0127816
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0127816
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130682
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130682
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0127816
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0127816
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0130682
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0130682
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0127847
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0127710
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130872
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130642
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0127847
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0127710
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0130872
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0130642
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126931
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126931
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126931
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126931
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127670
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127670
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0128625
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0128625
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0129170
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0130572, upper bound: 0.0129170
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128625
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128625
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129170
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.07
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129169

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0078612, 0.0185899, -0.0104970, 0.0106921
1: -0.0039673, 0.0010577, -0.0040267, 0.0011182, -0.0050855, 0.0050844
2: 0.0034788, 0.0092297, 0.0034657, 0.0093932, -0.0059144, 0.0057640
3: -0.0010035, 0.0036382, -0.0010985, 0.0036865, -0.0046901, 0.0047367
4: -0.0049200, -0.0011302, -0.0049890, -0.0011051, -0.0034178, 0.0032394
5: -0.0005518, 0.0040088, -0.0005575, 0.0041116, -0.0046634, 0.0045663
6: -0.0066797, 0.0007150, -0.0066939, 0.0008959, -0.0075756, 0.0074089
7: -0.0242347, -0.0024411, -0.0246223, -0.0022947, -0.0191839, 0.0169624
8: 0.9733176, 0.9943494, 0.9730005, 0.9945157, -0.0211981, 0.0213488
9: -0.0063017, 0.0082136, -0.0064020, 0.0084639, -0.0127849, 0.0134599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127172
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127816
time: 1.50 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0078612, 0.0185899, -0.0103335, 0.0107000
1: -0.0039485, 0.0010368, -0.0040267, 0.0011182, -0.0050668, 0.0050635
2: 0.0034755, 0.0091027, 0.0034657, 0.0093932, -0.0059178, 0.0056369
3: -0.0008895, 0.0036775, -0.0010985, 0.0036865, -0.0045761, 0.0047759
4: -0.0049572, -0.0011940, -0.0049890, -0.0011051, -0.0034766, 0.0032091
5: -0.0005470, 0.0039456, -0.0005575, 0.0041116, -0.0046586, 0.0045031
6: -0.0066028, 0.0008008, -0.0066939, 0.0008959, -0.0074986, 0.0074948
7: -0.0244393, -0.0027993, -0.0246223, -0.0022947, -0.0194695, 0.0168424
8: 0.9731785, 0.9940299, 0.9730005, 0.9945157, -0.0213372, 0.0210294
9: -0.0060726, 0.0083402, -0.0064020, 0.0084639, -0.0126567, 0.0136448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127172
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127816
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0080695, 0.0188152, -0.0107223, 0.0104838
1: -0.0039673, 0.0010577, -0.0039797, 0.0012173, -0.0051846, 0.0050374
2: 0.0034788, 0.0092297, 0.0033890, 0.0092414, -0.0057626, 0.0058407
3: -0.0010035, 0.0036382, -0.0010061, 0.0037298, -0.0047333, 0.0046443
4: -0.0049200, -0.0011302, -0.0051954, -0.0011550, -0.0033925, 0.0034962
5: -0.0005518, 0.0040088, -0.0006189, 0.0040227, -0.0045745, 0.0046277
6: -0.0066797, 0.0007150, -0.0066353, 0.0010696, -0.0077492, 0.0073503
7: -0.0242347, -0.0024411, -0.0258514, -0.0025773, -0.0189888, 0.0185128
8: 0.9733176, 0.9943494, 0.9718860, 0.9942452, -0.0209275, 0.0224633
9: -0.0063017, 0.0082136, -0.0062211, 0.0092663, -0.0137317, 0.0133514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0129857
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130682
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0080695, 0.0188152, -0.0105588, 0.0104916
1: -0.0039485, 0.0010368, -0.0039797, 0.0012173, -0.0051658, 0.0050166
2: 0.0034755, 0.0091027, 0.0033890, 0.0092414, -0.0057659, 0.0057136
3: -0.0008895, 0.0036775, -0.0010061, 0.0037298, -0.0046193, 0.0046836
4: -0.0049572, -0.0011940, -0.0051954, -0.0011550, -0.0034512, 0.0034660
5: -0.0005470, 0.0039456, -0.0006189, 0.0040227, -0.0045697, 0.0045645
6: -0.0066028, 0.0008008, -0.0066353, 0.0010696, -0.0076723, 0.0074362
7: -0.0244393, -0.0027993, -0.0258514, -0.0025773, -0.0192745, 0.0183928
8: 0.9731785, 0.9940299, 0.9718860, 0.9942452, -0.0210667, 0.0221439
9: -0.0060726, 0.0083402, -0.0062211, 0.0092663, -0.0136036, 0.0135363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0129857
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130682
time: 1.77 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0078817, 0.0185843, -0.0117756, 0.0106707
1: -0.0042306, 0.0012934, -0.0040217, 0.0011110, -0.0053416, 0.0053150
2: 0.0034796, 0.0101545, 0.0034678, 0.0093788, -0.0058992, 0.0066867
3: -0.0015647, 0.0036803, -0.0010904, 0.0036834, -0.0052481, 0.0047706
4: -0.0049252, -0.0009405, -0.0049796, -0.0011070, -0.0034325, 0.0034828
5: -0.0005544, 0.0045500, -0.0005567, 0.0041026, -0.0046570, 0.0051068
6: -0.0068399, 0.0010449, -0.0066925, 0.0008779, -0.0077178, 0.0077375
7: -0.0242427, -0.0013184, -0.0245771, -0.0023076, -0.0192571, 0.0183925
8: 0.9732838, 0.9955425, 0.9730402, 0.9945000, -0.0212162, 0.0225022
9: -0.0070691, 0.0082311, -0.0063927, 0.0084343, -0.0137155, 0.0135068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0127109
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0127109
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0078817, 0.0185843, -0.0116216, 0.0106779
1: -0.0042152, 0.0012591, -0.0040217, 0.0011110, -0.0053262, 0.0052808
2: 0.0034765, 0.0100375, 0.0034678, 0.0093788, -0.0059023, 0.0065698
3: -0.0014544, 0.0037225, -0.0010904, 0.0036834, -0.0051378, 0.0048129
4: -0.0049666, -0.0009945, -0.0049796, -0.0011070, -0.0034931, 0.0036560
5: -0.0005493, 0.0044909, -0.0005567, 0.0041026, -0.0046518, 0.0050477
6: -0.0067734, 0.0011315, -0.0066925, 0.0008779, -0.0076513, 0.0078241
7: -0.0244683, -0.0016217, -0.0245771, -0.0023076, -0.0195731, 0.0205747
8: 0.9731277, 0.9952570, 0.9730402, 0.9945000, -0.0213723, 0.0222168
9: -0.0068703, 0.0083713, -0.0063927, 0.0084343, -0.0143502, 0.0137012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0127109
time: 1.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0127109
time: 1.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0080966, 0.0188115, -0.0120027, 0.0104558
1: -0.0042306, 0.0012934, -0.0039736, 0.0012120, -0.0054426, 0.0052669
2: 0.0034796, 0.0101545, 0.0033906, 0.0092222, -0.0057426, 0.0067639
3: -0.0015647, 0.0036803, -0.0009961, 0.0037263, -0.0052909, 0.0046764
4: -0.0049252, -0.0009405, -0.0051905, -0.0011575, -0.0034062, 0.0037439
5: -0.0005544, 0.0045500, -0.0006182, 0.0040111, -0.0045655, 0.0051683
6: -0.0068399, 0.0010449, -0.0066341, 0.0010511, -0.0078910, 0.0076790
7: -0.0242427, -0.0013184, -0.0258253, -0.0025931, -0.0190561, 0.0199558
8: 0.9732838, 0.9955425, 0.9719090, 0.9942254, -0.0209417, 0.0236335
9: -0.0070691, 0.0082311, -0.0062098, 0.0092495, -0.0146849, 0.0133942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129542
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129542
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0080966, 0.0188115, -0.0118487, 0.0104631
1: -0.0042152, 0.0012591, -0.0039736, 0.0012120, -0.0054271, 0.0052327
2: 0.0034765, 0.0100375, 0.0033906, 0.0092222, -0.0057457, 0.0066470
3: -0.0014544, 0.0037225, -0.0009961, 0.0037263, -0.0051807, 0.0047186
4: -0.0049666, -0.0009945, -0.0051905, -0.0011575, -0.0034668, 0.0039012
5: -0.0005493, 0.0044909, -0.0006182, 0.0040111, -0.0045603, 0.0051092
6: -0.0067734, 0.0011315, -0.0066341, 0.0010511, -0.0078245, 0.0077656
7: -0.0244683, -0.0016217, -0.0258253, -0.0025931, -0.0193720, 0.0219527
8: 0.9731277, 0.9952570, 0.9719090, 0.9942254, -0.0210977, 0.0233480
9: -0.0068703, 0.0083713, -0.0062098, 0.0092495, -0.0152625, 0.0135886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129542
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129542
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0080239, 0.0185974, -0.0105045, 0.0105294
1: -0.0039673, 0.0010577, -0.0040117, 0.0010961, -0.0050634, 0.0050694
2: 0.0034788, 0.0092297, 0.0034624, 0.0092661, -0.0057873, 0.0057673
3: -0.0010035, 0.0036382, -0.0009836, 0.0037321, -0.0047356, 0.0046218
4: -0.0049200, -0.0011302, -0.0050278, -0.0011686, -0.0033782, 0.0033097
5: -0.0005518, 0.0040088, -0.0005526, 0.0040497, -0.0046016, 0.0045614
6: -0.0066797, 0.0007150, -0.0066172, 0.0009942, -0.0076739, 0.0073322
7: -0.0242347, -0.0024411, -0.0248398, -0.0026507, -0.0189579, 0.0174004
8: 0.9733176, 0.9943494, 0.9728556, 0.9941984, -0.0208808, 0.0214938
9: -0.0063017, 0.0082136, -0.0061724, 0.0085985, -0.0130029, 0.0132973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127106
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127847
time: 1.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0080239, 0.0185974, -0.0103410, 0.0105372
1: -0.0039485, 0.0010368, -0.0040117, 0.0010961, -0.0050446, 0.0050486
2: 0.0034755, 0.0091027, 0.0034624, 0.0092661, -0.0057906, 0.0056403
3: -0.0008895, 0.0036775, -0.0009836, 0.0037321, -0.0046216, 0.0046611
4: -0.0049572, -0.0011940, -0.0050278, -0.0011686, -0.0033928, 0.0032165
5: -0.0005470, 0.0039456, -0.0005526, 0.0040497, -0.0045967, 0.0044982
6: -0.0066028, 0.0008008, -0.0066172, 0.0009942, -0.0075970, 0.0074181
7: -0.0244393, -0.0027993, -0.0248398, -0.0026507, -0.0190180, 0.0168031
8: 0.9731785, 0.9940299, 0.9728556, 0.9941984, -0.0210199, 0.0211744
9: -0.0060726, 0.0083402, -0.0061724, 0.0085985, -0.0126879, 0.0133548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127007
time: 1.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127710
time: 1.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0081960, 0.0188245, -0.0107315, 0.0103573
1: -0.0039673, 0.0010577, -0.0039710, 0.0012024, -0.0051697, 0.0050287
2: 0.0034788, 0.0092297, 0.0033848, 0.0091405, -0.0056616, 0.0058449
3: -0.0010035, 0.0036382, -0.0009138, 0.0037767, -0.0047802, 0.0045520
4: -0.0049200, -0.0011302, -0.0052431, -0.0012046, -0.0033698, 0.0035759
5: -0.0005518, 0.0040088, -0.0006156, 0.0039755, -0.0045274, 0.0046244
6: -0.0066797, 0.0007150, -0.0065694, 0.0011851, -0.0078647, 0.0072844
7: -0.0242347, -0.0024411, -0.0261207, -0.0028622, -0.0188720, 0.0189906
8: 0.9733176, 0.9943494, 0.9716934, 0.9940014, -0.0206838, 0.0226560
9: -0.0063017, 0.0082136, -0.0060329, 0.0094339, -0.0139911, 0.0132505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130010
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130872
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0081960, 0.0188245, -0.0105680, 0.0103651
1: -0.0039485, 0.0010368, -0.0039710, 0.0012024, -0.0051509, 0.0050078
2: 0.0034755, 0.0091027, 0.0033848, 0.0091405, -0.0056650, 0.0057179
3: -0.0008895, 0.0036775, -0.0009138, 0.0037767, -0.0046662, 0.0045913
4: -0.0049572, -0.0011940, -0.0052431, -0.0012046, -0.0033822, 0.0034780
5: -0.0005470, 0.0039456, -0.0006156, 0.0039755, -0.0045225, 0.0045612
6: -0.0066028, 0.0008008, -0.0065694, 0.0011851, -0.0077878, 0.0073703
7: -0.0244393, -0.0027993, -0.0261207, -0.0028622, -0.0189150, 0.0183734
8: 0.9731785, 0.9940299, 0.9716934, 0.9940014, -0.0208229, 0.0223365
9: -0.0060726, 0.0083402, -0.0060329, 0.0094339, -0.0136552, 0.0132981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0129817
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130642
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0080430, 0.0185916, -0.0117829, 0.0105094
1: -0.0042306, 0.0012934, -0.0040073, 0.0010892, -0.0053199, 0.0053007
2: 0.0034796, 0.0101545, 0.0034645, 0.0092525, -0.0057729, 0.0066900
3: -0.0015647, 0.0036803, -0.0009760, 0.0037293, -0.0052940, 0.0046562
4: -0.0049252, -0.0009405, -0.0050205, -0.0011705, -0.0033927, 0.0035538
5: -0.0005544, 0.0045500, -0.0005517, 0.0040415, -0.0045958, 0.0051018
6: -0.0068399, 0.0010449, -0.0066158, 0.0009769, -0.0078168, 0.0076607
7: -0.0242427, -0.0013184, -0.0247996, -0.0026624, -0.0190315, 0.0188346
8: 0.9732838, 0.9955425, 0.9728891, 0.9941831, -0.0208994, 0.0226534
9: -0.0070691, 0.0082311, -0.0061640, 0.0085722, -0.0139372, 0.0133441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0126971
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0126971
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0080430, 0.0185916, -0.0116289, 0.0105166
1: -0.0042152, 0.0012591, -0.0040073, 0.0010892, -0.0053044, 0.0052664
2: 0.0034765, 0.0100375, 0.0034645, 0.0092525, -0.0057761, 0.0065730
3: -0.0014544, 0.0037225, -0.0009760, 0.0037293, -0.0051837, 0.0046985
4: -0.0049666, -0.0009945, -0.0050205, -0.0011705, -0.0034093, 0.0036718
5: -0.0005493, 0.0044909, -0.0005517, 0.0040415, -0.0045907, 0.0050427
6: -0.0067734, 0.0011315, -0.0066158, 0.0009769, -0.0077504, 0.0077473
7: -0.0244683, -0.0016217, -0.0247996, -0.0026624, -0.0191183, 0.0206564
8: 0.9731277, 0.9952570, 0.9728891, 0.9941831, -0.0210554, 0.0223680
9: -0.0068703, 0.0083713, -0.0061640, 0.0085722, -0.0144104, 0.0134090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0126904
time: 1.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0126904
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0082217, 0.0188207, -0.0120119, 0.0103307
1: -0.0042306, 0.0012934, -0.0039652, 0.0011971, -0.0054278, 0.0052586
2: 0.0034796, 0.0101545, 0.0033862, 0.0091222, -0.0056426, 0.0067683
3: -0.0015647, 0.0036803, -0.0009042, 0.0037733, -0.0053380, 0.0045845
4: -0.0049252, -0.0009405, -0.0052392, -0.0012072, -0.0033835, 0.0038238
5: -0.0005544, 0.0045500, -0.0006149, 0.0039644, -0.0045188, 0.0051649
6: -0.0068399, 0.0010449, -0.0065680, 0.0011661, -0.0080060, 0.0076130
7: -0.0242427, -0.0013184, -0.0260971, -0.0028778, -0.0189385, 0.0204444
8: 0.9732838, 0.9955425, 0.9717151, 0.9939815, -0.0206977, 0.0238274
9: -0.0070691, 0.0082311, -0.0060216, 0.0094185, -0.0149447, 0.0132934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129607
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129607
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0082217, 0.0188207, -0.0118579, 0.0103380
1: -0.0042152, 0.0012591, -0.0039652, 0.0011971, -0.0054123, 0.0052243
2: 0.0034765, 0.0100375, 0.0033862, 0.0091222, -0.0056458, 0.0066513
3: -0.0014544, 0.0037225, -0.0009042, 0.0037733, -0.0052277, 0.0046267
4: -0.0049666, -0.0009945, -0.0052392, -0.0012072, -0.0033979, 0.0039249
5: -0.0005493, 0.0044909, -0.0006149, 0.0039644, -0.0045137, 0.0051058
6: -0.0067734, 0.0011315, -0.0065680, 0.0011661, -0.0079395, 0.0076996
7: -0.0244683, -0.0016217, -0.0260971, -0.0028778, -0.0190086, 0.0220795
8: 0.9731277, 0.9952570, 0.9717151, 0.9939815, -0.0208538, 0.0235419
9: -0.0068703, 0.0083713, -0.0060216, 0.0094185, -0.0153491, 0.0133483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129490
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129490
time: 1.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0082006, 0.0185661, -0.0102606, 0.0105761
1: -0.0039174, 0.0011571, -0.0039645, 0.0010567, -0.0049741, 0.0051216
2: 0.0034030, 0.0090757, 0.0034723, 0.0091412, -0.0057382, 0.0056033
3: -0.0009109, 0.0036682, -0.0009118, 0.0037130, -0.0046239, 0.0045800
4: -0.0051203, -0.0011813, -0.0050060, -0.0011893, -0.0035934, 0.0034524
5: -0.0006129, 0.0039166, -0.0005472, 0.0039708, -0.0045836, 0.0044639
6: -0.0066209, 0.0008647, -0.0066061, 0.0008964, -0.0075173, 0.0074708
7: -0.0254298, -0.0027351, -0.0247159, -0.0027709, -0.0201535, 0.0192974
8: 0.9722265, 0.9940638, 0.9729701, 0.9940659, -0.0218394, 0.0210937
9: -0.0061136, 0.0089945, -0.0060926, 0.0085146, -0.0135385, 0.0140894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129857, upper bound: 0.0126848
time: 1.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129857, upper bound: 0.0126931
time: 1.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0082006, 0.0185661, -0.0116227, 0.0105813
1: -0.0041974, 0.0013851, -0.0039645, 0.0010567, -0.0052540, 0.0053496
2: 0.0034013, 0.0100584, 0.0034723, 0.0091412, -0.0057399, 0.0065860
3: -0.0015091, 0.0037173, -0.0009118, 0.0037130, -0.0052221, 0.0046291
4: -0.0051387, -0.0009777, -0.0050060, -0.0011893, -0.0036257, 0.0037034
5: -0.0006160, 0.0044893, -0.0005472, 0.0039708, -0.0045868, 0.0050365
6: -0.0067939, 0.0012106, -0.0066061, 0.0008964, -0.0076903, 0.0078167
7: -0.0255066, -0.0015269, -0.0247159, -0.0027709, -0.0203115, 0.0206946
8: 0.9721356, 0.9953537, 0.9729701, 0.9940659, -0.0219303, 0.0223836
9: -0.0069351, 0.0090564, -0.0060926, 0.0085146, -0.0145086, 0.0141970

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129857, upper bound: 0.0126848
time: 1.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129857, upper bound: 0.0126931
time: 1.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0068974, 0.0185647, -0.0102592, 0.0118794
1: -0.0039174, 0.0011571, -0.0042330, 0.0012826, -0.0052000, 0.0053901
2: 0.0034030, 0.0090757, 0.0034733, 0.0100833, -0.0066803, 0.0056023
3: -0.0009109, 0.0036682, -0.0014806, 0.0037621, -0.0046730, 0.0051488
4: -0.0051203, -0.0011813, -0.0050100, -0.0009890, -0.0038317, 0.0034759
5: -0.0006129, 0.0039166, -0.0005496, 0.0045201, -0.0051330, 0.0044662
6: -0.0066209, 0.0008647, -0.0067770, 0.0012379, -0.0078588, 0.0076417
7: -0.0254298, -0.0027351, -0.0247173, -0.0015889, -0.0214728, 0.0194191
8: 0.9722265, 0.9940638, 0.9729388, 0.9952987, -0.0230722, 0.0211250
9: -0.0061136, 0.0089945, -0.0068935, 0.0085286, -0.0136122, 0.0150176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127573
time: 1.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0068974, 0.0185647, -0.0116213, 0.0118846
1: -0.0041974, 0.0013851, -0.0042330, 0.0012826, -0.0054800, 0.0056181
2: 0.0034013, 0.0100584, 0.0034733, 0.0100833, -0.0066820, 0.0065851
3: -0.0015091, 0.0037173, -0.0014806, 0.0037621, -0.0052712, 0.0051979
4: -0.0051387, -0.0009777, -0.0050100, -0.0009890, -0.0038470, 0.0037102
5: -0.0006160, 0.0044893, -0.0005496, 0.0045201, -0.0051361, 0.0050389
6: -0.0067939, 0.0012106, -0.0067770, 0.0012379, -0.0080318, 0.0079876
7: -0.0255066, -0.0015269, -0.0247173, -0.0015889, -0.0217041, 0.0208923
8: 0.9721356, 0.9953537, 0.9729388, 0.9952987, -0.0231631, 0.0224149
9: -0.0069351, 0.0090564, -0.0068935, 0.0085286, -0.0145398, 0.0150825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126848
time: 1.75 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126931
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0081232, 0.0185697, -0.0101412, 0.0106628
1: -0.0039089, 0.0011429, -0.0039856, 0.0010719, -0.0049808, 0.0051285
2: 0.0033987, 0.0089764, 0.0034706, 0.0091953, -0.0057965, 0.0055057
3: -0.0008197, 0.0037132, -0.0009399, 0.0037392, -0.0045589, 0.0046531
4: -0.0051675, -0.0012301, -0.0050280, -0.0011835, -0.0036248, 0.0034606
5: -0.0006096, 0.0038723, -0.0005482, 0.0040053, -0.0046149, 0.0044205
6: -0.0065548, 0.0009793, -0.0066081, 0.0009823, -0.0075371, 0.0075874
7: -0.0256950, -0.0030151, -0.0248342, -0.0027339, -0.0203240, 0.0193329
8: 0.9720360, 0.9938253, 0.9728831, 0.9941160, -0.0220800, 0.0209422
9: -0.0059282, 0.0091599, -0.0061191, 0.0085898, -0.0135367, 0.0142164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130010, upper bound: 0.0126848
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130010, upper bound: 0.0126848
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0081232, 0.0185697, -0.0115349, 0.0106680
1: -0.0041962, 0.0013731, -0.0039856, 0.0010719, -0.0052681, 0.0053587
2: 0.0033970, 0.0099868, 0.0034706, 0.0091953, -0.0057983, 0.0065162
3: -0.0014337, 0.0037640, -0.0009399, 0.0037392, -0.0051730, 0.0047039
4: -0.0051883, -0.0010215, -0.0050280, -0.0011835, -0.0036561, 0.0037151
5: -0.0006126, 0.0044600, -0.0005482, 0.0040053, -0.0046179, 0.0050082
6: -0.0067363, 0.0013294, -0.0066081, 0.0009823, -0.0077186, 0.0079375
7: -0.0257798, -0.0017740, -0.0248342, -0.0027339, -0.0204773, 0.0207431
8: 0.9719387, 0.9951462, 0.9728831, 0.9941160, -0.0221773, 0.0222631
9: -0.0067789, 0.0092272, -0.0061191, 0.0085898, -0.0145310, 0.0143202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130010, upper bound: 0.0126848
time: 2.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130010, upper bound: 0.0126848
time: 2.04 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0068217, 0.0185682, -0.0101396, 0.0119643
1: -0.0039089, 0.0011429, -0.0042533, 0.0013018, -0.0052108, 0.0053962
2: 0.0033987, 0.0089764, 0.0034716, 0.0101364, -0.0067376, 0.0055048
3: -0.0008197, 0.0037132, -0.0015084, 0.0037888, -0.0046084, 0.0052216
4: -0.0051675, -0.0012301, -0.0050347, -0.0009833, -0.0038632, 0.0034842
5: -0.0006096, 0.0038723, -0.0005506, 0.0045540, -0.0051636, 0.0044229
6: -0.0065548, 0.0009793, -0.0067789, 0.0013238, -0.0078786, 0.0077583
7: -0.0256950, -0.0030151, -0.0248467, -0.0015529, -0.0216435, 0.0194624
8: 0.9720360, 0.9938253, 0.9728436, 0.9953477, -0.0233117, 0.0209817
9: -0.0059282, 0.0091599, -0.0069195, 0.0086104, -0.0136140, 0.0151409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0068217, 0.0185682, -0.0115334, 0.0119695
1: -0.0041962, 0.0013731, -0.0042533, 0.0013018, -0.0054981, 0.0056264
2: 0.0033970, 0.0099868, 0.0034716, 0.0101364, -0.0067394, 0.0065152
3: -0.0014337, 0.0037640, -0.0015084, 0.0037888, -0.0052225, 0.0052724
4: -0.0051883, -0.0010215, -0.0050347, -0.0009833, -0.0038776, 0.0037225
5: -0.0006126, 0.0044600, -0.0005506, 0.0045540, -0.0051666, 0.0050106
6: -0.0067363, 0.0013294, -0.0067789, 0.0013238, -0.0080600, 0.0081084
7: -0.0257798, -0.0017740, -0.0248467, -0.0015529, -0.0218742, 0.0209455
8: 0.9719387, 0.9951462, 0.9728436, 0.9953477, -0.0234090, 0.0223026
9: -0.0067789, 0.0092272, -0.0069195, 0.0086104, -0.0145663, 0.0152018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126848
time: 1.75 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126848
time: 1.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0083568, 0.0187905, -0.0104850, 0.0104200
1: -0.0039174, 0.0011571, -0.0039289, 0.0011624, -0.0050798, 0.0050861
2: 0.0034030, 0.0090757, 0.0033961, 0.0090266, -0.0056235, 0.0056795
3: -0.0009109, 0.0036682, -0.0008474, 0.0037561, -0.0046670, 0.0045156
4: -0.0051203, -0.0011813, -0.0052092, -0.0012242, -0.0034787, 0.0035938
5: -0.0006129, 0.0039166, -0.0006098, 0.0039046, -0.0045175, 0.0045265
6: -0.0066209, 0.0008647, -0.0065582, 0.0010844, -0.0077053, 0.0074228
7: -0.0254298, -0.0027351, -0.0259289, -0.0029789, -0.0193964, 0.0200219
8: 0.9722265, 0.9940638, 0.9718639, 0.9938715, -0.0216449, 0.0221999
9: -0.0061136, 0.0089945, -0.0059534, 0.0093075, -0.0141063, 0.0136921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130822, upper bound: 0.0128568
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130822, upper bound: 0.0128625
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0083568, 0.0187905, -0.0118471, 0.0104252
1: -0.0041974, 0.0013851, -0.0039289, 0.0011624, -0.0053597, 0.0053141
2: 0.0034013, 0.0100584, 0.0033961, 0.0090266, -0.0056253, 0.0066622
3: -0.0015091, 0.0037173, -0.0008474, 0.0037561, -0.0052652, 0.0045647
4: -0.0051387, -0.0009777, -0.0052092, -0.0012242, -0.0035081, 0.0038437
5: -0.0006160, 0.0044893, -0.0006098, 0.0039046, -0.0045207, 0.0050992
6: -0.0067939, 0.0012106, -0.0065582, 0.0010844, -0.0078783, 0.0077688
7: -0.0255066, -0.0015269, -0.0259289, -0.0029789, -0.0195430, 0.0214119
8: 0.9721356, 0.9953537, 0.9718639, 0.9938715, -0.0217358, 0.0234898
9: -0.0069351, 0.0090564, -0.0059534, 0.0093075, -0.0150735, 0.0137932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130822, upper bound: 0.0128568
time: 1.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130822, upper bound: 0.0128625
time: 1.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0069692, 0.0187955, -0.0104900, 0.0118076
1: -0.0039174, 0.0011571, -0.0042135, 0.0013943, -0.0053117, 0.0053707
2: 0.0034030, 0.0090757, 0.0033944, 0.0100326, -0.0066296, 0.0056812
3: -0.0009109, 0.0036682, -0.0014595, 0.0038055, -0.0047164, 0.0051277
4: -0.0051203, -0.0011813, -0.0052279, -0.0010157, -0.0037295, 0.0036238
5: -0.0006129, 0.0039166, -0.0006129, 0.0044893, -0.0051022, 0.0045295
6: -0.0066209, 0.0008647, -0.0067396, 0.0014358, -0.0080567, 0.0076043
7: -0.0254298, -0.0027351, -0.0260001, -0.0017402, -0.0207862, 0.0201786
8: 0.9722265, 0.9940638, 0.9717728, 0.9951893, -0.0229628, 0.0222909
9: -0.0061136, 0.0089945, -0.0068028, 0.0093666, -0.0142078, 0.0146734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0129087
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0129170
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0069692, 0.0187955, -0.0118521, 0.0118128
1: -0.0041974, 0.0013851, -0.0042135, 0.0013943, -0.0055916, 0.0055987
2: 0.0034013, 0.0100584, 0.0033944, 0.0100326, -0.0066313, 0.0066640
3: -0.0015091, 0.0037173, -0.0014595, 0.0038055, -0.0053146, 0.0051768
4: -0.0051387, -0.0009777, -0.0052279, -0.0010157, -0.0037426, 0.0038566
5: -0.0006160, 0.0044893, -0.0006129, 0.0044893, -0.0051053, 0.0051022
6: -0.0067939, 0.0012106, -0.0067396, 0.0014358, -0.0082297, 0.0079502
7: -0.0255066, -0.0015269, -0.0260001, -0.0017402, -0.0210063, 0.0216465
8: 0.9721356, 0.9953537, 0.9717728, 0.9951893, -0.0230537, 0.0235808
9: -0.0069351, 0.0090564, -0.0068028, 0.0093666, -0.0151325, 0.0147332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0128568
time: 1.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0128625
time: 1.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0082872, 0.0187946, -0.0103660, 0.0104988
1: -0.0039089, 0.0011429, -0.0039479, 0.0011771, -0.0050860, 0.0050908
2: 0.0033987, 0.0089764, 0.0033941, 0.0090752, -0.0056764, 0.0055823
3: -0.0008197, 0.0037132, -0.0008729, 0.0037831, -0.0046027, 0.0045861
4: -0.0051675, -0.0012301, -0.0052345, -0.0012188, -0.0035067, 0.0035991
5: -0.0006096, 0.0038723, -0.0006108, 0.0039357, -0.0045453, 0.0044830
6: -0.0065548, 0.0009793, -0.0065601, 0.0011731, -0.0077279, 0.0075394
7: -0.0256950, -0.0030151, -0.0260690, -0.0029448, -0.0195268, 0.0200364
8: 0.9720360, 0.9938253, 0.9717584, 0.9939172, -0.0218812, 0.0220669
9: -0.0059282, 0.0091599, -0.0059781, 0.0093962, -0.0141015, 0.0138048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130934, upper bound: 0.0128568
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130934, upper bound: 0.0128568
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0082872, 0.0187946, -0.0117598, 0.0105041
1: -0.0041962, 0.0013731, -0.0039479, 0.0011771, -0.0053733, 0.0053210
2: 0.0033970, 0.0099868, 0.0033941, 0.0090752, -0.0056782, 0.0065927
3: -0.0014337, 0.0037640, -0.0008729, 0.0037831, -0.0052168, 0.0046369
4: -0.0051883, -0.0010215, -0.0052345, -0.0012188, -0.0035367, 0.0038502
5: -0.0006126, 0.0044600, -0.0006108, 0.0039357, -0.0045483, 0.0050707
6: -0.0067363, 0.0013294, -0.0065601, 0.0011731, -0.0079094, 0.0078895
7: -0.0257798, -0.0017740, -0.0260690, -0.0029448, -0.0196845, 0.0214271
8: 0.9719387, 0.9951462, 0.9717584, 0.9939172, -0.0219784, 0.0233877
9: -0.0067789, 0.0092272, -0.0059781, 0.0093962, -0.0150830, 0.0139064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130934, upper bound: 0.0128568
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130934, upper bound: 0.0128568
time: 2.22 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0069017, 0.0187996, -0.0103711, 0.0118843
1: -0.0039089, 0.0011429, -0.0042315, 0.0014111, -0.0053200, 0.0053744
2: 0.0033987, 0.0089764, 0.0033924, 0.0100798, -0.0066810, 0.0055840
3: -0.0008197, 0.0037132, -0.0014844, 0.0038328, -0.0046525, 0.0051977
4: -0.0051675, -0.0012301, -0.0052537, -0.0010103, -0.0037556, 0.0036293
5: -0.0006096, 0.0038723, -0.0006138, 0.0045196, -0.0051292, 0.0044861
6: -0.0065548, 0.0009793, -0.0067415, 0.0015231, -0.0080779, 0.0077208
7: -0.0256950, -0.0030151, -0.0261429, -0.0017066, -0.0209185, 0.0201930
8: 0.9720360, 0.9938253, 0.9716690, 0.9952324, -0.0231964, 0.0221563
9: -0.0059282, 0.0091599, -0.0068265, 0.0094569, -0.0142039, 0.0147752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129087
time: 1.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129087
time: 1.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0069017, 0.0187996, -0.0117648, 0.0118895
1: -0.0041962, 0.0013731, -0.0042315, 0.0014111, -0.0056073, 0.0056046
2: 0.0033970, 0.0099868, 0.0033924, 0.0100798, -0.0066828, 0.0065944
3: -0.0014337, 0.0037640, -0.0014844, 0.0038328, -0.0052666, 0.0052484
4: -0.0051883, -0.0010215, -0.0052537, -0.0010103, -0.0037692, 0.0038639
5: -0.0006126, 0.0044600, -0.0006138, 0.0045196, -0.0051322, 0.0050738
6: -0.0067363, 0.0013294, -0.0067415, 0.0015231, -0.0082593, 0.0080709
7: -0.0257798, -0.0017740, -0.0261429, -0.0017066, -0.0211532, 0.0216575
8: 0.9719387, 0.9951462, 0.9716690, 0.9952324, -0.0232937, 0.0234771
9: -0.0067789, 0.0092272, -0.0068265, 0.0094569, -0.0151441, 0.0148354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128568
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128568
time: 2.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.90 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127172
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127816
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127172
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127816
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0129857
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130682
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0129857
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130682
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0127109
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0127109
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0127109
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0127109
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129542
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129542
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129542
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129542
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127106
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127847
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127007
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126904, upper bound: 0.0127710
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130010
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130872
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0129817
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0126848, upper bound: 0.0130642
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0126971
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0126971
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0126904
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127709, upper bound: 0.0126904
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129607
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129607
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129490
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0127573, upper bound: 0.0129490
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129857, upper bound: 0.0126848
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129857, upper bound: 0.0126931
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129857, upper bound: 0.0126848
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129857, upper bound: 0.0126931
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127573
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0127670
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126848
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129542, upper bound: 0.0126931
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130010, upper bound: 0.0126848
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130010, upper bound: 0.0126848
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130010, upper bound: 0.0126848
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130010, upper bound: 0.0126848
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0127573
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126848
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0129607, upper bound: 0.0126848
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130822, upper bound: 0.0128568
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130822, upper bound: 0.0128625
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130822, upper bound: 0.0128568
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130822, upper bound: 0.0128625
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0129087
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0129170
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0128568
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130573, upper bound: 0.0128625
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130934, upper bound: 0.0128568
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130934, upper bound: 0.0128568
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130934, upper bound: 0.0128568
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130934, upper bound: 0.0128568
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129087
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0129087
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128568
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.90
Output dim: 8, lower bound: -0.0130558, upper bound: 0.0128568

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0080929, 0.0185533, -0.0104604, 0.0104604
1: -0.0039673, 0.0010577, -0.0039673, 0.0010577, -0.0050250, 0.0050250
2: 0.0034788, 0.0092297, 0.0034788, 0.0092297, -0.0057509, 0.0057509
3: -0.0010035, 0.0036382, -0.0010035, 0.0036382, -0.0046417, 0.0046417
4: -0.0049200, -0.0011302, -0.0049200, -0.0011302, -0.0031663, 0.0031663
5: -0.0005518, 0.0040088, -0.0005518, 0.0040088, -0.0045607, 0.0045607
6: -0.0066797, 0.0007150, -0.0066797, 0.0007150, -0.0073947, 0.0073947
7: -0.0242347, -0.0024411, -0.0242347, -0.0024411, -0.0165468, 0.0165468
8: 0.9733176, 0.9943494, 0.9733176, 0.9943494, -0.0210317, 0.0210317
9: -0.0063017, 0.0082136, -0.0063017, 0.0082136, -0.0125206, 0.0125206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0120182
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126003
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0068087, 0.0185524, -0.0104595, 0.0117446
1: -0.0039673, 0.0010577, -0.0042306, 0.0012934, -0.0052607, 0.0052883
2: 0.0034788, 0.0092297, 0.0034796, 0.0101545, -0.0066756, 0.0057501
3: -0.0010035, 0.0036382, -0.0015647, 0.0036803, -0.0046838, 0.0052029
4: -0.0049200, -0.0011302, -0.0049252, -0.0009405, -0.0034122, 0.0031946
5: -0.0005518, 0.0040088, -0.0005544, 0.0045500, -0.0051019, 0.0045632
6: -0.0066797, 0.0007150, -0.0068399, 0.0010449, -0.0077246, 0.0075549
7: -0.0242347, -0.0024411, -0.0242427, -0.0013184, -0.0179779, 0.0167367
8: 0.9733176, 0.9943494, 0.9732838, 0.9955425, -0.0222248, 0.0210656
9: -0.0063017, 0.0082136, -0.0070691, 0.0082311, -0.0126087, 0.0134655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0120273
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126644
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0080929, 0.0185533, -0.0102969, 0.0104682
1: -0.0039485, 0.0010368, -0.0039673, 0.0010577, -0.0050062, 0.0050041
2: 0.0034755, 0.0091027, 0.0034788, 0.0092297, -0.0057543, 0.0056238
3: -0.0008895, 0.0036775, -0.0010035, 0.0036382, -0.0045278, 0.0046810
4: -0.0049572, -0.0011940, -0.0049200, -0.0011302, -0.0032362, 0.0031361
5: -0.0005470, 0.0039456, -0.0005518, 0.0040088, -0.0045558, 0.0044974
6: -0.0066028, 0.0008008, -0.0066797, 0.0007150, -0.0073177, 0.0074805
7: -0.0244393, -0.0027993, -0.0242347, -0.0024411, -0.0169889, 0.0164269
8: 0.9731785, 0.9940299, 0.9733176, 0.9943494, -0.0211709, 0.0207123
9: -0.0060726, 0.0083402, -0.0063017, 0.0082136, -0.0123924, 0.0127398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0120182
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126003
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0068087, 0.0185524, -0.0102959, 0.0117524
1: -0.0039485, 0.0010368, -0.0042306, 0.0012934, -0.0052419, 0.0052675
2: 0.0034755, 0.0091027, 0.0034796, 0.0101545, -0.0066790, 0.0056231
3: -0.0008895, 0.0036775, -0.0015647, 0.0036803, -0.0045698, 0.0052421
4: -0.0049572, -0.0011940, -0.0049252, -0.0009405, -0.0034820, 0.0031644
5: -0.0005470, 0.0039456, -0.0005544, 0.0045500, -0.0050970, 0.0045000
6: -0.0066028, 0.0008008, -0.0068399, 0.0010449, -0.0076477, 0.0076408
7: -0.0244393, -0.0027993, -0.0242427, -0.0013184, -0.0184200, 0.0166168
8: 0.9731785, 0.9940299, 0.9732838, 0.9955425, -0.0223640, 0.0207462
9: -0.0060726, 0.0083402, -0.0070691, 0.0082311, -0.0124806, 0.0136848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0120278
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126644
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0083055, 0.0187768, -0.0106839, 0.0102479
1: -0.0039673, 0.0010577, -0.0039174, 0.0011571, -0.0051244, 0.0049751
2: 0.0034788, 0.0092297, 0.0034030, 0.0090757, -0.0055968, 0.0058267
3: -0.0010035, 0.0036382, -0.0009109, 0.0036682, -0.0046717, 0.0045491
4: -0.0049200, -0.0011302, -0.0051203, -0.0011813, -0.0033633, 0.0034185
5: -0.0005518, 0.0040088, -0.0006129, 0.0039166, -0.0044685, 0.0046217
6: -0.0066797, 0.0007150, -0.0066209, 0.0008647, -0.0075444, 0.0073358
7: -0.0242347, -0.0024411, -0.0254298, -0.0027351, -0.0188232, 0.0180696
8: 0.9733176, 0.9943494, 0.9722265, 0.9940638, -0.0207462, 0.0221229
9: -0.0063017, 0.0082136, -0.0061136, 0.0089945, -0.0134521, 0.0132312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0123770
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0128582
time: 1.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0069434, 0.0187820, -0.0106891, 0.0116099
1: -0.0039673, 0.0010577, -0.0041974, 0.0013851, -0.0053524, 0.0052550
2: 0.0034788, 0.0092297, 0.0034013, 0.0100584, -0.0065795, 0.0058284
3: -0.0010035, 0.0036382, -0.0015091, 0.0037173, -0.0047208, 0.0051473
4: -0.0049200, -0.0011302, -0.0051387, -0.0009777, -0.0036143, 0.0034547
5: -0.0005518, 0.0040088, -0.0006160, 0.0044893, -0.0050411, 0.0046249
6: -0.0066797, 0.0007150, -0.0067939, 0.0012106, -0.0078903, 0.0075089
7: -0.0242347, -0.0024411, -0.0255066, -0.0015269, -0.0202204, 0.0182912
8: 0.9733176, 0.9943494, 0.9721356, 0.9953537, -0.0220361, 0.0222138
9: -0.0063017, 0.0082136, -0.0069351, 0.0090564, -0.0135809, 0.0142013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0124020
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0129421
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0083055, 0.0187768, -0.0105203, 0.0102557
1: -0.0039485, 0.0010368, -0.0039174, 0.0011571, -0.0051057, 0.0049543
2: 0.0034755, 0.0091027, 0.0034030, 0.0090757, -0.0056002, 0.0056996
3: -0.0008895, 0.0036775, -0.0009109, 0.0036682, -0.0045577, 0.0045883
4: -0.0049572, -0.0011940, -0.0051203, -0.0011813, -0.0034220, 0.0033883
5: -0.0005470, 0.0039456, -0.0006129, 0.0039166, -0.0044636, 0.0045585
6: -0.0066028, 0.0008008, -0.0066209, 0.0008647, -0.0074674, 0.0074217
7: -0.0244393, -0.0027993, -0.0254298, -0.0027351, -0.0191088, 0.0179497
8: 0.9731785, 0.9940299, 0.9722265, 0.9940638, -0.0208853, 0.0218034
9: -0.0060726, 0.0083402, -0.0061136, 0.0089945, -0.0133240, 0.0134161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0123770
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128582
time: 1.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0069434, 0.0187820, -0.0105255, 0.0116177
1: -0.0039485, 0.0010368, -0.0041974, 0.0013851, -0.0053337, 0.0052342
2: 0.0034755, 0.0091027, 0.0034013, 0.0100584, -0.0065829, 0.0057014
3: -0.0008895, 0.0036775, -0.0015091, 0.0037173, -0.0046068, 0.0051866
4: -0.0049572, -0.0011940, -0.0051387, -0.0009777, -0.0036730, 0.0034244
5: -0.0005470, 0.0039456, -0.0006160, 0.0044893, -0.0050363, 0.0045617
6: -0.0066028, 0.0008008, -0.0067939, 0.0012106, -0.0078134, 0.0075947
7: -0.0244393, -0.0027993, -0.0255066, -0.0015269, -0.0205060, 0.0181712
8: 0.9731785, 0.9940299, 0.9721356, 0.9953537, -0.0221752, 0.0218943
9: -0.0060726, 0.0083402, -0.0069351, 0.0090564, -0.0134528, 0.0143862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0124020
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129421
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0080929, 0.0185533, -0.0117446, 0.0104595
1: -0.0042306, 0.0012934, -0.0039673, 0.0010577, -0.0052883, 0.0052607
2: 0.0034796, 0.0101545, 0.0034788, 0.0092297, -0.0057501, 0.0066756
3: -0.0015647, 0.0036803, -0.0010035, 0.0036382, -0.0052029, 0.0046838
4: -0.0049252, -0.0009405, -0.0049200, -0.0011302, -0.0031946, 0.0034122
5: -0.0005544, 0.0045500, -0.0005518, 0.0040088, -0.0045632, 0.0051019
6: -0.0068399, 0.0010449, -0.0066797, 0.0007150, -0.0075549, 0.0077246
7: -0.0242427, -0.0013184, -0.0242347, -0.0024411, -0.0167367, 0.0179779
8: 0.9732838, 0.9955425, 0.9733176, 0.9943494, -0.0210656, 0.0222248
9: -0.0070691, 0.0082311, -0.0063017, 0.0082136, -0.0134655, 0.0126087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120061
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125937
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0068087, 0.0185524, -0.0117437, 0.0117437
1: -0.0042306, 0.0012934, -0.0042306, 0.0012934, -0.0055240, 0.0055240
2: 0.0034796, 0.0101545, 0.0034796, 0.0101545, -0.0066749, 0.0066749
3: -0.0015647, 0.0036803, -0.0015647, 0.0036803, -0.0052450, 0.0052450
4: -0.0049252, -0.0009405, -0.0049252, -0.0009405, -0.0033730, 0.0033730
5: -0.0005544, 0.0045500, -0.0005544, 0.0045500, -0.0051044, 0.0051044
6: -0.0068399, 0.0010449, -0.0068399, 0.0010449, -0.0078848, 0.0078848
7: -0.0242427, -0.0013184, -0.0242427, -0.0013184, -0.0176704, 0.0176704
8: 0.9732838, 0.9955425, 0.9732838, 0.9955425, -0.0222587, 0.0222587
9: -0.0070691, 0.0082311, -0.0070691, 0.0082311, -0.0133348, 0.0133348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120061
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125937
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0080929, 0.0185533, -0.0115906, 0.0104667
1: -0.0042152, 0.0012591, -0.0039673, 0.0010577, -0.0052729, 0.0052264
2: 0.0034765, 0.0100375, 0.0034788, 0.0092297, -0.0057533, 0.0065587
3: -0.0014544, 0.0037225, -0.0010035, 0.0036382, -0.0050926, 0.0047260
4: -0.0049666, -0.0009945, -0.0049200, -0.0011302, -0.0032657, 0.0035872
5: -0.0005493, 0.0044909, -0.0005518, 0.0040088, -0.0045581, 0.0050428
6: -0.0067734, 0.0011315, -0.0066797, 0.0007150, -0.0074884, 0.0078112
7: -0.0244683, -0.0016217, -0.0242347, -0.0024411, -0.0171755, 0.0201076
8: 0.9731277, 0.9952570, 0.9733176, 0.9943494, -0.0212216, 0.0219394
9: -0.0068703, 0.0083713, -0.0063017, 0.0082136, -0.0141035, 0.0128313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110302, upper bound: 0.0120062
time: 1.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0068087, 0.0185524, -0.0115896, 0.0117509
1: -0.0042152, 0.0012591, -0.0042306, 0.0012934, -0.0055085, 0.0054897
2: 0.0034765, 0.0100375, 0.0034796, 0.0101545, -0.0066780, 0.0065579
3: -0.0014544, 0.0037225, -0.0015647, 0.0036803, -0.0051347, 0.0052872
4: -0.0049666, -0.0009945, -0.0049252, -0.0009405, -0.0034453, 0.0035916
5: -0.0005493, 0.0044909, -0.0005544, 0.0045500, -0.0050993, 0.0050453
6: -0.0067734, 0.0011315, -0.0068399, 0.0010449, -0.0078184, 0.0079714
7: -0.0244683, -0.0016217, -0.0242427, -0.0013184, -0.0181135, 0.0202772
8: 0.9731277, 0.9952570, 0.9732838, 0.9955425, -0.0224147, 0.0219733
9: -0.0068703, 0.0083713, -0.0070691, 0.0082311, -0.0141281, 0.0135576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110302, upper bound: 0.0120062
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0083055, 0.0187768, -0.0119681, 0.0102469
1: -0.0042306, 0.0012934, -0.0039174, 0.0011571, -0.0053878, 0.0052108
2: 0.0034796, 0.0101545, 0.0034030, 0.0090757, -0.0055960, 0.0067514
3: -0.0015647, 0.0036803, -0.0009109, 0.0036682, -0.0052328, 0.0045912
4: -0.0049252, -0.0009405, -0.0051203, -0.0011813, -0.0033838, 0.0036644
5: -0.0005544, 0.0045500, -0.0006129, 0.0039166, -0.0044710, 0.0051629
6: -0.0068399, 0.0010449, -0.0066209, 0.0008647, -0.0077046, 0.0076658
7: -0.0242427, -0.0013184, -0.0254298, -0.0027351, -0.0189205, 0.0195007
8: 0.9732838, 0.9955425, 0.9722265, 0.9940638, -0.0207800, 0.0233160
9: -0.0070691, 0.0082311, -0.0061136, 0.0089945, -0.0143971, 0.0132959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123413
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128285
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0069434, 0.0187820, -0.0119732, 0.0116090
1: -0.0042306, 0.0012934, -0.0041974, 0.0013851, -0.0056158, 0.0054907
2: 0.0034796, 0.0101545, 0.0034013, 0.0100584, -0.0065787, 0.0067532
3: -0.0015647, 0.0036803, -0.0015091, 0.0037173, -0.0052820, 0.0051894
4: -0.0049252, -0.0009405, -0.0051387, -0.0009777, -0.0036186, 0.0036339
5: -0.0005544, 0.0045500, -0.0006160, 0.0044893, -0.0050437, 0.0051661
6: -0.0068399, 0.0010449, -0.0067939, 0.0012106, -0.0080505, 0.0078388
7: -0.0242427, -0.0013184, -0.0255066, -0.0015269, -0.0203928, 0.0192324
8: 0.9732838, 0.9955425, 0.9721356, 0.9953537, -0.0220699, 0.0234069
9: -0.0070691, 0.0082311, -0.0069351, 0.0090564, -0.0143011, 0.0142254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123413
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128285
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0083055, 0.0187768, -0.0118140, 0.0102542
1: -0.0042152, 0.0012591, -0.0039174, 0.0011571, -0.0053723, 0.0051765
2: 0.0034765, 0.0100375, 0.0034030, 0.0090757, -0.0055992, 0.0066345
3: -0.0014544, 0.0037225, -0.0009109, 0.0036682, -0.0051226, 0.0046334
4: -0.0049666, -0.0009945, -0.0051203, -0.0011813, -0.0034451, 0.0038252
5: -0.0005493, 0.0044909, -0.0006129, 0.0039166, -0.0044659, 0.0051038
6: -0.0067734, 0.0011315, -0.0066209, 0.0008647, -0.0076381, 0.0077524
7: -0.0244683, -0.0016217, -0.0254298, -0.0027351, -0.0192364, 0.0214403
8: 0.9731277, 0.9952570, 0.9722265, 0.9940638, -0.0209361, 0.0230305
9: -0.0068703, 0.0083713, -0.0061136, 0.0089945, -0.0149902, 0.0134910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110302, upper bound: 0.0123413
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0069434, 0.0187820, -0.0118192, 0.0116163
1: -0.0042152, 0.0012591, -0.0041974, 0.0013851, -0.0056003, 0.0054564
2: 0.0034765, 0.0100375, 0.0034013, 0.0100584, -0.0065819, 0.0066362
3: -0.0014544, 0.0037225, -0.0015091, 0.0037173, -0.0051717, 0.0052316
4: -0.0049666, -0.0009945, -0.0051387, -0.0009777, -0.0036792, 0.0038405
5: -0.0005493, 0.0044909, -0.0006160, 0.0044893, -0.0050386, 0.0051070
6: -0.0067734, 0.0011315, -0.0067939, 0.0012106, -0.0079841, 0.0079254
7: -0.0244683, -0.0016217, -0.0255066, -0.0015269, -0.0207092, 0.0216716
8: 0.9731277, 0.9952570, 0.9721356, 0.9953537, -0.0222260, 0.0231214
9: -0.0068703, 0.0083713, -0.0069351, 0.0090564, -0.0150552, 0.0144199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0110302, upper bound: 0.0123413
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0082565, 0.0185611, -0.0104682, 0.0102969
1: -0.0039673, 0.0010577, -0.0039485, 0.0010368, -0.0050041, 0.0050062
2: 0.0034788, 0.0092297, 0.0034755, 0.0091027, -0.0056238, 0.0057543
3: -0.0010035, 0.0036382, -0.0008895, 0.0036775, -0.0046810, 0.0045278
4: -0.0049200, -0.0011302, -0.0049572, -0.0011940, -0.0031361, 0.0032362
5: -0.0005518, 0.0040088, -0.0005470, 0.0039456, -0.0044974, 0.0045558
6: -0.0066797, 0.0007150, -0.0066028, 0.0008008, -0.0074805, 0.0073177
7: -0.0242347, -0.0024411, -0.0244393, -0.0027993, -0.0164269, 0.0169889
8: 0.9733176, 0.9943494, 0.9731785, 0.9940299, -0.0207123, 0.0211709
9: -0.0063017, 0.0082136, -0.0060726, 0.0083402, -0.0127398, 0.0123924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109176, upper bound: 0.0120215
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0125934
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0069627, 0.0185597, -0.0104667, 0.0115906
1: -0.0039673, 0.0010577, -0.0042152, 0.0012591, -0.0052264, 0.0052729
2: 0.0034788, 0.0092297, 0.0034765, 0.0100375, -0.0065587, 0.0057533
3: -0.0010035, 0.0036382, -0.0014544, 0.0037225, -0.0047260, 0.0050926
4: -0.0049200, -0.0011302, -0.0049666, -0.0009945, -0.0035872, 0.0032657
5: -0.0005518, 0.0040088, -0.0005493, 0.0044909, -0.0050428, 0.0045581
6: -0.0066797, 0.0007150, -0.0067734, 0.0011315, -0.0078112, 0.0074884
7: -0.0242347, -0.0024411, -0.0244683, -0.0016217, -0.0201076, 0.0171754
8: 0.9733176, 0.9943494, 0.9731277, 0.9952570, -0.0219394, 0.0212216
9: -0.0063017, 0.0082136, -0.0068703, 0.0083713, -0.0128313, 0.0141035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109176, upper bound: 0.0120299
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126659
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0082565, 0.0185611, -0.0103047, 0.0103047
1: -0.0039485, 0.0010368, -0.0039485, 0.0010368, -0.0049854, 0.0049854
2: 0.0034755, 0.0091027, 0.0034755, 0.0091027, -0.0056272, 0.0056272
3: -0.0008895, 0.0036775, -0.0008895, 0.0036775, -0.0045670, 0.0045670
4: -0.0049572, -0.0011940, -0.0049572, -0.0011940, -0.0031432, 0.0031432
5: -0.0005470, 0.0039456, -0.0005470, 0.0039456, -0.0044926, 0.0044926
6: -0.0066028, 0.0008008, -0.0066028, 0.0008008, -0.0074036, 0.0074036
7: -0.0244393, -0.0027993, -0.0244393, -0.0027993, -0.0163868, 0.0163868
8: 0.9731785, 0.9940299, 0.9731785, 0.9940299, -0.0208514, 0.0208514
9: -0.0060726, 0.0083402, -0.0060726, 0.0083402, -0.0124258, 0.0124258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0120655
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0125840
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0069627, 0.0185597, -0.0103032, 0.0115984
1: -0.0039485, 0.0010368, -0.0042152, 0.0012591, -0.0052076, 0.0052520
2: 0.0034755, 0.0091027, 0.0034765, 0.0100375, -0.0065621, 0.0056262
3: -0.0008895, 0.0036775, -0.0014544, 0.0037225, -0.0046120, 0.0051318
4: -0.0049572, -0.0011940, -0.0049666, -0.0009945, -0.0036021, 0.0031734
5: -0.0005470, 0.0039456, -0.0005493, 0.0044909, -0.0050379, 0.0044949
6: -0.0066028, 0.0008008, -0.0067734, 0.0011315, -0.0077343, 0.0075743
7: -0.0244393, -0.0027993, -0.0244683, -0.0016217, -0.0201676, 0.0165809
8: 0.9731785, 0.9940299, 0.9731277, 0.9952570, -0.0220785, 0.0209022
9: -0.0060726, 0.0083402, -0.0068703, 0.0083713, -0.0125183, 0.0141589

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0120668
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126532
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0084286, 0.0187860, -0.0106931, 0.0101248
1: -0.0039673, 0.0010577, -0.0039089, 0.0011429, -0.0051102, 0.0049666
2: 0.0034788, 0.0092297, 0.0033987, 0.0089764, -0.0054975, 0.0058310
3: -0.0010035, 0.0036382, -0.0008197, 0.0037132, -0.0047167, 0.0044579
4: -0.0049200, -0.0011302, -0.0051675, -0.0012301, -0.0033415, 0.0034966
5: -0.0005518, 0.0040088, -0.0006096, 0.0038723, -0.0044241, 0.0046185
6: -0.0066797, 0.0007150, -0.0065548, 0.0009793, -0.0076590, 0.0072698
7: -0.0242347, -0.0024411, -0.0256950, -0.0030151, -0.0187112, 0.0185493
8: 0.9733176, 0.9943494, 0.9720360, 0.9938253, -0.0205077, 0.0223134
9: -0.0063017, 0.0082136, -0.0059282, 0.0091599, -0.0137060, 0.0131351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109176, upper bound: 0.0123984
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0128720
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0080929, 0.0185533, 0.0070348, 0.0187912, -0.0106983, 0.0115185
1: -0.0039673, 0.0010577, -0.0041962, 0.0013731, -0.0053404, 0.0052539
2: 0.0034788, 0.0092297, 0.0033970, 0.0099868, -0.0065080, 0.0058328
3: -0.0010035, 0.0036382, -0.0014337, 0.0037640, -0.0047675, 0.0050720
4: -0.0049200, -0.0011302, -0.0051883, -0.0010215, -0.0035961, 0.0035372
5: -0.0005518, 0.0040088, -0.0006126, 0.0044600, -0.0050118, 0.0046214
6: -0.0066797, 0.0007150, -0.0067363, 0.0013294, -0.0080091, 0.0074512
7: -0.0242347, -0.0024411, -0.0257798, -0.0017740, -0.0201215, 0.0187952
8: 0.9733176, 0.9943494, 0.9719387, 0.9951462, -0.0218285, 0.0224106
9: -0.0063017, 0.0082136, -0.0067789, 0.0092272, -0.0138439, 0.0141293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0109176, upper bound: 0.0124276
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0129595
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0084286, 0.0187860, -0.0105295, 0.0101326
1: -0.0039485, 0.0010368, -0.0039089, 0.0011429, -0.0050914, 0.0049457
2: 0.0034755, 0.0091027, 0.0033987, 0.0089764, -0.0055009, 0.0057039
3: -0.0008895, 0.0036775, -0.0008197, 0.0037132, -0.0046028, 0.0044971
4: -0.0049572, -0.0011940, -0.0051675, -0.0012301, -0.0033539, 0.0034004
5: -0.0005470, 0.0039456, -0.0006096, 0.0038723, -0.0044193, 0.0045552
6: -0.0066028, 0.0008008, -0.0065548, 0.0009793, -0.0075821, 0.0073556
7: -0.0244393, -0.0027993, -0.0256950, -0.0030151, -0.0187534, 0.0179298
8: 0.9731785, 0.9940299, 0.9720360, 0.9938253, -0.0206468, 0.0219939
9: -0.0060726, 0.0083402, -0.0059282, 0.0091599, -0.0133754, 0.0131826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0124059
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128543
time: 1.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0082565, 0.0185611, 0.0070348, 0.0187912, -0.0105348, 0.0115263
1: -0.0039485, 0.0010368, -0.0041962, 0.0013731, -0.0053217, 0.0052330
2: 0.0034755, 0.0091027, 0.0033970, 0.0099868, -0.0065114, 0.0057057
3: -0.0008895, 0.0036775, -0.0014337, 0.0037640, -0.0046535, 0.0051112
4: -0.0049572, -0.0011940, -0.0051883, -0.0010215, -0.0036074, 0.0034369
5: -0.0005470, 0.0039456, -0.0006126, 0.0044600, -0.0050070, 0.0045582
6: -0.0066028, 0.0008008, -0.0067363, 0.0013294, -0.0079322, 0.0075371
7: -0.0244393, -0.0027993, -0.0257798, -0.0017740, -0.0201635, 0.0181487
8: 0.9731785, 0.9940299, 0.9719387, 0.9951462, -0.0219676, 0.0220912
9: -0.0060726, 0.0083402, -0.0067789, 0.0092272, -0.0135035, 0.0141726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0124342
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129376
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0082565, 0.0185611, -0.0117524, 0.0102959
1: -0.0042306, 0.0012934, -0.0039485, 0.0010368, -0.0052675, 0.0052419
2: 0.0034796, 0.0101545, 0.0034755, 0.0091027, -0.0056231, 0.0066790
3: -0.0015647, 0.0036803, -0.0008895, 0.0036775, -0.0052421, 0.0045698
4: -0.0049252, -0.0009405, -0.0049572, -0.0011940, -0.0031644, 0.0034820
5: -0.0005544, 0.0045500, -0.0005470, 0.0039456, -0.0045000, 0.0050970
6: -0.0068399, 0.0010449, -0.0066028, 0.0008008, -0.0076408, 0.0076477
7: -0.0242427, -0.0013184, -0.0244393, -0.0027993, -0.0166168, 0.0184200
8: 0.9732838, 0.9955425, 0.9731785, 0.9940299, -0.0207462, 0.0223640
9: -0.0070691, 0.0082311, -0.0060726, 0.0083402, -0.0136848, 0.0124806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120104
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0069627, 0.0185597, -0.0117509, 0.0115896
1: -0.0042306, 0.0012934, -0.0042152, 0.0012591, -0.0054897, 0.0055085
2: 0.0034796, 0.0101545, 0.0034765, 0.0100375, -0.0065579, 0.0066780
3: -0.0015647, 0.0036803, -0.0014544, 0.0037225, -0.0052872, 0.0051347
4: -0.0049252, -0.0009405, -0.0049666, -0.0009945, -0.0035916, 0.0034453
5: -0.0005544, 0.0045500, -0.0005493, 0.0044909, -0.0050453, 0.0050993
6: -0.0068399, 0.0010449, -0.0067734, 0.0011315, -0.0079714, 0.0078184
7: -0.0242427, -0.0013184, -0.0244683, -0.0016217, -0.0202772, 0.0181135
8: 0.9732838, 0.9955425, 0.9731277, 0.9952570, -0.0219733, 0.0224147
9: -0.0070691, 0.0082311, -0.0068703, 0.0083713, -0.0135576, 0.0141281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120104
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0082565, 0.0185611, -0.0115984, 0.0103032
1: -0.0042152, 0.0012591, -0.0039485, 0.0010368, -0.0052520, 0.0052076
2: 0.0034765, 0.0100375, 0.0034755, 0.0091027, -0.0056262, 0.0065621
3: -0.0014544, 0.0037225, -0.0008895, 0.0036775, -0.0051318, 0.0046120
4: -0.0049666, -0.0009945, -0.0049572, -0.0011940, -0.0031734, 0.0036021
5: -0.0005493, 0.0044909, -0.0005470, 0.0039456, -0.0044949, 0.0050379
6: -0.0067734, 0.0011315, -0.0066028, 0.0008008, -0.0075743, 0.0077343
7: -0.0244683, -0.0016217, -0.0244393, -0.0027993, -0.0165809, 0.0201676
8: 0.9731277, 0.9952570, 0.9731785, 0.9940299, -0.0209022, 0.0220785
9: -0.0068703, 0.0083713, -0.0060726, 0.0083402, -0.0141589, 0.0125183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0120515
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
time: 1.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0069627, 0.0185597, -0.0115969, 0.0115969
1: -0.0042152, 0.0012591, -0.0042152, 0.0012591, -0.0054742, 0.0054742
2: 0.0034765, 0.0100375, 0.0034765, 0.0100375, -0.0065611, 0.0065611
3: -0.0014544, 0.0037225, -0.0014544, 0.0037225, -0.0051769, 0.0051769
4: -0.0049666, -0.0009945, -0.0049666, -0.0009945, -0.0036083, 0.0036083
5: -0.0005493, 0.0044909, -0.0005493, 0.0044909, -0.0050402, 0.0050402
6: -0.0067734, 0.0011315, -0.0067734, 0.0011315, -0.0079050, 0.0079050
7: -0.0244683, -0.0016217, -0.0244683, -0.0016217, -0.0203653, 0.0203653
8: 0.9731277, 0.9952570, 0.9731277, 0.9952570, -0.0221293, 0.0221293
9: -0.0068703, 0.0083713, -0.0068703, 0.0083713, -0.0141905, 0.0141905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0120515
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0084286, 0.0187860, -0.0119773, 0.0101238
1: -0.0042306, 0.0012934, -0.0039089, 0.0011429, -0.0053735, 0.0052023
2: 0.0034796, 0.0101545, 0.0033987, 0.0089764, -0.0054968, 0.0067557
3: -0.0015647, 0.0036803, -0.0008197, 0.0037132, -0.0052779, 0.0045000
4: -0.0049252, -0.0009405, -0.0051675, -0.0012301, -0.0033621, 0.0037425
5: -0.0005544, 0.0045500, -0.0006096, 0.0038723, -0.0044267, 0.0051597
6: -0.0068399, 0.0010449, -0.0065548, 0.0009793, -0.0078192, 0.0075997
7: -0.0242427, -0.0013184, -0.0256950, -0.0030151, -0.0188085, 0.0199803
8: 0.9732838, 0.9955425, 0.9720360, 0.9938253, -0.0205415, 0.0235065
9: -0.0070691, 0.0082311, -0.0059282, 0.0091599, -0.0146509, 0.0131998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123586
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0068087, 0.0185524, 0.0070348, 0.0187912, -0.0119825, 0.0115176
1: -0.0042306, 0.0012934, -0.0041962, 0.0013731, -0.0056038, 0.0054896
2: 0.0034796, 0.0101545, 0.0033970, 0.0099868, -0.0065072, 0.0067575
3: -0.0015647, 0.0036803, -0.0014337, 0.0037640, -0.0053287, 0.0051140
4: -0.0049252, -0.0009405, -0.0051883, -0.0010215, -0.0036009, 0.0037144
5: -0.0005544, 0.0045500, -0.0006126, 0.0044600, -0.0050143, 0.0051626
6: -0.0068399, 0.0010449, -0.0067363, 0.0013294, -0.0081693, 0.0077812
7: -0.0242427, -0.0013184, -0.0257798, -0.0017740, -0.0202912, 0.0197146
8: 0.9732838, 0.9955425, 0.9719387, 0.9951462, -0.0218624, 0.0236037
9: -0.0070691, 0.0082311, -0.0067789, 0.0092272, -0.0145634, 0.0141538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123586
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
time: 1.55 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0084286, 0.0187860, -0.0118232, 0.0101311
1: -0.0042152, 0.0012591, -0.0039089, 0.0011429, -0.0053581, 0.0051680
2: 0.0034765, 0.0100375, 0.0033987, 0.0089764, -0.0054999, 0.0066388
3: -0.0014544, 0.0037225, -0.0008197, 0.0037132, -0.0051676, 0.0045422
4: -0.0049666, -0.0009945, -0.0051675, -0.0012301, -0.0033770, 0.0038494
5: -0.0005493, 0.0044909, -0.0006096, 0.0038723, -0.0044215, 0.0051006
6: -0.0067734, 0.0011315, -0.0065548, 0.0009793, -0.0077528, 0.0076863
7: -0.0244683, -0.0016217, -0.0256950, -0.0030151, -0.0188771, 0.0215750
8: 0.9731277, 0.9952570, 0.9720360, 0.9938253, -0.0206976, 0.0232210
9: -0.0068703, 0.0083713, -0.0059282, 0.0091599, -0.0150821, 0.0132563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0123698
time: 1.20 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
time: 1.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069627, 0.0185597, 0.0070348, 0.0187912, -0.0118285, 0.0115248
1: -0.0042152, 0.0012591, -0.0041962, 0.0013731, -0.0055883, 0.0054553
2: 0.0034765, 0.0100375, 0.0033970, 0.0099868, -0.0065104, 0.0066406
3: -0.0014544, 0.0037225, -0.0014337, 0.0037640, -0.0052184, 0.0051562
4: -0.0049666, -0.0009945, -0.0051883, -0.0010215, -0.0036135, 0.0038638
5: -0.0005493, 0.0044909, -0.0006126, 0.0044600, -0.0050092, 0.0051035
6: -0.0067734, 0.0011315, -0.0067363, 0.0013294, -0.0081029, 0.0078678
7: -0.0244683, -0.0016217, -0.0257798, -0.0017740, -0.0203621, 0.0218055
8: 0.9731277, 0.9952570, 0.9719387, 0.9951462, -0.0220184, 0.0233183
9: -0.0068703, 0.0083713, -0.0067789, 0.0092272, -0.0151430, 0.0142037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0123698
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0080929, 0.0185533, -0.0102479, 0.0106839
1: -0.0039174, 0.0011571, -0.0039673, 0.0010577, -0.0049751, 0.0051244
2: 0.0034030, 0.0090757, 0.0034788, 0.0092297, -0.0058267, 0.0055968
3: -0.0009109, 0.0036682, -0.0010035, 0.0036382, -0.0045491, 0.0046717
4: -0.0051203, -0.0011813, -0.0049200, -0.0011302, -0.0034185, 0.0033633
5: -0.0006129, 0.0039166, -0.0005518, 0.0040088, -0.0046217, 0.0044685
6: -0.0066209, 0.0008647, -0.0066797, 0.0007150, -0.0073358, 0.0075444
7: -0.0254298, -0.0027351, -0.0242347, -0.0024411, -0.0180696, 0.0188232
8: 0.9722265, 0.9940638, 0.9733176, 0.9943494, -0.0221229, 0.0207462
9: -0.0061136, 0.0089945, -0.0063017, 0.0082136, -0.0132312, 0.0134521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0120674
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126054
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0082565, 0.0185611, -0.0102557, 0.0105203
1: -0.0039174, 0.0011571, -0.0039485, 0.0010368, -0.0049543, 0.0051057
2: 0.0034030, 0.0090757, 0.0034755, 0.0091027, -0.0056996, 0.0056002
3: -0.0009109, 0.0036682, -0.0008895, 0.0036775, -0.0045883, 0.0045577
4: -0.0051203, -0.0011813, -0.0049572, -0.0011940, -0.0033883, 0.0034220
5: -0.0006129, 0.0039166, -0.0005470, 0.0039456, -0.0045585, 0.0044636
6: -0.0066209, 0.0008647, -0.0066028, 0.0008008, -0.0074217, 0.0074674
7: -0.0254298, -0.0027351, -0.0244393, -0.0027993, -0.0179497, 0.0191088
8: 0.9722265, 0.9940638, 0.9731785, 0.9940299, -0.0218034, 0.0208853
9: -0.0061136, 0.0089945, -0.0060726, 0.0083402, -0.0134161, 0.0133240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0120789
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126152
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0080929, 0.0185533, -0.0116099, 0.0106891
1: -0.0041974, 0.0013851, -0.0039673, 0.0010577, -0.0052550, 0.0053524
2: 0.0034013, 0.0100584, 0.0034788, 0.0092297, -0.0058284, 0.0065795
3: -0.0015091, 0.0037173, -0.0010035, 0.0036382, -0.0051473, 0.0047208
4: -0.0051387, -0.0009777, -0.0049200, -0.0011302, -0.0034547, 0.0036143
5: -0.0006160, 0.0044893, -0.0005518, 0.0040088, -0.0046249, 0.0050411
6: -0.0067939, 0.0012106, -0.0066797, 0.0007150, -0.0075089, 0.0078903
7: -0.0255066, -0.0015269, -0.0242347, -0.0024411, -0.0182912, 0.0202204
8: 0.9721356, 0.9953537, 0.9733176, 0.9943494, -0.0222138, 0.0220361
9: -0.0069351, 0.0090564, -0.0063017, 0.0082136, -0.0142013, 0.0135809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120003, upper bound: 0.0120038
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125672
time: 1.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0082565, 0.0185611, -0.0116177, 0.0105255
1: -0.0041974, 0.0013851, -0.0039485, 0.0010368, -0.0052342, 0.0053337
2: 0.0034013, 0.0100584, 0.0034755, 0.0091027, -0.0057014, 0.0065829
3: -0.0015091, 0.0037173, -0.0008895, 0.0036775, -0.0051866, 0.0046068
4: -0.0051387, -0.0009777, -0.0049572, -0.0011940, -0.0034244, 0.0036730
5: -0.0006160, 0.0044893, -0.0005470, 0.0039456, -0.0045617, 0.0050363
6: -0.0067939, 0.0012106, -0.0066028, 0.0008008, -0.0075947, 0.0078134
7: -0.0255066, -0.0015269, -0.0244393, -0.0027993, -0.0181712, 0.0205060
8: 0.9721356, 0.9953537, 0.9731785, 0.9940299, -0.0218943, 0.0221752
9: -0.0069351, 0.0090564, -0.0060726, 0.0083402, -0.0143862, 0.0134528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120003, upper bound: 0.0120141
time: 1.58 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
time: 1.62 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0068087, 0.0185524, -0.0102469, 0.0119681
1: -0.0039174, 0.0011571, -0.0042306, 0.0012934, -0.0052108, 0.0053878
2: 0.0034030, 0.0090757, 0.0034796, 0.0101545, -0.0067514, 0.0055960
3: -0.0009109, 0.0036682, -0.0015647, 0.0036803, -0.0045912, 0.0052328
4: -0.0051203, -0.0011813, -0.0049252, -0.0009405, -0.0036644, 0.0033838
5: -0.0006129, 0.0039166, -0.0005544, 0.0045500, -0.0051629, 0.0044710
6: -0.0066209, 0.0008647, -0.0068399, 0.0010449, -0.0076658, 0.0077046
7: -0.0254298, -0.0027351, -0.0242427, -0.0013184, -0.0195007, 0.0189205
8: 0.9722265, 0.9940638, 0.9732838, 0.9955425, -0.0233160, 0.0207800
9: -0.0061136, 0.0089945, -0.0070691, 0.0082311, -0.0132959, 0.0143971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0120242
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126380
time: 1.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0069627, 0.0185597, -0.0102542, 0.0118140
1: -0.0039174, 0.0011571, -0.0042152, 0.0012591, -0.0051765, 0.0053723
2: 0.0034030, 0.0090757, 0.0034765, 0.0100375, -0.0066345, 0.0055992
3: -0.0009109, 0.0036682, -0.0014544, 0.0037225, -0.0046334, 0.0051226
4: -0.0051203, -0.0011813, -0.0049666, -0.0009945, -0.0038252, 0.0034451
5: -0.0006129, 0.0039166, -0.0005493, 0.0044909, -0.0051038, 0.0044659
6: -0.0066209, 0.0008647, -0.0067734, 0.0011315, -0.0077524, 0.0076381
7: -0.0254298, -0.0027351, -0.0244683, -0.0016217, -0.0214403, 0.0192364
8: 0.9722265, 0.9940638, 0.9731277, 0.9952570, -0.0230305, 0.0209361
9: -0.0061136, 0.0089945, -0.0068703, 0.0083713, -0.0134910, 0.0149902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0120315
time: 1.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126465
time: 1.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0068087, 0.0185524, -0.0116090, 0.0119732
1: -0.0041974, 0.0013851, -0.0042306, 0.0012934, -0.0054907, 0.0056158
2: 0.0034013, 0.0100584, 0.0034796, 0.0101545, -0.0067532, 0.0065787
3: -0.0015091, 0.0037173, -0.0015647, 0.0036803, -0.0051894, 0.0052820
4: -0.0051387, -0.0009777, -0.0049252, -0.0009405, -0.0036339, 0.0036186
5: -0.0006160, 0.0044893, -0.0005544, 0.0045500, -0.0051661, 0.0050437
6: -0.0067939, 0.0012106, -0.0068399, 0.0010449, -0.0078388, 0.0080505
7: -0.0255066, -0.0015269, -0.0242427, -0.0013184, -0.0192324, 0.0203928
8: 0.9721356, 0.9953537, 0.9732838, 0.9955425, -0.0234069, 0.0220699
9: -0.0069351, 0.0090564, -0.0070691, 0.0082311, -0.0142254, 0.0143011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120000, upper bound: 0.0120025
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125672
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0069627, 0.0185597, -0.0116163, 0.0118192
1: -0.0041974, 0.0013851, -0.0042152, 0.0012591, -0.0054564, 0.0056003
2: 0.0034013, 0.0100584, 0.0034765, 0.0100375, -0.0066362, 0.0065819
3: -0.0015091, 0.0037173, -0.0014544, 0.0037225, -0.0052316, 0.0051717
4: -0.0051387, -0.0009777, -0.0049666, -0.0009945, -0.0038405, 0.0036792
5: -0.0006160, 0.0044893, -0.0005493, 0.0044909, -0.0051070, 0.0050386
6: -0.0067939, 0.0012106, -0.0067734, 0.0011315, -0.0079254, 0.0079841
7: -0.0255066, -0.0015269, -0.0244683, -0.0016217, -0.0216717, 0.0207092
8: 0.9721356, 0.9953537, 0.9731277, 0.9952570, -0.0231214, 0.0222260
9: -0.0069351, 0.0090564, -0.0068703, 0.0083713, -0.0144198, 0.0150552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120000, upper bound: 0.0120122
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0080929, 0.0185533, -0.0101248, 0.0106931
1: -0.0039089, 0.0011429, -0.0039673, 0.0010577, -0.0049666, 0.0051102
2: 0.0033987, 0.0089764, 0.0034788, 0.0092297, -0.0058310, 0.0054975
3: -0.0008197, 0.0037132, -0.0010035, 0.0036382, -0.0044579, 0.0047167
4: -0.0051675, -0.0012301, -0.0049200, -0.0011302, -0.0034966, 0.0033415
5: -0.0006096, 0.0038723, -0.0005518, 0.0040088, -0.0046185, 0.0044241
6: -0.0065548, 0.0009793, -0.0066797, 0.0007150, -0.0072698, 0.0076590
7: -0.0256950, -0.0030151, -0.0242347, -0.0024411, -0.0185493, 0.0187112
8: 0.9720360, 0.9938253, 0.9733176, 0.9943494, -0.0223134, 0.0205077
9: -0.0059282, 0.0091599, -0.0063017, 0.0082136, -0.0131351, 0.0137060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120418, upper bound: 0.0120690
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128991, upper bound: 0.0126054
time: 1.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0082565, 0.0185611, -0.0101326, 0.0105295
1: -0.0039089, 0.0011429, -0.0039485, 0.0010368, -0.0049457, 0.0050914
2: 0.0033987, 0.0089764, 0.0034755, 0.0091027, -0.0057039, 0.0055009
3: -0.0008197, 0.0037132, -0.0008895, 0.0036775, -0.0044971, 0.0046028
4: -0.0051675, -0.0012301, -0.0049572, -0.0011940, -0.0034004, 0.0033539
5: -0.0006096, 0.0038723, -0.0005470, 0.0039456, -0.0045552, 0.0044193
6: -0.0065548, 0.0009793, -0.0066028, 0.0008008, -0.0073556, 0.0075821
7: -0.0256950, -0.0030151, -0.0244393, -0.0027993, -0.0179298, 0.0187534
8: 0.9720360, 0.9938253, 0.9731785, 0.9940299, -0.0219939, 0.0206468
9: -0.0059282, 0.0091599, -0.0060726, 0.0083402, -0.0131827, 0.0133754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120418, upper bound: 0.0121212
time: 1.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128991, upper bound: 0.0126060
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0080929, 0.0185533, -0.0115185, 0.0106983
1: -0.0041962, 0.0013731, -0.0039673, 0.0010577, -0.0052539, 0.0053404
2: 0.0033970, 0.0099868, 0.0034788, 0.0092297, -0.0058328, 0.0065080
3: -0.0014337, 0.0037640, -0.0010035, 0.0036382, -0.0050720, 0.0047675
4: -0.0051883, -0.0010215, -0.0049200, -0.0011302, -0.0035372, 0.0035961
5: -0.0006126, 0.0044600, -0.0005518, 0.0040088, -0.0046214, 0.0050118
6: -0.0067363, 0.0013294, -0.0066797, 0.0007150, -0.0074512, 0.0080091
7: -0.0257798, -0.0017740, -0.0242347, -0.0024411, -0.0187952, 0.0201215
8: 0.9719387, 0.9951462, 0.9733176, 0.9943494, -0.0224106, 0.0218285
9: -0.0067789, 0.0092272, -0.0063017, 0.0082136, -0.0141293, 0.0138439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120718, upper bound: 0.0120063
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
time: 1.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0082565, 0.0185611, -0.0115263, 0.0105348
1: -0.0041962, 0.0013731, -0.0039485, 0.0010368, -0.0052330, 0.0053217
2: 0.0033970, 0.0099868, 0.0034755, 0.0091027, -0.0057057, 0.0065114
3: -0.0014337, 0.0037640, -0.0008895, 0.0036775, -0.0051112, 0.0046535
4: -0.0051883, -0.0010215, -0.0049572, -0.0011940, -0.0034369, 0.0036074
5: -0.0006126, 0.0044600, -0.0005470, 0.0039456, -0.0045582, 0.0050070
6: -0.0067363, 0.0013294, -0.0066028, 0.0008008, -0.0075371, 0.0079322
7: -0.0257798, -0.0017740, -0.0244393, -0.0027993, -0.0181487, 0.0201635
8: 0.9719387, 0.9951462, 0.9731785, 0.9940299, -0.0220912, 0.0219676
9: -0.0067789, 0.0092272, -0.0060726, 0.0083402, -0.0141726, 0.0135035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120718, upper bound: 0.0120541
time: 1.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0068087, 0.0185524, -0.0101238, 0.0119773
1: -0.0039089, 0.0011429, -0.0042306, 0.0012934, -0.0052023, 0.0053735
2: 0.0033987, 0.0089764, 0.0034796, 0.0101545, -0.0067557, 0.0054968
3: -0.0008197, 0.0037132, -0.0015647, 0.0036803, -0.0045000, 0.0052779
4: -0.0051675, -0.0012301, -0.0049252, -0.0009405, -0.0037425, 0.0033621
5: -0.0006096, 0.0038723, -0.0005544, 0.0045500, -0.0051597, 0.0044267
6: -0.0065548, 0.0009793, -0.0068399, 0.0010449, -0.0075997, 0.0078192
7: -0.0256950, -0.0030151, -0.0242427, -0.0013184, -0.0199803, 0.0188086
8: 0.9720360, 0.9938253, 0.9732838, 0.9955425, -0.0235065, 0.0205415
9: -0.0059282, 0.0091599, -0.0070691, 0.0082311, -0.0131998, 0.0146509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119569, upper bound: 0.0120279
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
time: 1.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0069627, 0.0185597, -0.0101311, 0.0118232
1: -0.0039089, 0.0011429, -0.0042152, 0.0012591, -0.0051680, 0.0053581
2: 0.0033987, 0.0089764, 0.0034765, 0.0100375, -0.0066388, 0.0054999
3: -0.0008197, 0.0037132, -0.0014544, 0.0037225, -0.0045422, 0.0051676
4: -0.0051675, -0.0012301, -0.0049666, -0.0009945, -0.0038494, 0.0033770
5: -0.0006096, 0.0038723, -0.0005493, 0.0044909, -0.0051006, 0.0044215
6: -0.0065548, 0.0009793, -0.0067734, 0.0011315, -0.0076863, 0.0077528
7: -0.0256950, -0.0030151, -0.0244683, -0.0016217, -0.0215749, 0.0188771
8: 0.9720360, 0.9938253, 0.9731277, 0.9952570, -0.0232210, 0.0206976
9: -0.0059282, 0.0091599, -0.0068703, 0.0083713, -0.0132563, 0.0150821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0119569, upper bound: 0.0120668
time: 1.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
time: 1.67 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0068087, 0.0185524, -0.0115176, 0.0119825
1: -0.0041962, 0.0013731, -0.0042306, 0.0012934, -0.0054896, 0.0056038
2: 0.0033970, 0.0099868, 0.0034796, 0.0101545, -0.0067575, 0.0065072
3: -0.0014337, 0.0037640, -0.0015647, 0.0036803, -0.0051140, 0.0053287
4: -0.0051883, -0.0010215, -0.0049252, -0.0009405, -0.0037144, 0.0036009
5: -0.0006126, 0.0044600, -0.0005544, 0.0045500, -0.0051626, 0.0050143
6: -0.0067363, 0.0013294, -0.0068399, 0.0010449, -0.0077812, 0.0081693
7: -0.0257798, -0.0017740, -0.0242427, -0.0013184, -0.0197146, 0.0202912
8: 0.9719387, 0.9951462, 0.9732838, 0.9955425, -0.0236037, 0.0218624
9: -0.0067789, 0.0092272, -0.0070691, 0.0082311, -0.0141538, 0.0145634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120715, upper bound: 0.0120053
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
time: 1.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0069627, 0.0185597, -0.0115248, 0.0118285
1: -0.0041962, 0.0013731, -0.0042152, 0.0012591, -0.0054553, 0.0055883
2: 0.0033970, 0.0099868, 0.0034765, 0.0100375, -0.0066406, 0.0065104
3: -0.0014337, 0.0037640, -0.0014544, 0.0037225, -0.0051562, 0.0052184
4: -0.0051883, -0.0010215, -0.0049666, -0.0009945, -0.0038638, 0.0036135
5: -0.0006126, 0.0044600, -0.0005493, 0.0044909, -0.0051035, 0.0050092
6: -0.0067363, 0.0013294, -0.0067734, 0.0011315, -0.0078678, 0.0081029
7: -0.0257798, -0.0017740, -0.0244683, -0.0016217, -0.0218055, 0.0203621
8: 0.9719387, 0.9951462, 0.9731277, 0.9952570, -0.0233183, 0.0220184
9: -0.0067789, 0.0092272, -0.0068703, 0.0083713, -0.0142037, 0.0151430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120715, upper bound: 0.0120516
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0083055, 0.0187768, -0.0104713, 0.0104713
1: -0.0039174, 0.0011571, -0.0039174, 0.0011571, -0.0050746, 0.0050746
2: 0.0034030, 0.0090757, 0.0034030, 0.0090757, -0.0056726, 0.0056726
3: -0.0009109, 0.0036682, -0.0009109, 0.0036682, -0.0045791, 0.0045791
4: -0.0051203, -0.0011813, -0.0051203, -0.0011813, -0.0034988, 0.0034988
5: -0.0006129, 0.0039166, -0.0006129, 0.0039166, -0.0045295, 0.0045295
6: -0.0066209, 0.0008647, -0.0066209, 0.0008647, -0.0074855, 0.0074855
7: -0.0254298, -0.0027351, -0.0254298, -0.0027351, -0.0195166, 0.0195166
8: 0.9722265, 0.9940638, 0.9722265, 0.9940638, -0.0218373, 0.0218373
9: -0.0061136, 0.0089945, -0.0061136, 0.0089945, -0.0137772, 0.0137772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125273
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127449
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0084286, 0.0187860, -0.0104805, 0.0103482
1: -0.0039174, 0.0011571, -0.0039089, 0.0011429, -0.0050603, 0.0050660
2: 0.0034030, 0.0090757, 0.0033987, 0.0089764, -0.0055734, 0.0056769
3: -0.0009109, 0.0036682, -0.0008197, 0.0037132, -0.0046241, 0.0044879
4: -0.0051203, -0.0011813, -0.0051675, -0.0012301, -0.0034719, 0.0035666
5: -0.0006129, 0.0039166, -0.0006096, 0.0038723, -0.0044852, 0.0045263
6: -0.0066209, 0.0008647, -0.0065548, 0.0009793, -0.0076002, 0.0074195
7: -0.0254298, -0.0027351, -0.0256950, -0.0030151, -0.0193615, 0.0198710
8: 0.9722265, 0.9940638, 0.9720360, 0.9938253, -0.0215988, 0.0220278
9: -0.0061136, 0.0089945, -0.0059282, 0.0091599, -0.0140035, 0.0136636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125314
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127518
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0083055, 0.0187768, -0.0118334, 0.0104765
1: -0.0041974, 0.0013851, -0.0039174, 0.0011571, -0.0053545, 0.0053026
2: 0.0034013, 0.0100584, 0.0034030, 0.0090757, -0.0056744, 0.0066553
3: -0.0015091, 0.0037173, -0.0009109, 0.0036682, -0.0051773, 0.0046282
4: -0.0051387, -0.0009777, -0.0051203, -0.0011813, -0.0035282, 0.0037487
5: -0.0006160, 0.0044893, -0.0006129, 0.0039166, -0.0045327, 0.0051022
6: -0.0067939, 0.0012106, -0.0066209, 0.0008647, -0.0076586, 0.0078315
7: -0.0255066, -0.0015269, -0.0254298, -0.0027351, -0.0196633, 0.0209066
8: 0.9721356, 0.9953537, 0.9722265, 0.9940638, -0.0219282, 0.0231272
9: -0.0069351, 0.0090564, -0.0061136, 0.0089945, -0.0147443, 0.0138783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125170
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127174
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0084286, 0.0187860, -0.0118426, 0.0103534
1: -0.0041974, 0.0013851, -0.0039089, 0.0011429, -0.0053403, 0.0052940
2: 0.0034013, 0.0100584, 0.0033987, 0.0089764, -0.0055751, 0.0066596
3: -0.0015091, 0.0037173, -0.0008197, 0.0037132, -0.0052223, 0.0045370
4: -0.0051387, -0.0009777, -0.0051675, -0.0012301, -0.0035013, 0.0038165
5: -0.0006160, 0.0044893, -0.0006096, 0.0038723, -0.0044883, 0.0050989
6: -0.0067939, 0.0012106, -0.0065548, 0.0009793, -0.0077732, 0.0077654
7: -0.0255066, -0.0015269, -0.0256950, -0.0030151, -0.0195081, 0.0212610
8: 0.9721356, 0.9953537, 0.9720360, 0.9938253, -0.0216897, 0.0233177
9: -0.0069351, 0.0090564, -0.0059282, 0.0091599, -0.0149707, 0.0137648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0069434, 0.0187820, -0.0104765, 0.0118334
1: -0.0039174, 0.0011571, -0.0041974, 0.0013851, -0.0053026, 0.0053545
2: 0.0034030, 0.0090757, 0.0034013, 0.0100584, -0.0066553, 0.0056744
3: -0.0009109, 0.0036682, -0.0015091, 0.0037173, -0.0046282, 0.0051773
4: -0.0051203, -0.0011813, -0.0051387, -0.0009777, -0.0037487, 0.0035282
5: -0.0006129, 0.0039166, -0.0006160, 0.0044893, -0.0051022, 0.0045327
6: -0.0066209, 0.0008647, -0.0067939, 0.0012106, -0.0078315, 0.0076586
7: -0.0254298, -0.0027351, -0.0255066, -0.0015269, -0.0209066, 0.0196633
8: 0.9722265, 0.9940638, 0.9721356, 0.9953537, -0.0231272, 0.0219282
9: -0.0061136, 0.0089945, -0.0069351, 0.0090564, -0.0138783, 0.0147443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125544
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127713
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0083055, 0.0187768, 0.0070348, 0.0187912, -0.0104858, 0.0117420
1: -0.0039174, 0.0011571, -0.0041962, 0.0013731, -0.0052906, 0.0053533
2: 0.0034030, 0.0090757, 0.0033970, 0.0099868, -0.0065838, 0.0056787
3: -0.0009109, 0.0036682, -0.0014337, 0.0037640, -0.0046749, 0.0051019
4: -0.0051203, -0.0011813, -0.0051883, -0.0010215, -0.0037230, 0.0035955
5: -0.0006129, 0.0039166, -0.0006126, 0.0044600, -0.0050729, 0.0045292
6: -0.0066209, 0.0008647, -0.0067363, 0.0013294, -0.0079503, 0.0076009
7: -0.0254298, -0.0027351, -0.0257798, -0.0017740, -0.0207522, 0.0200182
8: 0.9722265, 0.9940638, 0.9719387, 0.9951462, -0.0229197, 0.0221251
9: -0.0061136, 0.0089945, -0.0067789, 0.0092272, -0.0140993, 0.0146451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125577
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127786
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0069434, 0.0187820, -0.0118386, 0.0118386
1: -0.0041974, 0.0013851, -0.0041974, 0.0013851, -0.0055825, 0.0055825
2: 0.0034013, 0.0100584, 0.0034013, 0.0100584, -0.0066571, 0.0066571
3: -0.0015091, 0.0037173, -0.0015091, 0.0037173, -0.0052264, 0.0052264
4: -0.0051387, -0.0009777, -0.0051387, -0.0009777, -0.0037611, 0.0037611
5: -0.0006160, 0.0044893, -0.0006160, 0.0044893, -0.0051054, 0.0051054
6: -0.0067939, 0.0012106, -0.0067939, 0.0012106, -0.0080045, 0.0080045
7: -0.0255066, -0.0015269, -0.0255066, -0.0015269, -0.0211298, 0.0211298
8: 0.9721356, 0.9953537, 0.9721356, 0.9953537, -0.0232181, 0.0232181
9: -0.0069351, 0.0090564, -0.0069351, 0.0090564, -0.0148033, 0.0148033

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125170
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127174
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0069434, 0.0187820, 0.0070348, 0.0187912, -0.0118478, 0.0117472
1: -0.0041974, 0.0013851, -0.0041962, 0.0013731, -0.0055705, 0.0055814
2: 0.0034013, 0.0100584, 0.0033970, 0.0099868, -0.0065855, 0.0066614
3: -0.0015091, 0.0037173, -0.0014337, 0.0037640, -0.0052731, 0.0051510
4: -0.0051387, -0.0009777, -0.0051883, -0.0010215, -0.0037361, 0.0038284
5: -0.0006160, 0.0044893, -0.0006126, 0.0044600, -0.0050760, 0.0051019
6: -0.0067939, 0.0012106, -0.0067363, 0.0013294, -0.0081233, 0.0079469
7: -0.0255066, -0.0015269, -0.0257798, -0.0017740, -0.0209722, 0.0214839
8: 0.9721356, 0.9953537, 0.9719387, 0.9951462, -0.0230106, 0.0234150
9: -0.0069351, 0.0090564, -0.0067789, 0.0092272, -0.0150235, 0.0147049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
time: 1.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0083055, 0.0187768, -0.0103482, 0.0104805
1: -0.0039089, 0.0011429, -0.0039174, 0.0011571, -0.0050660, 0.0050603
2: 0.0033987, 0.0089764, 0.0034030, 0.0090757, -0.0056769, 0.0055734
3: -0.0008197, 0.0037132, -0.0009109, 0.0036682, -0.0044879, 0.0046241
4: -0.0051675, -0.0012301, -0.0051203, -0.0011813, -0.0035666, 0.0034719
5: -0.0006096, 0.0038723, -0.0006129, 0.0039166, -0.0045263, 0.0044852
6: -0.0065548, 0.0009793, -0.0066209, 0.0008647, -0.0074195, 0.0076002
7: -0.0256950, -0.0030151, -0.0254298, -0.0027351, -0.0198710, 0.0193615
8: 0.9720360, 0.9938253, 0.9722265, 0.9940638, -0.0220278, 0.0215988
9: -0.0059282, 0.0091599, -0.0061136, 0.0089945, -0.0136636, 0.0140035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123800, upper bound: 0.0125273
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129743, upper bound: 0.0127449
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0084286, 0.0187860, -0.0103574, 0.0103574
1: -0.0039089, 0.0011429, -0.0039089, 0.0011429, -0.0050518, 0.0050518
2: 0.0033987, 0.0089764, 0.0033987, 0.0089764, -0.0055776, 0.0055776
3: -0.0008197, 0.0037132, -0.0008197, 0.0037132, -0.0045329, 0.0045329
4: -0.0051675, -0.0012301, -0.0051675, -0.0012301, -0.0034933, 0.0034933
5: -0.0006096, 0.0038723, -0.0006096, 0.0038723, -0.0044819, 0.0044819
6: -0.0065548, 0.0009793, -0.0065548, 0.0009793, -0.0075341, 0.0075341
7: -0.0256950, -0.0030151, -0.0256950, -0.0030151, -0.0194595, 0.0194595
8: 0.9720360, 0.9938253, 0.9720360, 0.9938253, -0.0217893, 0.0217893
9: -0.0059282, 0.0091599, -0.0059282, 0.0091599, -0.0137476, 0.0137476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123800, upper bound: 0.0125280
time: 1.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129743, upper bound: 0.0127449
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0083055, 0.0187768, -0.0117420, 0.0104858
1: -0.0041962, 0.0013731, -0.0039174, 0.0011571, -0.0053533, 0.0052906
2: 0.0033970, 0.0099868, 0.0034030, 0.0090757, -0.0056787, 0.0065838
3: -0.0014337, 0.0037640, -0.0009109, 0.0036682, -0.0051019, 0.0046749
4: -0.0051883, -0.0010215, -0.0051203, -0.0011813, -0.0035955, 0.0037230
5: -0.0006126, 0.0044600, -0.0006129, 0.0039166, -0.0045292, 0.0050729
6: -0.0067363, 0.0013294, -0.0066209, 0.0008647, -0.0076009, 0.0079503
7: -0.0257798, -0.0017740, -0.0254298, -0.0027351, -0.0200182, 0.0207522
8: 0.9719387, 0.9951462, 0.9722265, 0.9940638, -0.0221251, 0.0229197
9: -0.0067789, 0.0092272, -0.0061136, 0.0089945, -0.0146451, 0.0140993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125170
time: 1.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0084286, 0.0187860, -0.0117512, 0.0103627
1: -0.0041962, 0.0013731, -0.0039089, 0.0011429, -0.0053391, 0.0052820
2: 0.0033970, 0.0099868, 0.0033987, 0.0089764, -0.0055794, 0.0065881
3: -0.0014337, 0.0037640, -0.0008197, 0.0037132, -0.0051470, 0.0045837
4: -0.0051883, -0.0010215, -0.0051675, -0.0012301, -0.0035233, 0.0037425
5: -0.0006126, 0.0044600, -0.0006096, 0.0038723, -0.0044849, 0.0050696
6: -0.0067363, 0.0013294, -0.0065548, 0.0009793, -0.0077156, 0.0078842
7: -0.0257798, -0.0017740, -0.0256950, -0.0030151, -0.0196173, 0.0208531
8: 0.9719387, 0.9951462, 0.9720360, 0.9938253, -0.0218866, 0.0231102
9: -0.0067789, 0.0092272, -0.0059282, 0.0091599, -0.0147187, 0.0138492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125175
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0069434, 0.0187820, -0.0103534, 0.0118426
1: -0.0039089, 0.0011429, -0.0041974, 0.0013851, -0.0052940, 0.0053403
2: 0.0033987, 0.0089764, 0.0034013, 0.0100584, -0.0066596, 0.0055751
3: -0.0008197, 0.0037132, -0.0015091, 0.0037173, -0.0045370, 0.0052223
4: -0.0051675, -0.0012301, -0.0051387, -0.0009777, -0.0038165, 0.0035013
5: -0.0006096, 0.0038723, -0.0006160, 0.0044893, -0.0050989, 0.0044883
6: -0.0065548, 0.0009793, -0.0067939, 0.0012106, -0.0077654, 0.0077732
7: -0.0256950, -0.0030151, -0.0255066, -0.0015269, -0.0212610, 0.0195081
8: 0.9720360, 0.9938253, 0.9721356, 0.9953537, -0.0233177, 0.0216897
9: -0.0059282, 0.0091599, -0.0069351, 0.0090564, -0.0137647, 0.0149707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125544
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0084286, 0.0187860, 0.0070348, 0.0187912, -0.0103627, 0.0117512
1: -0.0039089, 0.0011429, -0.0041962, 0.0013731, -0.0052820, 0.0053391
2: 0.0033987, 0.0089764, 0.0033970, 0.0099868, -0.0065881, 0.0055794
3: -0.0008197, 0.0037132, -0.0014337, 0.0037640, -0.0045837, 0.0051470
4: -0.0051675, -0.0012301, -0.0051883, -0.0010215, -0.0037425, 0.0035233
5: -0.0006096, 0.0038723, -0.0006126, 0.0044600, -0.0050696, 0.0044849
6: -0.0065548, 0.0009793, -0.0067363, 0.0013294, -0.0078842, 0.0077156
7: -0.0256950, -0.0030151, -0.0257798, -0.0017740, -0.0208531, 0.0196173
8: 0.9720360, 0.9938253, 0.9719387, 0.9951462, -0.0231102, 0.0218866
9: -0.0059282, 0.0091599, -0.0067789, 0.0092272, -0.0138492, 0.0147187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125550
time: 1.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
time: 1.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0069434, 0.0187820, -0.0117472, 0.0118478
1: -0.0041962, 0.0013731, -0.0041974, 0.0013851, -0.0055814, 0.0055705
2: 0.0033970, 0.0099868, 0.0034013, 0.0100584, -0.0066614, 0.0065855
3: -0.0014337, 0.0037640, -0.0015091, 0.0037173, -0.0051510, 0.0052731
4: -0.0051883, -0.0010215, -0.0051387, -0.0009777, -0.0038284, 0.0037361
5: -0.0006126, 0.0044600, -0.0006160, 0.0044893, -0.0051019, 0.0050760
6: -0.0067363, 0.0013294, -0.0067939, 0.0012106, -0.0079469, 0.0081233
7: -0.0257798, -0.0017740, -0.0255066, -0.0015269, -0.0214839, 0.0209722
8: 0.9719387, 0.9951462, 0.9721356, 0.9953537, -0.0234150, 0.0230106
9: -0.0067789, 0.0092272, -0.0069351, 0.0090564, -0.0147049, 0.0150235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125170
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0070348, 0.0187912, 0.0070348, 0.0187912, -0.0117564, 0.0117564
1: -0.0041962, 0.0013731, -0.0041962, 0.0013731, -0.0055693, 0.0055693
2: 0.0033970, 0.0099868, 0.0033970, 0.0099868, -0.0065898, 0.0065898
3: -0.0014337, 0.0037640, -0.0014337, 0.0037640, -0.0051978, 0.0051978
4: -0.0051883, -0.0010215, -0.0051883, -0.0010215, -0.0037562, 0.0037562
5: -0.0006126, 0.0044600, -0.0006126, 0.0044600, -0.0050726, 0.0050726
6: -0.0067363, 0.0013294, -0.0067363, 0.0013294, -0.0080657, 0.0080657
7: -0.0257798, -0.0017740, -0.0257798, -0.0017740, -0.0210878, 0.0210878
8: 0.9719387, 0.9951462, 0.9719387, 0.9951462, -0.0232074, 0.0232074
9: -0.0067789, 0.0092272, -0.0067789, 0.0092272, -0.0147793, 0.0147793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125175
time: 1.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
time: 1.19 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.17 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0120182
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126003
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0120273
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126644
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0120182
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126003
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0120278
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126644
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0123770
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0128582
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0109168, upper bound: 0.0124020
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0129421
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0123770
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128582
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0110483, upper bound: 0.0124020
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129421
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120061
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125937
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120061
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125937
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0110302, upper bound: 0.0120062
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0110302, upper bound: 0.0120062
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125937
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123413
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128285
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123413
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128285
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0110302, upper bound: 0.0123413
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0110302, upper bound: 0.0123413
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128285
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0109176, upper bound: 0.0120215
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0125934
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0109176, upper bound: 0.0120299
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125937, upper bound: 0.0126659
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0120655
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0125840
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0120668
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125802, upper bound: 0.0126532
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0109176, upper bound: 0.0123984
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0128720
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0109176, upper bound: 0.0124276
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125885, upper bound: 0.0129595
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0124059
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0128543
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0111738, upper bound: 0.0124342
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0125750, upper bound: 0.0129376
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120104
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0120104
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0120515
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0120515
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123586
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0108986, upper bound: 0.0123586
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0123698
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0111561, upper bound: 0.0123698
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0120674
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126054
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0119762, upper bound: 0.0120789
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126152
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120003, upper bound: 0.0120038
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125672
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120003, upper bound: 0.0120141
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0120242
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126380
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0118908, upper bound: 0.0120315
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126465
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120000, upper bound: 0.0120025
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125672
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120000, upper bound: 0.0120122
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120418, upper bound: 0.0120690
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0128991, upper bound: 0.0126054
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120418, upper bound: 0.0121212
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0128991, upper bound: 0.0126060
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120718, upper bound: 0.0120063
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120718, upper bound: 0.0120541
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0119569, upper bound: 0.0120279
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0119569, upper bound: 0.0120668
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120715, upper bound: 0.0120053
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0120715, upper bound: 0.0120516
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125273
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127449
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125314
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127518
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125170
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127174
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125544
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127713
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125577
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127786
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125170
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127174
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0123800, upper bound: 0.0125273
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129743, upper bound: 0.0127449
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0123800, upper bound: 0.0125280
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129743, upper bound: 0.0127449
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125170
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125175
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125544
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125550
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125170
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125175
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.17
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0080929, 0.0185533, -0.0103982, 0.0104302
1: -0.0039517, 0.0010343, -0.0039673, 0.0010577, -0.0050094, 0.0050016
2: 0.0034888, 0.0091861, 0.0034788, 0.0092297, -0.0057409, 0.0057073
3: -0.0009779, 0.0036276, -0.0010035, 0.0036382, -0.0046161, 0.0046311
4: -0.0048988, -0.0011364, -0.0049200, -0.0011302, -0.0030353, 0.0031593
5: -0.0005395, 0.0039815, -0.0005518, 0.0040088, -0.0045484, 0.0045333
6: -0.0066769, 0.0006627, -0.0066797, 0.0007150, -0.0073918, 0.0073424
7: -0.0241124, -0.0024790, -0.0242347, -0.0024411, -0.0155837, 0.0165085
8: 0.9734337, 0.9943037, 0.9733176, 0.9943494, -0.0209156, 0.0209861
9: -0.0062750, 0.0081323, -0.0063017, 0.0082136, -0.0124911, 0.0120602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0110390
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0126328
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0068087, 0.0185524, -0.0103973, 0.0117144
1: -0.0039517, 0.0010343, -0.0042306, 0.0012934, -0.0052451, 0.0052649
2: 0.0034888, 0.0091861, 0.0034796, 0.0101545, -0.0066656, 0.0057065
3: -0.0009779, 0.0036276, -0.0015647, 0.0036803, -0.0046581, 0.0051922
4: -0.0048988, -0.0011364, -0.0049252, -0.0009405, -0.0032933, 0.0031876
5: -0.0005395, 0.0039815, -0.0005544, 0.0045500, -0.0050896, 0.0045358
6: -0.0066769, 0.0006627, -0.0068399, 0.0010449, -0.0077218, 0.0075027
7: -0.0241124, -0.0024790, -0.0242427, -0.0013184, -0.0171099, 0.0166984
8: 0.9734337, 0.9943037, 0.9732838, 0.9955425, -0.0221087, 0.0210199
9: -0.0062750, 0.0081323, -0.0070691, 0.0082311, -0.0125793, 0.0130499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0109371
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0126644
time: 1.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083241, 0.0185306, 0.0080929, 0.0185533, -0.0102292, 0.0104376
1: -0.0039318, 0.0010127, -0.0039673, 0.0010577, -0.0049895, 0.0049800
2: 0.0034854, 0.0090551, 0.0034788, 0.0092297, -0.0057443, 0.0055763
3: -0.0008622, 0.0036666, -0.0010035, 0.0036382, -0.0045005, 0.0046701
4: -0.0049351, -0.0012007, -0.0049200, -0.0011302, -0.0031150, 0.0031283
5: -0.0005348, 0.0039160, -0.0005518, 0.0040088, -0.0045437, 0.0044679
6: -0.0065998, 0.0007472, -0.0066797, 0.0007150, -0.0073148, 0.0074269
7: -0.0243155, -0.0028397, -0.0242347, -0.0024411, -0.0161123, 0.0163833
8: 0.9732951, 0.9939808, 0.9733176, 0.9943494, -0.0210543, 0.0206631
9: -0.0060440, 0.0082579, -0.0063017, 0.0082136, -0.0123601, 0.0123099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120789, upper bound: 0.0110390
time: 1.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120789, upper bound: 0.0126328
time: 1.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083241, 0.0185306, 0.0068087, 0.0185524, -0.0102283, 0.0117218
1: -0.0039318, 0.0010127, -0.0042306, 0.0012934, -0.0052252, 0.0052433
2: 0.0034854, 0.0090551, 0.0034796, 0.0101545, -0.0066691, 0.0055755
3: -0.0008622, 0.0036666, -0.0015647, 0.0036803, -0.0045425, 0.0052312
4: -0.0049351, -0.0012007, -0.0049252, -0.0009405, -0.0033730, 0.0031566
5: -0.0005348, 0.0039160, -0.0005544, 0.0045500, -0.0050849, 0.0044704
6: -0.0065998, 0.0007472, -0.0068399, 0.0010449, -0.0076448, 0.0075871
7: -0.0243155, -0.0028397, -0.0242427, -0.0013184, -0.0176385, 0.0165732
8: 0.9732951, 0.9939808, 0.9732838, 0.9955425, -0.0222474, 0.0206970
9: -0.0060440, 0.0082579, -0.0070691, 0.0082311, -0.0124482, 0.0132996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120123, upper bound: 0.0109371
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120123, upper bound: 0.0126644
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0083055, 0.0187768, -0.0106217, 0.0102177
1: -0.0039517, 0.0010343, -0.0039174, 0.0011571, -0.0051089, 0.0049517
2: 0.0034888, 0.0091861, 0.0034030, 0.0090757, -0.0055868, 0.0057831
3: -0.0009779, 0.0036276, -0.0009109, 0.0036682, -0.0046460, 0.0045385
4: -0.0048988, -0.0011364, -0.0051203, -0.0011813, -0.0032973, 0.0034115
5: -0.0005395, 0.0039815, -0.0006129, 0.0039166, -0.0044562, 0.0045943
6: -0.0066769, 0.0006627, -0.0066209, 0.0008647, -0.0075415, 0.0072836
7: -0.0241124, -0.0024790, -0.0254298, -0.0027351, -0.0186095, 0.0180313
8: 0.9734337, 0.9943037, 0.9722265, 0.9940638, -0.0206301, 0.0220772
9: -0.0062750, 0.0081323, -0.0061136, 0.0089945, -0.0134227, 0.0130137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0119762
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0128846
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0069434, 0.0187820, -0.0106268, 0.0115797
1: -0.0039517, 0.0010343, -0.0041974, 0.0013851, -0.0053369, 0.0052317
2: 0.0034888, 0.0091861, 0.0034013, 0.0100584, -0.0065695, 0.0057848
3: -0.0009779, 0.0036276, -0.0015091, 0.0037173, -0.0046951, 0.0051367
4: -0.0048988, -0.0011364, -0.0051387, -0.0009777, -0.0035532, 0.0034477
5: -0.0005395, 0.0039815, -0.0006160, 0.0044893, -0.0050288, 0.0045975
6: -0.0066769, 0.0006627, -0.0067939, 0.0012106, -0.0078875, 0.0074566
7: -0.0241124, -0.0024790, -0.0255066, -0.0015269, -0.0200112, 0.0182528
8: 0.9734337, 0.9943037, 0.9721356, 0.9953537, -0.0219200, 0.0221681
9: -0.0062750, 0.0081323, -0.0069351, 0.0090564, -0.0135515, 0.0140002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0120003
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0129421
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083241, 0.0185306, 0.0083055, 0.0187768, -0.0104527, 0.0102251
1: -0.0039318, 0.0010127, -0.0039174, 0.0011571, -0.0050889, 0.0049301
2: 0.0034854, 0.0090551, 0.0034030, 0.0090757, -0.0055902, 0.0056521
3: -0.0008622, 0.0036666, -0.0009109, 0.0036682, -0.0045304, 0.0045774
4: -0.0049351, -0.0012007, -0.0051203, -0.0011813, -0.0033581, 0.0033805
5: -0.0005348, 0.0039160, -0.0006129, 0.0039166, -0.0044515, 0.0045289
6: -0.0065998, 0.0007472, -0.0066209, 0.0008647, -0.0074645, 0.0073681
7: -0.0243155, -0.0028397, -0.0254298, -0.0027351, -0.0188934, 0.0179061
8: 0.9732951, 0.9939808, 0.9722265, 0.9940638, -0.0207687, 0.0217543
9: -0.0060440, 0.0082579, -0.0061136, 0.0089945, -0.0132916, 0.0132026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120789, upper bound: 0.0119762
time: 1.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120789, upper bound: 0.0128846
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083241, 0.0185306, 0.0069434, 0.0187820, -0.0104579, 0.0115872
1: -0.0039318, 0.0010127, -0.0041974, 0.0013851, -0.0053170, 0.0052100
2: 0.0034854, 0.0090551, 0.0034013, 0.0100584, -0.0065729, 0.0056538
3: -0.0008622, 0.0036666, -0.0015091, 0.0037173, -0.0045795, 0.0051757
4: -0.0049351, -0.0012007, -0.0051387, -0.0009777, -0.0036140, 0.0034167
5: -0.0005348, 0.0039160, -0.0006160, 0.0044893, -0.0050242, 0.0045321
6: -0.0065998, 0.0007472, -0.0067939, 0.0012106, -0.0078105, 0.0075411
7: -0.0243155, -0.0028397, -0.0255066, -0.0015269, -0.0202950, 0.0181276
8: 0.9732951, 0.9939808, 0.9721356, 0.9953537, -0.0220586, 0.0218452
9: -0.0060440, 0.0082579, -0.0069351, 0.0090564, -0.0134204, 0.0141891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120141, upper bound: 0.0120003
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120141, upper bound: 0.0129421
time: 1.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0080929, 0.0185533, -0.0116810, 0.0104289
1: -0.0042149, 0.0012676, -0.0039673, 0.0010577, -0.0052725, 0.0052349
2: 0.0034898, 0.0101097, 0.0034788, 0.0092297, -0.0057399, 0.0066309
3: -0.0015385, 0.0036695, -0.0010035, 0.0036382, -0.0051768, 0.0046730
4: -0.0049032, -0.0009466, -0.0049200, -0.0011302, -0.0030701, 0.0034051
5: -0.0005420, 0.0045222, -0.0005518, 0.0040088, -0.0045508, 0.0050740
6: -0.0068371, 0.0009945, -0.0066797, 0.0007150, -0.0075521, 0.0076742
7: -0.0241178, -0.0013553, -0.0242347, -0.0024411, -0.0158504, 0.0179388
8: 0.9734010, 0.9954968, 0.9733176, 0.9943494, -0.0209484, 0.0221792
9: -0.0070430, 0.0081484, -0.0063017, 0.0082136, -0.0134361, 0.0121664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0109168
time: 1.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0125937
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0068087, 0.0185524, -0.0116801, 0.0117131
1: -0.0042149, 0.0012676, -0.0042306, 0.0012934, -0.0055082, 0.0054983
2: 0.0034898, 0.0101097, 0.0034796, 0.0101545, -0.0066647, 0.0066301
3: -0.0015385, 0.0036695, -0.0015647, 0.0036803, -0.0052188, 0.0052341
4: -0.0049032, -0.0009466, -0.0049252, -0.0009405, -0.0032547, 0.0033660
5: -0.0005420, 0.0045222, -0.0005544, 0.0045500, -0.0050920, 0.0050766
6: -0.0068371, 0.0009945, -0.0068399, 0.0010449, -0.0078820, 0.0078344
7: -0.0241178, -0.0013553, -0.0242427, -0.0013184, -0.0167873, 0.0176324
8: 0.9734010, 0.9954968, 0.9732838, 0.9955425, -0.0221415, 0.0222130
9: -0.0070430, 0.0081484, -0.0070691, 0.0082311, -0.0133055, 0.0129189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0108956
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0125937
time: 1.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0070363, 0.0185290, 0.0080929, 0.0185533, -0.0115171, 0.0104361
1: -0.0041970, 0.0012314, -0.0039673, 0.0010577, -0.0052547, 0.0051987
2: 0.0034867, 0.0099856, 0.0034788, 0.0092297, -0.0057430, 0.0065067
3: -0.0014245, 0.0037094, -0.0010035, 0.0036382, -0.0050627, 0.0047129
4: -0.0049444, -0.0010014, -0.0049200, -0.0011302, -0.0031549, 0.0033752
5: -0.0005369, 0.0044588, -0.0005518, 0.0040088, -0.0045458, 0.0050106
6: -0.0067705, 0.0010740, -0.0066797, 0.0007150, -0.0074854, 0.0077537
7: -0.0243427, -0.0016636, -0.0242347, -0.0024411, -0.0163879, 0.0177992
8: 0.9732470, 0.9952059, 0.9733176, 0.9943494, -0.0211024, 0.0218883
9: -0.0068406, 0.0082877, -0.0063017, 0.0082136, -0.0133124, 0.0124297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0109176
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0125937
time: 1.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0070363, 0.0185290, 0.0068087, 0.0185524, -0.0115161, 0.0117203
1: -0.0041970, 0.0012314, -0.0042306, 0.0012934, -0.0054904, 0.0054620
2: 0.0034867, 0.0099856, 0.0034796, 0.0101545, -0.0066678, 0.0065059
3: -0.0014245, 0.0037094, -0.0015647, 0.0036803, -0.0051047, 0.0052741
4: -0.0049444, -0.0010014, -0.0049252, -0.0009405, -0.0033385, 0.0033362
5: -0.0005369, 0.0044588, -0.0005544, 0.0045500, -0.0050870, 0.0050132
6: -0.0067705, 0.0010740, -0.0068399, 0.0010449, -0.0078154, 0.0079139
7: -0.0243427, -0.0016636, -0.0242427, -0.0013184, -0.0173312, 0.0175141
8: 0.9732470, 0.9952059, 0.9732838, 0.9955425, -0.0222955, 0.0219221
9: -0.0068406, 0.0082877, -0.0070691, 0.0082311, -0.0131818, 0.0131756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0108956
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0125937
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0083055, 0.0187768, -0.0119045, 0.0102163
1: -0.0042149, 0.0012676, -0.0039174, 0.0011571, -0.0053720, 0.0051851
2: 0.0034898, 0.0101097, 0.0034030, 0.0090757, -0.0055859, 0.0067067
3: -0.0015385, 0.0036695, -0.0009109, 0.0036682, -0.0052067, 0.0045804
4: -0.0049032, -0.0009466, -0.0051203, -0.0011813, -0.0033195, 0.0036573
5: -0.0005420, 0.0045222, -0.0006129, 0.0039166, -0.0044586, 0.0051351
6: -0.0068371, 0.0009945, -0.0066209, 0.0008647, -0.0077018, 0.0076154
7: -0.0241178, -0.0013553, -0.0254298, -0.0027351, -0.0187049, 0.0194617
8: 0.9734010, 0.9954968, 0.9722265, 0.9940638, -0.0206628, 0.0232703
9: -0.0070430, 0.0081484, -0.0061136, 0.0089945, -0.0143677, 0.0130813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0118908
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128285
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068723, 0.0185218, 0.0069434, 0.0187820, -0.0119097, 0.0115784
1: -0.0042149, 0.0012676, -0.0041974, 0.0013851, -0.0056000, 0.0054650
2: 0.0034898, 0.0101097, 0.0034013, 0.0100584, -0.0065686, 0.0067084
3: -0.0015385, 0.0036695, -0.0015091, 0.0037173, -0.0052558, 0.0051786
4: -0.0049032, -0.0009466, -0.0051387, -0.0009777, -0.0035584, 0.0036270
5: -0.0005420, 0.0045222, -0.0006160, 0.0044893, -0.0050313, 0.0051382
6: -0.0068371, 0.0009945, -0.0067939, 0.0012106, -0.0080478, 0.0077884
7: -0.0241178, -0.0013553, -0.0255066, -0.0015269, -0.0201823, 0.0191944
8: 0.9734010, 0.9954968, 0.9721356, 0.9953537, -0.0219527, 0.0233612
9: -0.0070430, 0.0081484, -0.0069351, 0.0090564, -0.0142718, 0.0140261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0118908
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128285
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0070363, 0.0185290, 0.0083055, 0.0187768, -0.0117405, 0.0102236
1: -0.0041970, 0.0012314, -0.0039174, 0.0011571, -0.0053541, 0.0051488
2: 0.0034867, 0.0099856, 0.0034030, 0.0090757, -0.0055890, 0.0065825
3: -0.0014245, 0.0037094, -0.0009109, 0.0036682, -0.0050926, 0.0046203
4: -0.0049444, -0.0010014, -0.0051203, -0.0011813, -0.0033825, 0.0036274
5: -0.0005369, 0.0044588, -0.0006129, 0.0039166, -0.0044536, 0.0050717
6: -0.0067705, 0.0010740, -0.0066209, 0.0008647, -0.0076352, 0.0076949
7: -0.0243427, -0.0016636, -0.0254298, -0.0027351, -0.0190187, 0.0193220
8: 0.9732470, 0.9952059, 0.9722265, 0.9940638, -0.0208168, 0.0229794
9: -0.0068406, 0.0082877, -0.0061136, 0.0089945, -0.0142440, 0.0132796

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120315, upper bound: 0.0118908
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120315, upper bound: 0.0128285
time: 1.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0070363, 0.0185290, 0.0069434, 0.0187820, -0.0117457, 0.0115856
1: -0.0041970, 0.0012314, -0.0041974, 0.0013851, -0.0055821, 0.0054287
2: 0.0034867, 0.0099856, 0.0034013, 0.0100584, -0.0065717, 0.0065843
3: -0.0014245, 0.0037094, -0.0015091, 0.0037173, -0.0051418, 0.0052185
4: -0.0049444, -0.0010014, -0.0051387, -0.0009777, -0.0036210, 0.0035972
5: -0.0005369, 0.0044588, -0.0006160, 0.0044893, -0.0050262, 0.0050749
6: -0.0067705, 0.0010740, -0.0067939, 0.0012106, -0.0079811, 0.0078679
7: -0.0243427, -0.0016636, -0.0255066, -0.0015269, -0.0204962, 0.0190761
8: 0.9732470, 0.9952059, 0.9721356, 0.9953537, -0.0221067, 0.0230703
9: -0.0068406, 0.0082877, -0.0069351, 0.0090564, -0.0141481, 0.0142237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120300, upper bound: 0.0118908
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120300, upper bound: 0.0128285
time: 1.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0082565, 0.0185611, -0.0104060, 0.0102667
1: -0.0039517, 0.0010343, -0.0039485, 0.0010368, -0.0049886, 0.0049829
2: 0.0034888, 0.0091861, 0.0034755, 0.0091027, -0.0056138, 0.0057106
3: -0.0009779, 0.0036276, -0.0008895, 0.0036775, -0.0046553, 0.0045171
4: -0.0048988, -0.0011364, -0.0049572, -0.0011940, -0.0030180, 0.0032292
5: -0.0005395, 0.0039815, -0.0005470, 0.0039456, -0.0044851, 0.0045284
6: -0.0066769, 0.0006627, -0.0066028, 0.0008008, -0.0074777, 0.0072655
7: -0.0241124, -0.0024790, -0.0244393, -0.0027993, -0.0155572, 0.0169506
8: 0.9734337, 0.9943037, 0.9731785, 0.9940299, -0.0205962, 0.0211252
9: -0.0062750, 0.0081323, -0.0060726, 0.0083402, -0.0127104, 0.0119757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0111862
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0126255
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0069627, 0.0185597, -0.0104045, 0.0115604
1: -0.0039517, 0.0010343, -0.0042152, 0.0012591, -0.0052108, 0.0052495
2: 0.0034888, 0.0091861, 0.0034765, 0.0100375, -0.0065487, 0.0057096
3: -0.0009779, 0.0036276, -0.0014544, 0.0037225, -0.0047004, 0.0050820
4: -0.0048988, -0.0011364, -0.0049666, -0.0009945, -0.0035227, 0.0032587
5: -0.0005395, 0.0039815, -0.0005493, 0.0044909, -0.0050305, 0.0045307
6: -0.0066769, 0.0006627, -0.0067734, 0.0011315, -0.0078084, 0.0074362
7: -0.0241124, -0.0024790, -0.0244683, -0.0016217, -0.0198915, 0.0171371
8: 0.9734337, 0.9943037, 0.9731277, 0.9952570, -0.0218233, 0.0211760
9: -0.0062750, 0.0081323, -0.0068703, 0.0083713, -0.0128018, 0.0138887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0110722
time: 1.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0126659
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083241, 0.0185306, 0.0082565, 0.0185611, -0.0102370, 0.0102741
1: -0.0039318, 0.0010127, -0.0039485, 0.0010368, -0.0049686, 0.0049612
2: 0.0034854, 0.0090551, 0.0034755, 0.0091027, -0.0056173, 0.0055797
3: -0.0008622, 0.0036666, -0.0008895, 0.0036775, -0.0045397, 0.0045561
4: -0.0049351, -0.0012007, -0.0049572, -0.0011940, -0.0030145, 0.0031358
5: -0.0005348, 0.0039160, -0.0005470, 0.0039456, -0.0044805, 0.0044630
6: -0.0065998, 0.0007472, -0.0066028, 0.0008008, -0.0074007, 0.0073499
7: -0.0243155, -0.0028397, -0.0244393, -0.0027993, -0.0154343, 0.0163456
8: 0.9732951, 0.9939808, 0.9731785, 0.9940299, -0.0207348, 0.0208023
9: -0.0060440, 0.0082579, -0.0060726, 0.0083402, -0.0123945, 0.0119757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121236, upper bound: 0.0113139
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121236, upper bound: 0.0126167
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083241, 0.0185306, 0.0069627, 0.0185597, -0.0102356, 0.0115678
1: -0.0039318, 0.0010127, -0.0042152, 0.0012591, -0.0051909, 0.0052278
2: 0.0034854, 0.0090551, 0.0034765, 0.0100375, -0.0065521, 0.0055787
3: -0.0008622, 0.0036666, -0.0014544, 0.0037225, -0.0045848, 0.0051209
4: -0.0049351, -0.0012007, -0.0049666, -0.0009945, -0.0035345, 0.0031660
5: -0.0005348, 0.0039160, -0.0005493, 0.0044909, -0.0050258, 0.0044653
6: -0.0065998, 0.0007472, -0.0067734, 0.0011315, -0.0077314, 0.0075206
7: -0.0243155, -0.0028397, -0.0244683, -0.0016217, -0.0199458, 0.0165396
8: 0.9732951, 0.9939808, 0.9731277, 0.9952570, -0.0219619, 0.0208530
9: -0.0060440, 0.0082579, -0.0068703, 0.0083713, -0.0124870, 0.0139372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120561, upper bound: 0.0111859
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120561, upper bound: 0.0126532
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0084286, 0.0187860, -0.0106309, 0.0100946
1: -0.0039517, 0.0010343, -0.0039089, 0.0011429, -0.0050946, 0.0049432
2: 0.0034888, 0.0091861, 0.0033987, 0.0089764, -0.0054875, 0.0057874
3: -0.0009779, 0.0036276, -0.0008197, 0.0037132, -0.0046911, 0.0044472
4: -0.0048988, -0.0011364, -0.0051675, -0.0012301, -0.0032781, 0.0034896
5: -0.0005395, 0.0039815, -0.0006096, 0.0038723, -0.0044118, 0.0045911
6: -0.0066769, 0.0006627, -0.0065548, 0.0009793, -0.0076562, 0.0072175
7: -0.0241124, -0.0024790, -0.0256950, -0.0030151, -0.0185029, 0.0185110
8: 0.9734337, 0.9943037, 0.9720360, 0.9938253, -0.0203916, 0.0222677
9: -0.0062750, 0.0081323, -0.0059282, 0.0091599, -0.0136765, 0.0129264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0120418
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0128991
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0081551, 0.0185231, 0.0070348, 0.0187912, -0.0106361, 0.0114883
1: -0.0039517, 0.0010343, -0.0041962, 0.0013731, -0.0053249, 0.0052305
2: 0.0034888, 0.0091861, 0.0033970, 0.0099868, -0.0064980, 0.0057891
3: -0.0009779, 0.0036276, -0.0014337, 0.0037640, -0.0047419, 0.0050613
4: -0.0048988, -0.0011364, -0.0051883, -0.0010215, -0.0035373, 0.0035302
5: -0.0005395, 0.0039815, -0.0006126, 0.0044600, -0.0049995, 0.0045940
6: -0.0066769, 0.0006627, -0.0067363, 0.0013294, -0.0080063, 0.0073990
7: -0.0241124, -0.0024790, -0.0257798, -0.0017740, -0.0199195, 0.0187569
8: 0.9734337, 0.9943037, 0.9719387, 0.9951462, -0.0217124, 0.0223650
9: -0.0062750, 0.0081323, -0.0067789, 0.0092272, -0.0138144, 0.0139329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0120718
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0129595
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0083241, 0.0185306, 0.0084286, 0.0187860, -0.0104619, 0.0101020
1: -0.0039318, 0.0010127, -0.0039089, 0.0011429, -0.0050747, 0.0049216
2: 0.0034854, 0.0090551, 0.0033987, 0.0089764, -0.0054910, 0.0056564
3: -0.0008622, 0.0036666, -0.0008197, 0.0037132, -0.0045755, 0.0044862
4: -0.0049351, -0.0012007, -0.0051675, -0.0012301, -0.0032887, 0.0033930
5: -0.0005348, 0.0039160, -0.0006096, 0.0038723, -0.0044071, 0.0045257
6: -0.0065998, 0.0007472, -0.0065548, 0.0009793, -0.0075792, 0.0073020
7: -0.0243155, -0.0028397, -0.0256950, -0.0030151, -0.0185360, 0.0178886
8: 0.9732951, 0.9939808, 0.9720360, 0.9938253, -0.0205302, 0.0219448
9: -0.0060440, 0.0082579, -0.0059282, 0.0091599, -0.0133441, 0.0129668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0121236, upper bound: 0.0120473
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0121236, upper bound: 0.0128809
time: 1.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0083241, 0.0185306, 0.0070348, 0.0187912, -0.0104671, 0.0114958
1: -0.0039318, 0.0010127, -0.0041962, 0.0013731, -0.0053049, 0.0052089
2: 0.0034854, 0.0090551, 0.0033970, 0.0099868, -0.0065014, 0.0056581
3: -0.0008622, 0.0036666, -0.0014337, 0.0037640, -0.0046263, 0.0051003
4: -0.0049351, -0.0012007, -0.0051883, -0.0010215, -0.0035476, 0.0034295
5: -0.0005348, 0.0039160, -0.0006126, 0.0044600, -0.0049948, 0.0045286
6: -0.0065998, 0.0007472, -0.0067363, 0.0013294, -0.0079293, 0.0074834
7: -0.0243155, -0.0028397, -0.0257798, -0.0017740, -0.0199548, 0.0181075
8: 0.9732951, 0.9939808, 0.9719387, 0.9951462, -0.0218511, 0.0220420
9: -0.0060440, 0.0082579, -0.0067789, 0.0092272, -0.0134722, 0.0139745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 86

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0120562, upper bound: 0.0120622
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0120562, upper bound: 0.0129376
time: 1.73 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.18 seconds
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0110390
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0126328
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0109371
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0126644
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120789, upper bound: 0.0110390
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120789, upper bound: 0.0126328
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120123, upper bound: 0.0109371
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120123, upper bound: 0.0126644
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0119762
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0128846
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0120003
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0129421
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120789, upper bound: 0.0119762
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120789, upper bound: 0.0128846
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120141, upper bound: 0.0120003
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120141, upper bound: 0.0129421
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0109168
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0125937
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0108956
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120261, upper bound: 0.0125937
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0109176
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120299, upper bound: 0.0125937
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0108956
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120273, upper bound: 0.0125937
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0118908
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128285
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0118908
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120298, upper bound: 0.0128285
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120315, upper bound: 0.0118908
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120315, upper bound: 0.0128285
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120300, upper bound: 0.0118908
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120300, upper bound: 0.0128285
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0111862
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0126255
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0110722
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120084, upper bound: 0.0126659
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0121236, upper bound: 0.0113139
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0121236, upper bound: 0.0126167
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120561, upper bound: 0.0111859
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120561, upper bound: 0.0126532
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0120418
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120751, upper bound: 0.0128991
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0120718
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120117, upper bound: 0.0129595
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0121236, upper bound: 0.0120473
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0121236, upper bound: 0.0128809
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120562, upper bound: 0.0120622
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 8, lower bound: -0.0120562, upper bound: 0.0129376
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0126644, upper bound: 0.0125802
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0126659, upper bound: 0.0125735
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0126498, upper bound: 0.0128323
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0126465, upper bound: 0.0128217
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126054
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0128846, upper bound: 0.0126152
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125672
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126380
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0128285, upper bound: 0.0126465
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125672
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129421, upper bound: 0.0125750
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0128991, upper bound: 0.0126054
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0128991, upper bound: 0.0126060
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0128323, upper bound: 0.0126380
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129595, upper bound: 0.0125672
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125273
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127449
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0123432, upper bound: 0.0125314
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129646, upper bound: 0.0127518
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125170
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127174
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125544
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127713
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0123131, upper bound: 0.0125577
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129180, upper bound: 0.0127786
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125170
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127174
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0124169, upper bound: 0.0125207
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0130127, upper bound: 0.0127229
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0123800, upper bound: 0.0125273
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129743, upper bound: 0.0127449
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0123800, upper bound: 0.0125280
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129743, upper bound: 0.0127449
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125170
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125175
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125544
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0123429, upper bound: 0.0125550
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0129155, upper bound: 0.0127713
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125170
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0124597, upper bound: 0.0125175
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.18
Output dim: 8, lower bound: -0.0130274, upper bound: 0.0127174

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.61 + 598.03 = 601.64 seconds
