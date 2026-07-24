## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0157495625


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9730924, 0.9962217, 0.9730924, 0.9962217, -0.0231293, 0.0231293)
1: (-0.0046436, -0.0022054, -0.0046436, -0.0022054, -0.0024382, 0.0024382)
2: (0.0016335, 0.0145545, 0.0016335, 0.0145545, -0.0129210, 0.0129210)
3: (-0.0081189, -0.0020166, -0.0081189, -0.0020166, -0.0061023, 0.0061023)
4: (0.0008441, 0.0043016, 0.0008441, 0.0043016, -0.0034575, 0.0034575)
5: (0.0010140, 0.0271774, 0.0010140, 0.0271774, -0.0261635, 0.0261635)
6: (-0.0028412, 0.0012835, -0.0028412, 0.0012835, -0.0041247, 0.0041247)
7: (-0.0104888, 0.0001831, -0.0104888, 0.0001831, -0.0106719, 0.0106719)
8: (-0.0050801, 0.0005322, -0.0050801, 0.0005322, -0.0056123, 0.0056123)
9: (-0.0024809, 0.0040268, -0.0024809, 0.0040268, -0.0065077, 0.0065077)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.33 + 2.26 = 3.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0179995, upper bound: 0.0179995

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166656, upper bound: 0.0161717
time: 1.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160624, upper bound: 0.0160623
time: 1.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.00
Output dim: 0, lower bound: -0.0166656, upper bound: 0.0161717
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.00
Output dim: 0, lower bound: -0.0160624, upper bound: 0.0160623

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9733640, 0.9960400, 0.9730924, 0.9962217, -0.0228577, 0.0229476
1: -0.0046396, -0.0022507, -0.0046436, -0.0022054, -0.0024342, 0.0023929
2: 0.0018734, 0.0145335, 0.0016335, 0.0145545, -0.0126810, 0.0129000
3: -0.0081051, -0.0021258, -0.0081189, -0.0020166, -0.0060885, 0.0059930
4: 0.0008905, 0.0042792, 0.0008441, 0.0043016, -0.0034111, 0.0034351
5: 0.0013158, 0.0269612, 0.0010140, 0.0271774, -0.0258617, 0.0259472
6: -0.0028346, 0.0012069, -0.0028412, 0.0012835, -0.0041180, 0.0040481
7: -0.0104715, -0.0000151, -0.0104888, 0.0001831, -0.0106546, 0.0104737
8: -0.0050710, 0.0004279, -0.0050801, 0.0005322, -0.0056032, 0.0055080
9: -0.0023601, 0.0040162, -0.0024809, 0.0040268, -0.0063868, 0.0064971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160624, upper bound: 0.0160624
time: 1.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160624, upper bound: 0.0160623
time: 1.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9439012, 0.9955350, 0.9733753, 0.9960068, -0.0521056, 0.0221598
1: -0.0050679, -0.0019527, -0.0046395, -0.0022590, -0.0028090, 0.0026867
2: 0.0025403, 0.0168035, 0.0019173, 0.0145327, -0.0119923, 0.0148862
3: -0.0095982, -0.0024294, -0.0081045, -0.0021458, -0.0074524, 0.0056752
4: 0.0010196, 0.0067076, 0.0008990, 0.0042783, -0.0032587, 0.0058086
5: 0.0021545, 0.0504245, 0.0013708, 0.0269523, -0.0247977, 0.0490536
6: -0.0035592, 0.0015914, -0.0028343, 0.0011929, -0.0047521, 0.0044257
7: -0.0123463, -0.0005659, -0.0104708, -0.0000512, -0.0122951, 0.0099049
8: -0.0060570, 0.0093595, -0.0050706, 0.0004089, -0.0064659, 0.0144302
9: -0.0020242, 0.0051595, -0.0023380, 0.0040158, -0.0060400, 0.0074975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0150382, upper bound: 0.0149961
time: 1.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158412, upper bound: 0.0158412
time: 1.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.81 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 0, lower bound: -0.0160624, upper bound: 0.0160624
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 0, lower bound: -0.0160624, upper bound: 0.0160623
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 3.81
Output dim: 0, lower bound: -0.0150382, upper bound: 0.0149961
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 0, lower bound: -0.0158412, upper bound: 0.0158412

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9733640, 0.9960400, 0.9733640, 0.9960400, -0.0226760, 0.0226760
1: -0.0046396, -0.0022507, -0.0046396, -0.0022507, -0.0023889, 0.0023889
2: 0.0018734, 0.0145335, 0.0018734, 0.0145335, -0.0126601, 0.0126601
3: -0.0081051, -0.0021258, -0.0081051, -0.0021258, -0.0059793, 0.0059793
4: 0.0008905, 0.0042792, 0.0008905, 0.0042792, -0.0033887, 0.0033887
5: 0.0013158, 0.0269612, 0.0013158, 0.0269612, -0.0256454, 0.0256454
6: -0.0028346, 0.0012069, -0.0028346, 0.0012069, -0.0040414, 0.0040414
7: -0.0104715, -0.0000151, -0.0104715, -0.0000151, -0.0104564, 0.0104564
8: -0.0050710, 0.0004279, -0.0050710, 0.0004279, -0.0054989, 0.0054989
9: -0.0023601, 0.0040162, -0.0023601, 0.0040162, -0.0063763, 0.0063763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158753, upper bound: 0.0152317
time: 1.42 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164437, upper bound: 0.0159519
time: 1.39 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9733640, 0.9960400, 0.9439012, 0.9955350, -0.0221710, 0.0521387
1: -0.0046396, -0.0022507, -0.0050679, -0.0019527, -0.0026869, 0.0028173
2: 0.0018734, 0.0145335, 0.0025403, 0.0168035, -0.0149300, 0.0119932
3: -0.0081051, -0.0021258, -0.0095982, -0.0024294, -0.0056757, 0.0074723
4: 0.0008905, 0.0042792, 0.0010196, 0.0067076, -0.0058171, 0.0032596
5: 0.0013158, 0.0269612, 0.0021545, 0.0504245, -0.0491087, 0.0248066
6: -0.0028346, 0.0012069, -0.0035592, 0.0015914, -0.0044260, 0.0047661
7: -0.0104715, -0.0000151, -0.0123463, -0.0005659, -0.0099056, 0.0123313
8: -0.0050710, 0.0004279, -0.0060570, 0.0093595, -0.0144305, 0.0064849
9: -0.0023601, 0.0040162, -0.0020242, 0.0051595, -0.0075195, 0.0060404

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0158753, upper bound: 0.0152317
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164437, upper bound: 0.0159519
time: 1.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9439012, 0.9955350, 0.9736376, 0.9958029, -0.0519016, 0.0218974
1: -0.0050679, -0.0019527, -0.0046356, -0.0023098, -0.0027582, 0.0026829
2: 0.0025403, 0.0168035, 0.0021866, 0.0145124, -0.0119721, 0.0146169
3: -0.0095982, -0.0024294, -0.0080912, -0.0022684, -0.0073298, 0.0056619
4: 0.0010196, 0.0067076, 0.0009511, 0.0042566, -0.0032371, 0.0057565
5: 0.0021545, 0.0504245, 0.0017096, 0.0267432, -0.0245887, 0.0487149
6: -0.0035592, 0.0015914, -0.0028278, 0.0011069, -0.0046661, 0.0044192
7: -0.0123463, -0.0005659, -0.0104541, -0.0002737, -0.0120726, 0.0098882
8: -0.0060570, 0.0093595, -0.0050618, 0.0002919, -0.0063489, 0.0144214
9: -0.0020242, 0.0051595, -0.0022023, 0.0040056, -0.0060298, 0.0073618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0149961, upper bound: 0.0150382
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0149961, upper bound: 0.0158412
time: 1.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.82 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0158753, upper bound: 0.0152317
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0164437, upper bound: 0.0159519
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0158753, upper bound: 0.0152317
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0164437, upper bound: 0.0159519
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0149961, upper bound: 0.0150382
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 0, lower bound: -0.0149961, upper bound: 0.0158412

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9672669, 0.9953922, 0.9739169, 0.9958201, -0.0285532, 0.0214753
1: -0.0047283, -0.0024121, -0.0046316, -0.0023055, -0.0024228, 0.0022195
2: 0.0027289, 0.0150033, 0.0021638, 0.0144909, -0.0117620, 0.0128395
3: -0.0084141, -0.0025152, -0.0080771, -0.0022580, -0.0061561, 0.0055619
4: 0.0010561, 0.0047817, 0.0009467, 0.0042336, -0.0031775, 0.0038350
5: 0.0023917, 0.0318167, 0.0016809, 0.0265208, -0.0241291, 0.0301357
6: -0.0029845, 0.0009338, -0.0028210, 0.0011142, -0.0040987, 0.0037548
7: -0.0108595, -0.0007216, -0.0104363, -0.0002549, -0.0106046, 0.0097147
8: -0.0052750, 0.0018343, -0.0050525, 0.0003018, -0.0055769, 0.0068868
9: -0.0019292, 0.0042528, -0.0022138, 0.0039948, -0.0059240, 0.0064666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168153, upper bound: 0.0168671
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168185, upper bound: 0.0167500
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9736274, 0.9958317, 0.9733640, 0.9960400, -0.0224126, 0.0224677
1: -0.0046358, -0.0023026, -0.0046396, -0.0022507, -0.0023851, 0.0023371
2: 0.0021484, 0.0145133, 0.0018734, 0.0145335, -0.0123852, 0.0126398
3: -0.0080918, -0.0022510, -0.0081051, -0.0021258, -0.0059659, 0.0058542
4: 0.0009437, 0.0042575, 0.0008905, 0.0042792, -0.0033355, 0.0033670
5: 0.0016615, 0.0267515, 0.0013158, 0.0269612, -0.0252996, 0.0254357
6: -0.0028281, 0.0011191, -0.0028346, 0.0012069, -0.0040350, 0.0039537
7: -0.0104548, -0.0002421, -0.0104715, -0.0000151, -0.0104397, 0.0102294
8: -0.0050622, 0.0003085, -0.0050710, 0.0004279, -0.0054901, 0.0053795
9: -0.0022216, 0.0040060, -0.0023601, 0.0040162, -0.0062378, 0.0063661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171804, upper bound: 0.0172268
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0171804, upper bound: 0.0176115
time: 1.69 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9672669, 0.9953922, 0.9444634, 0.9953202, -0.0280533, 0.0509288
1: -0.0047283, -0.0024121, -0.0050598, -0.0019849, -0.0027433, 0.0026477
2: 0.0027289, 0.0150033, 0.0028239, 0.0167602, -0.0140312, 0.0121794
3: -0.0084141, -0.0025152, -0.0095697, -0.0025584, -0.0058556, 0.0070545
4: 0.0010561, 0.0047817, 0.0010744, 0.0066613, -0.0056052, 0.0037073
5: 0.0023917, 0.0318167, 0.0025112, 0.0499769, -0.0475852, 0.0293055
6: -0.0029845, 0.0009338, -0.0035454, 0.0015385, -0.0045230, 0.0044791
7: -0.0108595, -0.0007216, -0.0123106, -0.0008001, -0.0100594, 0.0115889
8: -0.0052750, 0.0018343, -0.0060381, 0.0091785, -0.0144536, 0.0078724
9: -0.0019292, 0.0042528, -0.0018814, 0.0051377, -0.0070669, 0.0061342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155161, upper bound: 0.0150015
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0155392, upper bound: 0.0149035
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9736274, 0.9958317, 0.9439012, 0.9955350, -0.0219076, 0.0519305
1: -0.0046358, -0.0023026, -0.0050679, -0.0019527, -0.0026831, 0.0027654
2: 0.0021484, 0.0145133, 0.0025403, 0.0168035, -0.0146551, 0.0119729
3: -0.0080918, -0.0022510, -0.0095982, -0.0024294, -0.0056624, 0.0073472
4: 0.0009437, 0.0042575, 0.0010196, 0.0067076, -0.0057639, 0.0032379
5: 0.0016615, 0.0267515, 0.0021545, 0.0504245, -0.0487630, 0.0245970
6: -0.0028281, 0.0011191, -0.0035592, 0.0015914, -0.0044195, 0.0046783
7: -0.0104548, -0.0002421, -0.0123463, -0.0005659, -0.0098889, 0.0121042
8: -0.0050622, 0.0003085, -0.0060570, 0.0093595, -0.0144217, 0.0063655
9: -0.0022216, 0.0040060, -0.0020242, 0.0051595, -0.0073811, 0.0060302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157498, upper bound: 0.0151418
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0157498, upper bound: 0.0159519
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9441786, 0.9953268, 0.9736376, 0.9958029, -0.0516243, 0.0216892
1: -0.0050639, -0.0019686, -0.0046356, -0.0023098, -0.0027541, 0.0026670
2: 0.0028151, 0.0167821, 0.0021866, 0.0145124, -0.0116974, 0.0145955
3: -0.0095841, -0.0025544, -0.0080912, -0.0022684, -0.0073157, 0.0055368
4: 0.0010727, 0.0066847, 0.0009511, 0.0042566, -0.0031839, 0.0057336
5: 0.0025001, 0.0502036, 0.0017096, 0.0267432, -0.0242431, 0.0484940
6: -0.0035524, 0.0015653, -0.0028278, 0.0011069, -0.0046593, 0.0043931
7: -0.0123287, -0.0007928, -0.0104541, -0.0002737, -0.0120550, 0.0096613
8: -0.0060477, 0.0092702, -0.0050618, 0.0002919, -0.0063396, 0.0143321
9: -0.0018858, 0.0051487, -0.0022023, 0.0040056, -0.0058914, 0.0073511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0141912, upper bound: 0.0153128
time: 1.47 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0141606, upper bound: 0.0151192
time: 1.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.14 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0168153, upper bound: 0.0168671
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0168185, upper bound: 0.0167500
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0171804, upper bound: 0.0172268
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0171804, upper bound: 0.0176115
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0155161, upper bound: 0.0150015
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0155392, upper bound: 0.0149035
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0157498, upper bound: 0.0151418
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0157498, upper bound: 0.0159519
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0141912, upper bound: 0.0153128
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.14
Output dim: 0, lower bound: -0.0141606, upper bound: 0.0151192

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9673066, 0.9953672, 0.9743503, 0.9955707, -0.0282640, 0.0210169
1: -0.0047277, -0.0024183, -0.0046253, -0.0023676, -0.0023600, 0.0022069
2: 0.0027619, 0.0150002, 0.0024933, 0.0144576, -0.0116957, 0.0125070
3: -0.0084121, -0.0025302, -0.0080551, -0.0024079, -0.0060041, 0.0055249
4: 0.0010624, 0.0047785, 0.0010105, 0.0041979, -0.0031355, 0.0037680
5: 0.0024332, 0.0317851, 0.0020953, 0.0261757, -0.0237425, 0.0296898
6: -0.0029835, 0.0009233, -0.0028103, 0.0010090, -0.0039926, 0.0037336
7: -0.0108570, -0.0007489, -0.0104088, -0.0005270, -0.0103300, 0.0096599
8: -0.0052737, 0.0018215, -0.0050380, 0.0001587, -0.0054324, 0.0068595
9: -0.0019126, 0.0042513, -0.0020479, 0.0039780, -0.0058906, 0.0062992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166060, upper bound: 0.0164796
time: 1.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165606, upper bound: 0.0165689
time: 1.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9675197, 0.9952746, 0.9665877, 0.9955329, -0.0280132, 0.0286869
1: -0.0047246, -0.0024414, -0.0047381, -0.0023771, -0.0023475, 0.0022967
2: 0.0028841, 0.0149838, 0.0025431, 0.0150556, -0.0121715, 0.0124407
3: -0.0084013, -0.0025858, -0.0084485, -0.0024306, -0.0059706, 0.0058627
4: 0.0010861, 0.0047609, 0.0010201, 0.0048377, -0.0037516, 0.0037408
5: 0.0025869, 0.0316154, 0.0021580, 0.0323576, -0.0297707, 0.0294574
6: -0.0029783, 0.0008843, -0.0030012, 0.0009931, -0.0039714, 0.0038855
7: -0.0108434, -0.0008498, -0.0109027, -0.0005682, -0.0102752, 0.0100529
8: -0.0052666, 0.0017529, -0.0052978, 0.0020531, -0.0073196, 0.0070507
9: -0.0018510, 0.0042430, -0.0020228, 0.0042792, -0.0061302, 0.0062658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166108, upper bound: 0.0164009
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165641, upper bound: 0.0164923
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9736274, 0.9958317, 0.9672669, 0.9953922, -0.0217648, 0.0285648
1: -0.0046358, -0.0023026, -0.0047283, -0.0024121, -0.0022237, 0.0024257
2: 0.0021484, 0.0145133, 0.0027289, 0.0150033, -0.0128549, 0.0117843
3: -0.0080918, -0.0022510, -0.0084141, -0.0025152, -0.0055766, 0.0061631
4: 0.0009437, 0.0042575, 0.0010561, 0.0047817, -0.0038380, 0.0032014
5: 0.0016615, 0.0267515, 0.0023917, 0.0318167, -0.0301551, 0.0243598
6: -0.0028281, 0.0011191, -0.0029845, 0.0009338, -0.0037619, 0.0041036
7: -0.0104548, -0.0002421, -0.0108595, -0.0007216, -0.0097331, 0.0106174
8: -0.0050622, 0.0003085, -0.0052750, 0.0018343, -0.0068965, 0.0055836
9: -0.0022216, 0.0040060, -0.0019292, 0.0042528, -0.0064744, 0.0059352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168671, upper bound: 0.0168153
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167500, upper bound: 0.0168185
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9736274, 0.9958317, 0.9736274, 0.9958317, -0.0222043, 0.0222043
1: -0.0046358, -0.0023026, -0.0046358, -0.0023026, -0.0023332, 0.0023332
2: 0.0021484, 0.0145133, 0.0021484, 0.0145133, -0.0123649, 0.0123649
3: -0.0080918, -0.0022510, -0.0080918, -0.0022510, -0.0058408, 0.0058408
4: 0.0009437, 0.0042575, 0.0009437, 0.0042575, -0.0033138, 0.0033138
5: 0.0016615, 0.0267515, 0.0016615, 0.0267515, -0.0250900, 0.0250900
6: -0.0028281, 0.0011191, -0.0028281, 0.0011191, -0.0039472, 0.0039472
7: -0.0104548, -0.0002421, -0.0104548, -0.0002421, -0.0102126, 0.0102126
8: -0.0050622, 0.0003085, -0.0050622, 0.0003085, -0.0053707, 0.0053707
9: -0.0022216, 0.0040060, -0.0022216, 0.0040060, -0.0062276, 0.0062276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168671, upper bound: 0.0171201
time: 2.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167500, upper bound: 0.0171236
time: 1.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9736274, 0.9958317, 0.9377449, 0.9948803, -0.0212529, 0.0580868
1: -0.0046358, -0.0023026, -0.0051574, -0.0015999, -0.0030359, 0.0028549
2: 0.0021484, 0.0145133, 0.0034047, 0.0172778, -0.0151294, 0.0111085
3: -0.0080918, -0.0022510, -0.0099101, -0.0028228, -0.0052690, 0.0076592
4: 0.0009437, 0.0042575, 0.0011869, 0.0072150, -0.0062713, 0.0030706
5: 0.0016615, 0.0267515, 0.0032417, 0.0553273, -0.0536657, 0.0235098
6: -0.0028281, 0.0011191, -0.0037106, 0.0021713, -0.0049994, 0.0048297
7: -0.0104548, -0.0002421, -0.0127381, -0.0012798, -0.0091750, 0.0124959
8: -0.0050622, 0.0003085, -0.0062630, 0.0113423, -0.0164045, 0.0065715
9: -0.0022216, 0.0040060, -0.0015888, 0.0053984, -0.0076200, 0.0055948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=6, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0152672, upper bound: 0.0145745
time: 1.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0151435, upper bound: 0.0145589
time: 1.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9736274, 0.9958317, 0.9441786, 0.9953268, -0.0216994, 0.0516531
1: -0.0046358, -0.0023026, -0.0050639, -0.0019686, -0.0026672, 0.0027614
2: 0.0021484, 0.0145133, 0.0028151, 0.0167821, -0.0146337, 0.0116982
3: -0.0080918, -0.0022510, -0.0095841, -0.0025544, -0.0055374, 0.0073331
4: 0.0009437, 0.0042575, 0.0010727, 0.0066847, -0.0057410, 0.0031848
5: 0.0016615, 0.0267515, 0.0025001, 0.0502036, -0.0485421, 0.0242514
6: -0.0028281, 0.0011191, -0.0035524, 0.0015653, -0.0043934, 0.0046715
7: -0.0104548, -0.0002421, -0.0123287, -0.0007928, -0.0096620, 0.0120865
8: -0.0050622, 0.0003085, -0.0060477, 0.0092702, -0.0143324, 0.0063562
9: -0.0022216, 0.0040060, -0.0018858, 0.0051487, -0.0073703, 0.0058918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=5, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0152672, upper bound: 0.0152617
time: 1.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0151435, upper bound: 0.0152515
time: 1.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.43 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0166060, upper bound: 0.0164796
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0165606, upper bound: 0.0165689
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0166108, upper bound: 0.0164009
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0165641, upper bound: 0.0164923
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0168671, upper bound: 0.0168153
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0167500, upper bound: 0.0168185
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0168671, upper bound: 0.0171201
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0167500, upper bound: 0.0171236
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0152672, upper bound: 0.0145745
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0151435, upper bound: 0.0145589
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0152672, upper bound: 0.0152617
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.43
Output dim: 0, lower bound: -0.0151435, upper bound: 0.0152515

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9679046, 0.9950094, 0.9745117, 0.9954812, -0.0275766, 0.0204977
1: -0.0047190, -0.0025075, -0.0046229, -0.0023899, -0.0023290, 0.0021154
2: 0.0032343, 0.0149542, 0.0026114, 0.0144451, -0.0112108, 0.0123428
3: -0.0083818, -0.0027452, -0.0080470, -0.0024617, -0.0059201, 0.0053017
4: 0.0011539, 0.0047292, 0.0010333, 0.0041846, -0.0030307, 0.0036959
5: 0.0030273, 0.0313089, 0.0022439, 0.0260471, -0.0230198, 0.0290650
6: -0.0029688, 0.0007725, -0.0028063, 0.0009713, -0.0039401, 0.0035788
7: -0.0108189, -0.0011390, -0.0103985, -0.0006246, -0.0101944, 0.0092595
8: -0.0052537, 0.0016290, -0.0050326, 0.0001074, -0.0053611, 0.0066616
9: -0.0016747, 0.0042281, -0.0019884, 0.0039717, -0.0056464, 0.0062165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164796
time: 1.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9633706, 0.9951404, 0.9745053, 0.9954847, -0.0321141, 0.0206351
1: -0.0047849, -0.0024748, -0.0046230, -0.0023890, -0.0023959, 0.0021482
2: 0.0030613, 0.0153035, 0.0026066, 0.0144456, -0.0113843, 0.0126968
3: -0.0086115, -0.0026665, -0.0080473, -0.0024595, -0.0061520, 0.0053808
4: 0.0011204, 0.0051029, 0.0010324, 0.0041851, -0.0030647, 0.0040705
5: 0.0028097, 0.0349197, 0.0022379, 0.0260522, -0.0232425, 0.0326818
6: -0.0030804, 0.0008277, -0.0028065, 0.0009728, -0.0040532, 0.0036342
7: -0.0111074, -0.0009961, -0.0103989, -0.0006206, -0.0104868, 0.0094028
8: -0.0054054, 0.0030892, -0.0050328, 0.0001095, -0.0055149, 0.0081220
9: -0.0017618, 0.0044040, -0.0019908, 0.0039719, -0.0057338, 0.0063948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165455
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165689
time: 2.14 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9681103, 0.9949163, 0.9667473, 0.9954416, -0.0273313, 0.0281690
1: -0.0047160, -0.0025307, -0.0047358, -0.0023998, -0.0023162, 0.0022051
2: 0.0033572, 0.0149383, 0.0026636, 0.0150433, -0.0116861, 0.0122747
3: -0.0083714, -0.0028012, -0.0084404, -0.0024855, -0.0058859, 0.0056392
4: 0.0011777, 0.0047122, 0.0010434, 0.0048246, -0.0036469, 0.0036688
5: 0.0031820, 0.0311451, 0.0023096, 0.0322306, -0.0290487, 0.0288355
6: -0.0029638, 0.0007332, -0.0029973, 0.0009546, -0.0039184, 0.0037305
7: -0.0108058, -0.0012406, -0.0108926, -0.0006677, -0.0101381, 0.0096520
8: -0.0052468, 0.0015627, -0.0052924, 0.0020017, -0.0072485, 0.0068551
9: -0.0016128, 0.0042201, -0.0019621, 0.0042730, -0.0058857, 0.0061822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165209, upper bound: 0.0163842
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165209, upper bound: 0.0164009
time: 1.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9635813, 0.9950516, 0.9667398, 0.9954507, -0.0318694, 0.0283118
1: -0.0047818, -0.0024970, -0.0047359, -0.0023975, -0.0023843, 0.0022390
2: 0.0031785, 0.0152872, 0.0026516, 0.0150439, -0.0118654, 0.0126356
3: -0.0086009, -0.0027199, -0.0084408, -0.0024800, -0.0061208, 0.0057209
4: 0.0011431, 0.0050855, 0.0010411, 0.0048252, -0.0036821, 0.0040444
5: 0.0029572, 0.0347519, 0.0022945, 0.0322365, -0.0292792, 0.0324574
6: -0.0030752, 0.0007903, -0.0029975, 0.0009585, -0.0040336, 0.0037877
7: -0.0110940, -0.0010930, -0.0108930, -0.0006578, -0.0104362, 0.0098000
8: -0.0053984, 0.0030213, -0.0052927, 0.0020041, -0.0074025, 0.0083140
9: -0.0017027, 0.0043958, -0.0019681, 0.0042733, -0.0059760, 0.0063640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164737, upper bound: 0.0164737
time: 2.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164737, upper bound: 0.0164923
time: 2.49 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9740576, 0.9955914, 0.9673066, 0.9953672, -0.0213096, 0.0282848
1: -0.0046295, -0.0023625, -0.0047277, -0.0024183, -0.0022112, 0.0023652
2: 0.0024657, 0.0144801, 0.0027619, 0.0150002, -0.0125345, 0.0117182
3: -0.0080700, -0.0023954, -0.0084121, -0.0025302, -0.0055397, 0.0060167
4: 0.0010051, 0.0042220, 0.0010624, 0.0047785, -0.0037733, 0.0031596
5: 0.0020607, 0.0264088, 0.0024332, 0.0317851, -0.0297243, 0.0239756
6: -0.0028175, 0.0010178, -0.0029835, 0.0009233, -0.0037408, 0.0040013
7: -0.0104274, -0.0005043, -0.0108570, -0.0007489, -0.0096785, 0.0103527
8: -0.0050478, 0.0001707, -0.0052737, 0.0018215, -0.0068693, 0.0054444
9: -0.0020617, 0.0039893, -0.0019126, 0.0042513, -0.0063130, 0.0059019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164796, upper bound: 0.0166060
time: 1.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165689, upper bound: 0.0165606
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9662870, 0.9955361, 0.9675197, 0.9952746, -0.0289876, 0.0280164
1: -0.0047425, -0.0023762, -0.0047246, -0.0024414, -0.0023011, 0.0023483
2: 0.0025388, 0.0150788, 0.0028841, 0.0149838, -0.0124451, 0.0121947
3: -0.0084638, -0.0024287, -0.0084013, -0.0025858, -0.0058779, 0.0059726
4: 0.0010193, 0.0048625, 0.0010861, 0.0047609, -0.0037416, 0.0037764
5: 0.0021525, 0.0325970, 0.0025869, 0.0316154, -0.0294629, 0.0300101
6: -0.0030086, 0.0009945, -0.0029783, 0.0008843, -0.0038929, 0.0039728
7: -0.0109218, -0.0005646, -0.0108434, -0.0008498, -0.0100720, 0.0102788
8: -0.0053078, 0.0021499, -0.0052666, 0.0017529, -0.0070607, 0.0074164
9: -0.0020250, 0.0042908, -0.0018510, 0.0042430, -0.0062680, 0.0061419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164009, upper bound: 0.0166108
time: 1.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164923, upper bound: 0.0165641
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9740576, 0.9955914, 0.9736679, 0.9958081, -0.0217505, 0.0219235
1: -0.0046295, -0.0023625, -0.0046352, -0.0023085, -0.0023211, 0.0022727
2: 0.0024657, 0.0144801, 0.0021797, 0.0145101, -0.0120444, 0.0123004
3: -0.0080700, -0.0023954, -0.0080897, -0.0022652, -0.0058047, 0.0056943
4: 0.0010051, 0.0042220, 0.0009498, 0.0042542, -0.0032490, 0.0032723
5: 0.0020607, 0.0264088, 0.0017009, 0.0267192, -0.0246584, 0.0247079
6: -0.0028175, 0.0010178, -0.0028271, 0.0011091, -0.0039266, 0.0038449
7: -0.0104274, -0.0005043, -0.0104522, -0.0002680, -0.0101594, 0.0099479
8: -0.0050478, 0.0001707, -0.0050608, 0.0002949, -0.0053427, 0.0052315
9: -0.0020617, 0.0039893, -0.0022058, 0.0040044, -0.0060662, 0.0061951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167785, upper bound: 0.0168764
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0168596, upper bound: 0.0167886
time: 1.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9662870, 0.9955361, 0.9738868, 0.9957046, -0.0294176, 0.0216493
1: -0.0047425, -0.0023762, -0.0046320, -0.0023343, -0.0024082, 0.0022558
2: 0.0025388, 0.0150788, 0.0023163, 0.0144933, -0.0119545, 0.0127624
3: -0.0084638, -0.0024287, -0.0080786, -0.0023274, -0.0061363, 0.0056500
4: 0.0010193, 0.0048625, 0.0009762, 0.0042361, -0.0032168, 0.0038863
5: 0.0021525, 0.0325970, 0.0018728, 0.0265448, -0.0243923, 0.0307242
6: -0.0030086, 0.0009945, -0.0028217, 0.0010655, -0.0040741, 0.0038162
7: -0.0109218, -0.0005646, -0.0104382, -0.0003809, -0.0105410, 0.0098737
8: -0.0053078, 0.0021499, -0.0050535, 0.0002356, -0.0055434, 0.0072034
9: -0.0020250, 0.0042908, -0.0021370, 0.0039959, -0.0060209, 0.0064278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166894, upper bound: 0.0168827
time: 1.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167666, upper bound: 0.0167952
time: 1.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.81 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164796
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165455
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165689
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0165209, upper bound: 0.0163842
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0165209, upper bound: 0.0164009
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0164737, upper bound: 0.0164737
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0164737, upper bound: 0.0164923
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0164796, upper bound: 0.0166060
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0165689, upper bound: 0.0165606
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0164009, upper bound: 0.0166108
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0164923, upper bound: 0.0165641
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0167785, upper bound: 0.0168764
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0168596, upper bound: 0.0167886
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0166894, upper bound: 0.0168827
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.81
Output dim: 0, lower bound: -0.0167666, upper bound: 0.0167952

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9679046, 0.9950094, 0.9678258, 0.9950408, -0.0271362, 0.0271837
1: -0.0047190, -0.0025075, -0.0047201, -0.0024997, -0.0022193, 0.0022126
2: 0.0032343, 0.0149542, 0.0031929, 0.0149602, -0.0117260, 0.0117612
3: -0.0083818, -0.0027452, -0.0083858, -0.0027264, -0.0056554, 0.0056405
4: 0.0011539, 0.0047292, 0.0011459, 0.0047357, -0.0035818, 0.0035833
5: 0.0030273, 0.0313089, 0.0029753, 0.0313717, -0.0283444, 0.0283336
6: -0.0029688, 0.0007725, -0.0029708, 0.0007857, -0.0037545, 0.0037432
7: -0.0108189, -0.0011390, -0.0108239, -0.0011049, -0.0097141, 0.0096849
8: -0.0052537, 0.0016290, -0.0052563, 0.0016543, -0.0069080, 0.0068853
9: -0.0016747, 0.0042281, -0.0016955, 0.0042311, -0.0059058, 0.0059236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9679046, 0.9950094, 0.9742222, 0.9955032, -0.0275986, 0.0207872
1: -0.0047190, -0.0025075, -0.0046271, -0.0023844, -0.0023345, 0.0021197
2: 0.0032343, 0.0149542, 0.0025822, 0.0144674, -0.0112331, 0.0123719
3: -0.0083818, -0.0027452, -0.0080616, -0.0024484, -0.0059333, 0.0053164
4: 0.0011539, 0.0047292, 0.0010277, 0.0042085, -0.0030546, 0.0037015
5: 0.0030273, 0.0313089, 0.0022072, 0.0262777, -0.0232504, 0.0291017
6: -0.0029688, 0.0007725, -0.0028135, 0.0009806, -0.0039495, 0.0035859
7: -0.0108189, -0.0011390, -0.0104169, -0.0006005, -0.0102185, 0.0092779
8: -0.0052537, 0.0016290, -0.0050423, 0.0001201, -0.0053738, 0.0066712
9: -0.0016747, 0.0042281, -0.0020031, 0.0039829, -0.0056576, 0.0062312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164796
time: 1.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164796
time: 2.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9633706, 0.9951404, 0.9678178, 0.9950483, -0.0316778, 0.0273226
1: -0.0047849, -0.0024748, -0.0047202, -0.0024978, -0.0022871, 0.0022454
2: 0.0030613, 0.0153035, 0.0031829, 0.0149608, -0.0118996, 0.0121206
3: -0.0086115, -0.0026665, -0.0083862, -0.0027218, -0.0058897, 0.0057197
4: 0.0011204, 0.0051029, 0.0011439, 0.0047363, -0.0036159, 0.0039590
5: 0.0028097, 0.0349197, 0.0029627, 0.0313780, -0.0285683, 0.0319570
6: -0.0030804, 0.0008277, -0.0029710, 0.0007889, -0.0038692, 0.0037987
7: -0.0111074, -0.0009961, -0.0108244, -0.0010966, -0.0100108, 0.0098283
8: -0.0054054, 0.0030892, -0.0052566, 0.0016569, -0.0070623, 0.0083458
9: -0.0017618, 0.0044040, -0.0017006, 0.0042314, -0.0059933, 0.0061046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165455
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165455
time: 1.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9633706, 0.9951404, 0.9742166, 0.9955028, -0.0321322, 0.0209238
1: -0.0047849, -0.0024748, -0.0046272, -0.0023845, -0.0024004, 0.0021524
2: 0.0030613, 0.0153035, 0.0025828, 0.0144678, -0.0114066, 0.0127207
3: -0.0086115, -0.0026665, -0.0080619, -0.0024487, -0.0061628, 0.0053954
4: 0.0011204, 0.0051029, 0.0010278, 0.0042089, -0.0030885, 0.0040751
5: 0.0028097, 0.0349197, 0.0022079, 0.0262822, -0.0234724, 0.0327117
6: -0.0030804, 0.0008277, -0.0028136, 0.0009804, -0.0040608, 0.0036413
7: -0.0111074, -0.0009961, -0.0104173, -0.0006009, -0.0105065, 0.0094211
8: -0.0054054, 0.0030892, -0.0050425, 0.0001198, -0.0055253, 0.0081317
9: -0.0017618, 0.0044040, -0.0020028, 0.0039831, -0.0057450, 0.0064068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165689
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165689
time: 1.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9681103, 0.9949163, 0.9595149, 0.9950256, -0.0269153, 0.0354014
1: -0.0047160, -0.0025307, -0.0048410, -0.0025034, -0.0022126, 0.0023103
2: 0.0033572, 0.0149383, 0.0032128, 0.0156005, -0.0122433, 0.0117255
3: -0.0083714, -0.0028012, -0.0088069, -0.0027355, -0.0056359, 0.0060057
4: 0.0011777, 0.0047122, 0.0011497, 0.0054207, -0.0042430, 0.0035625
5: 0.0031820, 0.0311451, 0.0030004, 0.0379902, -0.0348082, 0.0281447
6: -0.0029638, 0.0007332, -0.0031752, 0.0007793, -0.0037431, 0.0039084
7: -0.0108058, -0.0012406, -0.0113528, -0.0011213, -0.0096845, 0.0101122
8: -0.0052468, 0.0015627, -0.0055345, 0.0043309, -0.0095778, 0.0070972
9: -0.0016128, 0.0042201, -0.0016855, 0.0045536, -0.0061664, 0.0059056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162773
time: 2.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162843
time: 2.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9681103, 0.9949163, 0.9664495, 0.9954472, -0.0273368, 0.0284668
1: -0.0047160, -0.0025307, -0.0047401, -0.0023984, -0.0023176, 0.0022095
2: 0.0033572, 0.0149383, 0.0026563, 0.0150663, -0.0117090, 0.0122821
3: -0.0083714, -0.0028012, -0.0084555, -0.0024821, -0.0058892, 0.0056543
4: 0.0011777, 0.0047122, 0.0010420, 0.0048491, -0.0036714, 0.0036702
5: 0.0031820, 0.0311451, 0.0023003, 0.0324677, -0.0292858, 0.0288448
6: -0.0029638, 0.0007332, -0.0030046, 0.0009570, -0.0039208, 0.0037378
7: -0.0108058, -0.0012406, -0.0109115, -0.0006616, -0.0101442, 0.0096709
8: -0.0052468, 0.0015627, -0.0053024, 0.0020976, -0.0073444, 0.0068651
9: -0.0016128, 0.0042201, -0.0019658, 0.0042845, -0.0058973, 0.0061859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162931
time: 2.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162998
time: 1.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9635813, 0.9950516, 0.9595081, 0.9950368, -0.0314556, 0.0355435
1: -0.0047818, -0.0024970, -0.0048411, -0.0025007, -0.0022812, 0.0023441
2: 0.0031785, 0.0152872, 0.0031981, 0.0156011, -0.0124225, 0.0120891
3: -0.0086009, -0.0027199, -0.0088073, -0.0027288, -0.0058721, 0.0060874
4: 0.0011431, 0.0050855, 0.0011469, 0.0054212, -0.0042781, 0.0039386
5: 0.0029572, 0.0347519, 0.0029819, 0.0379956, -0.0350384, 0.0317700
6: -0.0030752, 0.0007903, -0.0031753, 0.0007840, -0.0038592, 0.0039656
7: -0.0110940, -0.0010930, -0.0113532, -0.0011092, -0.0099848, 0.0102602
8: -0.0053984, 0.0030213, -0.0055347, 0.0043331, -0.0097315, 0.0085560
9: -0.0017027, 0.0043958, -0.0016929, 0.0045539, -0.0062566, 0.0060887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163487, upper bound: 0.0163651
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163688, upper bound: 0.0163688
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9635813, 0.9950516, 0.9664431, 0.9954506, -0.0318693, 0.0286086
1: -0.0047818, -0.0024970, -0.0047402, -0.0023976, -0.0023843, 0.0022433
2: 0.0031785, 0.0152872, 0.0026518, 0.0150668, -0.0118882, 0.0126355
3: -0.0086009, -0.0027199, -0.0084558, -0.0024801, -0.0061208, 0.0057360
4: 0.0011431, 0.0050855, 0.0010411, 0.0048496, -0.0037065, 0.0040444
5: 0.0029572, 0.0347519, 0.0022947, 0.0324727, -0.0295155, 0.0324572
6: -0.0030752, 0.0007903, -0.0030048, 0.0009584, -0.0040336, 0.0037950
7: -0.0110940, -0.0010930, -0.0109119, -0.0006579, -0.0104361, 0.0098189
8: -0.0053984, 0.0030213, -0.0053026, 0.0020996, -0.0074980, 0.0083239
9: -0.0017027, 0.0043958, -0.0019681, 0.0042848, -0.0059875, 0.0063639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163487, upper bound: 0.0163841
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163688, upper bound: 0.0163868
time: 1.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9742222, 0.9955032, 0.9679046, 0.9950094, -0.0207872, 0.0275986
1: -0.0046271, -0.0023844, -0.0047190, -0.0025075, -0.0021197, 0.0023345
2: 0.0025822, 0.0144674, 0.0032343, 0.0149542, -0.0123719, 0.0112331
3: -0.0080616, -0.0024484, -0.0083818, -0.0027452, -0.0053164, 0.0059333
4: 0.0010277, 0.0042085, 0.0011539, 0.0047292, -0.0037015, 0.0030546
5: 0.0022072, 0.0262777, 0.0030273, 0.0313089, -0.0291017, 0.0232504
6: -0.0028135, 0.0009806, -0.0029688, 0.0007725, -0.0035859, 0.0039495
7: -0.0104169, -0.0006005, -0.0108189, -0.0011390, -0.0092779, 0.0102185
8: -0.0050423, 0.0001201, -0.0052537, 0.0016290, -0.0066712, 0.0053738
9: -0.0020031, 0.0039829, -0.0016747, 0.0042281, -0.0062312, 0.0056576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
time: 2.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164864
time: 2.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9742166, 0.9955028, 0.9633706, 0.9951404, -0.0209238, 0.0321322
1: -0.0046272, -0.0023845, -0.0047849, -0.0024748, -0.0021524, 0.0024004
2: 0.0025828, 0.0144678, 0.0030613, 0.0153035, -0.0127207, 0.0114066
3: -0.0080619, -0.0024487, -0.0086115, -0.0026665, -0.0053954, 0.0061628
4: 0.0010278, 0.0042089, 0.0011204, 0.0051029, -0.0040751, 0.0030885
5: 0.0022079, 0.0262822, 0.0028097, 0.0349197, -0.0327117, 0.0234724
6: -0.0028136, 0.0009804, -0.0030804, 0.0008277, -0.0036413, 0.0040608
7: -0.0104173, -0.0006009, -0.0111074, -0.0009961, -0.0094211, 0.0105065
8: -0.0050425, 0.0001198, -0.0054054, 0.0030892, -0.0081317, 0.0055253
9: -0.0020028, 0.0039831, -0.0017618, 0.0044040, -0.0064068, 0.0057450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164661
time: 1.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164446
time: 1.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9664495, 0.9954472, 0.9681103, 0.9949163, -0.0284668, 0.0273368
1: -0.0047401, -0.0023984, -0.0047160, -0.0025307, -0.0022095, 0.0023176
2: 0.0026563, 0.0150663, 0.0033572, 0.0149383, -0.0122821, 0.0117090
3: -0.0084555, -0.0024821, -0.0083714, -0.0028012, -0.0056543, 0.0058892
4: 0.0010420, 0.0048491, 0.0011777, 0.0047122, -0.0036702, 0.0036714
5: 0.0023003, 0.0324677, 0.0031820, 0.0311451, -0.0288448, 0.0292858
6: -0.0030046, 0.0009570, -0.0029638, 0.0007332, -0.0037378, 0.0039208
7: -0.0109115, -0.0006616, -0.0108058, -0.0012406, -0.0096709, 0.0101442
8: -0.0053024, 0.0020976, -0.0052468, 0.0015627, -0.0068651, 0.0073444
9: -0.0019658, 0.0042845, -0.0016128, 0.0042201, -0.0061859, 0.0058973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164009, upper bound: 0.0164893
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164009, upper bound: 0.0165641
time: 2.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9664431, 0.9954506, 0.9635813, 0.9950516, -0.0286086, 0.0318693
1: -0.0047402, -0.0023976, -0.0047818, -0.0024970, -0.0022433, 0.0023843
2: 0.0026518, 0.0150668, 0.0031785, 0.0152872, -0.0126355, 0.0118882
3: -0.0084558, -0.0024801, -0.0086009, -0.0027199, -0.0057360, 0.0061208
4: 0.0010411, 0.0048496, 0.0011431, 0.0050855, -0.0040444, 0.0037065
5: 0.0022947, 0.0324727, 0.0029572, 0.0347519, -0.0324572, 0.0295155
6: -0.0030048, 0.0009584, -0.0030752, 0.0007903, -0.0037950, 0.0040336
7: -0.0109119, -0.0006579, -0.0110940, -0.0010930, -0.0098189, 0.0104361
8: -0.0053026, 0.0020996, -0.0053984, 0.0030213, -0.0083239, 0.0074980
9: -0.0019681, 0.0042848, -0.0017027, 0.0043958, -0.0063639, 0.0059875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163868, upper bound: 0.0164640
time: 1.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163868, upper bound: 0.0164593
time: 1.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9742222, 0.9955032, 0.9743497, 0.9954617, -0.0212395, 0.0211535
1: -0.0046271, -0.0023844, -0.0046253, -0.0023948, -0.0022324, 0.0022408
2: 0.0025822, 0.0144674, 0.0026371, 0.0144576, -0.0118754, 0.0118303
3: -0.0080616, -0.0024484, -0.0080552, -0.0024734, -0.0055882, 0.0056067
4: 0.0010277, 0.0042085, 0.0010383, 0.0041979, -0.0031703, 0.0031702
5: 0.0022072, 0.0262777, 0.0022762, 0.0261761, -0.0239689, 0.0240015
6: -0.0028135, 0.0009806, -0.0028103, 0.0009631, -0.0037766, 0.0037909
7: -0.0104169, -0.0006005, -0.0104088, -0.0006458, -0.0097711, 0.0098083
8: -0.0050423, 0.0001201, -0.0050380, 0.0000962, -0.0051385, 0.0051581
9: -0.0020031, 0.0039829, -0.0019754, 0.0039780, -0.0059811, 0.0059584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167834
time: 1.83 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167584
time: 1.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9742166, 0.9955028, 0.9699495, 0.9955721, -0.0213555, 0.0255533
1: -0.0046272, -0.0023845, -0.0046893, -0.0023673, -0.0022599, 0.0023047
2: 0.0025828, 0.0144678, 0.0024914, 0.0147966, -0.0122138, 0.0119765
3: -0.0080619, -0.0024487, -0.0082782, -0.0024071, -0.0056548, 0.0058295
4: 0.0010278, 0.0042089, 0.0010101, 0.0045606, -0.0035328, 0.0031988
5: 0.0022079, 0.0262822, 0.0020929, 0.0296804, -0.0274725, 0.0241892
6: -0.0028136, 0.0009804, -0.0029185, 0.0010096, -0.0038232, 0.0038990
7: -0.0104173, -0.0006009, -0.0106888, -0.0005254, -0.0098918, 0.0100879
8: -0.0050425, 0.0001198, -0.0051853, 0.0009703, -0.0060128, 0.0053051
9: -0.0020028, 0.0039831, -0.0020489, 0.0041487, -0.0061515, 0.0060320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167495, upper bound: 0.0166949
time: 1.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167494, upper bound: 0.0166713
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9664495, 0.9954472, 0.9745618, 0.9953547, -0.0289052, 0.0208853
1: -0.0047401, -0.0023984, -0.0046222, -0.0024214, -0.0023187, 0.0022238
2: 0.0026563, 0.0150663, 0.0027783, 0.0144413, -0.0117850, 0.0122880
3: -0.0084555, -0.0024821, -0.0080444, -0.0025377, -0.0059178, 0.0055623
4: 0.0010420, 0.0048491, 0.0010656, 0.0041805, -0.0031385, 0.0037835
5: 0.0023003, 0.0324677, 0.0024538, 0.0260073, -0.0237070, 0.0300139
6: -0.0030046, 0.0009570, -0.0028051, 0.0009180, -0.0039227, 0.0037621
7: -0.0109115, -0.0006616, -0.0103953, -0.0007624, -0.0101491, 0.0097337
8: -0.0053024, 0.0020976, -0.0050309, 0.0000349, -0.0053373, 0.0071285
9: -0.0019658, 0.0042845, -0.0019043, 0.0039698, -0.0059356, 0.0061889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166894, upper bound: 0.0167222
time: 1.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166894, upper bound: 0.0167952
time: 2.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9664431, 0.9954506, 0.9701653, 0.9954686, -0.0290256, 0.0252853
1: -0.0047402, -0.0023976, -0.0046861, -0.0023931, -0.0023472, 0.0022886
2: 0.0026518, 0.0150668, 0.0026279, 0.0147800, -0.0121282, 0.0124388
3: -0.0084558, -0.0024801, -0.0082672, -0.0024692, -0.0059866, 0.0057871
4: 0.0010411, 0.0048496, 0.0010365, 0.0045428, -0.0035017, 0.0038131
5: 0.0022947, 0.0324727, 0.0022647, 0.0295085, -0.0272139, 0.0302081
6: -0.0030048, 0.0009584, -0.0029132, 0.0009660, -0.0039708, 0.0038717
7: -0.0109119, -0.0006579, -0.0106751, -0.0006382, -0.0102737, 0.0100172
8: -0.0053026, 0.0020996, -0.0051781, 0.0009008, -0.0062035, 0.0072777
9: -0.0019681, 0.0042848, -0.0019801, 0.0041403, -0.0061084, 0.0062649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166639, upper bound: 0.0166962
time: 1.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166639, upper bound: 0.0166901
time: 2.41 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.77 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164796
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164796
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165455
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165455
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165689
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164598, upper bound: 0.0165689
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162773
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162843
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162931
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162998
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0163487, upper bound: 0.0163651
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0163688, upper bound: 0.0163688
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0163487, upper bound: 0.0163841
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0163688, upper bound: 0.0163868
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164864
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164661
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164446
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164009, upper bound: 0.0164893
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0164009, upper bound: 0.0165641
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0163868, upper bound: 0.0164640
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0163868, upper bound: 0.0164593
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167834
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167584
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0167495, upper bound: 0.0166949
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0167494, upper bound: 0.0166713
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0166894, upper bound: 0.0167222
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0166894, upper bound: 0.0167952
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0166639, upper bound: 0.0166962
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -0.0166639, upper bound: 0.0166901

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9682679, 0.9947788, 0.9678258, 0.9950408, -0.0267729, 0.0269530
1: -0.0047137, -0.0025650, -0.0047201, -0.0024997, -0.0022140, 0.0021552
2: 0.0035389, 0.0149262, 0.0031929, 0.0149602, -0.0114213, 0.0117333
3: -0.0083634, -0.0028839, -0.0083858, -0.0027264, -0.0056370, 0.0055019
4: 0.0012128, 0.0046992, 0.0011459, 0.0047357, -0.0035228, 0.0035534
5: 0.0034105, 0.0310196, 0.0029753, 0.0313717, -0.0279612, 0.0280443
6: -0.0029599, 0.0006752, -0.0029708, 0.0007857, -0.0037456, 0.0036460
7: -0.0107958, -0.0013906, -0.0108239, -0.0011049, -0.0096909, 0.0094333
8: -0.0052415, 0.0015120, -0.0052563, 0.0016543, -0.0068959, 0.0067683
9: -0.0015212, 0.0042140, -0.0016955, 0.0042311, -0.0057524, 0.0059095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
time: 1.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
time: 2.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9599677, 0.9947580, 0.9678258, 0.9950408, -0.0350730, 0.0269322
1: -0.0048344, -0.0025701, -0.0047201, -0.0024997, -0.0023347, 0.0021500
2: 0.0035663, 0.0155656, 0.0031929, 0.0149602, -0.0113939, 0.0123727
3: -0.0087840, -0.0028964, -0.0083858, -0.0027264, -0.0060576, 0.0054894
4: 0.0012181, 0.0053834, 0.0011459, 0.0047357, -0.0035175, 0.0042375
5: 0.0034450, 0.0376295, 0.0029753, 0.0313717, -0.0279267, 0.0346543
6: -0.0031640, 0.0006665, -0.0029708, 0.0007857, -0.0039497, 0.0036372
7: -0.0113240, -0.0014133, -0.0108239, -0.0011049, -0.0102191, 0.0094106
8: -0.0055193, 0.0041851, -0.0052563, 0.0016543, -0.0071736, 0.0094414
9: -0.0015074, 0.0045361, -0.0016955, 0.0042311, -0.0057386, 0.0062316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
time: 1.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9682679, 0.9947788, 0.9742222, 0.9955032, -0.0272353, 0.0205566
1: -0.0047137, -0.0025650, -0.0046271, -0.0023844, -0.0023293, 0.0020622
2: 0.0035389, 0.0149262, 0.0025822, 0.0144674, -0.0109285, 0.0123439
3: -0.0083634, -0.0028839, -0.0080616, -0.0024484, -0.0059149, 0.0051777
4: 0.0012128, 0.0046992, 0.0010277, 0.0042085, -0.0029956, 0.0036716
5: 0.0034105, 0.0310196, 0.0022072, 0.0262777, -0.0228672, 0.0288124
6: -0.0029599, 0.0006752, -0.0028135, 0.0009806, -0.0039405, 0.0034887
7: -0.0107958, -0.0013906, -0.0104169, -0.0006005, -0.0101953, 0.0090263
8: -0.0052415, 0.0015120, -0.0050423, 0.0001201, -0.0053616, 0.0065543
9: -0.0015212, 0.0042140, -0.0020031, 0.0039829, -0.0055042, 0.0062171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165136, upper bound: 0.0163835
time: 1.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164864, upper bound: 0.0163835
time: 1.91 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9599677, 0.9947580, 0.9742222, 0.9955032, -0.0355355, 0.0205358
1: -0.0048344, -0.0025701, -0.0046271, -0.0023844, -0.0024499, 0.0020570
2: 0.0035663, 0.0155656, 0.0025822, 0.0144674, -0.0109011, 0.0129834
3: -0.0087840, -0.0028964, -0.0080616, -0.0024484, -0.0063355, 0.0051653
4: 0.0012181, 0.0053834, 0.0010277, 0.0042085, -0.0029903, 0.0043557
5: 0.0034450, 0.0376295, 0.0022072, 0.0262777, -0.0228327, 0.0354223
6: -0.0031640, 0.0006665, -0.0028135, 0.0009806, -0.0041447, 0.0034799
7: -0.0113240, -0.0014133, -0.0104169, -0.0006005, -0.0107235, 0.0090036
8: -0.0055193, 0.0041851, -0.0050423, 0.0001201, -0.0056394, 0.0092274
9: -0.0015074, 0.0045361, -0.0020031, 0.0039829, -0.0054904, 0.0065391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165136, upper bound: 0.0163835
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164864, upper bound: 0.0163835
time: 1.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9637401, 0.9949022, 0.9678178, 0.9950483, -0.0313083, 0.0270844
1: -0.0047795, -0.0025342, -0.0047202, -0.0024978, -0.0022818, 0.0021861
2: 0.0033758, 0.0152750, 0.0031829, 0.0149608, -0.0115850, 0.0120921
3: -0.0085928, -0.0028097, -0.0083862, -0.0027218, -0.0058710, 0.0055765
4: 0.0011813, 0.0050724, 0.0011439, 0.0047363, -0.0035551, 0.0039285
5: 0.0032054, 0.0346254, 0.0029627, 0.0313780, -0.0281727, 0.0316627
6: -0.0030713, 0.0007273, -0.0029710, 0.0007889, -0.0038601, 0.0036982
7: -0.0110839, -0.0012559, -0.0108244, -0.0010966, -0.0099873, 0.0095685
8: -0.0053931, 0.0029702, -0.0052566, 0.0016569, -0.0070500, 0.0082268
9: -0.0016034, 0.0043897, -0.0017006, 0.0042314, -0.0058348, 0.0060902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163542, upper bound: 0.0164431
time: 1.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163487, upper bound: 0.0164431
time: 1.52 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9552293, 0.9948976, 0.9678178, 0.9950483, -0.0398191, 0.0270798
1: -0.0049033, -0.0025354, -0.0047202, -0.0024978, -0.0024055, 0.0021849
2: 0.0033820, 0.0159307, 0.0031829, 0.0149608, -0.0115788, 0.0127478
3: -0.0090241, -0.0028125, -0.0083862, -0.0027218, -0.0063023, 0.0055737
4: 0.0011825, 0.0057739, 0.0011439, 0.0047363, -0.0035539, 0.0046300
5: 0.0032132, 0.0414032, 0.0029627, 0.0313780, -0.0281648, 0.0384405
6: -0.0032806, 0.0007253, -0.0029710, 0.0007889, -0.0040694, 0.0036963
7: -0.0116255, -0.0012611, -0.0108244, -0.0010966, -0.0105289, 0.0095634
8: -0.0056779, 0.0057112, -0.0052566, 0.0016569, -0.0073348, 0.0109678
9: -0.0016002, 0.0047199, -0.0017006, 0.0042314, -0.0058317, 0.0064205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163542, upper bound: 0.0164431
time: 1.74 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163487, upper bound: 0.0164431
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9637401, 0.9949022, 0.9742166, 0.9955028, -0.0317627, 0.0206856
1: -0.0047795, -0.0025342, -0.0046272, -0.0023845, -0.0023950, 0.0020930
2: 0.0033758, 0.0152750, 0.0025828, 0.0144678, -0.0110920, 0.0126922
3: -0.0085928, -0.0028097, -0.0080619, -0.0024487, -0.0061441, 0.0052523
4: 0.0011813, 0.0050724, 0.0010278, 0.0042089, -0.0030277, 0.0040446
5: 0.0032054, 0.0346254, 0.0022079, 0.0262822, -0.0230768, 0.0324175
6: -0.0030713, 0.0007273, -0.0028136, 0.0009804, -0.0040517, 0.0035409
7: -0.0110839, -0.0012559, -0.0104173, -0.0006009, -0.0104830, 0.0091613
8: -0.0053931, 0.0029702, -0.0050425, 0.0001198, -0.0055129, 0.0080127
9: -0.0016034, 0.0043897, -0.0020028, 0.0039831, -0.0055865, 0.0063925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164661, upper bound: 0.0164664
time: 1.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164446, upper bound: 0.0164664
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9552293, 0.9948976, 0.9742166, 0.9955028, -0.0402735, 0.0206810
1: -0.0049033, -0.0025354, -0.0046272, -0.0023845, -0.0025187, 0.0020919
2: 0.0033820, 0.0159307, 0.0025828, 0.0144678, -0.0110858, 0.0133479
3: -0.0090241, -0.0028125, -0.0080619, -0.0024487, -0.0065754, 0.0052494
4: 0.0011825, 0.0057739, 0.0010278, 0.0042089, -0.0030265, 0.0047461
5: 0.0032132, 0.0414032, 0.0022079, 0.0262822, -0.0230690, 0.0391953
6: -0.0032806, 0.0007253, -0.0028136, 0.0009804, -0.0042610, 0.0035389
7: -0.0116255, -0.0012611, -0.0104173, -0.0006009, -0.0110245, 0.0091562
8: -0.0056779, 0.0057112, -0.0050425, 0.0001198, -0.0057977, 0.0107537
9: -0.0016002, 0.0047199, -0.0020028, 0.0039831, -0.0055834, 0.0067227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164661, upper bound: 0.0164664
time: 1.55 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164446, upper bound: 0.0164664
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9686901, 0.9947357, 0.9596381, 0.9949889, -0.0262988, 0.0350976
1: -0.0047076, -0.0025757, -0.0048392, -0.0025126, -0.0021950, 0.0022635
2: 0.0035957, 0.0148936, 0.0032615, 0.0155910, -0.0119953, 0.0116321
3: -0.0083420, -0.0029097, -0.0088007, -0.0027576, -0.0055843, 0.0058909
4: 0.0012238, 0.0046644, 0.0011591, 0.0054105, -0.0041867, 0.0035053
5: 0.0034819, 0.0306833, 0.0030616, 0.0378921, -0.0344101, 0.0276217
6: -0.0029495, 0.0006571, -0.0031721, 0.0007638, -0.0037133, 0.0038292
7: -0.0107689, -0.0014376, -0.0113449, -0.0011615, -0.0096074, 0.0099074
8: -0.0052274, 0.0013759, -0.0055303, 0.0042913, -0.0095187, 0.0069063
9: -0.0014926, 0.0041976, -0.0016610, 0.0045488, -0.0060415, 0.0058586

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162773
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162773
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9649394, 0.9947543, 0.9597356, 0.9949495, -0.0300100, 0.0350187
1: -0.0047621, -0.0025711, -0.0048377, -0.0025224, -0.0022397, 0.0022667
2: 0.0035712, 0.0151826, 0.0033134, 0.0155835, -0.0120123, 0.0118692
3: -0.0085320, -0.0028986, -0.0087957, -0.0027812, -0.0057508, 0.0058972
4: 0.0012191, 0.0049736, 0.0011692, 0.0054025, -0.0041834, 0.0038044
5: 0.0034511, 0.0336702, 0.0031269, 0.0378145, -0.0343634, 0.0305434
6: -0.0030418, 0.0006649, -0.0031697, 0.0007472, -0.0037890, 0.0038346
7: -0.0110076, -0.0014173, -0.0113387, -0.0012044, -0.0098032, 0.0099214
8: -0.0053529, 0.0025839, -0.0055271, 0.0042599, -0.0096128, 0.0081110
9: -0.0015050, 0.0043431, -0.0016348, 0.0045451, -0.0060500, 0.0059779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162843
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162843
time: 2.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9686901, 0.9947357, 0.9665819, 0.9954106, -0.0267205, 0.0281538
1: -0.0047076, -0.0025757, -0.0047382, -0.0024075, -0.0023000, 0.0021625
2: 0.0035957, 0.0148936, 0.0027045, 0.0150561, -0.0114603, 0.0121891
3: -0.0083420, -0.0029097, -0.0084488, -0.0025041, -0.0058379, 0.0055391
4: 0.0012238, 0.0046644, 0.0010513, 0.0048382, -0.0036144, 0.0036131
5: 0.0034819, 0.0306833, 0.0023610, 0.0323622, -0.0288803, 0.0283222
6: -0.0029495, 0.0006571, -0.0030014, 0.0009416, -0.0038911, 0.0036584
7: -0.0107689, -0.0014376, -0.0109031, -0.0007015, -0.0100674, 0.0094655
8: -0.0052274, 0.0013759, -0.0052980, 0.0020549, -0.0072823, 0.0066739
9: -0.0014926, 0.0041976, -0.0019415, 0.0042794, -0.0057720, 0.0061391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165042, upper bound: 0.0162931
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165042, upper bound: 0.0162931
time: 2.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9649394, 0.9947543, 0.9666794, 0.9953604, -0.0304210, 0.0280749
1: -0.0047621, -0.0025711, -0.0047368, -0.0024200, -0.0023421, 0.0021657
2: 0.0035712, 0.0151826, 0.0027708, 0.0150485, -0.0114773, 0.0124118
3: -0.0085320, -0.0028986, -0.0084439, -0.0025343, -0.0059978, 0.0055453
4: 0.0012191, 0.0049736, 0.0010642, 0.0048302, -0.0036111, 0.0039094
5: 0.0034511, 0.0336702, 0.0024444, 0.0322846, -0.0288335, 0.0312258
6: -0.0030418, 0.0006649, -0.0029990, 0.0009204, -0.0039622, 0.0036639
7: -0.0110076, -0.0014173, -0.0108969, -0.0007562, -0.0102514, 0.0094796
8: -0.0053529, 0.0025839, -0.0052947, 0.0020235, -0.0073765, 0.0078786
9: -0.0015050, 0.0043431, -0.0019081, 0.0042756, -0.0057806, 0.0062512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165042, upper bound: 0.0162998
time: 1.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165042, upper bound: 0.0162998
time: 1.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9641901, 0.9948631, 0.9596313, 0.9950002, -0.0308101, 0.0352318
1: -0.0047730, -0.0025439, -0.0048393, -0.0025098, -0.0022632, 0.0022953
2: 0.0034275, 0.0152403, 0.0032465, 0.0155916, -0.0121641, 0.0119939
3: -0.0085700, -0.0028332, -0.0088010, -0.0027508, -0.0058192, 0.0059679
4: 0.0011913, 0.0050353, 0.0011562, 0.0054111, -0.0042198, 0.0038791
5: 0.0032704, 0.0342670, 0.0030427, 0.0378975, -0.0346272, 0.0312243
6: -0.0030602, 0.0007108, -0.0031723, 0.0007686, -0.0038288, 0.0038831
7: -0.0110553, -0.0012986, -0.0113454, -0.0011491, -0.0099062, 0.0100467
8: -0.0053780, 0.0028252, -0.0055306, 0.0042935, -0.0096715, 0.0083558
9: -0.0015774, 0.0043722, -0.0016685, 0.0045491, -0.0061265, 0.0060407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162843, upper bound: 0.0163651
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162843, upper bound: 0.0163651
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9605731, 0.9948773, 0.9597291, 0.9949598, -0.0343866, 0.0351483
1: -0.0048256, -0.0025404, -0.0048378, -0.0025198, -0.0023057, 0.0022975
2: 0.0034087, 0.0155190, 0.0032998, 0.0155840, -0.0121753, 0.0122192
3: -0.0087533, -0.0028246, -0.0087961, -0.0027750, -0.0059783, 0.0059715
4: 0.0011876, 0.0053335, 0.0011666, 0.0054030, -0.0042154, 0.0041669
5: 0.0032467, 0.0371475, 0.0031097, 0.0378197, -0.0345730, 0.0340378
6: -0.0031492, 0.0007168, -0.0031699, 0.0007516, -0.0039007, 0.0038867
7: -0.0112854, -0.0012831, -0.0113391, -0.0011931, -0.0100923, 0.0100561
8: -0.0054990, 0.0039901, -0.0055273, 0.0042620, -0.0097610, 0.0095174
9: -0.0015868, 0.0045126, -0.0016417, 0.0045453, -0.0061321, 0.0061543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162843, upper bound: 0.0163688
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162843, upper bound: 0.0163688
time: 2.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9641901, 0.9948631, 0.9665756, 0.9954144, -0.0312243, 0.0282875
1: -0.0047730, -0.0025439, -0.0047383, -0.0024066, -0.0023664, 0.0021944
2: 0.0034275, 0.0152403, 0.0026995, 0.0150566, -0.0116291, 0.0125408
3: -0.0085700, -0.0028332, -0.0084491, -0.0025018, -0.0060682, 0.0056160
4: 0.0011913, 0.0050353, 0.0010504, 0.0048387, -0.0036475, 0.0039850
5: 0.0032704, 0.0342670, 0.0023548, 0.0323674, -0.0290970, 0.0319122
6: -0.0030602, 0.0007108, -0.0030015, 0.0009432, -0.0040034, 0.0037123
7: -0.0110553, -0.0012986, -0.0109035, -0.0006974, -0.0103579, 0.0096049
8: -0.0053780, 0.0028252, -0.0052982, 0.0020570, -0.0074350, 0.0081234
9: -0.0015774, 0.0043722, -0.0019440, 0.0042796, -0.0058570, 0.0063162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162185, upper bound: 0.0159699
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162149, upper bound: 0.0161313
time: 1.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9605731, 0.9948773, 0.9666731, 0.9953633, -0.0347902, 0.0282043
1: -0.0048256, -0.0025404, -0.0047369, -0.0024193, -0.0024063, 0.0021965
2: 0.0034087, 0.0155190, 0.0027670, 0.0150490, -0.0116403, 0.0127520
3: -0.0087533, -0.0028246, -0.0084442, -0.0025325, -0.0062208, 0.0056196
4: 0.0011876, 0.0053335, 0.0010634, 0.0048307, -0.0036431, 0.0042700
5: 0.0032467, 0.0371475, 0.0024396, 0.0322896, -0.0290429, 0.0347079
6: -0.0031492, 0.0007168, -0.0029991, 0.0009216, -0.0040708, 0.0037159
7: -0.0112854, -0.0012831, -0.0108973, -0.0007531, -0.0105324, 0.0096142
8: -0.0054990, 0.0039901, -0.0052949, 0.0020256, -0.0075246, 0.0092851
9: -0.0015868, 0.0045126, -0.0019100, 0.0042759, -0.0058627, 0.0064226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162185, upper bound: 0.0159751
time: 1.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162149, upper bound: 0.0161320
time: 2.17 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9748567, 0.9953405, 0.9680188, 0.9949713, -0.0201146, 0.0273216
1: -0.0046179, -0.0024250, -0.0047173, -0.0025170, -0.0021010, 0.0022923
2: 0.0027972, 0.0144185, 0.0032845, 0.0149454, -0.0121482, 0.0111340
3: -0.0080295, -0.0025463, -0.0083760, -0.0027681, -0.0052614, 0.0058297
4: 0.0010693, 0.0041562, 0.0011636, 0.0047198, -0.0036505, 0.0029926
5: 0.0024776, 0.0257724, 0.0030906, 0.0312179, -0.0287403, 0.0226819
6: -0.0027979, 0.0009120, -0.0029660, 0.0007564, -0.0035543, 0.0038780
7: -0.0103765, -0.0007780, -0.0108116, -0.0011805, -0.0091960, 0.0100336
8: -0.0050211, 0.0000267, -0.0052499, 0.0015921, -0.0066132, 0.0052766
9: -0.0018948, 0.0039583, -0.0016493, 0.0042236, -0.0061185, 0.0056077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
time: 1.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
time: 2.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9718309, 0.9952995, 0.9681312, 0.9949226, -0.0230917, 0.0271683
1: -0.0046619, -0.0024352, -0.0047157, -0.0025291, -0.0021328, 0.0022805
2: 0.0028513, 0.0146517, 0.0033488, 0.0149367, -0.0120854, 0.0113028
3: -0.0081828, -0.0025709, -0.0083703, -0.0027974, -0.0053854, 0.0057994
4: 0.0010797, 0.0044056, 0.0011760, 0.0047105, -0.0036308, 0.0032295
5: 0.0025456, 0.0281821, 0.0031714, 0.0311284, -0.0285828, 0.0250107
6: -0.0028723, 0.0008947, -0.0029633, 0.0007359, -0.0036082, 0.0038580
7: -0.0105691, -0.0008227, -0.0108045, -0.0012336, -0.0093354, 0.0099818
8: -0.0051223, 0.0003644, -0.0052461, 0.0015559, -0.0066783, 0.0056106
9: -0.0018676, 0.0040757, -0.0016170, 0.0042193, -0.0060869, 0.0056927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164864
time: 1.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164864
time: 2.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9748513, 0.9953412, 0.9634892, 0.9951025, -0.0202512, 0.0318520
1: -0.0046180, -0.0024248, -0.0047832, -0.0024843, -0.0021337, 0.0023584
2: 0.0027961, 0.0144190, 0.0031115, 0.0152943, -0.0124982, 0.0113075
3: -0.0080297, -0.0025458, -0.0086055, -0.0026893, -0.0053404, 0.0060597
4: 0.0010691, 0.0041566, 0.0011301, 0.0050931, -0.0040240, 0.0030265
5: 0.0024762, 0.0257767, 0.0028729, 0.0348252, -0.0323490, 0.0229039
6: -0.0027980, 0.0009123, -0.0030774, 0.0008117, -0.0036096, 0.0039898
7: -0.0103769, -0.0007771, -0.0110999, -0.0010376, -0.0093393, 0.0103228
8: -0.0050212, 0.0000272, -0.0054015, 0.0030510, -0.0080722, 0.0054287
9: -0.0018954, 0.0039585, -0.0017365, 0.0043994, -0.0062948, 0.0056950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164661
time: 1.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164661
time: 1.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9718214, 0.9952992, 0.9636061, 0.9950519, -0.0232305, 0.0316930
1: -0.0046620, -0.0024353, -0.0047815, -0.0024969, -0.0021652, 0.0023462
2: 0.0028517, 0.0146524, 0.0031781, 0.0152853, -0.0124336, 0.0114743
3: -0.0081833, -0.0025711, -0.0085996, -0.0027196, -0.0054636, 0.0060285
4: 0.0010798, 0.0044063, 0.0011430, 0.0050835, -0.0040036, 0.0032633
5: 0.0025462, 0.0281896, 0.0029567, 0.0347321, -0.0321859, 0.0252330
6: -0.0028725, 0.0008946, -0.0030746, 0.0007904, -0.0036629, 0.0039692
7: -0.0105697, -0.0008230, -0.0110924, -0.0010926, -0.0094771, 0.0102694
8: -0.0051226, 0.0003675, -0.0053976, 0.0030133, -0.0081359, 0.0057650
9: -0.0018673, 0.0040761, -0.0017030, 0.0043949, -0.0062622, 0.0057791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164446
time: 1.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164446
time: 1.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9669439, 0.9951854, 0.9681103, 0.9949163, -0.0279725, 0.0270751
1: -0.0047330, -0.0024636, -0.0047160, -0.0025307, -0.0022023, 0.0022523
2: 0.0030020, 0.0150282, 0.0033572, 0.0149383, -0.0119363, 0.0116709
3: -0.0084305, -0.0026395, -0.0083714, -0.0028012, -0.0056293, 0.0057319
4: 0.0011089, 0.0048084, 0.0011777, 0.0047122, -0.0036033, 0.0036307
5: 0.0027352, 0.0320741, 0.0031820, 0.0311451, -0.0284099, 0.0288921
6: -0.0029925, 0.0008466, -0.0029638, 0.0007332, -0.0037257, 0.0038104
7: -0.0108801, -0.0009472, -0.0108058, -0.0012406, -0.0096395, 0.0098587
8: -0.0052859, 0.0019384, -0.0052468, 0.0015627, -0.0068486, 0.0071852
9: -0.0017917, 0.0042654, -0.0016128, 0.0042201, -0.0060118, 0.0058781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162931, upper bound: 0.0165041
time: 1.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162998, upper bound: 0.0165042
time: 1.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9623438, 0.9953049, 0.9681103, 0.9949163, -0.0325725, 0.0271946
1: -0.0047998, -0.0024339, -0.0047160, -0.0025307, -0.0022691, 0.0022821
2: 0.0028441, 0.0153826, 0.0033572, 0.0149383, -0.0120942, 0.0120253
3: -0.0086636, -0.0025676, -0.0083714, -0.0028012, -0.0058624, 0.0058037
4: 0.0010784, 0.0051875, 0.0011777, 0.0047122, -0.0036339, 0.0040098
5: 0.0025366, 0.0357373, 0.0031820, 0.0311451, -0.0286085, 0.0325553
6: -0.0031056, 0.0008970, -0.0029638, 0.0007332, -0.0038388, 0.0038608
7: -0.0111728, -0.0008168, -0.0108058, -0.0012406, -0.0099322, 0.0099891
8: -0.0054398, 0.0034198, -0.0052468, 0.0015627, -0.0070025, 0.0086666
9: -0.0018712, 0.0044439, -0.0016128, 0.0042201, -0.0060913, 0.0060566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162931, upper bound: 0.0165046
time: 1.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162998, upper bound: 0.0165046
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9671127, 0.9952742, 0.9637039, 0.9950135, -0.0279008, 0.0315703
1: -0.0047305, -0.0024415, -0.0047801, -0.0025065, -0.0022240, 0.0023386
2: 0.0028846, 0.0150152, 0.0032289, 0.0152778, -0.0123932, 0.0117862
3: -0.0084219, -0.0025861, -0.0085947, -0.0027428, -0.0056791, 0.0060086
4: 0.0010862, 0.0047944, 0.0011528, 0.0050754, -0.0039892, 0.0036416
5: 0.0025876, 0.0319395, 0.0030206, 0.0346542, -0.0320666, 0.0289189
6: -0.0029883, 0.0008841, -0.0030722, 0.0007742, -0.0037625, 0.0039562
7: -0.0108693, -0.0008502, -0.0110862, -0.0011346, -0.0097347, 0.0102360
8: -0.0052802, 0.0018840, -0.0053943, 0.0029818, -0.0082620, 0.0072782
9: -0.0018508, 0.0042588, -0.0016774, 0.0043911, -0.0062419, 0.0059362

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163654, upper bound: 0.0164640
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163654, upper bound: 0.0164640
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9627867, 0.9952946, 0.9638140, 0.9949643, -0.0321776, 0.0314806
1: -0.0047934, -0.0024364, -0.0047785, -0.0025187, -0.0022747, 0.0023420
2: 0.0028578, 0.0153485, 0.0032938, 0.0152693, -0.0124115, 0.0120546
3: -0.0086411, -0.0025739, -0.0085891, -0.0027723, -0.0058688, 0.0060152
4: 0.0010810, 0.0051510, 0.0011654, 0.0050663, -0.0039853, 0.0039856
5: 0.0025538, 0.0353847, 0.0031022, 0.0345666, -0.0320128, 0.0322824
6: -0.0030947, 0.0008927, -0.0030694, 0.0007535, -0.0038482, 0.0039621
7: -0.0111446, -0.0008281, -0.0110792, -0.0011882, -0.0099564, 0.0102512
8: -0.0054250, 0.0032772, -0.0053906, 0.0029464, -0.0083713, 0.0086678
9: -0.0018643, 0.0044267, -0.0016447, 0.0043868, -0.0062511, 0.0060713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163654, upper bound: 0.0164593
time: 1.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163654, upper bound: 0.0164593
time: 1.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9748567, 0.9953405, 0.9744686, 0.9954266, -0.0205699, 0.0208718
1: -0.0046179, -0.0024250, -0.0046236, -0.0024035, -0.0022144, 0.0021986
2: 0.0027972, 0.0144185, 0.0026834, 0.0144484, -0.0116512, 0.0117351
3: -0.0080295, -0.0025463, -0.0080491, -0.0024945, -0.0055350, 0.0055029
4: 0.0010693, 0.0041562, 0.0010473, 0.0041882, -0.0031189, 0.0031089
5: 0.0024776, 0.0257724, 0.0023345, 0.0260815, -0.0236039, 0.0234379
6: -0.0027979, 0.0009120, -0.0028074, 0.0009483, -0.0037462, 0.0037194
7: -0.0103765, -0.0007780, -0.0104012, -0.0006841, -0.0096925, 0.0096232
8: -0.0050211, 0.0000267, -0.0050340, 0.0000761, -0.0050972, 0.0050608
9: -0.0018948, 0.0039583, -0.0019521, 0.0039734, -0.0058682, 0.0059104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167834
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167834
time: 1.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9718309, 0.9952995, 0.9745802, 0.9953652, -0.0235343, 0.0207193
1: -0.0046619, -0.0024352, -0.0046219, -0.0024188, -0.0022431, 0.0021867
2: 0.0028513, 0.0146517, 0.0027645, 0.0144398, -0.0115886, 0.0118872
3: -0.0081828, -0.0025709, -0.0080435, -0.0025314, -0.0056514, 0.0054726
4: 0.0010797, 0.0044056, 0.0010629, 0.0041790, -0.0030992, 0.0033426
5: 0.0025456, 0.0281821, 0.0024364, 0.0259927, -0.0234471, 0.0257457
6: -0.0028723, 0.0008947, -0.0028047, 0.0009224, -0.0037947, 0.0036994
7: -0.0105691, -0.0008227, -0.0103941, -0.0007510, -0.0098181, 0.0095715
8: -0.0051223, 0.0003644, -0.0050303, 0.0000409, -0.0051632, 0.0053948
9: -0.0018676, 0.0040757, -0.0019113, 0.0039690, -0.0058366, 0.0059870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167584
time: 2.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167584
time: 2.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9748513, 0.9953412, 0.9700713, 0.9955373, -0.0206860, 0.0252700
1: -0.0046180, -0.0024248, -0.0046875, -0.0023759, -0.0022421, 0.0022627
2: 0.0027961, 0.0144190, 0.0025372, 0.0147872, -0.0119912, 0.0118818
3: -0.0080297, -0.0025458, -0.0082720, -0.0024279, -0.0056018, 0.0057262
4: 0.0010691, 0.0041566, 0.0010190, 0.0045506, -0.0034815, 0.0031377
5: 0.0024762, 0.0257767, 0.0021505, 0.0295834, -0.0271072, 0.0236262
6: -0.0027980, 0.0009123, -0.0029156, 0.0009950, -0.0037930, 0.0038279
7: -0.0103769, -0.0007771, -0.0106810, -0.0005632, -0.0098136, 0.0099039
8: -0.0050212, 0.0000272, -0.0051812, 0.0009311, -0.0059524, 0.0052084
9: -0.0018954, 0.0039585, -0.0020258, 0.0041440, -0.0060394, 0.0059843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167495, upper bound: 0.0166949
time: 2.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167495, upper bound: 0.0166949
time: 1.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9718214, 0.9952992, 0.9701877, 0.9954744, -0.0236530, 0.0251114
1: -0.0046620, -0.0024353, -0.0046858, -0.0023916, -0.0022704, 0.0022505
2: 0.0028517, 0.0146524, 0.0026203, 0.0147783, -0.0119265, 0.0120320
3: -0.0081833, -0.0025711, -0.0082661, -0.0024658, -0.0057175, 0.0056950
4: 0.0010798, 0.0044063, 0.0010351, 0.0045410, -0.0034612, 0.0033713
5: 0.0025462, 0.0281896, 0.0022552, 0.0294907, -0.0269445, 0.0259345
6: -0.0028725, 0.0008946, -0.0029127, 0.0009685, -0.0038410, 0.0038073
7: -0.0105697, -0.0008230, -0.0106736, -0.0006320, -0.0099377, 0.0098506
8: -0.0051226, 0.0003675, -0.0051773, 0.0008936, -0.0060163, 0.0055448
9: -0.0018673, 0.0040761, -0.0019839, 0.0041395, -0.0060068, 0.0060600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167494, upper bound: 0.0166713
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0167494, upper bound: 0.0166713
time: 2.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9669439, 0.9951854, 0.9745618, 0.9953547, -0.0284109, 0.0206236
1: -0.0047330, -0.0024636, -0.0046222, -0.0024214, -0.0023115, 0.0021586
2: 0.0030020, 0.0150282, 0.0027783, 0.0144413, -0.0114393, 0.0122499
3: -0.0084305, -0.0026395, -0.0080444, -0.0025377, -0.0058928, 0.0054049
4: 0.0011089, 0.0048084, 0.0010656, 0.0041805, -0.0030716, 0.0037428
5: 0.0027352, 0.0320741, 0.0024538, 0.0260073, -0.0232722, 0.0296203
6: -0.0029925, 0.0008466, -0.0028051, 0.0009180, -0.0039105, 0.0036517
7: -0.0108801, -0.0009472, -0.0103953, -0.0007624, -0.0101177, 0.0094481
8: -0.0052859, 0.0019384, -0.0050309, 0.0000349, -0.0053208, 0.0069693
9: -0.0017917, 0.0042654, -0.0019043, 0.0039698, -0.0057614, 0.0061697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165893, upper bound: 0.0167774
time: 1.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165884, upper bound: 0.0167773
time: 1.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9623438, 0.9953049, 0.9745618, 0.9953547, -0.0330109, 0.0207431
1: -0.0047998, -0.0024339, -0.0046222, -0.0024214, -0.0023784, 0.0021884
2: 0.0028441, 0.0153826, 0.0027783, 0.0144413, -0.0115972, 0.0126043
3: -0.0086636, -0.0025676, -0.0080444, -0.0025377, -0.0061259, 0.0054768
4: 0.0010784, 0.0051875, 0.0010656, 0.0041805, -0.0031021, 0.0041219
5: 0.0025366, 0.0357373, 0.0024538, 0.0260073, -0.0234708, 0.0332835
6: -0.0031056, 0.0008970, -0.0028051, 0.0009180, -0.0040236, 0.0037021
7: -0.0111728, -0.0008168, -0.0103953, -0.0007624, -0.0104104, 0.0095785
8: -0.0054398, 0.0034198, -0.0050309, 0.0000349, -0.0054747, 0.0084507
9: -0.0018712, 0.0044439, -0.0019043, 0.0039698, -0.0058410, 0.0063482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165893, upper bound: 0.0167774
time: 2.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165884, upper bound: 0.0167774
time: 2.17 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9671127, 0.9952742, 0.9702910, 0.9954327, -0.0283200, 0.0249832
1: -0.0047305, -0.0024415, -0.0046843, -0.0024020, -0.0023285, 0.0022428
2: 0.0028846, 0.0150152, 0.0026754, 0.0147703, -0.0118857, 0.0123397
3: -0.0084219, -0.0025861, -0.0082608, -0.0024909, -0.0059310, 0.0056748
4: 0.0010862, 0.0047944, 0.0010457, 0.0045325, -0.0034463, 0.0037487
5: 0.0025876, 0.0319395, 0.0023244, 0.0294083, -0.0268208, 0.0296151
6: -0.0029883, 0.0008841, -0.0029101, 0.0009509, -0.0039392, 0.0037942
7: -0.0108693, -0.0008502, -0.0106671, -0.0006774, -0.0101919, 0.0098168
8: -0.0052802, 0.0018840, -0.0051738, 0.0008603, -0.0061405, 0.0070578
9: -0.0018508, 0.0042588, -0.0019561, 0.0041355, -0.0059862, 0.0062149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166426, upper bound: 0.0166962
time: 2.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166426, upper bound: 0.0166962
time: 2.28 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9627867, 0.9952946, 0.9704022, 0.9953749, -0.0325882, 0.0248924
1: -0.0047934, -0.0024364, -0.0046827, -0.0024164, -0.0023770, 0.0022462
2: 0.0028578, 0.0153485, 0.0027517, 0.0147617, -0.0119039, 0.0125968
3: -0.0086411, -0.0025739, -0.0082552, -0.0025256, -0.0061156, 0.0056813
4: 0.0010810, 0.0051510, 0.0010605, 0.0045233, -0.0034423, 0.0040905
5: 0.0025538, 0.0353847, 0.0024204, 0.0293198, -0.0267661, 0.0329643
6: -0.0030947, 0.0008927, -0.0029074, 0.0009265, -0.0040212, 0.0038001
7: -0.0111446, -0.0008281, -0.0106600, -0.0007404, -0.0104041, 0.0098319
8: -0.0054250, 0.0032772, -0.0051701, 0.0008245, -0.0062495, 0.0084473
9: -0.0018643, 0.0044267, -0.0019177, 0.0041312, -0.0059955, 0.0063444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 231
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166426, upper bound: 0.0166901
time: 1.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0166425, upper bound: 0.0166901
time: 2.09 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.44 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165055, upper bound: 0.0164573
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165136, upper bound: 0.0163835
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164864, upper bound: 0.0163835
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165136, upper bound: 0.0163835
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164864, upper bound: 0.0163835
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163542, upper bound: 0.0164431
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163487, upper bound: 0.0164431
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163542, upper bound: 0.0164431
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163487, upper bound: 0.0164431
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164661, upper bound: 0.0164664
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164446, upper bound: 0.0164664
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164661, upper bound: 0.0164664
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164446, upper bound: 0.0164664
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162773
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162773
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162843
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164186, upper bound: 0.0162843
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165042, upper bound: 0.0162931
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165042, upper bound: 0.0162931
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165042, upper bound: 0.0162998
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165042, upper bound: 0.0162998
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162843, upper bound: 0.0163651
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162843, upper bound: 0.0163651
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162843, upper bound: 0.0163688
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162843, upper bound: 0.0163688
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162185, upper bound: 0.0159699
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162149, upper bound: 0.0161313
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162185, upper bound: 0.0159751
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162149, upper bound: 0.0161320
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164864
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164864
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164661
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164661
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164446
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0164664, upper bound: 0.0164446
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162931, upper bound: 0.0165041
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162998, upper bound: 0.0165042
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162931, upper bound: 0.0165046
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0162998, upper bound: 0.0165046
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163654, upper bound: 0.0164640
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163654, upper bound: 0.0164640
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163654, upper bound: 0.0164593
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0163654, upper bound: 0.0164593
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167834
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167834
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167584
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0166721, upper bound: 0.0167584
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0167495, upper bound: 0.0166949
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0167495, upper bound: 0.0166949
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0167494, upper bound: 0.0166713
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0167494, upper bound: 0.0166713
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165893, upper bound: 0.0167774
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165884, upper bound: 0.0167773
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165893, upper bound: 0.0167774
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0165884, upper bound: 0.0167774
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0166426, upper bound: 0.0166962
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0166426, upper bound: 0.0166962
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0166426, upper bound: 0.0166901
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.44
Output dim: 0, lower bound: -0.0166425, upper bound: 0.0166901

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9682679, 0.9947788, 0.9682679, 0.9947788, -0.0265109, 0.0265109
1: -0.0047137, -0.0025650, -0.0047137, -0.0025650, -0.0021487, 0.0021487
2: 0.0035389, 0.0149262, 0.0035389, 0.0149262, -0.0113872, 0.0113872
3: -0.0083634, -0.0028839, -0.0083634, -0.0028839, -0.0054795, 0.0054795
4: 0.0012128, 0.0046992, 0.0012128, 0.0046992, -0.0034864, 0.0034864
5: 0.0034105, 0.0310196, 0.0034105, 0.0310196, -0.0276091, 0.0276091
6: -0.0029599, 0.0006752, -0.0029599, 0.0006752, -0.0036351, 0.0036351
7: -0.0107958, -0.0013906, -0.0107958, -0.0013906, -0.0094052, 0.0094052
8: -0.0052415, 0.0015120, -0.0052415, 0.0015120, -0.0067535, 0.0067535
9: -0.0015212, 0.0042140, -0.0015212, 0.0042140, -0.0057352, 0.0057352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165755, upper bound: 0.0164120
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165755, upper bound: 0.0164091
time: 1.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9682679, 0.9947788, 0.9637401, 0.9949022, -0.0266343, 0.0310386
1: -0.0047137, -0.0025650, -0.0047795, -0.0025342, -0.0021795, 0.0022146
2: 0.0035389, 0.0149262, 0.0033758, 0.0152750, -0.0117361, 0.0115503
3: -0.0083634, -0.0028839, -0.0085928, -0.0028097, -0.0055537, 0.0057089
4: 0.0012128, 0.0046992, 0.0011813, 0.0050724, -0.0038596, 0.0035180
5: 0.0034105, 0.0310196, 0.0032054, 0.0346254, -0.0312149, 0.0278143
6: -0.0029599, 0.0006752, -0.0030713, 0.0007273, -0.0036872, 0.0037465
7: -0.0107958, -0.0013906, -0.0110839, -0.0012559, -0.0095399, 0.0096933
8: -0.0052415, 0.0015120, -0.0053931, 0.0029702, -0.0082117, 0.0069050
9: -0.0015212, 0.0042140, -0.0016034, 0.0043897, -0.0059109, 0.0058174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165755, upper bound: 0.0164120
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0165755, upper bound: 0.0164091
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9599677, 0.9947580, 0.9682679, 0.9947788, -0.0348110, 0.0264901
1: -0.0048344, -0.0025701, -0.0047137, -0.0025650, -0.0022694, 0.0021436
2: 0.0035663, 0.0155656, 0.0035389, 0.0149262, -0.0113598, 0.0120267
3: -0.0087840, -0.0028964, -0.0083634, -0.0028839, -0.0059001, 0.0054670
4: 0.0012181, 0.0053834, 0.0012128, 0.0046992, -0.0034811, 0.0041705
5: 0.0034450, 0.0376295, 0.0034105, 0.0310196, -0.0275747, 0.0342191
6: -0.0031640, 0.0006665, -0.0029599, 0.0006752, -0.0038393, 0.0036264
7: -0.0113240, -0.0014133, -0.0107958, -0.0013906, -0.0099333, 0.0093825
8: -0.0055193, 0.0041851, -0.0052415, 0.0015120, -0.0070313, 0.0094266
9: -0.0015074, 0.0045361, -0.0015212, 0.0042140, -0.0057214, 0.0060573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163970, upper bound: 0.0163460
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163970, upper bound: 0.0163618
time: 1.78 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9599677, 0.9947580, 0.9637401, 0.9949022, -0.0349345, 0.0310178
1: -0.0048344, -0.0025701, -0.0047795, -0.0025342, -0.0023002, 0.0022094
2: 0.0035663, 0.0155656, 0.0033758, 0.0152750, -0.0117087, 0.0121898
3: -0.0087840, -0.0028964, -0.0085928, -0.0028097, -0.0059743, 0.0056964
4: 0.0012181, 0.0053834, 0.0011813, 0.0050724, -0.0038543, 0.0042021
5: 0.0034450, 0.0376295, 0.0032054, 0.0346254, -0.0311804, 0.0344242
6: -0.0031640, 0.0006665, -0.0030713, 0.0007273, -0.0038913, 0.0037377
7: -0.0113240, -0.0014133, -0.0110839, -0.0012559, -0.0100680, 0.0096706
8: -0.0055193, 0.0041851, -0.0053931, 0.0029702, -0.0084895, 0.0095782
9: -0.0015074, 0.0045361, -0.0016034, 0.0043897, -0.0058971, 0.0061394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163970, upper bound: 0.0163460
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163970, upper bound: 0.0163618
time: 1.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9683803, 0.9947430, 0.9748567, 0.9953405, -0.0269602, 0.0198863
1: -0.0047121, -0.0025739, -0.0046179, -0.0024250, -0.0022871, 0.0020440
2: 0.0035861, 0.0149175, 0.0027972, 0.0144185, -0.0108324, 0.0121203
3: -0.0083577, -0.0029054, -0.0080295, -0.0025463, -0.0058114, 0.0051241
4: 0.0012220, 0.0046900, 0.0010693, 0.0041562, -0.0029342, 0.0036207
5: 0.0034699, 0.0309300, 0.0024776, 0.0257724, -0.0223026, 0.0284525
6: -0.0029571, 0.0006601, -0.0027979, 0.0009120, -0.0038691, 0.0034580
7: -0.0107886, -0.0014296, -0.0103765, -0.0007780, -0.0100106, 0.0089469
8: -0.0052378, 0.0014757, -0.0050211, 0.0000267, -0.0052645, 0.0064968
9: -0.0014975, 0.0042096, -0.0018948, 0.0039583, -0.0054558, 0.0061044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163317, upper bound: 0.0162184
time: 2.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164786, upper bound: 0.0162184
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9685006, 0.9946887, 0.9718309, 0.9952995, -0.0267990, 0.0228578
1: -0.0047103, -0.0025874, -0.0046619, -0.0024352, -0.0022751, 0.0020745
2: 0.0036577, 0.0149082, 0.0028513, 0.0146517, -0.0109940, 0.0120570
3: -0.0083516, -0.0029379, -0.0081828, -0.0025709, -0.0057807, 0.0052449
4: 0.0012358, 0.0046801, 0.0010797, 0.0044056, -0.0031697, 0.0036003
5: 0.0035598, 0.0308343, 0.0025456, 0.0281821, -0.0246223, 0.0282888
6: -0.0029542, 0.0006373, -0.0028723, 0.0008947, -0.0038489, 0.0035096
7: -0.0107810, -0.0014887, -0.0105691, -0.0008227, -0.0099583, 0.0090804
8: -0.0052338, 0.0014370, -0.0051223, 0.0003644, -0.0055982, 0.0065593
9: -0.0014614, 0.0042050, -0.0018676, 0.0040757, -0.0055372, 0.0060725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163084, upper bound: 0.0162180
time: 1.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164616, upper bound: 0.0162180
time: 2.09 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9600872, 0.9947204, 0.9748567, 0.9953405, -0.0352532, 0.0198637
1: -0.0048326, -0.0025795, -0.0046179, -0.0024250, -0.0024076, 0.0020384
2: 0.0036160, 0.0155564, 0.0027972, 0.0144185, -0.0108026, 0.0127593
3: -0.0087779, -0.0029190, -0.0080295, -0.0025463, -0.0062316, 0.0051105
4: 0.0012278, 0.0053735, 0.0010693, 0.0041562, -0.0029284, 0.0043042
5: 0.0035074, 0.0375344, 0.0024776, 0.0257724, -0.0222650, 0.0350569
6: -0.0031611, 0.0006506, -0.0027979, 0.0009120, -0.0040731, 0.0034485
7: -0.0113164, -0.0014543, -0.0103765, -0.0007780, -0.0105384, 0.0089223
8: -0.0055153, 0.0041466, -0.0050211, 0.0000267, -0.0055420, 0.0091677
9: -0.0014824, 0.0045314, -0.0018948, 0.0039583, -0.0054407, 0.0064262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0161707, upper bound: 0.0161562
time: 1.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162873, upper bound: 0.0161540
time: 2.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9601806, 0.9946814, 0.9718309, 0.9952995, -0.0351189, 0.0228505
1: -0.0048313, -0.0025892, -0.0046619, -0.0024352, -0.0023961, 0.0020727
2: 0.0036674, 0.0155492, 0.0028513, 0.0146517, -0.0109843, 0.0126980
3: -0.0087732, -0.0029424, -0.0081828, -0.0025709, -0.0062023, 0.0052404
4: 0.0012377, 0.0053658, 0.0010797, 0.0044056, -0.0031679, 0.0042861
5: 0.0035721, 0.0374601, 0.0025456, 0.0281821, -0.0246100, 0.0349146
6: -0.0031588, 0.0006342, -0.0028723, 0.0008947, -0.0040535, 0.0035065
7: -0.0113104, -0.0014968, -0.0105691, -0.0008227, -0.0104877, 0.0090723
8: -0.0055122, 0.0041166, -0.0051223, 0.0003644, -0.0058766, 0.0092389
9: -0.0014565, 0.0045278, -0.0018676, 0.0040757, -0.0055322, 0.0063954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160097, upper bound: 0.0161562
time: 1.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162627, upper bound: 0.0161540
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9638573, 0.9948661, 0.9684075, 0.9948835, -0.0310262, 0.0264586
1: -0.0047778, -0.0025432, -0.0047117, -0.0025389, -0.0022390, 0.0021685
2: 0.0034235, 0.0152660, 0.0034006, 0.0149154, -0.0114919, 0.0118654
3: -0.0085869, -0.0028314, -0.0083563, -0.0028209, -0.0057659, 0.0055249
4: 0.0011905, 0.0050628, 0.0011861, 0.0046877, -0.0034972, 0.0038767
5: 0.0032654, 0.0345321, 0.0032365, 0.0309084, -0.0276430, 0.0312956
6: -0.0030684, 0.0007121, -0.0029565, 0.0007194, -0.0037877, 0.0036685
7: -0.0110765, -0.0012953, -0.0107869, -0.0012764, -0.0098000, 0.0094916
8: -0.0053891, 0.0029325, -0.0052369, 0.0014670, -0.0068561, 0.0081693
9: -0.0015794, 0.0043851, -0.0015909, 0.0042086, -0.0057879, 0.0059760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160899, upper bound: 0.0162496
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162517, upper bound: 0.0162490
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9639812, 0.9948114, 0.9646998, 0.9948804, -0.0308993, 0.0301116
1: -0.0047760, -0.0025569, -0.0047656, -0.0025396, -0.0022364, 0.0022087
2: 0.0034959, 0.0152564, 0.0034046, 0.0152011, -0.0117051, 0.0118518
3: -0.0085806, -0.0028643, -0.0085442, -0.0028228, -0.0057578, 0.0056799
4: 0.0012045, 0.0050526, 0.0011868, 0.0049933, -0.0037888, 0.0038657
5: 0.0033564, 0.0344335, 0.0032416, 0.0338610, -0.0305046, 0.0311919
6: -0.0030653, 0.0006889, -0.0030477, 0.0007181, -0.0037834, 0.0037366
7: -0.0110686, -0.0013551, -0.0110228, -0.0012797, -0.0097889, 0.0096677
8: -0.0053850, 0.0028926, -0.0053609, 0.0026611, -0.0080461, 0.0082535
9: -0.0015429, 0.0043803, -0.0015889, 0.0043524, -0.0058953, 0.0059692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160877, upper bound: 0.0162488
time: 2.00 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162482, upper bound: 0.0162482
time: 1.61 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9553528, 0.9948589, 0.9684075, 0.9948835, -0.0395308, 0.0264514
1: -0.0049015, -0.0025450, -0.0047117, -0.0025389, -0.0023626, 0.0021667
2: 0.0034332, 0.0159212, 0.0034006, 0.0149154, -0.0114823, 0.0125206
3: -0.0090178, -0.0028357, -0.0083563, -0.0028209, -0.0061969, 0.0055206
4: 0.0011924, 0.0057637, 0.0011861, 0.0046877, -0.0034954, 0.0045777
5: 0.0032774, 0.0413048, 0.0032365, 0.0309084, -0.0276309, 0.0380683
6: -0.0032775, 0.0007090, -0.0029565, 0.0007194, -0.0039969, 0.0036654
7: -0.0116176, -0.0013033, -0.0107869, -0.0012764, -0.0103412, 0.0094836
8: -0.0056737, 0.0056714, -0.0052369, 0.0014670, -0.0071407, 0.0109083
9: -0.0015745, 0.0047151, -0.0015909, 0.0042086, -0.0057831, 0.0063060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159605, upper bound: 0.0162021
time: 2.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160958, upper bound: 0.0161953
time: 1.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9554510, 0.9948141, 0.9646998, 0.9948804, -0.0394295, 0.0301143
1: -0.0049000, -0.0025561, -0.0047656, -0.0025396, -0.0023604, 0.0022094
2: 0.0034922, 0.0159136, 0.0034046, 0.0152011, -0.0117089, 0.0125090
3: -0.0090129, -0.0028626, -0.0085442, -0.0028228, -0.0061901, 0.0056816
4: 0.0012038, 0.0057556, 0.0011868, 0.0049933, -0.0037895, 0.0045688
5: 0.0033517, 0.0412266, 0.0032416, 0.0338610, -0.0305094, 0.0379851
6: -0.0032751, 0.0006901, -0.0030477, 0.0007181, -0.0039932, 0.0037378
7: -0.0116114, -0.0013520, -0.0110228, -0.0012797, -0.0103317, 0.0096708
8: -0.0056705, 0.0056398, -0.0053609, 0.0026611, -0.0083315, 0.0110008
9: -0.0015448, 0.0047113, -0.0015889, 0.0043524, -0.0058972, 0.0063002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159555, upper bound: 0.0162021
time: 1.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160909, upper bound: 0.0161953
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9638573, 0.9948661, 0.9748513, 0.9953412, -0.0314839, 0.0200148
1: -0.0047778, -0.0025432, -0.0046180, -0.0024248, -0.0023530, 0.0020748
2: 0.0034235, 0.0152660, 0.0027961, 0.0144190, -0.0109954, 0.0124699
3: -0.0085869, -0.0028314, -0.0080297, -0.0025458, -0.0060411, 0.0051984
4: 0.0011905, 0.0050628, 0.0010691, 0.0041566, -0.0029661, 0.0039937
5: 0.0032654, 0.0345321, 0.0024762, 0.0257767, -0.0225114, 0.0320559
6: -0.0030684, 0.0007121, -0.0027980, 0.0009123, -0.0039807, 0.0035100
7: -0.0110765, -0.0012953, -0.0103769, -0.0007771, -0.0102994, 0.0090815
8: -0.0053891, 0.0029325, -0.0050212, 0.0000272, -0.0054163, 0.0079537
9: -0.0015794, 0.0043851, -0.0018954, 0.0039585, -0.0055379, 0.0062805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162516, upper bound: 0.0162930
time: 1.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163775, upper bound: 0.0162929
time: 1.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9639812, 0.9948114, 0.9718214, 0.9952992, -0.0313180, 0.0229899
1: -0.0047760, -0.0025569, -0.0046620, -0.0024353, -0.0023407, 0.0021052
2: 0.0034959, 0.0152564, 0.0028517, 0.0146524, -0.0111565, 0.0124047
3: -0.0085806, -0.0028643, -0.0081833, -0.0025711, -0.0060095, 0.0053190
4: 0.0012045, 0.0050526, 0.0010798, 0.0044063, -0.0032018, 0.0039727
5: 0.0033564, 0.0344335, 0.0025462, 0.0281896, -0.0248332, 0.0318873
6: -0.0030653, 0.0006889, -0.0028725, 0.0008946, -0.0039599, 0.0035614
7: -0.0110686, -0.0013551, -0.0105697, -0.0008230, -0.0102455, 0.0092145
8: -0.0053850, 0.0028926, -0.0051226, 0.0003675, -0.0057525, 0.0080152
9: -0.0015429, 0.0043803, -0.0018673, 0.0040761, -0.0056190, 0.0062477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162314, upper bound: 0.0162922
time: 1.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163624, upper bound: 0.0162922
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9553528, 0.9948589, 0.9748513, 0.9953412, -0.0399885, 0.0200076
1: -0.0049015, -0.0025450, -0.0046180, -0.0024248, -0.0024767, 0.0020730
2: 0.0034332, 0.0159212, 0.0027961, 0.0144190, -0.0109858, 0.0131251
3: -0.0090178, -0.0028357, -0.0080297, -0.0025458, -0.0064721, 0.0051940
4: 0.0011924, 0.0057637, 0.0010691, 0.0041566, -0.0029642, 0.0046947
5: 0.0032774, 0.0413048, 0.0024762, 0.0257767, -0.0224993, 0.0388286
6: -0.0032775, 0.0007090, -0.0027980, 0.0009123, -0.0041899, 0.0035070
7: -0.0116176, -0.0013033, -0.0103769, -0.0007771, -0.0108405, 0.0090736
8: -0.0056737, 0.0056714, -0.0050212, 0.0000272, -0.0057009, 0.0106926
9: -0.0015745, 0.0047151, -0.0018954, 0.0039585, -0.0055330, 0.0066105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0161200, upper bound: 0.0162329
time: 1.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162242, upper bound: 0.0162273
time: 1.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9554510, 0.9948141, 0.9718214, 0.9952992, -0.0398482, 0.0229927
1: -0.0049000, -0.0025561, -0.0046620, -0.0024353, -0.0024648, 0.0021059
2: 0.0034922, 0.0159136, 0.0028517, 0.0146524, -0.0111602, 0.0130619
3: -0.0090129, -0.0028626, -0.0081833, -0.0025711, -0.0064418, 0.0053207
4: 0.0012038, 0.0057556, 0.0010798, 0.0044063, -0.0032026, 0.0046758
5: 0.0033517, 0.0412266, 0.0025462, 0.0281896, -0.0248380, 0.0386805
6: -0.0032751, 0.0006901, -0.0028725, 0.0008946, -0.0041697, 0.0035626
7: -0.0116114, -0.0013520, -0.0105697, -0.0008230, -0.0107883, 0.0092176
8: -0.0056705, 0.0056398, -0.0051226, 0.0003675, -0.0060379, 0.0107624
9: -0.0015448, 0.0047113, -0.0018673, 0.0040761, -0.0056209, 0.0065787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160959, upper bound: 0.0162329
time: 1.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162018, upper bound: 0.0162273
time: 2.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9686901, 0.9947357, 0.9600866, 0.9947204, -0.0260303, 0.0346491
1: -0.0047076, -0.0025757, -0.0048326, -0.0025795, -0.0021281, 0.0022570
2: 0.0035957, 0.0148936, 0.0036160, 0.0155565, -0.0119608, 0.0112777
3: -0.0083420, -0.0029097, -0.0087780, -0.0029190, -0.0054230, 0.0058682
4: 0.0012238, 0.0046644, 0.0012278, 0.0053736, -0.0041497, 0.0034367
5: 0.0034819, 0.0306833, 0.0035074, 0.0375349, -0.0340530, 0.0271759
6: -0.0029495, 0.0006571, -0.0031611, 0.0006506, -0.0036001, 0.0038182
7: -0.0107689, -0.0014376, -0.0113164, -0.0014543, -0.0093147, 0.0098788
8: -0.0052274, 0.0013759, -0.0055153, 0.0041468, -0.0093742, 0.0068913
9: -0.0014926, 0.0041976, -0.0014824, 0.0045314, -0.0060241, 0.0056800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163969, upper bound: 0.0162700
time: 1.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163969, upper bound: 0.0162773
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9686901, 0.9947357, 0.9553486, 0.9948498, -0.0261597, 0.0393872
1: -0.0047076, -0.0025757, -0.0049015, -0.0025472, -0.0021603, 0.0023258
2: 0.0035957, 0.0148936, 0.0034450, 0.0159215, -0.0123258, 0.0114486
3: -0.0083420, -0.0029097, -0.0090181, -0.0028411, -0.0055008, 0.0061083
4: 0.0012238, 0.0046644, 0.0011947, 0.0057641, -0.0045402, 0.0034698
5: 0.0034819, 0.0306833, 0.0032924, 0.0413081, -0.0378262, 0.0273909
6: -0.0029495, 0.0006571, -0.0032776, 0.0007052, -0.0036547, 0.0039347
7: -0.0107689, -0.0014376, -0.0116179, -0.0013131, -0.0094558, 0.0101803
8: -0.0052274, 0.0013759, -0.0056739, 0.0056727, -0.0109002, 0.0070498
9: -0.0014926, 0.0041976, -0.0015685, 0.0047153, -0.0062079, 0.0057661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163969, upper bound: 0.0162700
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163969, upper bound: 0.0162773
time: 1.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9649394, 0.9947543, 0.9601798, 0.9946814, -0.0297420, 0.0345745
1: -0.0047621, -0.0025711, -0.0048313, -0.0025892, -0.0021729, 0.0022602
2: 0.0035712, 0.0151826, 0.0036674, 0.0155493, -0.0119781, 0.0115152
3: -0.0085320, -0.0028986, -0.0087732, -0.0029424, -0.0055897, 0.0058746
4: 0.0012191, 0.0049736, 0.0012377, 0.0053659, -0.0041468, 0.0037359
5: 0.0034511, 0.0336702, 0.0035721, 0.0374606, -0.0340095, 0.0300981
6: -0.0030418, 0.0006649, -0.0031588, 0.0006342, -0.0036760, 0.0038237
7: -0.0110076, -0.0014173, -0.0113105, -0.0014968, -0.0095108, 0.0098931
8: -0.0053529, 0.0025839, -0.0055122, 0.0041168, -0.0094697, 0.0080961
9: -0.0015050, 0.0043431, -0.0014565, 0.0045278, -0.0060328, 0.0057997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163969, upper bound: 0.0162680
time: 1.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163969, upper bound: 0.0162843
time: 2.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9649394, 0.9947543, 0.9554466, 0.9948087, -0.0298693, 0.0393077
1: -0.0047621, -0.0025711, -0.0049001, -0.0025575, -0.0022046, 0.0023290
2: 0.0035712, 0.0151826, 0.0034994, 0.0159140, -0.0123428, 0.0116832
3: -0.0085320, -0.0028986, -0.0090131, -0.0028659, -0.0056661, 0.0061145
4: 0.0012191, 0.0049736, 0.0012052, 0.0057560, -0.0045369, 0.0037684
5: 0.0034511, 0.0336702, 0.0033608, 0.0412300, -0.0377789, 0.0303094
6: -0.0030418, 0.0006649, -0.0032752, 0.0006878, -0.0037296, 0.0039401
7: -0.0110076, -0.0014173, -0.0116116, -0.0013580, -0.0096496, 0.0101943
8: -0.0053529, 0.0025839, -0.0056706, 0.0056412, -0.0109941, 0.0082545
9: -0.0015050, 0.0043431, -0.0015411, 0.0047115, -0.0062165, 0.0058843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163969, upper bound: 0.0162680
time: 2.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163969, upper bound: 0.0162843
time: 2.30 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9686901, 0.9947357, 0.9670730, 0.9951481, -0.0264580, 0.0276627
1: -0.0047076, -0.0025757, -0.0047311, -0.0024729, -0.0022347, 0.0021554
2: 0.0035957, 0.0148936, 0.0030511, 0.0150182, -0.0114225, 0.0118425
3: -0.0083420, -0.0029097, -0.0084239, -0.0026619, -0.0056801, 0.0055142
4: 0.0012238, 0.0046644, 0.0011184, 0.0047977, -0.0035739, 0.0035460
5: 0.0034819, 0.0306833, 0.0027969, 0.0319712, -0.0284892, 0.0278863
6: -0.0029495, 0.0006571, -0.0029893, 0.0008309, -0.0037805, 0.0036464
7: -0.0107689, -0.0014376, -0.0108718, -0.0009877, -0.0097812, 0.0094343
8: -0.0052274, 0.0013759, -0.0052815, 0.0018968, -0.0071242, 0.0066575
9: -0.0014926, 0.0041976, -0.0017669, 0.0042603, -0.0057530, 0.0059645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164858, upper bound: 0.0162857
time: 1.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164858, upper bound: 0.0162931
time: 1.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9686901, 0.9947357, 0.9624758, 0.9952683, -0.0265782, 0.0322599
1: -0.0047076, -0.0025757, -0.0047979, -0.0024430, -0.0022646, 0.0022222
2: 0.0035957, 0.0148936, 0.0028924, 0.0153724, -0.0117767, 0.0120012
3: -0.0083420, -0.0029097, -0.0086569, -0.0025896, -0.0057523, 0.0057471
4: 0.0012238, 0.0046644, 0.0010877, 0.0051766, -0.0039528, 0.0035767
5: 0.0034819, 0.0306833, 0.0025973, 0.0356322, -0.0321502, 0.0280860
6: -0.0029495, 0.0006571, -0.0031024, 0.0008816, -0.0038311, 0.0037594
7: -0.0107689, -0.0014376, -0.0111644, -0.0008567, -0.0099123, 0.0097268
8: -0.0052274, 0.0013759, -0.0054354, 0.0033773, -0.0086047, 0.0068113
9: -0.0014926, 0.0041976, -0.0018469, 0.0044387, -0.0059314, 0.0060445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164858, upper bound: 0.0162857
time: 1.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164858, upper bound: 0.0162931
time: 1.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9649394, 0.9947543, 0.9671664, 0.9950976, -0.0301582, 0.0275879
1: -0.0047621, -0.0025711, -0.0047297, -0.0024855, -0.0022766, 0.0021587
2: 0.0035712, 0.0151826, 0.0031180, 0.0150110, -0.0114398, 0.0120646
3: -0.0085320, -0.0028986, -0.0084192, -0.0026923, -0.0058397, 0.0055206
4: 0.0012191, 0.0049736, 0.0011314, 0.0047900, -0.0035709, 0.0038422
5: 0.0034511, 0.0336702, 0.0028811, 0.0318968, -0.0284457, 0.0307892
6: -0.0030418, 0.0006649, -0.0029870, 0.0008096, -0.0038513, 0.0036519
7: -0.0110076, -0.0014173, -0.0108659, -0.0010430, -0.0099646, 0.0094486
8: -0.0053529, 0.0025839, -0.0052784, 0.0018667, -0.0072196, 0.0078623
9: -0.0015050, 0.0043431, -0.0017332, 0.0042567, -0.0057617, 0.0060764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164858, upper bound: 0.0162825
time: 1.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164858, upper bound: 0.0162998
time: 1.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9649394, 0.9947543, 0.9625733, 0.9952161, -0.0302767, 0.0321810
1: -0.0047621, -0.0025711, -0.0047965, -0.0024560, -0.0023061, 0.0022254
2: 0.0035712, 0.0151826, 0.0029613, 0.0153649, -0.0117937, 0.0122213
3: -0.0085320, -0.0028986, -0.0086519, -0.0026210, -0.0059110, 0.0057534
4: 0.0012191, 0.0049736, 0.0011010, 0.0051686, -0.0039495, 0.0038725
5: 0.0034511, 0.0336702, 0.0026840, 0.0355546, -0.0321035, 0.0309862
6: -0.0030418, 0.0006649, -0.0031000, 0.0008596, -0.0039014, 0.0037649
7: -0.0110076, -0.0014173, -0.0111582, -0.0009136, -0.0100940, 0.0097408
8: -0.0053529, 0.0025839, -0.0054321, 0.0033460, -0.0086989, 0.0080160
9: -0.0015050, 0.0043431, -0.0018122, 0.0044349, -0.0059399, 0.0061553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164858, upper bound: 0.0162825
time: 1.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0164858, upper bound: 0.0162998
time: 1.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9641901, 0.9948631, 0.9600866, 0.9947204, -0.0305303, 0.0347764
1: -0.0047730, -0.0025439, -0.0048326, -0.0025795, -0.0021935, 0.0022887
2: 0.0034275, 0.0152403, 0.0036160, 0.0155565, -0.0121290, 0.0116244
3: -0.0085700, -0.0028332, -0.0087780, -0.0029190, -0.0056511, 0.0059448
4: 0.0011913, 0.0050353, 0.0012278, 0.0053736, -0.0041823, 0.0038076
5: 0.0032704, 0.0342670, 0.0035074, 0.0375349, -0.0342646, 0.0307596
6: -0.0030602, 0.0007108, -0.0031611, 0.0006506, -0.0037108, 0.0038719
7: -0.0110553, -0.0012986, -0.0113164, -0.0014543, -0.0096010, 0.0100178
8: -0.0053780, 0.0028252, -0.0055153, 0.0041468, -0.0095248, 0.0083405
9: -0.0015774, 0.0043722, -0.0014824, 0.0045314, -0.0061088, 0.0058546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162680, upper bound: 0.0163524
time: 1.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162680, upper bound: 0.0163651
time: 1.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9641901, 0.9948631, 0.9553486, 0.9948498, -0.0306597, 0.0395145
1: -0.0047730, -0.0025439, -0.0049015, -0.0025472, -0.0022257, 0.0023576
2: 0.0034275, 0.0152403, 0.0034450, 0.0159215, -0.0124940, 0.0117953
3: -0.0085700, -0.0028332, -0.0090181, -0.0028411, -0.0057289, 0.0061849
4: 0.0011913, 0.0050353, 0.0011947, 0.0057641, -0.0045728, 0.0038407
5: 0.0032704, 0.0342670, 0.0032924, 0.0413081, -0.0380378, 0.0309746
6: -0.0030602, 0.0007108, -0.0032776, 0.0007052, -0.0037654, 0.0039884
7: -0.0110553, -0.0012986, -0.0116179, -0.0013131, -0.0097422, 0.0103193
8: -0.0053780, 0.0028252, -0.0056739, 0.0056727, -0.0110508, 0.0084991
9: -0.0015774, 0.0043722, -0.0015685, 0.0047153, -0.0062926, 0.0059407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162680, upper bound: 0.0163524
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162680, upper bound: 0.0163651
time: 1.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9605731, 0.9948773, 0.9601798, 0.9946814, -0.0341083, 0.0346975
1: -0.0048256, -0.0025404, -0.0048313, -0.0025892, -0.0022364, 0.0022909
2: 0.0034087, 0.0155190, 0.0036674, 0.0155493, -0.0121406, 0.0118516
3: -0.0087533, -0.0028246, -0.0087732, -0.0029424, -0.0058109, 0.0059486
4: 0.0011876, 0.0053335, 0.0012377, 0.0053659, -0.0041782, 0.0040958
5: 0.0032467, 0.0371475, 0.0035721, 0.0374606, -0.0342139, 0.0335754
6: -0.0031492, 0.0007168, -0.0031588, 0.0006342, -0.0037833, 0.0038756
7: -0.0112854, -0.0012831, -0.0113105, -0.0014968, -0.0097887, 0.0100274
8: -0.0054990, 0.0039901, -0.0055122, 0.0041168, -0.0096158, 0.0095023
9: -0.0015868, 0.0045126, -0.0014565, 0.0045278, -0.0061146, 0.0059691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162680, upper bound: 0.0163487
time: 1.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162680, upper bound: 0.0163688
time: 1.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9605731, 0.9948773, 0.9554466, 0.9948087, -0.0342355, 0.0394307
1: -0.0048256, -0.0025404, -0.0049001, -0.0025575, -0.0022681, 0.0023597
2: 0.0034087, 0.0155190, 0.0034994, 0.0159140, -0.0125053, 0.0120196
3: -0.0087533, -0.0028246, -0.0090131, -0.0028659, -0.0058874, 0.0061885
4: 0.0011876, 0.0053335, 0.0012052, 0.0057560, -0.0045684, 0.0041283
5: 0.0032467, 0.0371475, 0.0033608, 0.0412300, -0.0379833, 0.0337867
6: -0.0031492, 0.0007168, -0.0032752, 0.0006878, -0.0038370, 0.0039920
7: -0.0112854, -0.0012831, -0.0116116, -0.0013580, -0.0099274, 0.0103286
8: -0.0054990, 0.0039901, -0.0056706, 0.0056412, -0.0111402, 0.0096607
9: -0.0015868, 0.0045126, -0.0015411, 0.0047115, -0.0062983, 0.0060537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162680, upper bound: 0.0163487
time: 2.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162680, upper bound: 0.0163688
time: 1.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9652723, 0.9948539, 0.9705098, 0.9953772, -0.0301049, 0.0243441
1: -0.0047573, -0.0025462, -0.0046811, -0.0024159, -0.0023414, 0.0021349
2: 0.0034396, 0.0151569, 0.0027487, 0.0147534, -0.0113139, 0.0124082
3: -0.0085152, -0.0028387, -0.0082497, -0.0025242, -0.0059909, 0.0054111
4: 0.0011936, 0.0049461, 0.0010599, 0.0045145, -0.0033208, 0.0038862
5: 0.0032855, 0.0334052, 0.0024166, 0.0292342, -0.0259487, 0.0309885
6: -0.0030336, 0.0007069, -0.0029048, 0.0009275, -0.0039610, 0.0036117
7: -0.0109864, -0.0013086, -0.0106531, -0.0007380, -0.0102484, 0.0093446
8: -0.0053418, 0.0024767, -0.0051665, 0.0007899, -0.0061317, 0.0076432
9: -0.0015713, 0.0043302, -0.0019192, 0.0041270, -0.0056983, 0.0062494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162019, upper bound: 0.0159610
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162019, upper bound: 0.0159699
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9649186, 0.9948573, 0.9686911, 0.9956214, -0.0307028, 0.0261661
1: -0.0047624, -0.0025454, -0.0047075, -0.0023550, -0.0024074, 0.0021622
2: 0.0034352, 0.0151842, 0.0024262, 0.0148935, -0.0114583, 0.0127580
3: -0.0085331, -0.0028367, -0.0083419, -0.0023774, -0.0061557, 0.0055052
4: 0.0011928, 0.0049753, 0.0009975, 0.0046643, -0.0034716, 0.0039778
5: 0.0032800, 0.0336868, 0.0020109, 0.0306825, -0.0274024, 0.0316759
6: -0.0030423, 0.0007083, -0.0029495, 0.0010304, -0.0040727, 0.0036578
7: -0.0110089, -0.0013050, -0.0107689, -0.0004716, -0.0105373, 0.0094639
8: -0.0053536, 0.0025906, -0.0052274, 0.0013756, -0.0067292, 0.0078180
9: -0.0015735, 0.0043439, -0.0020817, 0.0041976, -0.0057710, 0.0064256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162007, upper bound: 0.0161198
time: 1.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162007, upper bound: 0.0161313
time: 1.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9616556, 0.9948673, 0.9706045, 0.9953268, -0.0336712, 0.0242628
1: -0.0048098, -0.0025429, -0.0046797, -0.0024284, -0.0023814, 0.0021368
2: 0.0034220, 0.0154356, 0.0028154, 0.0147462, -0.0113241, 0.0126203
3: -0.0086985, -0.0028307, -0.0082450, -0.0025546, -0.0061439, 0.0054143
4: 0.0011902, 0.0052442, 0.0010728, 0.0045066, -0.0033164, 0.0041714
5: 0.0032635, 0.0362855, 0.0025004, 0.0291588, -0.0258954, 0.0337851
6: -0.0031225, 0.0007125, -0.0029024, 0.0009062, -0.0040287, 0.0036150
7: -0.0112166, -0.0012941, -0.0106471, -0.0007930, -0.0104235, 0.0093530
8: -0.0054628, 0.0036415, -0.0051634, 0.0007594, -0.0062222, 0.0088049
9: -0.0015801, 0.0044706, -0.0018857, 0.0041233, -0.0057034, 0.0063562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162019, upper bound: 0.0159600
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162019, upper bound: 0.0159751
time: 1.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9613091, 0.9948705, 0.9687921, 0.9955729, -0.0342638, 0.0260785
1: -0.0048149, -0.0025421, -0.0047061, -0.0023671, -0.0024478, 0.0021640
2: 0.0034176, 0.0154623, 0.0024904, 0.0148858, -0.0114682, 0.0129719
3: -0.0087160, -0.0028287, -0.0083368, -0.0024066, -0.0063094, 0.0055081
4: 0.0011894, 0.0052728, 0.0010099, 0.0046560, -0.0034667, 0.0042629
5: 0.0032578, 0.0365613, 0.0020917, 0.0306021, -0.0273442, 0.0344697
6: -0.0031310, 0.0007140, -0.0029470, 0.0010099, -0.0041410, 0.0036610
7: -0.0112386, -0.0012904, -0.0107624, -0.0005246, -0.0107140, 0.0094720
8: -0.0054744, 0.0037531, -0.0052240, 0.0013431, -0.0068175, 0.0089771
9: -0.0015824, 0.0044840, -0.0020493, 0.0041936, -0.0057760, 0.0065334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162007, upper bound: 0.0161147
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162007, upper bound: 0.0161320
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9748567, 0.9953405, 0.9683803, 0.9947430, -0.0198863, 0.0269602
1: -0.0046179, -0.0024250, -0.0047121, -0.0025739, -0.0020440, 0.0022871
2: 0.0027972, 0.0144185, 0.0035861, 0.0149175, -0.0121203, 0.0108324
3: -0.0080295, -0.0025463, -0.0083577, -0.0029054, -0.0051241, 0.0058114
4: 0.0010693, 0.0041562, 0.0012220, 0.0046900, -0.0036207, 0.0029342
5: 0.0024776, 0.0257724, 0.0034699, 0.0309300, -0.0284525, 0.0223026
6: -0.0027979, 0.0009120, -0.0029571, 0.0006601, -0.0034580, 0.0038691
7: -0.0103765, -0.0007780, -0.0107886, -0.0014296, -0.0089469, 0.0100106
8: -0.0050211, 0.0000267, -0.0052378, 0.0014757, -0.0064968, 0.0052645
9: -0.0018948, 0.0039583, -0.0014975, 0.0042096, -0.0061044, 0.0054558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
time: 1.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
time: 2.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9748567, 0.9953405, 0.9600872, 0.9947204, -0.0198637, 0.0352532
1: -0.0046179, -0.0024250, -0.0048326, -0.0025795, -0.0020384, 0.0024076
2: 0.0027972, 0.0144185, 0.0036160, 0.0155564, -0.0127593, 0.0108026
3: -0.0080295, -0.0025463, -0.0087779, -0.0029190, -0.0051105, 0.0062316
4: 0.0010693, 0.0041562, 0.0012278, 0.0053735, -0.0043042, 0.0029284
5: 0.0024776, 0.0257724, 0.0035074, 0.0375344, -0.0350569, 0.0222650
6: -0.0027979, 0.0009120, -0.0031611, 0.0006506, -0.0034485, 0.0040731
7: -0.0103765, -0.0007780, -0.0113164, -0.0014543, -0.0089223, 0.0105384
8: -0.0050211, 0.0000267, -0.0055153, 0.0041466, -0.0091677, 0.0055420
9: -0.0018948, 0.0039583, -0.0014824, 0.0045314, -0.0064262, 0.0054407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0165136
time: 1.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9718309, 0.9952995, 0.9685006, 0.9946887, -0.0228578, 0.0267990
1: -0.0046619, -0.0024352, -0.0047103, -0.0025874, -0.0020745, 0.0022751
2: 0.0028513, 0.0146517, 0.0036577, 0.0149082, -0.0120570, 0.0109940
3: -0.0081828, -0.0025709, -0.0083516, -0.0029379, -0.0052449, 0.0057807
4: 0.0010797, 0.0044056, 0.0012358, 0.0046801, -0.0036003, 0.0031697
5: 0.0025456, 0.0281821, 0.0035598, 0.0308343, -0.0282888, 0.0246223
6: -0.0028723, 0.0008947, -0.0029542, 0.0006373, -0.0035096, 0.0038489
7: -0.0105691, -0.0008227, -0.0107810, -0.0014887, -0.0090804, 0.0099583
8: -0.0051223, 0.0003644, -0.0052338, 0.0014370, -0.0065593, 0.0055982
9: -0.0018676, 0.0040757, -0.0014614, 0.0042050, -0.0060725, 0.0055372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164862
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164864
time: 1.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9718309, 0.9952995, 0.9601806, 0.9946814, -0.0228505, 0.0351189
1: -0.0046619, -0.0024352, -0.0048313, -0.0025892, -0.0020727, 0.0023961
2: 0.0028513, 0.0146517, 0.0036674, 0.0155492, -0.0126980, 0.0109843
3: -0.0081828, -0.0025709, -0.0087732, -0.0029424, -0.0052404, 0.0062023
4: 0.0010797, 0.0044056, 0.0012377, 0.0053658, -0.0042861, 0.0031679
5: 0.0025456, 0.0281821, 0.0035721, 0.0374601, -0.0349146, 0.0246100
6: -0.0028723, 0.0008947, -0.0031588, 0.0006342, -0.0035065, 0.0040535
7: -0.0105691, -0.0008227, -0.0113104, -0.0014968, -0.0090723, 0.0104877
8: -0.0051223, 0.0003644, -0.0055122, 0.0041166, -0.0092389, 0.0058766
9: -0.0018676, 0.0040757, -0.0014565, 0.0045278, -0.0063954, 0.0055322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164862
time: 1.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164864
time: 1.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9748513, 0.9953412, 0.9638573, 0.9948661, -0.0200148, 0.0314839
1: -0.0046180, -0.0024248, -0.0047778, -0.0025432, -0.0020748, 0.0023530
2: 0.0027961, 0.0144190, 0.0034235, 0.0152660, -0.0124699, 0.0109954
3: -0.0080297, -0.0025458, -0.0085869, -0.0028314, -0.0051984, 0.0060411
4: 0.0010691, 0.0041566, 0.0011905, 0.0050628, -0.0039937, 0.0029661
5: 0.0024762, 0.0257767, 0.0032654, 0.0345321, -0.0320559, 0.0225114
6: -0.0027980, 0.0009123, -0.0030684, 0.0007121, -0.0035100, 0.0039807
7: -0.0103769, -0.0007771, -0.0110765, -0.0012953, -0.0090815, 0.0102994
8: -0.0050212, 0.0000272, -0.0053891, 0.0029325, -0.0079537, 0.0054163
9: -0.0018954, 0.0039585, -0.0015794, 0.0043851, -0.0062805, 0.0055379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0163985
time: 1.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164661
time: 2.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9748513, 0.9953412, 0.9553528, 0.9948589, -0.0200076, 0.0399885
1: -0.0046180, -0.0024248, -0.0049015, -0.0025450, -0.0020730, 0.0024767
2: 0.0027961, 0.0144190, 0.0034332, 0.0159212, -0.0131251, 0.0109858
3: -0.0080297, -0.0025458, -0.0090178, -0.0028357, -0.0051940, 0.0064721
4: 0.0010691, 0.0041566, 0.0011924, 0.0057637, -0.0046947, 0.0029642
5: 0.0024762, 0.0257767, 0.0032774, 0.0413048, -0.0388286, 0.0224993
6: -0.0027980, 0.0009123, -0.0032775, 0.0007090, -0.0035070, 0.0041899
7: -0.0103769, -0.0007771, -0.0116176, -0.0013033, -0.0090736, 0.0108405
8: -0.0050212, 0.0000272, -0.0056737, 0.0056714, -0.0106926, 0.0057009
9: -0.0018954, 0.0039585, -0.0015745, 0.0047151, -0.0066105, 0.0055330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0163985
time: 2.48 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0163835, upper bound: 0.0164661
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9718214, 0.9952992, 0.9639812, 0.9948114, -0.0229899, 0.0313180
1: -0.0046620, -0.0024353, -0.0047760, -0.0025569, -0.0021052, 0.0023407
2: 0.0028517, 0.0146524, 0.0034959, 0.0152564, -0.0124047, 0.0111565
3: -0.0081833, -0.0025711, -0.0085806, -0.0028643, -0.0053190, 0.0060095
4: 0.0010798, 0.0044063, 0.0012045, 0.0050526, -0.0039727, 0.0032018
5: 0.0025462, 0.0281896, 0.0033564, 0.0344335, -0.0318873, 0.0248332
6: -0.0028725, 0.0008946, -0.0030653, 0.0006889, -0.0035614, 0.0039599
7: -0.0105697, -0.0008230, -0.0110686, -0.0013551, -0.0092145, 0.0102455
8: -0.0051226, 0.0003675, -0.0053850, 0.0028926, -0.0080152, 0.0057525
9: -0.0018673, 0.0040761, -0.0015429, 0.0043803, -0.0062477, 0.0056190

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160508, upper bound: 0.0162029
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162273, upper bound: 0.0162018
time: 1.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9718214, 0.9952992, 0.9554510, 0.9948141, -0.0229927, 0.0398482
1: -0.0046620, -0.0024353, -0.0049000, -0.0025561, -0.0021059, 0.0024648
2: 0.0028517, 0.0146524, 0.0034922, 0.0159136, -0.0130619, 0.0111602
3: -0.0081833, -0.0025711, -0.0090129, -0.0028626, -0.0053207, 0.0064418
4: 0.0010798, 0.0044063, 0.0012038, 0.0057556, -0.0046758, 0.0032026
5: 0.0025462, 0.0281896, 0.0033517, 0.0412266, -0.0386805, 0.0248380
6: -0.0028725, 0.0008946, -0.0032751, 0.0006901, -0.0035626, 0.0041697
7: -0.0105697, -0.0008230, -0.0116114, -0.0013520, -0.0092176, 0.0107883
8: -0.0051226, 0.0003675, -0.0056705, 0.0056398, -0.0107624, 0.0060379
9: -0.0018673, 0.0040761, -0.0015448, 0.0047113, -0.0065787, 0.0056209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160508, upper bound: 0.0162029
time: 1.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162273, upper bound: 0.0162018
time: 2.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9670730, 0.9951481, 0.9686901, 0.9947357, -0.0276627, 0.0264580
1: -0.0047311, -0.0024729, -0.0047076, -0.0025757, -0.0021554, 0.0022347
2: 0.0030511, 0.0150182, 0.0035957, 0.0148936, -0.0118425, 0.0114225
3: -0.0084239, -0.0026619, -0.0083420, -0.0029097, -0.0055142, 0.0056801
4: 0.0011184, 0.0047977, 0.0012238, 0.0046644, -0.0035460, 0.0035739
5: 0.0027969, 0.0319712, 0.0034819, 0.0306833, -0.0278863, 0.0284892
6: -0.0029893, 0.0008309, -0.0029495, 0.0006571, -0.0036464, 0.0037805
7: -0.0108718, -0.0009877, -0.0107689, -0.0014376, -0.0094343, 0.0097812
8: -0.0052815, 0.0018968, -0.0052274, 0.0013759, -0.0066575, 0.0071242
9: -0.0017669, 0.0042603, -0.0014926, 0.0041976, -0.0059645, 0.0057530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160633, upper bound: 0.0163197
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162444, upper bound: 0.0163197
time: 1.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9671664, 0.9950976, 0.9649394, 0.9947543, -0.0275879, 0.0301582
1: -0.0047297, -0.0024855, -0.0047621, -0.0025711, -0.0021587, 0.0022766
2: 0.0031180, 0.0150110, 0.0035712, 0.0151826, -0.0120646, 0.0114398
3: -0.0084192, -0.0026923, -0.0085320, -0.0028986, -0.0055206, 0.0058397
4: 0.0011314, 0.0047900, 0.0012191, 0.0049736, -0.0038422, 0.0035709
5: 0.0028811, 0.0318968, 0.0034511, 0.0336702, -0.0307892, 0.0284457
6: -0.0029870, 0.0008096, -0.0030418, 0.0006649, -0.0036519, 0.0038513
7: -0.0108659, -0.0010430, -0.0110076, -0.0014173, -0.0094486, 0.0099646
8: -0.0052784, 0.0018667, -0.0053529, 0.0025839, -0.0078623, 0.0072196
9: -0.0017332, 0.0042567, -0.0015050, 0.0043431, -0.0060764, 0.0057617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160679, upper bound: 0.0163197
time: 1.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0162481, upper bound: 0.0163197
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9624758, 0.9952683, 0.9686901, 0.9947357, -0.0322599, 0.0265782
1: -0.0047979, -0.0024430, -0.0047076, -0.0025757, -0.0022222, 0.0022646
2: 0.0028924, 0.0153724, 0.0035957, 0.0148936, -0.0120012, 0.0117767
3: -0.0086569, -0.0025896, -0.0083420, -0.0029097, -0.0057471, 0.0057523
4: 0.0010877, 0.0051766, 0.0012238, 0.0046644, -0.0035767, 0.0039528
5: 0.0025973, 0.0356322, 0.0034819, 0.0306833, -0.0280860, 0.0321502
6: -0.0031024, 0.0008816, -0.0029495, 0.0006571, -0.0037594, 0.0038311
7: -0.0111644, -0.0008567, -0.0107689, -0.0014376, -0.0097268, 0.0099123
8: -0.0054354, 0.0033773, -0.0052274, 0.0013759, -0.0068113, 0.0086047
9: -0.0018469, 0.0044387, -0.0014926, 0.0041976, -0.0060445, 0.0059314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159047, upper bound: 0.0162788
time: 1.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159047, upper bound: 0.0162786
time: 2.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9625733, 0.9952161, 0.9649394, 0.9947543, -0.0321810, 0.0302767
1: -0.0047965, -0.0024560, -0.0047621, -0.0025711, -0.0022254, 0.0023061
2: 0.0029613, 0.0153649, 0.0035712, 0.0151826, -0.0122213, 0.0117937
3: -0.0086519, -0.0026210, -0.0085320, -0.0028986, -0.0057534, 0.0059110
4: 0.0011010, 0.0051686, 0.0012191, 0.0049736, -0.0038725, 0.0039495
5: 0.0026840, 0.0355546, 0.0034511, 0.0336702, -0.0309862, 0.0321035
6: -0.0031000, 0.0008596, -0.0030418, 0.0006649, -0.0037649, 0.0039014
7: -0.0111582, -0.0009136, -0.0110076, -0.0014173, -0.0097408, 0.0100940
8: -0.0054321, 0.0033460, -0.0053529, 0.0025839, -0.0080160, 0.0086989
9: -0.0018122, 0.0044349, -0.0015050, 0.0043431, -0.0061553, 0.0059399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159104, upper bound: 0.0162788
time: 1.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0160470, upper bound: 0.0162786
time: 2.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9671127, 0.9952742, 0.9638573, 0.9948661, -0.0277534, 0.0314169
1: -0.0047305, -0.0024415, -0.0047778, -0.0025432, -0.0021873, 0.0023363
2: 0.0028846, 0.0150152, 0.0034235, 0.0152660, -0.0123813, 0.0115916
3: -0.0084219, -0.0025861, -0.0085869, -0.0028314, -0.0055905, 0.0060008
4: 0.0010862, 0.0047944, 0.0011905, 0.0050628, -0.0039766, 0.0036039
5: 0.0025876, 0.0319395, 0.0032654, 0.0345321, -0.0319446, 0.0286741
6: -0.0029883, 0.0008841, -0.0030684, 0.0007121, -0.0037004, 0.0039525
7: -0.0108693, -0.0008502, -0.0110765, -0.0012953, -0.0095740, 0.0102262
8: -0.0052802, 0.0018840, -0.0053891, 0.0029325, -0.0082127, 0.0072731
9: -0.0018508, 0.0042588, -0.0015794, 0.0043851, -0.0062359, 0.0058382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 231
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0159600, upper bound: 0.0162232
time: 2.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0161148, upper bound: 0.0162200
time: 1.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9671127, 0.9952742, 0.9553528, 0.9948589, -0.0277461, 0.0399214
1: -0.0047305, -0.0024415, -0.0049015, -0.0025450, -0.0021855, 0.0024600
2: 0.0028846, 0.0150152, 0.0034332, 0.0159212, -0.0130365, 0.0115820
3: -0.0084219, -0.0025861, -0.0090178, -0.0028357, -0.0055862, 0.0064317
4: 0.0010862, 0.0047944, 0.0011924, 0.0057637, -0.0046775, 0.0036021
5: 0.0025876, 0.0319395, 0.0032774, 0.0413048, -0.0387172, 0.0286621
6: -0.0029883, 0.0008841, -0.0032775, 0.0007090, -0.0036973, 0.0041616
7: -0.0108693, -0.0008502, -0.0116176, -0.0013033, -0.0095660, 0.0107674
8: -0.0052802, 0.0018840, -0.0056737, 0.0056714, -0.0109516, 0.0075577
9: -0.0018508, 0.0042588, -0.0015745, 0.0047151, -0.0065659, 0.0058333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.59 + 597.68 = 601.27 seconds
