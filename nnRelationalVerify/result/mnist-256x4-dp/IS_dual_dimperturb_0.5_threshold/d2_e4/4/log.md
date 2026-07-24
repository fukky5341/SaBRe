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
Threshold: 0.0148662


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0007405, 0.0331512, 0.0007405, 0.0331512, -0.0324107, 0.0324107)
1: (-0.0070848, 0.0073948, -0.0070848, 0.0073948, -0.0144796, 0.0144796)
2: (-0.0090910, 0.0142565, -0.0090910, 0.0142565, -0.0233475, 0.0233475)
3: (-0.0046189, 0.0075591, -0.0046189, 0.0075591, -0.0121780, 0.0121780)
4: (-0.0061589, -0.0001321, -0.0061589, -0.0001321, -0.0060269, 0.0060269)
5: (-0.0058564, 0.0078750, -0.0058564, 0.0078750, -0.0137314, 0.0137314)
6: (-0.0149815, 0.0081529, -0.0149815, 0.0081529, -0.0231344, 0.0231344)
7: (-0.0223283, 0.0094589, -0.0223283, 0.0094589, -0.0317872, 0.0317872)
8: (0.9751642, 1.0054737, 0.9751642, 1.0054737, -0.0303096, 0.0303096)
9: (-0.0218918, 0.0071018, -0.0218918, 0.0071018, -0.0289935, 0.0289935)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 2.80 = 4.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0252723, upper bound: 0.0252723

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0246898, upper bound: 0.0246768
time: 1.21 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0246326, upper bound: 0.0246326
time: 1.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.73
Output dim: 8, lower bound: -0.0246898, upper bound: 0.0246768
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.73
Output dim: 8, lower bound: -0.0246326, upper bound: 0.0246326

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0007731, 0.0299660, 0.0007441, 0.0327817, -0.0320086, 0.0292219
1: -0.0068850, 0.0060243, -0.0070609, 0.0072347, -0.0141198, 0.0130851
2: -0.0067486, 0.0139324, -0.0088212, 0.0142036, -0.0209522, 0.0227537
3: -0.0044611, 0.0073536, -0.0046005, 0.0075341, -0.0119953, 0.0119541
4: -0.0057376, -0.0003255, -0.0061102, -0.0001625, -0.0055751, 0.0057847
5: -0.0047425, 0.0076561, -0.0057281, 0.0078397, -0.0125822, 0.0133843
6: -0.0134490, 0.0080770, -0.0147943, 0.0081445, -0.0215935, 0.0228713
7: -0.0220335, 0.0073861, -0.0222934, 0.0092130, -0.0312465, 0.0296795
8: 0.9762204, 1.0014770, 0.9753261, 1.0050001, -0.0287797, 0.0261509
9: -0.0172937, 0.0068355, -0.0213527, 0.0070670, -0.0243607, 0.0281882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0244399, upper bound: 0.0243901
time: 1.22 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0244336, upper bound: 0.0244079
time: 1.53 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0006865, 0.0253751, 0.0007709, 0.0291021, -0.0284156, 0.0246042
1: -0.0066295, 0.0042257, -0.0068089, 0.0057107, -0.0123402, 0.0110346
2: -0.0033427, 0.0139803, -0.0061141, 0.0139337, -0.0172764, 0.0200944
3: -0.0042769, 0.0071299, -0.0044243, 0.0072836, -0.0115604, 0.0115542
4: -0.0051977, -0.0003527, -0.0056357, -0.0003289, -0.0048688, 0.0052830
5: -0.0031327, 0.0077069, -0.0044299, 0.0076575, -0.0107902, 0.0121367
6: -0.0115140, 0.0082783, -0.0130713, 0.0080823, -0.0195963, 0.0213495
7: -0.0220045, 0.0051524, -0.0219568, 0.0069457, -0.0289501, 0.0271092
8: 0.9763674, 0.9964545, 0.9763159, 1.0005637, -0.0241964, 0.0201386
9: -0.0115717, 0.0068088, -0.0162763, 0.0067800, -0.0183517, 0.0230851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=29, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0238830, upper bound: 0.0241098
time: 1.56 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0241398, upper bound: 0.0241398
time: 1.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.43
Output dim: 8, lower bound: -0.0244399, upper bound: 0.0243901
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.43
Output dim: 8, lower bound: -0.0244336, upper bound: 0.0244079
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.43
Output dim: 8, lower bound: -0.0238830, upper bound: 0.0241098
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.43
Output dim: 8, lower bound: -0.0241398, upper bound: 0.0241398

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0008702, 0.0293274, 0.0009757, 0.0312171, -0.0303469, 0.0283517
1: -0.0068555, 0.0057420, -0.0069630, 0.0065292, -0.0133847, 0.0127049
2: -0.0062772, 0.0138788, -0.0076798, 0.0138204, -0.0200976, 0.0215585
3: -0.0044164, 0.0073230, -0.0044905, 0.0074268, -0.0118433, 0.0118134
4: -0.0056192, -0.0003313, -0.0058275, -0.0003117, -0.0053075, 0.0054962
5: -0.0045145, 0.0075993, -0.0051803, 0.0075375, -0.0120520, 0.0127795
6: -0.0131279, 0.0078514, -0.0139350, 0.0076063, -0.0207342, 0.0217863
7: -0.0217965, 0.0069655, -0.0216913, 0.0081468, -0.0299434, 0.0286568
8: 0.9764253, 1.0006740, 0.9764702, 1.0029898, -0.0265644, 0.0242038
9: -0.0163918, 0.0066712, -0.0190923, 0.0065938, -0.0229857, 0.0257634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237002, upper bound: 0.0238625
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0239647, upper bound: 0.0238812
time: 1.24 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0008827, 0.0294697, 0.0009454, 0.0402968, -0.0394141, 0.0285243
1: -0.0068488, 0.0058129, -0.0077202, 0.0101397, -0.0169885, 0.0135331
2: -0.0063839, 0.0138719, -0.0142930, 0.0146345, -0.0210183, 0.0281649
3: -0.0044187, 0.0073169, -0.0050323, 0.0081218, -0.0125405, 0.0123491
4: -0.0056295, -0.0003321, -0.0069155, 0.0002317, -0.0058612, 0.0065834
5: -0.0045615, 0.0075920, -0.0084410, 0.0080755, -0.0126370, 0.0160330
6: -0.0131875, 0.0078224, -0.0189811, 0.0076766, -0.0208641, 0.0268035
7: -0.0217427, 0.0070672, -0.0225629, 0.0146217, -0.0363644, 0.0296301
8: 0.9764594, 1.0008855, 0.9735090, 1.0143464, -0.0378870, 0.0273765
9: -0.0166320, 0.0066360, -0.0316737, 0.0073788, -0.0240108, 0.0383097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0244336, upper bound: 0.0244079
time: 2.00 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0244336, upper bound: 0.0244079
time: 1.42 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: 0.0011271, 0.0202182, 0.0009549, 0.0253937, -0.0242666, 0.0192633
1: -0.0055451, 0.0029655, -0.0063702, 0.0043382, -0.0098833, 0.0093357
2: 0.0005557, 0.0137367, -0.0033403, 0.0138319, -0.0132762, 0.0170770
3: -0.0039772, 0.0058598, -0.0042522, 0.0067679, -0.0107451, 0.0101119
4: -0.0043280, -0.0003698, -0.0050413, -0.0003357, -0.0039923, 0.0046715
5: -0.0012555, 0.0074488, -0.0031268, 0.0075497, -0.0088052, 0.0105756
6: -0.0093956, 0.0072543, -0.0115528, 0.0076545, -0.0170501, 0.0188071
7: -0.0192969, 0.0036961, -0.0208184, 0.0053512, -0.0246481, 0.0245145
8: 0.9785156, 0.9930018, 0.9772064, 0.9968712, -0.0183556, 0.0157954
9: -0.0084597, 0.0051074, -0.0120799, 0.0060664, -0.0145262, 0.0171874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0236602, upper bound: 0.0237765
time: 1.36 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0236602, upper bound: 0.0238942
time: 1.60 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: 0.0008710, 0.0206175, 0.0008773, 0.0250103, -0.0241393, 0.0197402
1: -0.0058094, 0.0031716, -0.0064014, 0.0041826, -0.0099920, 0.0095730
2: 0.0002625, 0.0138783, -0.0030650, 0.0138749, -0.0136124, 0.0169433
3: -0.0043585, 0.0062625, -0.0042487, 0.0068528, -0.0112113, 0.0105112
4: -0.0045364, -0.0000749, -0.0050343, -0.0003355, -0.0042009, 0.0049594
5: -0.0014342, 0.0075988, -0.0030006, 0.0075951, -0.0090294, 0.0105994
6: -0.0101005, 0.0078495, -0.0113992, 0.0078350, -0.0179355, 0.0192488
7: -0.0205227, 0.0052041, -0.0211502, 0.0051841, -0.0257068, 0.0263543
8: 0.9777050, 0.9939639, 0.9770191, 0.9964216, -0.0187166, 0.0169448
9: -0.0094240, 0.0058427, -0.0115901, 0.0062586, -0.0156826, 0.0174329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0239327, upper bound: 0.0238176
time: 1.35 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0239327, upper bound: 0.0239327
time: 1.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.17 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 8, lower bound: -0.0237002, upper bound: 0.0238625
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 8, lower bound: -0.0239647, upper bound: 0.0238812
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 8, lower bound: -0.0244336, upper bound: 0.0244079
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 8, lower bound: -0.0244336, upper bound: 0.0244079
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 8, lower bound: -0.0236602, upper bound: 0.0237765
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 8, lower bound: -0.0236602, upper bound: 0.0238942
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 8, lower bound: -0.0239327, upper bound: 0.0238176
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.17
Output dim: 8, lower bound: -0.0239327, upper bound: 0.0239327

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0013076, 0.0207071, 0.0011536, 0.0274819, -0.0261743, 0.0195536
1: -0.0057562, 0.0029934, -0.0065212, 0.0051126, -0.0108688, 0.0095146
2: 0.0001915, 0.0136369, -0.0048915, 0.0137221, -0.0135306, 0.0185284
3: -0.0040149, 0.0060405, -0.0043221, 0.0069089, -0.0109238, 0.0103626
4: -0.0042613, -0.0003488, -0.0052197, -0.0003186, -0.0039427, 0.0048709
5: -0.0014635, 0.0073431, -0.0038598, 0.0074333, -0.0088968, 0.0112029
6: -0.0095865, 0.0068348, -0.0123981, 0.0071928, -0.0167793, 0.0192329
7: -0.0190403, 0.0038037, -0.0205493, 0.0062502, -0.0252906, 0.0243531
8: 0.9786041, 0.9930847, 0.9773751, 0.9990222, -0.0204182, 0.0157096
9: -0.0085285, 0.0049356, -0.0145702, 0.0058745, -0.0144030, 0.0195058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234836, upper bound: 0.0235667
time: 2.18 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234836, upper bound: 0.0236541
time: 2.43 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010664, 0.0211786, 0.0010782, 0.0270414, -0.0259750, 0.0201004
1: -0.0060567, 0.0032038, -0.0065599, 0.0049066, -0.0109633, 0.0097637
2: -0.0001530, 0.0137703, -0.0045726, 0.0137637, -0.0139168, 0.0183429
3: -0.0043952, 0.0064750, -0.0043172, 0.0070003, -0.0113955, 0.0107923
4: -0.0044727, -0.0000598, -0.0052016, -0.0003182, -0.0041545, 0.0051418
5: -0.0016707, 0.0074844, -0.0037086, 0.0074774, -0.0091481, 0.0111929
6: -0.0102562, 0.0073955, -0.0122168, 0.0073679, -0.0176240, 0.0196122
7: -0.0202936, 0.0052816, -0.0208904, 0.0059927, -0.0262863, 0.0261720
8: 0.9777656, 0.9940039, 0.9771718, 0.9984286, -0.0206631, 0.0168321
9: -0.0094735, 0.0056947, -0.0138597, 0.0060748, -0.0155483, 0.0195544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237872, upper bound: 0.0235871
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234836, upper bound: 0.0236822
time: 1.68 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0008827, 0.0294697, 0.0009648, 0.0373436, -0.0364609, 0.0285049
1: -0.0068488, 0.0058129, -0.0075356, 0.0089013, -0.0157501, 0.0133485
2: -0.0063839, 0.0138719, -0.0121532, 0.0142199, -0.0206038, 0.0260251
3: -0.0044187, 0.0073169, -0.0048676, 0.0079307, -0.0123494, 0.0121844
4: -0.0056295, -0.0003321, -0.0065345, -0.0000242, -0.0056053, 0.0062024
5: -0.0045615, 0.0075920, -0.0074329, 0.0078006, -0.0123621, 0.0150248
6: -0.0131875, 0.0078224, -0.0174246, 0.0076315, -0.0208189, 0.0252470
7: -0.0217427, 0.0070672, -0.0223118, 0.0123757, -0.0341184, 0.0293790
8: 0.9764594, 1.0008855, 0.9747532, 1.0104477, -0.0339884, 0.0261323
9: -0.0166320, 0.0066360, -0.0272792, 0.0071244, -0.0237564, 0.0339153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237000, upper bound: 0.0238666
time: 4.05 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0239503, upper bound: 0.0238810
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0008827, 0.0294697, 0.0008779, 0.0320900, -0.0312073, 0.0285918
1: -0.0068488, 0.0058129, -0.0071962, 0.0066981, -0.0135469, 0.0130091
2: -0.0063839, 0.0138719, -0.0082595, 0.0138745, -0.0202584, 0.0221314
3: -0.0044187, 0.0073169, -0.0046193, 0.0076163, -0.0120350, 0.0119361
4: -0.0056295, -0.0003321, -0.0058980, -0.0002896, -0.0053399, 0.0055659
5: -0.0045615, 0.0075920, -0.0055876, 0.0075948, -0.0121562, 0.0131796
6: -0.0131875, 0.0078224, -0.0148653, 0.0078335, -0.0210209, 0.0226877
7: -0.0217427, 0.0070672, -0.0221747, 0.0088679, -0.0306107, 0.0292419
8: 0.9764594, 1.0008855, 0.9759435, 1.0037756, -0.0273162, 0.0249420
9: -0.0166320, 0.0066360, -0.0197630, 0.0069419, -0.0235739, 0.0263991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237000, upper bound: 0.0238666
time: 1.41 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0239503, upper bound: 0.0238810
time: 1.35 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0012786, 0.0197903, 0.0013199, 0.0221560, -0.0208774, 0.0184703
1: -0.0052918, 0.0029067, -0.0057931, 0.0033025, -0.0085943, 0.0086998
2: 0.0008744, 0.0136529, -0.0009011, 0.0136301, -0.0127557, 0.0145540
3: -0.0039253, 0.0055489, -0.0040640, 0.0060464, -0.0099717, 0.0096129
4: -0.0041918, -0.0003770, -0.0044140, -0.0003468, -0.0038450, 0.0040370
5: -0.0010721, 0.0073600, -0.0019470, 0.0073358, -0.0084080, 0.0093070
6: -0.0092071, 0.0069021, -0.0101561, 0.0068061, -0.0160132, 0.0170582
7: -0.0185277, 0.0036593, -0.0189750, 0.0043075, -0.0228352, 0.0226343
8: 0.9790971, 0.9929398, 0.9785914, 0.9941738, -0.0150766, 0.0143484
9: -0.0084362, 0.0046225, -0.0094404, 0.0049105, -0.0133467, 0.0140628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0231884
time: 1.39 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0236889
time: 1.54 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0012267, 0.0198289, 0.0011339, 0.0245201, -0.0232934, 0.0186951
1: -0.0053141, 0.0029160, -0.0059969, 0.0042071, -0.0095212, 0.0089129
2: 0.0008488, 0.0136817, -0.0026572, 0.0137330, -0.0128842, 0.0163389
3: -0.0039334, 0.0056018, -0.0043816, 0.0063316, -0.0102650, 0.0099834
4: -0.0042301, -0.0003761, -0.0048334, -0.0001632, -0.0040670, 0.0044573
5: -0.0010887, 0.0073905, -0.0027994, 0.0074448, -0.0085335, 0.0101899
6: -0.0092270, 0.0070228, -0.0116258, 0.0072386, -0.0164656, 0.0186486
7: -0.0187558, 0.0036641, -0.0199479, 0.0061249, -0.0248807, 0.0236120
8: 0.9789642, 0.9929478, 0.9779144, 0.9969447, -0.0179805, 0.0150334
9: -0.0084393, 0.0047553, -0.0122048, 0.0055118, -0.0139510, 0.0169601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0232645
time: 1.57 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0238088
time: 2.07 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0010364, 0.0201594, 0.0012624, 0.0217674, -0.0207310, 0.0188970
1: -0.0055362, 0.0031047, -0.0058090, 0.0031957, -0.0087318, 0.0089137
2: 0.0006060, 0.0137869, -0.0006087, 0.0136619, -0.0130559, 0.0143956
3: -0.0043003, 0.0059295, -0.0040578, 0.0061092, -0.0104095, 0.0099873
4: -0.0043883, -0.0000819, -0.0044064, -0.0003467, -0.0040415, 0.0043245
5: -0.0012392, 0.0075019, -0.0018175, 0.0073695, -0.0086088, 0.0093195
6: -0.0099142, 0.0074652, -0.0100021, 0.0069398, -0.0168540, 0.0174672
7: -0.0196765, 0.0051682, -0.0192271, 0.0041563, -0.0238328, 0.0243953
8: 0.9783393, 0.9939021, 0.9784581, 0.9938465, -0.0155072, 0.0154440
9: -0.0094010, 0.0053137, -0.0091620, 0.0050483, -0.0144493, 0.0144756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235289, upper bound: 0.0231899
time: 1.42 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0238467, upper bound: 0.0237288
time: 1.41 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0009598, 0.0202665, 0.0010461, 0.0241713, -0.0232115, 0.0192204
1: -0.0056001, 0.0031237, -0.0060481, 0.0040583, -0.0096584, 0.0091718
2: 0.0005284, 0.0138292, -0.0024022, 0.0137815, -0.0132532, 0.0162314
3: -0.0043157, 0.0060249, -0.0043807, 0.0064341, -0.0107498, 0.0104056
4: -0.0044517, -0.0000813, -0.0048440, -0.0001632, -0.0042885, 0.0047626
5: -0.0012847, 0.0075468, -0.0026875, 0.0074963, -0.0087809, 0.0102343
6: -0.0099561, 0.0076432, -0.0114878, 0.0074427, -0.0173987, 0.0191310
7: -0.0200404, 0.0051713, -0.0203063, 0.0059539, -0.0259943, 0.0254776
8: 0.9781069, 0.9939141, 0.9776858, 0.9965228, -0.0184159, 0.0162283
9: -0.0094030, 0.0055377, -0.0117378, 0.0057301, -0.0151331, 0.0172755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235289, upper bound: 0.0232669
time: 1.39 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0238467, upper bound: 0.0238467
time: 1.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.48 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0234836, upper bound: 0.0235667
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0234836, upper bound: 0.0236541
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0237872, upper bound: 0.0235871
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0234836, upper bound: 0.0236822
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0237000, upper bound: 0.0238666
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0239503, upper bound: 0.0238810
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0237000, upper bound: 0.0238666
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0239503, upper bound: 0.0238810
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0231884
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0236889
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0232645
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0238088
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0235289, upper bound: 0.0231899
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0238467, upper bound: 0.0237288
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0235289, upper bound: 0.0232669
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.48
Output dim: 8, lower bound: -0.0238467, upper bound: 0.0238467

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0014588, 0.0203344, 0.0015141, 0.0242707, -0.0228119, 0.0188203
1: -0.0055238, 0.0029416, -0.0059511, 0.0039787, -0.0095025, 0.0088927
2: 0.0004737, 0.0135533, -0.0024780, 0.0135228, -0.0130490, 0.0160313
3: -0.0039707, 0.0057482, -0.0041379, 0.0061933, -0.0101640, 0.0098862
4: -0.0041292, -0.0003529, -0.0045712, -0.0003288, -0.0038004, 0.0042182
5: -0.0013039, 0.0072545, -0.0026996, 0.0072221, -0.0085260, 0.0099541
6: -0.0094286, 0.0064833, -0.0110258, 0.0063549, -0.0157834, 0.0175091
7: -0.0182933, 0.0037824, -0.0186951, 0.0050864, -0.0233797, 0.0224776
8: 0.9791623, 0.9930382, 0.9787609, 0.9961005, -0.0169381, 0.0142773
9: -0.0085149, 0.0044677, -0.0113449, 0.0047111, -0.0132260, 0.0158125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229778
time: 3.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233921, upper bound: 0.0234817
time: 4.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0014088, 0.0203601, 0.0013425, 0.0266409, -0.0252321, 0.0190176
1: -0.0055422, 0.0029494, -0.0061587, 0.0049792, -0.0105214, 0.0091081
2: 0.0004577, 0.0135810, -0.0042363, 0.0136176, -0.0131600, 0.0178173
3: -0.0039767, 0.0057963, -0.0044560, 0.0064841, -0.0104608, 0.0102523
4: -0.0041661, -0.0003533, -0.0050128, -0.0001464, -0.0040197, 0.0046595
5: -0.0013149, 0.0072838, -0.0035534, 0.0073226, -0.0086376, 0.0108372
6: -0.0094390, 0.0065996, -0.0124545, 0.0067537, -0.0161927, 0.0190541
7: -0.0185115, 0.0037808, -0.0196524, 0.0069991, -0.0255106, 0.0234332
8: 0.9790369, 0.9930407, 0.9780797, 0.9990351, -0.0199982, 0.0149611
9: -0.0085139, 0.0045964, -0.0146141, 0.0053133, -0.0138272, 0.0192105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230439
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233921, upper bound: 0.0235710
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0012314, 0.0207716, 0.0014575, 0.0237614, -0.0225300, 0.0193141
1: -0.0058025, 0.0031444, -0.0059726, 0.0037684, -0.0095709, 0.0091170
2: 0.0001563, 0.0136791, -0.0021011, 0.0135541, -0.0133977, 0.0157802
3: -0.0043451, 0.0061582, -0.0041276, 0.0062618, -0.0106069, 0.0102859
4: -0.0043273, -0.0000638, -0.0045402, -0.0003286, -0.0039988, 0.0044764
5: -0.0014966, 0.0073877, -0.0025276, 0.0072553, -0.0087518, 0.0099152
6: -0.0100930, 0.0070119, -0.0108145, 0.0064864, -0.0165794, 0.0178263
7: -0.0194679, 0.0052610, -0.0189528, 0.0048797, -0.0243475, 0.0242138
8: 0.9783785, 0.9939579, 0.9786218, 0.9955208, -0.0171424, 0.0153361
9: -0.0094604, 0.0051777, -0.0107210, 0.0048569, -0.0143173, 0.0158987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233733, upper bound: 0.0229787
time: 1.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237084, upper bound: 0.0235019
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0011557, 0.0208682, 0.0012521, 0.0262140, -0.0250584, 0.0196161
1: -0.0058627, 0.0031619, -0.0062165, 0.0047732, -0.0106359, 0.0093784
2: 0.0000860, 0.0137209, -0.0039279, 0.0136676, -0.0135816, 0.0176488
3: -0.0043586, 0.0062501, -0.0044529, 0.0065914, -0.0109500, 0.0107030
4: -0.0043914, -0.0000643, -0.0050043, -0.0001463, -0.0042451, 0.0049400
5: -0.0015378, 0.0074321, -0.0034085, 0.0073756, -0.0089134, 0.0108406
6: -0.0101285, 0.0071879, -0.0122806, 0.0069638, -0.0170923, 0.0194685
7: -0.0198209, 0.0052584, -0.0200192, 0.0067475, -0.0265684, 0.0252776
8: 0.9781435, 0.9939649, 0.9778378, 0.9984459, -0.0203024, 0.0161271
9: -0.0094587, 0.0053987, -0.0139068, 0.0055382, -0.0149969, 0.0193055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233733, upper bound: 0.0230446
time: 2.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237084, upper bound: 0.0235976
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0013161, 0.0206908, 0.0011432, 0.0333899, -0.0320738, 0.0195475
1: -0.0057493, 0.0029908, -0.0070591, 0.0073671, -0.0131164, 0.0100498
2: 0.0002033, 0.0136322, -0.0092055, 0.0137278, -0.0135245, 0.0228377
3: -0.0040123, 0.0060341, -0.0046699, 0.0073644, -0.0113767, 0.0107040
4: -0.0042524, -0.0003494, -0.0059016, -0.0002550, -0.0039974, 0.0055521
5: -0.0014564, 0.0073381, -0.0060470, 0.0074393, -0.0088958, 0.0133851
6: -0.0095792, 0.0068150, -0.0155823, 0.0072168, -0.0167959, 0.0223973
7: -0.0189785, 0.0038003, -0.0211246, 0.0100023, -0.0289809, 0.0249249
8: 0.9786394, 0.9930819, 0.9766309, 1.0058768, -0.0272374, 0.0164511
9: -0.0085264, 0.0049022, -0.0222128, 0.0062896, -0.0148160, 0.0271150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237922, upper bound: 0.0239239
time: 1.60 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237922, upper bound: 0.0240702
time: 2.34 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010810, 0.0212252, 0.0010702, 0.0331055, -0.0320244, 0.0201550
1: -0.0060507, 0.0032100, -0.0070645, 0.0072539, -0.0133046, 0.0102745
2: -0.0001891, 0.0137622, -0.0090102, 0.0137682, -0.0139572, 0.0227724
3: -0.0043937, 0.0064694, -0.0046679, 0.0074260, -0.0118197, 0.0111372
4: -0.0044708, -0.0000605, -0.0059066, -0.0002548, -0.0042159, 0.0058461
5: -0.0016853, 0.0074758, -0.0059442, 0.0074821, -0.0091674, 0.0134200
6: -0.0102728, 0.0073614, -0.0154572, 0.0073865, -0.0176593, 0.0228186
7: -0.0202330, 0.0052988, -0.0214270, 0.0098242, -0.0300573, 0.0267258
8: 0.9777987, 0.9940378, 0.9764699, 1.0055166, -0.0277179, 0.0175679
9: -0.0095065, 0.0056540, -0.0217953, 0.0064651, -0.0159716, 0.0274493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0241051, upper bound: 0.0239425
time: 2.04 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0241051, upper bound: 0.0241051
time: 2.53 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0013161, 0.0206908, 0.0010589, 0.0283011, -0.0269850, 0.0196319
1: -0.0057493, 0.0029908, -0.0067703, 0.0052470, -0.0109963, 0.0097611
2: 0.0002033, 0.0136322, -0.0054331, 0.0137745, -0.0135711, 0.0190653
3: -0.0040123, 0.0060341, -0.0044283, 0.0071108, -0.0111231, 0.0104625
4: -0.0042524, -0.0003494, -0.0052935, -0.0002964, -0.0039561, 0.0049441
5: -0.0014564, 0.0073381, -0.0042454, 0.0074888, -0.0089452, 0.0115834
6: -0.0095792, 0.0068150, -0.0132259, 0.0074129, -0.0169921, 0.0200409
7: -0.0189785, 0.0038003, -0.0210507, 0.0067773, -0.0257558, 0.0248510
8: 0.9786394, 0.9930819, 0.9768258, 0.9996965, -0.0210571, 0.0162562
9: -0.0085264, 0.0049022, -0.0150637, 0.0062374, -0.0147638, 0.0199659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234792, upper bound: 0.0235639
time: 2.68 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234792, upper bound: 0.0236577
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010810, 0.0212252, 0.0009801, 0.0278565, -0.0267754, 0.0202450
1: -0.0060507, 0.0032100, -0.0067560, 0.0050930, -0.0111437, 0.0099660
2: -0.0001891, 0.0137622, -0.0051122, 0.0138180, -0.0140071, 0.0188744
3: -0.0043937, 0.0064694, -0.0044201, 0.0071554, -0.0115491, 0.0108894
4: -0.0044708, -0.0000605, -0.0052842, -0.0002962, -0.0041746, 0.0052237
5: -0.0016853, 0.0074758, -0.0040890, 0.0075349, -0.0092202, 0.0115648
6: -0.0102728, 0.0073614, -0.0130504, 0.0075959, -0.0178687, 0.0204117
7: -0.0202330, 0.0052988, -0.0213370, 0.0065735, -0.0268066, 0.0266358
8: 0.9777987, 0.9940378, 0.9766886, 0.9992393, -0.0214406, 0.0173492
9: -0.0095065, 0.0056540, -0.0145335, 0.0064003, -0.0159068, 0.0201875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=28, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237737, upper bound: 0.0235847
time: 1.60 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237736, upper bound: 0.0236838
time: 1.79 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0013870, 0.0176023, 0.0015159, 0.0160930, -0.0147060, 0.0160864
1: -0.0042734, 0.0026804, -0.0035927, 0.0025093, -0.0067828, 0.0062731
2: 0.0024605, 0.0135930, 0.0035240, 0.0135218, -0.0110612, 0.0100690
3: -0.0037344, 0.0045259, -0.0036224, 0.0038298, -0.0075643, 0.0081483
4: -0.0039339, -0.0003926, -0.0037157, -0.0003608, -0.0035731, 0.0033230
5: -0.0001494, 0.0072965, 0.0005005, 0.0072210, -0.0073704, 0.0067960
6: -0.0084074, 0.0066501, -0.0079427, 0.0063506, -0.0147580, 0.0145928
7: -0.0169628, 0.0035794, -0.0156689, 0.0037423, -0.0207052, 0.0192482
8: 0.9805180, 0.9927316, 0.9813204, 0.9927093, -0.0121912, 0.0114112
9: -0.0083851, 0.0036024, -0.0084893, 0.0027690, -0.0111541, 0.0120917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0230953
time: 1.63 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0231884
time: 1.42 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0012937, 0.0194993, 0.0013735, 0.0202956, -0.0190019, 0.0181258
1: -0.0051616, 0.0028793, -0.0053408, 0.0029995, -0.0081611, 0.0082201
2: 0.0010849, 0.0136446, 0.0004686, 0.0136005, -0.0125156, 0.0131760
3: -0.0039022, 0.0054193, -0.0039632, 0.0055912, -0.0094933, 0.0093825
4: -0.0041577, -0.0003790, -0.0041894, -0.0003523, -0.0038054, 0.0038105
5: -0.0009489, 0.0073512, -0.0012222, 0.0073044, -0.0082533, 0.0085734
6: -0.0090927, 0.0068670, -0.0094004, 0.0066815, -0.0157742, 0.0162674
7: -0.0183206, 0.0036493, -0.0182365, 0.0039533, -0.0222739, 0.0218858
8: 0.9792901, 0.9929135, 0.9792789, 0.9933696, -0.0140795, 0.0136347
9: -0.0084298, 0.0044880, -0.0088167, 0.0044280, -0.0128578, 0.0133047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0235187
time: 1.38 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0236889
time: 1.46 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0013376, 0.0176417, 0.0013493, 0.0177191, -0.0163814, 0.0162924
1: -0.0042966, 0.0026884, -0.0037866, 0.0029456, -0.0072422, 0.0064751
2: 0.0024324, 0.0136203, 0.0023256, 0.0136139, -0.0111815, 0.0112947
3: -0.0037415, 0.0045769, -0.0038868, 0.0040963, -0.0078378, 0.0084637
4: -0.0039688, -0.0003916, -0.0040177, -0.0001823, -0.0037865, 0.0036261
5: -0.0001662, 0.0073255, -0.0001025, 0.0073186, -0.0074848, 0.0074280
6: -0.0084263, 0.0067650, -0.0091042, 0.0067378, -0.0151641, 0.0158692
7: -0.0171601, 0.0035847, -0.0165084, 0.0051271, -0.0222872, 0.0200931
8: 0.9803938, 0.9927399, 0.9809489, 0.9943305, -0.0139367, 0.0117909
9: -0.0083885, 0.0037218, -0.0099448, 0.0032850, -0.0116735, 0.0136666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0231384
time: 2.48 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0232645
time: 1.43 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0012415, 0.0195366, 0.0011868, 0.0225834, -0.0213418, 0.0183498
1: -0.0051837, 0.0028883, -0.0055599, 0.0037372, -0.0089209, 0.0084482
2: 0.0010599, 0.0136735, -0.0012323, 0.0137037, -0.0126438, 0.0149058
3: -0.0039100, 0.0054720, -0.0042729, 0.0058880, -0.0097980, 0.0097449
4: -0.0041962, -0.0003780, -0.0045911, -0.0001687, -0.0040275, 0.0042131
5: -0.0009649, 0.0073818, -0.0020527, 0.0074138, -0.0083787, 0.0094345
6: -0.0091120, 0.0069883, -0.0108368, 0.0071156, -0.0162276, 0.0178251
7: -0.0185510, 0.0036541, -0.0192301, 0.0056747, -0.0242257, 0.0228841
8: 0.9791576, 0.9929214, 0.9785935, 0.9958082, -0.0166506, 0.0143279
9: -0.0084329, 0.0046215, -0.0110611, 0.0050381, -0.0134710, 0.0156826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0235897
time: 1.24 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0238088
time: 2.24 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0011477, 0.0179670, 0.0014601, 0.0160677, -0.0149200, 0.0165069
1: -0.0045079, 0.0028568, -0.0035957, 0.0025074, -0.0070153, 0.0064526
2: 0.0022032, 0.0137254, 0.0035472, 0.0135526, -0.0113494, 0.0101782
3: -0.0040928, 0.0048855, -0.0036212, 0.0038617, -0.0079546, 0.0085066
4: -0.0041226, -0.0000967, -0.0037508, -0.0003603, -0.0037623, 0.0036540
5: -0.0003180, 0.0074367, 0.0005059, 0.0072537, -0.0075717, 0.0069308
6: -0.0091718, 0.0072065, -0.0079401, 0.0064803, -0.0156521, 0.0151466
7: -0.0180462, 0.0050925, -0.0158510, 0.0037447, -0.0217909, 0.0209435
8: 0.9798120, 0.9937083, 0.9811959, 0.9927102, -0.0128982, 0.0125124
9: -0.0093526, 0.0042574, -0.0084908, 0.0028811, -0.0122337, 0.0127482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0230815
time: 1.51 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0231255
time: 1.98 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0010507, 0.0198884, 0.0013153, 0.0199314, -0.0188806, 0.0185731
1: -0.0054151, 0.0030768, -0.0053691, 0.0029368, -0.0083519, 0.0084459
2: 0.0008025, 0.0137790, 0.0007434, 0.0136327, -0.0128302, 0.0130356
3: -0.0042767, 0.0058074, -0.0039586, 0.0056664, -0.0099431, 0.0097661
4: -0.0043560, -0.0000839, -0.0041872, -0.0003523, -0.0040038, 0.0041033
5: -0.0011233, 0.0074935, -0.0011033, 0.0073386, -0.0084618, 0.0085969
6: -0.0098075, 0.0074318, -0.0092583, 0.0068169, -0.0166244, 0.0166901
7: -0.0194812, 0.0051581, -0.0185020, 0.0038171, -0.0232983, 0.0236601
8: 0.9785232, 0.9938783, 0.9791225, 0.9930772, -0.0145540, 0.0147558
9: -0.0093946, 0.0051849, -0.0085701, 0.0045835, -0.0139781, 0.0137550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0238088, upper bound: 0.0235046
time: 1.41 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0238088, upper bound: 0.0236370
time: 1.54 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0010730, 0.0180512, 0.0012615, 0.0173226, -0.0162496, 0.0167897
1: -0.0045587, 0.0028729, -0.0038006, 0.0028184, -0.0073772, 0.0066735
2: 0.0021427, 0.0137666, 0.0026307, 0.0136624, -0.0115197, 0.0111360
3: -0.0041061, 0.0049705, -0.0038766, 0.0041558, -0.0082619, 0.0088471
4: -0.0041798, -0.0000960, -0.0040242, -0.0001819, -0.0039979, 0.0039282
5: -0.0003534, 0.0074805, 0.0000250, 0.0073701, -0.0077234, 0.0074555
6: -0.0092047, 0.0073800, -0.0089590, 0.0069419, -0.0161466, 0.0163390
7: -0.0183856, 0.0050962, -0.0168004, 0.0049707, -0.0233563, 0.0218966
8: 0.9796072, 0.9937186, 0.9807531, 0.9940109, -0.0144037, 0.0129655
9: -0.0093550, 0.0044573, -0.0096530, 0.0034678, -0.0128228, 0.0141103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0231188
time: 1.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0231828
time: 1.34 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0009741, 0.0200001, 0.0010982, 0.0222857, -0.0213116, 0.0189019
1: -0.0054812, 0.0030962, -0.0056248, 0.0036053, -0.0090865, 0.0087210
2: 0.0007220, 0.0138213, -0.0010099, 0.0137527, -0.0130307, 0.0148313
3: -0.0042925, 0.0059045, -0.0042753, 0.0060085, -0.0103010, 0.0101798
4: -0.0044200, -0.0000833, -0.0046107, -0.0001687, -0.0042513, 0.0045274
5: -0.0011707, 0.0075384, -0.0019599, 0.0074657, -0.0086364, 0.0094983
6: -0.0098508, 0.0076099, -0.0107166, 0.0073214, -0.0171722, 0.0183264
7: -0.0198473, 0.0051614, -0.0196215, 0.0055149, -0.0253622, 0.0247828
8: 0.9782894, 0.9938908, 0.9783434, 0.9954246, -0.0171352, 0.0155473
9: -0.0093967, 0.0054117, -0.0106530, 0.0052772, -0.0146739, 0.0160648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232669, upper bound: 0.0235289
time: 1.52 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232669, upper bound: 0.0238467
time: 1.99 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.06 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229778
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0233921, upper bound: 0.0234817
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230439
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0233921, upper bound: 0.0235710
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0233733, upper bound: 0.0229787
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0237084, upper bound: 0.0235019
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0233733, upper bound: 0.0230446
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0237084, upper bound: 0.0235976
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0237922, upper bound: 0.0239239
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0237922, upper bound: 0.0240702
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0241051, upper bound: 0.0239425
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0241051, upper bound: 0.0241051
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0234792, upper bound: 0.0235639
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0234792, upper bound: 0.0236577
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0237737, upper bound: 0.0235847
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0237736, upper bound: 0.0236838
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0230953
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0231884
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0235187
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0236889
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0231384
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0232645
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0235897
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0235716, upper bound: 0.0238088
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0230815
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0231255
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0238088, upper bound: 0.0235046
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0238088, upper bound: 0.0236370
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0231188
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0231828
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0232669, upper bound: 0.0235289
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 8, lower bound: -0.0232669, upper bound: 0.0238467

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0015673, 0.0181572, 0.0016927, 0.0179576, -0.0163903, 0.0164645
1: -0.0045143, 0.0027346, -0.0037437, 0.0029769, -0.0074913, 0.0064783
2: 0.0020552, 0.0134933, 0.0021256, 0.0134240, -0.0113689, 0.0113677
3: -0.0037971, 0.0047282, -0.0037141, 0.0039698, -0.0077669, 0.0084423
4: -0.0038709, -0.0003659, -0.0038381, -0.0003429, -0.0035280, 0.0034722
5: -0.0003822, 0.0071909, -0.0001517, 0.0071175, -0.0074997, 0.0073426
6: -0.0086228, 0.0062311, -0.0086097, 0.0059397, -0.0145625, 0.0148408
7: -0.0167116, 0.0037163, -0.0153508, 0.0044013, -0.0211129, 0.0190671
8: 0.9806167, 0.9928443, 0.9817119, 0.9939606, -0.0133439, 0.0111324
9: -0.0084727, 0.0034433, -0.0096079, 0.0025462, -0.0110188, 0.0130512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229138
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229778
time: 2.04 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0014740, 0.0200562, 0.0015672, 0.0224659, -0.0209919, 0.0184890
1: -0.0053992, 0.0029172, -0.0054990, 0.0035784, -0.0089777, 0.0084162
2: 0.0006755, 0.0135449, -0.0011560, 0.0134934, -0.0128179, 0.0147009
3: -0.0039503, 0.0056229, -0.0040413, 0.0057395, -0.0096898, 0.0096642
4: -0.0040959, -0.0003544, -0.0043430, -0.0003343, -0.0037615, 0.0039885
5: -0.0011855, 0.0072456, -0.0019948, 0.0071910, -0.0083765, 0.0092404
6: -0.0093183, 0.0064480, -0.0102755, 0.0062313, -0.0155496, 0.0167235
7: -0.0180884, 0.0037748, -0.0179409, 0.0047339, -0.0228223, 0.0217156
8: 0.9793526, 0.9930147, 0.9794595, 0.9951038, -0.0157512, 0.0135552
9: -0.0085100, 0.0043364, -0.0103746, 0.0042227, -0.0127328, 0.0147110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229722, upper bound: 0.0232263
time: 1.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229722, upper bound: 0.0234817
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0015200, 0.0181953, 0.0015428, 0.0199356, -0.0184155, 0.0166524
1: -0.0045368, 0.0027421, -0.0039399, 0.0037614, -0.0082982, 0.0066820
2: 0.0020283, 0.0135195, 0.0006647, 0.0135069, -0.0114785, 0.0128548
3: -0.0038029, 0.0047802, -0.0039899, 0.0042391, -0.0080420, 0.0087701
4: -0.0039047, -0.0003661, -0.0041770, -0.0001649, -0.0037398, 0.0038109
5: -0.0003983, 0.0072186, -0.0008712, 0.0072052, -0.0076035, 0.0080898
6: -0.0086373, 0.0063410, -0.0099252, 0.0062880, -0.0149253, 0.0162662
7: -0.0169102, 0.0037149, -0.0161892, 0.0059744, -0.0228845, 0.0199041
8: 0.9804956, 0.9928480, 0.9812030, 0.9962462, -0.0157505, 0.0116450
9: -0.0084718, 0.0035579, -0.0119153, 0.0030610, -0.0115328, 0.0154732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230439
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230439
time: 1.50 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0014236, 0.0200824, 0.0013948, 0.0247397, -0.0233161, 0.0186876
1: -0.0054176, 0.0029250, -0.0057236, 0.0045060, -0.0099237, 0.0086486
2: 0.0006591, 0.0135728, -0.0028433, 0.0135887, -0.0129296, 0.0164161
3: -0.0039562, 0.0056715, -0.0043528, 0.0060426, -0.0099988, 0.0100243
4: -0.0041335, -0.0003547, -0.0047612, -0.0001519, -0.0039816, 0.0044064
5: -0.0011967, 0.0072751, -0.0028188, 0.0072919, -0.0084886, 0.0100939
6: -0.0093289, 0.0065651, -0.0116798, 0.0066320, -0.0159609, 0.0182450
7: -0.0183107, 0.0037732, -0.0189300, 0.0065168, -0.0248274, 0.0227031
8: 0.9792259, 0.9930174, 0.9787811, 0.9978203, -0.0185944, 0.0142363
9: -0.0085090, 0.0044662, -0.0132895, 0.0048295, -0.0133385, 0.0177556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229722, upper bound: 0.0233187
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229722, upper bound: 0.0235710
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0013434, 0.0185585, 0.0016393, 0.0174301, -0.0160867, 0.0169192
1: -0.0047695, 0.0029163, -0.0037456, 0.0027931, -0.0075626, 0.0066618
2: 0.0017717, 0.0136171, 0.0025303, 0.0134535, -0.0116818, 0.0110868
3: -0.0041555, 0.0051057, -0.0036984, 0.0040029, -0.0081584, 0.0088041
4: -0.0040576, -0.0000756, -0.0038095, -0.0003422, -0.0037154, 0.0037338
5: -0.0005625, 0.0073221, 0.0000259, 0.0071487, -0.0077113, 0.0072962
6: -0.0093436, 0.0067515, -0.0084133, 0.0060637, -0.0154073, 0.0151648
7: -0.0178068, 0.0052004, -0.0155457, 0.0042123, -0.0220191, 0.0207460
8: 0.9799001, 0.9937816, 0.9815956, 0.9935328, -0.0136327, 0.0121860
9: -0.0094216, 0.0040989, -0.0092326, 0.0026641, -0.0120857, 0.0133316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228558, upper bound: 0.0224029
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228000, upper bound: 0.0223748
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0012459, 0.0205189, 0.0015091, 0.0219892, -0.0207433, 0.0190097
1: -0.0056890, 0.0031205, -0.0055345, 0.0033919, -0.0090809, 0.0086550
2: 0.0003400, 0.0136710, -0.0007980, 0.0135255, -0.0131854, 0.0144691
3: -0.0043251, 0.0060434, -0.0040341, 0.0058204, -0.0101454, 0.0100776
4: -0.0042967, -0.0000652, -0.0043191, -0.0003341, -0.0039626, 0.0042538
5: -0.0013885, 0.0073792, -0.0018378, 0.0072250, -0.0086134, 0.0092170
6: -0.0099915, 0.0069781, -0.0100797, 0.0063663, -0.0163578, 0.0170578
7: -0.0192808, 0.0052536, -0.0182225, 0.0045345, -0.0238153, 0.0234761
8: 0.9785537, 0.9939381, 0.9792914, 0.9945677, -0.0160140, 0.0146468
9: -0.0094556, 0.0050550, -0.0098364, 0.0043818, -0.0138374, 0.0148914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231216, upper bound: 0.0232314
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231216, upper bound: 0.0235019
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0012704, 0.0186403, 0.0014559, 0.0194319, -0.0181615, 0.0171844
1: -0.0048206, 0.0029315, -0.0039607, 0.0035419, -0.0083624, 0.0068923
2: 0.0017128, 0.0136575, 0.0010403, 0.0135549, -0.0118421, 0.0126171
3: -0.0041673, 0.0051908, -0.0039776, 0.0043041, -0.0084714, 0.0091683
4: -0.0041160, -0.0000761, -0.0041643, -0.0001643, -0.0039517, 0.0040882
5: -0.0005971, 0.0073648, -0.0007068, 0.0072562, -0.0078533, 0.0080717
6: -0.0093717, 0.0069212, -0.0097376, 0.0064900, -0.0158618, 0.0166589
7: -0.0181423, 0.0051981, -0.0164835, 0.0057656, -0.0239078, 0.0216815
8: 0.9796934, 0.9937875, 0.9810180, 0.9957287, -0.0160354, 0.0127695
9: -0.0094201, 0.0043040, -0.0113269, 0.0032452, -0.0126653, 0.0156309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228558, upper bound: 0.0224699
time: 1.38 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228000, upper bound: 0.0224538
time: 2.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0011698, 0.0206150, 0.0013031, 0.0243515, -0.0231818, 0.0193119
1: -0.0057498, 0.0031379, -0.0057979, 0.0043057, -0.0100555, 0.0089358
2: 0.0002700, 0.0137131, -0.0025611, 0.0136394, -0.0133695, 0.0162742
3: -0.0043385, 0.0061350, -0.0043528, 0.0061708, -0.0105093, 0.0104877
4: -0.0043607, -0.0000657, -0.0047602, -0.0001517, -0.0042089, 0.0046945
5: -0.0014296, 0.0074238, -0.0026924, 0.0073457, -0.0087752, 0.0101162
6: -0.0100266, 0.0071552, -0.0115196, 0.0068452, -0.0168718, 0.0186747
7: -0.0196335, 0.0052510, -0.0193342, 0.0062967, -0.0259302, 0.0245852
8: 0.9783211, 0.9939451, 0.9785060, 0.9972782, -0.0189571, 0.0154392
9: -0.0094540, 0.0052775, -0.0126341, 0.0050801, -0.0145341, 0.0179116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231216, upper bound: 0.0233211
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231216, upper bound: 0.0235976
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0014693, 0.0203177, 0.0015100, 0.0301748, -0.0287055, 0.0188078
1: -0.0055169, 0.0029387, -0.0064833, 0.0061772, -0.0116941, 0.0094220
2: 0.0004858, 0.0135475, -0.0067985, 0.0135251, -0.0130393, 0.0203460
3: -0.0039680, 0.0057417, -0.0044681, 0.0066445, -0.0106125, 0.0102098
4: -0.0041205, -0.0003536, -0.0052750, -0.0002654, -0.0038551, 0.0049214
5: -0.0012967, 0.0072483, -0.0048949, 0.0072245, -0.0085212, 0.0121432
6: -0.0094212, 0.0064589, -0.0141036, 0.0063645, -0.0157857, 0.0205625
7: -0.0182373, 0.0037790, -0.0192568, 0.0083087, -0.0265460, 0.0230358
8: 0.9791995, 0.9930354, 0.9780267, 1.0025463, -0.0233468, 0.0150086
9: -0.0085128, 0.0044374, -0.0184665, 0.0051272, -0.0136399, 0.0229039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235172, upper bound: 0.0233995
time: 1.31 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237089, upper bound: 0.0238397
time: 2.26 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0014138, 0.0203436, 0.0013211, 0.0325351, -0.0311213, 0.0190225
1: -0.0055353, 0.0029472, -0.0066833, 0.0072591, -0.0127944, 0.0096305
2: 0.0004696, 0.0135782, -0.0085442, 0.0136295, -0.0131599, 0.0221224
3: -0.0039745, 0.0057899, -0.0048110, 0.0069258, -0.0109003, 0.0106009
4: -0.0041578, -0.0003539, -0.0056966, -0.0000746, -0.0040832, 0.0053427
5: -0.0013077, 0.0072808, -0.0057471, 0.0073351, -0.0086429, 0.0130279
6: -0.0094317, 0.0065879, -0.0155562, 0.0068034, -0.0162351, 0.0221441
7: -0.0184517, 0.0037774, -0.0202232, 0.0106677, -0.0291195, 0.0240005
8: 0.9790645, 0.9930381, 0.9773424, 1.0058571, -0.0267926, 0.0156958
9: -0.0085117, 0.0045624, -0.0222468, 0.0057272, -0.0142389, 0.0268092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235172, upper bound: 0.0234943
time: 1.34 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237089, upper bound: 0.0239963
time: 1.71 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0012451, 0.0207564, 0.0014531, 0.0297946, -0.0285495, 0.0193033
1: -0.0057964, 0.0031410, -0.0064631, 0.0060366, -0.0118330, 0.0096042
2: 0.0001672, 0.0136715, -0.0065250, 0.0135565, -0.0133893, 0.0201965
3: -0.0043419, 0.0061524, -0.0044595, 0.0066842, -0.0110261, 0.0106120
4: -0.0043179, -0.0000646, -0.0052620, -0.0002653, -0.0040526, 0.0051975
5: -0.0014899, 0.0073796, -0.0047554, 0.0072578, -0.0087478, 0.0121350
6: -0.0100856, 0.0069799, -0.0139444, 0.0064966, -0.0165822, 0.0209243
7: -0.0194083, 0.0052571, -0.0194749, 0.0080909, -0.0274992, 0.0247320
8: 0.9784179, 0.9939544, 0.9779255, 1.0021192, -0.0237013, 0.0160289
9: -0.0094579, 0.0051420, -0.0179671, 0.0052459, -0.0147038, 0.0231091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237639, upper bound: 0.0234053
time: 3.19 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0240263, upper bound: 0.0238564
time: 1.85 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0011692, 0.0208532, 0.0012390, 0.0322445, -0.0310753, 0.0196142
1: -0.0058567, 0.0031586, -0.0066996, 0.0071370, -0.0129937, 0.0098582
2: 0.0000967, 0.0137134, -0.0083462, 0.0136749, -0.0135782, 0.0220596
3: -0.0043554, 0.0062444, -0.0048087, 0.0069968, -0.0113522, 0.0110531
4: -0.0043810, -0.0000651, -0.0057097, -0.0000747, -0.0043063, 0.0056446
5: -0.0015313, 0.0074241, -0.0056419, 0.0073832, -0.0089145, 0.0130660
6: -0.0101213, 0.0071564, -0.0154270, 0.0069942, -0.0171155, 0.0225834
7: -0.0197652, 0.0052545, -0.0205453, 0.0104739, -0.0302391, 0.0257999
8: 0.9781786, 0.9939615, 0.9771549, 1.0054721, -0.0272934, 0.0168065
9: -0.0094563, 0.0053606, -0.0217965, 0.0059171, -0.0153733, 0.0271571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237639, upper bound: 0.0235009
time: 3.38 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0240263, upper bound: 0.0240263
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0014693, 0.0203177, 0.0014271, 0.0251389, -0.0236696, 0.0188906
1: -0.0055169, 0.0029387, -0.0061887, 0.0041439, -0.0096608, 0.0091274
2: 0.0004858, 0.0135475, -0.0030664, 0.0135708, -0.0130850, 0.0166139
3: -0.0039680, 0.0057417, -0.0042311, 0.0063851, -0.0103531, 0.0099728
4: -0.0041205, -0.0003536, -0.0046687, -0.0003064, -0.0038141, 0.0043151
5: -0.0012967, 0.0072483, -0.0030945, 0.0072730, -0.0085697, 0.0103428
6: -0.0094212, 0.0064589, -0.0118358, 0.0065569, -0.0159782, 0.0182946
7: -0.0182373, 0.0037790, -0.0191792, 0.0055037, -0.0237410, 0.0229582
8: 0.9791995, 0.9930354, 0.9782253, 0.9967895, -0.0175900, 0.0148101
9: -0.0085128, 0.0044374, -0.0119040, 0.0050731, -0.0135858, 0.0163414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229990
time: 1.36 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233875, upper bound: 0.0234756
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0014138, 0.0203436, 0.0012332, 0.0274591, -0.0260453, 0.0191103
1: -0.0055353, 0.0029472, -0.0063934, 0.0051394, -0.0106747, 0.0093406
2: 0.0004696, 0.0135782, -0.0047777, 0.0136780, -0.0132085, 0.0183560
3: -0.0039745, 0.0057899, -0.0045753, 0.0066751, -0.0106496, 0.0103652
4: -0.0041578, -0.0003539, -0.0050930, -0.0001106, -0.0040472, 0.0047390
5: -0.0013077, 0.0072808, -0.0039356, 0.0073866, -0.0086944, 0.0112164
6: -0.0094317, 0.0065879, -0.0132557, 0.0070076, -0.0164393, 0.0198436
7: -0.0184517, 0.0037774, -0.0201591, 0.0075608, -0.0260125, 0.0239365
8: 0.9790645, 0.9930381, 0.9775357, 0.9997622, -0.0206977, 0.0155024
9: -0.0085117, 0.0045624, -0.0151710, 0.0056752, -0.0141870, 0.0197334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230734
time: 1.34 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233875, upper bound: 0.0235733
time: 2.06 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0012451, 0.0207564, 0.0013633, 0.0246103, -0.0233652, 0.0193931
1: -0.0057964, 0.0031410, -0.0061584, 0.0039705, -0.0097669, 0.0092994
2: 0.0001672, 0.0136715, -0.0026713, 0.0136061, -0.0134389, 0.0163428
3: -0.0043419, 0.0061524, -0.0042173, 0.0064150, -0.0107569, 0.0103697
4: -0.0043179, -0.0000646, -0.0046469, -0.0003063, -0.0040116, 0.0045823
5: -0.0014899, 0.0073796, -0.0029140, 0.0073104, -0.0088003, 0.0102936
6: -0.0100856, 0.0069799, -0.0116279, 0.0067052, -0.0167908, 0.0186078
7: -0.0194083, 0.0052571, -0.0193886, 0.0052995, -0.0247078, 0.0246457
8: 0.9784179, 0.9939544, 0.9781373, 0.9962928, -0.0178749, 0.0158172
9: -0.0094579, 0.0051420, -0.0113758, 0.0051875, -0.0146454, 0.0165178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233720, upper bound: 0.0230013
time: 1.61 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0236946, upper bound: 0.0234957
time: 2.97 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0011692, 0.0208532, 0.0011460, 0.0270063, -0.0258371, 0.0197072
1: -0.0058567, 0.0031586, -0.0063906, 0.0049749, -0.0108316, 0.0095492
2: 0.0000967, 0.0137134, -0.0044506, 0.0137263, -0.0136296, 0.0181640
3: -0.0043554, 0.0062444, -0.0045669, 0.0067274, -0.0110828, 0.0108113
4: -0.0043810, -0.0000651, -0.0050921, -0.0001107, -0.0042703, 0.0050271
5: -0.0015313, 0.0074241, -0.0037770, 0.0074377, -0.0089690, 0.0112011
6: -0.0101213, 0.0071564, -0.0130756, 0.0072103, -0.0173316, 0.0202321
7: -0.0197652, 0.0052545, -0.0204683, 0.0073454, -0.0271106, 0.0257229
8: 0.9781786, 0.9939615, 0.9773688, 0.9992830, -0.0211044, 0.0165926
9: -0.0094563, 0.0053606, -0.0146190, 0.0058599, -0.0153162, 0.0199797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233720, upper bound: 0.0230742
time: 1.47 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0236946, upper bound: 0.0235998
time: 1.60 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0013870, 0.0176023, 0.0017587, 0.0149694, -0.0135823, 0.0158436
1: -0.0042734, 0.0026804, -0.0029663, 0.0023326, -0.0066061, 0.0056467
2: 0.0024605, 0.0135930, 0.0043604, 0.0133875, -0.0109270, 0.0092326
3: -0.0037344, 0.0045259, -0.0034750, 0.0031063, -0.0068407, 0.0080009
4: -0.0039339, -0.0003926, -0.0034666, -0.0003707, -0.0035632, 0.0030740
5: -0.0001494, 0.0072965, 0.0009485, 0.0070788, -0.0072282, 0.0063480
6: -0.0084074, 0.0066501, -0.0075348, 0.0057862, -0.0141936, 0.0141849
7: -0.0169628, 0.0035794, -0.0142173, 0.0036917, -0.0206545, 0.0177967
8: 0.9805180, 0.9927316, 0.9818617, 0.9925863, -0.0120682, 0.0108699
9: -0.0083851, 0.0036024, -0.0084569, 0.0018627, -0.0102478, 0.0120593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232206, upper bound: 0.0230673
time: 1.56 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0230953
time: 2.75 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0013870, 0.0176023, 0.0015623, 0.0152608, -0.0138737, 0.0160400
1: -0.0042734, 0.0026804, -0.0031509, 0.0024584, -0.0067318, 0.0058314
2: 0.0024605, 0.0135930, 0.0041611, 0.0134961, -0.0110356, 0.0094319
3: -0.0037344, 0.0045259, -0.0037821, 0.0033803, -0.0071147, 0.0083081
4: -0.0039339, -0.0003926, -0.0036203, -0.0000898, -0.0038441, 0.0032276
5: -0.0001494, 0.0072965, 0.0008021, 0.0071939, -0.0073433, 0.0064945
6: -0.0084074, 0.0066501, -0.0083084, 0.0062428, -0.0146502, 0.0149585
7: -0.0169628, 0.0035794, -0.0150647, 0.0051280, -0.0220908, 0.0186441
8: 0.9805180, 0.9927316, 0.9814237, 0.9935305, -0.0130124, 0.0113078
9: -0.0083851, 0.0036024, -0.0093753, 0.0023870, -0.0107721, 0.0129777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232206, upper bound: 0.0231580
time: 1.36 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0231884
time: 2.42 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0012937, 0.0194993, 0.0016211, 0.0186549, -0.0173611, 0.0178783
1: -0.0051616, 0.0028793, -0.0046922, 0.0027757, -0.0079373, 0.0075715
2: 0.0010849, 0.0136446, 0.0017002, 0.0134636, -0.0123787, 0.0119443
3: -0.0039022, 0.0054193, -0.0038322, 0.0048421, -0.0087443, 0.0092515
4: -0.0041577, -0.0003790, -0.0038687, -0.0003626, -0.0037951, 0.0034898
5: -0.0009489, 0.0073512, -0.0005922, 0.0071594, -0.0081083, 0.0079434
6: -0.0090927, 0.0068670, -0.0087730, 0.0061061, -0.0151988, 0.0156399
7: -0.0183206, 0.0036493, -0.0167218, 0.0037331, -0.0220537, 0.0203711
8: 0.9792901, 0.9929135, 0.9804882, 0.9928889, -0.0135987, 0.0124253
9: -0.0084298, 0.0044880, -0.0084834, 0.0034711, -0.0119009, 0.0129714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0232574
time: 1.66 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0234412
time: 1.43 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0012937, 0.0194993, 0.0014160, 0.0190831, -0.0177893, 0.0180833
1: -0.0051616, 0.0028793, -0.0049538, 0.0029644, -0.0081260, 0.0078331
2: 0.0010849, 0.0136446, 0.0013887, 0.0135770, -0.0124921, 0.0122559
3: -0.0039022, 0.0054193, -0.0041981, 0.0052235, -0.0091257, 0.0096174
4: -0.0041577, -0.0003790, -0.0040499, -0.0000696, -0.0040881, 0.0036709
5: -0.0009489, 0.0073512, -0.0007809, 0.0072795, -0.0082284, 0.0081321
6: -0.0090927, 0.0068670, -0.0094841, 0.0065828, -0.0156754, 0.0163511
7: -0.0183206, 0.0036493, -0.0177665, 0.0052312, -0.0235518, 0.0214158
8: 0.9792901, 0.9929135, 0.9797751, 0.9938425, -0.0145524, 0.0131385
9: -0.0084298, 0.0044880, -0.0094413, 0.0041079, -0.0125377, 0.0139293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0233985
time: 1.63 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0236027
time: 1.52 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0013376, 0.0176417, 0.0016106, 0.0153629, -0.0140253, 0.0160310
1: -0.0042966, 0.0026884, -0.0031474, 0.0024411, -0.0067377, 0.0058358
2: 0.0024324, 0.0136203, 0.0040946, 0.0134694, -0.0110370, 0.0095257
3: -0.0037415, 0.0045769, -0.0036927, 0.0033528, -0.0070944, 0.0082696
4: -0.0039688, -0.0003916, -0.0035945, -0.0001923, -0.0037766, 0.0032029
5: -0.0001662, 0.0073255, 0.0007562, 0.0071655, -0.0073318, 0.0065693
6: -0.0084263, 0.0067650, -0.0082489, 0.0061304, -0.0145567, 0.0150138
7: -0.0171601, 0.0035847, -0.0149395, 0.0046040, -0.0217642, 0.0185242
8: 0.9803938, 0.9927399, 0.9815316, 0.9932545, -0.0128608, 0.0112083
9: -0.0083885, 0.0037218, -0.0090403, 0.0023099, -0.0106984, 0.0127621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227716, upper bound: 0.0226321
time: 1.40 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227362, upper bound: 0.0225536
time: 1.47 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0013376, 0.0176417, 0.0013536, 0.0157168, -0.0143792, 0.0162881
1: -0.0042966, 0.0026884, -0.0033738, 0.0025958, -0.0068924, 0.0060622
2: 0.0024324, 0.0136203, 0.0038501, 0.0136115, -0.0111791, 0.0097702
3: -0.0037415, 0.0045769, -0.0040439, 0.0037046, -0.0074461, 0.0086208
4: -0.0039688, -0.0003916, -0.0037958, 0.0001170, -0.0040859, 0.0034042
5: -0.0001662, 0.0073255, 0.0005808, 0.0073161, -0.0074823, 0.0067446
6: -0.0084263, 0.0067650, -0.0091258, 0.0067279, -0.0151542, 0.0158908
7: -0.0171601, 0.0035847, -0.0160832, 0.0061856, -0.0233458, 0.0196679
8: 0.9803938, 0.9927399, 0.9809584, 0.9943036, -0.0139099, 0.0117815
9: -0.0083885, 0.0037218, -0.0100516, 0.0030020, -0.0113905, 0.0137734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227716, upper bound: 0.0227154
time: 2.07 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227362, upper bound: 0.0226241
time: 1.79 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0012415, 0.0195366, 0.0014536, 0.0190396, -0.0177981, 0.0180830
1: -0.0051837, 0.0028883, -0.0048869, 0.0029056, -0.0080893, 0.0077753
2: 0.0010599, 0.0136735, 0.0014337, 0.0135562, -0.0124963, 0.0122398
3: -0.0039100, 0.0054720, -0.0040707, 0.0051111, -0.0090210, 0.0095427
4: -0.0041962, -0.0003780, -0.0040192, -0.0001789, -0.0040173, 0.0036412
5: -0.0009649, 0.0073818, -0.0007709, 0.0072575, -0.0082224, 0.0081526
6: -0.0091120, 0.0069883, -0.0094529, 0.0064954, -0.0156074, 0.0164413
7: -0.0185510, 0.0036541, -0.0175770, 0.0046722, -0.0232232, 0.0212311
8: 0.9791576, 0.9929214, 0.9799049, 0.9935696, -0.0144120, 0.0130165
9: -0.0084329, 0.0046215, -0.0090839, 0.0039976, -0.0124305, 0.0137055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0233251
time: 1.45 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0235071
time: 1.40 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0012415, 0.0195366, 0.0011888, 0.0195840, -0.0183425, 0.0183478
1: -0.0051837, 0.0028883, -0.0052226, 0.0031298, -0.0083135, 0.0081109
2: 0.0010599, 0.0136735, 0.0010352, 0.0137026, -0.0126427, 0.0126383
3: -0.0039100, 0.0054720, -0.0044830, 0.0055809, -0.0094909, 0.0099549
4: -0.0041962, -0.0003780, -0.0042520, 0.0001384, -0.0043346, 0.0038740
5: -0.0009649, 0.0073818, -0.0010093, 0.0074126, -0.0083775, 0.0083910
6: -0.0091120, 0.0069883, -0.0102554, 0.0071109, -0.0162229, 0.0172437
7: -0.0185510, 0.0036541, -0.0189331, 0.0062951, -0.0248460, 0.0225872
8: 0.9791576, 0.9929214, 0.9789847, 0.9946132, -0.0154556, 0.0139368
9: -0.0084329, 0.0046215, -0.0101216, 0.0048263, -0.0132592, 0.0147431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0235231
time: 1.72 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0237086
time: 2.07 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0011477, 0.0179670, 0.0017587, 0.0149694, -0.0138217, 0.0162083
1: -0.0045079, 0.0028568, -0.0029663, 0.0023326, -0.0068405, 0.0058231
2: 0.0022032, 0.0137254, 0.0043604, 0.0133875, -0.0111843, 0.0093650
3: -0.0040928, 0.0048855, -0.0034750, 0.0031063, -0.0071992, 0.0083604
4: -0.0041226, -0.0000967, -0.0034666, -0.0003707, -0.0037519, 0.0033699
5: -0.0003180, 0.0074367, 0.0009485, 0.0070788, -0.0073968, 0.0064883
6: -0.0091718, 0.0072065, -0.0075348, 0.0057862, -0.0149580, 0.0147413
7: -0.0180462, 0.0050925, -0.0142173, 0.0036917, -0.0217379, 0.0193098
8: 0.9798120, 0.9937083, 0.9818617, 0.9925863, -0.0127743, 0.0118465
9: -0.0093526, 0.0042574, -0.0084569, 0.0018627, -0.0112153, 0.0127143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234414, upper bound: 0.0230565
time: 1.56 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0230815
time: 1.52 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0011477, 0.0179670, 0.0015623, 0.0152608, -0.0141131, 0.0164047
1: -0.0045079, 0.0028568, -0.0031509, 0.0024584, -0.0069662, 0.0060078
2: 0.0022032, 0.0137254, 0.0041611, 0.0134961, -0.0112929, 0.0095643
3: -0.0040928, 0.0048855, -0.0037821, 0.0033803, -0.0074731, 0.0086676
4: -0.0041226, -0.0000967, -0.0036203, -0.0000898, -0.0038586, 0.0033115
5: -0.0003180, 0.0074367, 0.0008021, 0.0071939, -0.0075118, 0.0066347
6: -0.0091718, 0.0072065, -0.0083084, 0.0062428, -0.0154146, 0.0155149
7: -0.0180462, 0.0050925, -0.0150647, 0.0051280, -0.0228430, 0.0196673
8: 0.9798120, 0.9937083, 0.9814237, 0.9935305, -0.0137185, 0.0122845
9: -0.0093526, 0.0042574, -0.0093753, 0.0023870, -0.0111095, 0.0131904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234414, upper bound: 0.0230956
time: 1.33 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0231255
time: 1.41 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0010507, 0.0198884, 0.0016211, 0.0186549, -0.0176041, 0.0182673
1: -0.0054151, 0.0030768, -0.0046922, 0.0027757, -0.0081908, 0.0077690
2: 0.0008025, 0.0137790, 0.0017002, 0.0134636, -0.0126612, 0.0120787
3: -0.0042767, 0.0058074, -0.0038322, 0.0048421, -0.0091188, 0.0096396
4: -0.0043560, -0.0000839, -0.0038687, -0.0003626, -0.0039935, 0.0037848
5: -0.0011233, 0.0074935, -0.0005922, 0.0071594, -0.0082827, 0.0080858
6: -0.0098075, 0.0074318, -0.0087730, 0.0061061, -0.0159136, 0.0162048
7: -0.0194812, 0.0051581, -0.0167218, 0.0037331, -0.0232143, 0.0218799
8: 0.9785232, 0.9938783, 0.9804882, 0.9928889, -0.0143657, 0.0133901
9: -0.0093946, 0.0051849, -0.0084834, 0.0034711, -0.0128657, 0.0136683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0232415
time: 1.47 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0234349
time: 1.83 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0010507, 0.0198884, 0.0014160, 0.0190831, -0.0180323, 0.0184723
1: -0.0054151, 0.0030768, -0.0049538, 0.0029644, -0.0083794, 0.0080306
2: 0.0008025, 0.0137790, 0.0013887, 0.0135770, -0.0127745, 0.0123903
3: -0.0042767, 0.0058074, -0.0041981, 0.0052235, -0.0095002, 0.0100055
4: -0.0043560, -0.0000839, -0.0040499, -0.0000696, -0.0041882, 0.0037304
5: -0.0011233, 0.0074935, -0.0007809, 0.0072795, -0.0084028, 0.0082744
6: -0.0098075, 0.0074318, -0.0094841, 0.0065828, -0.0163902, 0.0169159
7: -0.0194812, 0.0051581, -0.0177665, 0.0052312, -0.0247123, 0.0223053
8: 0.9785232, 0.9938783, 0.9797751, 0.9938425, -0.0153193, 0.0141032
9: -0.0093946, 0.0051849, -0.0094413, 0.0041079, -0.0128788, 0.0144625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0233017
time: 1.45 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0235696
time: 1.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0010730, 0.0180512, 0.0016106, 0.0153629, -0.0142899, 0.0164405
1: -0.0045587, 0.0028729, -0.0031474, 0.0024411, -0.0069998, 0.0060203
2: 0.0021427, 0.0137666, 0.0040946, 0.0134694, -0.0113267, 0.0096720
3: -0.0041061, 0.0049705, -0.0036927, 0.0033528, -0.0074590, 0.0086632
4: -0.0041798, -0.0000960, -0.0035945, -0.0001923, -0.0039875, 0.0034985
5: -0.0003534, 0.0074805, 0.0007562, 0.0071655, -0.0075189, 0.0067243
6: -0.0092047, 0.0073800, -0.0082489, 0.0061304, -0.0153351, 0.0156289
7: -0.0183856, 0.0050962, -0.0149395, 0.0046040, -0.0229896, 0.0200357
8: 0.9796072, 0.9937186, 0.9815316, 0.9932545, -0.0136473, 0.0121871
9: -0.0093550, 0.0044573, -0.0090403, 0.0023099, -0.0116649, 0.0134976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229252, upper bound: 0.0226086
time: 2.25 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229004, upper bound: 0.0225329
time: 2.28 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0010730, 0.0180512, 0.0013536, 0.0157168, -0.0146438, 0.0166976
1: -0.0045587, 0.0028729, -0.0033738, 0.0025958, -0.0071546, 0.0062467
2: 0.0021427, 0.0137666, 0.0038501, 0.0136115, -0.0114688, 0.0099165
3: -0.0041061, 0.0049705, -0.0040439, 0.0037046, -0.0078107, 0.0090144
4: -0.0041798, -0.0000960, -0.0037958, 0.0001170, -0.0042063, 0.0034208
5: -0.0003534, 0.0074805, 0.0005808, 0.0073161, -0.0076695, 0.0068997
6: -0.0092047, 0.0073800, -0.0091258, 0.0067279, -0.0159326, 0.0165058
7: -0.0183856, 0.0050962, -0.0160832, 0.0061856, -0.0245713, 0.0202372
8: 0.9796072, 0.9937186, 0.9809584, 0.9943036, -0.0146964, 0.0127602
9: -0.0093550, 0.0044573, -0.0100516, 0.0030020, -0.0115127, 0.0143301

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229252, upper bound: 0.0226382
time: 1.59 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229004, upper bound: 0.0225518
time: 1.30 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0011660, 0.0154469, 0.0010982, 0.0222857, -0.0211197, 0.0143487
1: -0.0033299, 0.0024936, -0.0056248, 0.0036053, -0.0069351, 0.0081183
2: 0.0040264, 0.0137152, -0.0010099, 0.0137527, -0.0097263, 0.0147252
3: -0.0037944, 0.0037254, -0.0042753, 0.0060085, -0.0098029, 0.0080007
4: -0.0038899, -0.0001109, -0.0046107, -0.0001687, -0.0037212, 0.0044998
5: 0.0007207, 0.0074260, -0.0019599, 0.0074657, -0.0067450, 0.0093859
6: -0.0083727, 0.0071639, -0.0107166, 0.0073214, -0.0156941, 0.0178805
7: -0.0165476, 0.0050201, -0.0196215, 0.0055149, -0.0220625, 0.0246416
8: 0.9805401, 0.9934986, 0.9783434, 0.9954246, -0.0148845, 0.0151552
9: -0.0093064, 0.0032892, -0.0106530, 0.0052772, -0.0145836, 0.0139422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0214420, upper bound: 0.0215061
time: 1.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231004, upper bound: 0.0233571
time: 1.60 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010120, 0.0192896, 0.0010982, 0.0222857, -0.0212737, 0.0181914
1: -0.0051653, 0.0030198, -0.0056248, 0.0036053, -0.0087706, 0.0086446
2: 0.0012369, 0.0138003, -0.0010099, 0.0137527, -0.0125158, 0.0148103
3: -0.0042282, 0.0055863, -0.0042753, 0.0060085, -0.0102367, 0.0098615
4: -0.0043357, -0.0000885, -0.0046107, -0.0001687, -0.0041670, 0.0045222
5: -0.0008679, 0.0075162, -0.0019599, 0.0074657, -0.0083337, 0.0094760
6: -0.0095731, 0.0075217, -0.0107166, 0.0073214, -0.0168945, 0.0182382
7: -0.0193418, 0.0051348, -0.0196215, 0.0055149, -0.0248567, 0.0247563
8: 0.9787722, 0.9938284, 0.9783434, 0.9954246, -0.0166525, 0.0154850
9: -0.0093797, 0.0050769, -0.0106530, 0.0052772, -0.0146569, 0.0157299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0234940
time: 1.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0236411
time: 1.30 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.26 seconds
IS_A1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229138
IS_A1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229778
IS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0229722, upper bound: 0.0232263
IS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0229722, upper bound: 0.0234817
IS_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230439
IS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230439
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0229722, upper bound: 0.0233187
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0229722, upper bound: 0.0235710
IS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0228558, upper bound: 0.0224029
IS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0228000, upper bound: 0.0223748
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231216, upper bound: 0.0232314
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231216, upper bound: 0.0235019
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0228558, upper bound: 0.0224699
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0228000, upper bound: 0.0224538
IS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231216, upper bound: 0.0233211
IS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231216, upper bound: 0.0235976
IS_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0235172, upper bound: 0.0233995
IS_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0237089, upper bound: 0.0238397
IS_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0235172, upper bound: 0.0234943
IS_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0237089, upper bound: 0.0239963
IS_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0237639, upper bound: 0.0234053
IS_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0240263, upper bound: 0.0238564
IS_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0237639, upper bound: 0.0235009
IS_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0240263, upper bound: 0.0240263
IS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229990
IS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0233875, upper bound: 0.0234756
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230734
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0233875, upper bound: 0.0235733
IS_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0233720, upper bound: 0.0230013
IS_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0236946, upper bound: 0.0234957
IS_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0233720, upper bound: 0.0230742
IS_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0236946, upper bound: 0.0235998
IS_A2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232206, upper bound: 0.0230673
IS_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0230953
IS_A2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232206, upper bound: 0.0231580
IS_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232972, upper bound: 0.0231884
IS_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0232574
IS_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0234412
IS_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0233985
IS_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0236027
IS_A2_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0227716, upper bound: 0.0226321
IS_A2_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0227362, upper bound: 0.0225536
IS_A2_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0227716, upper bound: 0.0227154
IS_A2_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0227362, upper bound: 0.0226241
IS_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0233251
IS_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0235071
IS_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0235231
IS_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0237086
IS_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0234414, upper bound: 0.0230565
IS_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0230815
IS_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0234414, upper bound: 0.0230956
IS_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0235231, upper bound: 0.0231255
IS_A2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0232415
IS_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0234349
IS_A2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0233017
IS_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0235696
IS_A2_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0229252, upper bound: 0.0226086
IS_A2_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0229004, upper bound: 0.0225329
IS_A2_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0229252, upper bound: 0.0226382
IS_A2_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0229004, upper bound: 0.0225518
IS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0214420, upper bound: 0.0215061
IS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0231004, upper bound: 0.0233571
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0234940
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.26
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0236411

## BFS IS instance: IS_A1_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0015673, 0.0181572, 0.0019422, 0.0153030, -0.0137357, 0.0162150
1: -0.0045143, 0.0027346, -0.0031174, 0.0023815, -0.0068958, 0.0058520
2: 0.0020552, 0.0134933, 0.0041203, 0.0132861, -0.0112309, 0.0093731
3: -0.0037971, 0.0047282, -0.0035269, 0.0032478, -0.0070450, 0.0082551
4: -0.0038709, -0.0003659, -0.0033855, -0.0003529, -0.0035180, 0.0030197
5: -0.0003822, 0.0071909, 0.0008176, 0.0069713, -0.0073535, 0.0063733
6: -0.0086228, 0.0062311, -0.0076459, 0.0053598, -0.0139826, 0.0138770
7: -0.0167116, 0.0037163, -0.0138717, 0.0037827, -0.0204943, 0.0175880
8: 0.9806167, 0.9928443, 0.9822709, 0.9926553, -0.0120386, 0.0105734
9: -0.0084727, 0.0034433, -0.0085151, 0.0016261, -0.0100987, 0.0119583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226451, upper bound: 0.0223533
time: 1.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225882, upper bound: 0.0223298
time: 1.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0015673, 0.0181572, 0.0017492, 0.0155947, -0.0140274, 0.0164080
1: -0.0045143, 0.0027346, -0.0033071, 0.0025110, -0.0070253, 0.0060417
2: 0.0020552, 0.0134933, 0.0039192, 0.0133928, -0.0113377, 0.0095742
3: -0.0037971, 0.0047282, -0.0038364, 0.0035262, -0.0073233, 0.0085646
4: -0.0038709, -0.0003659, -0.0035399, -0.0000728, -0.0037981, 0.0031740
5: -0.0003822, 0.0071909, 0.0006732, 0.0070844, -0.0074666, 0.0065177
6: -0.0086228, 0.0062311, -0.0084003, 0.0058085, -0.0144313, 0.0146314
7: -0.0167116, 0.0037163, -0.0147489, 0.0052151, -0.0219267, 0.0184653
8: 0.9806167, 0.9928443, 0.9818404, 0.9935856, -0.0129690, 0.0110039
9: -0.0084727, 0.0034433, -0.0094311, 0.0021570, -0.0106296, 0.0128743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226451, upper bound: 0.0224027
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225882, upper bound: 0.0223740
time: 1.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0016612, 0.0155973, 0.0015672, 0.0224659, -0.0208047, 0.0140301
1: -0.0033247, 0.0024283, -0.0054990, 0.0035784, -0.0069031, 0.0079273
2: 0.0039085, 0.0134414, -0.0011560, 0.0134934, -0.0095849, 0.0145974
3: -0.0035545, 0.0035308, -0.0040413, 0.0057395, -0.0092940, 0.0075721
4: -0.0035879, -0.0003657, -0.0043430, -0.0003343, -0.0032536, 0.0039773
5: 0.0006931, 0.0071359, -0.0019948, 0.0071910, -0.0064979, 0.0091307
6: -0.0077547, 0.0060129, -0.0102755, 0.0062313, -0.0139860, 0.0162884
7: -0.0149530, 0.0037172, -0.0179409, 0.0047339, -0.0196869, 0.0216580
8: 0.9816443, 0.9926529, 0.9794595, 0.9951038, -0.0134596, 0.0131934
9: -0.0084732, 0.0023127, -0.0103746, 0.0042227, -0.0126959, 0.0126873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229546, upper bound: 0.0230950
time: 2.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229546, upper bound: 0.0232263
time: 1.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0015140, 0.0193209, 0.0015672, 0.0224659, -0.0209519, 0.0177537
1: -0.0050715, 0.0028500, -0.0054990, 0.0035784, -0.0086499, 0.0083491
2: 0.0012103, 0.0135228, -0.0011560, 0.0134934, -0.0122831, 0.0146788
3: -0.0038942, 0.0052939, -0.0040413, 0.0057395, -0.0096337, 0.0093352
4: -0.0040083, -0.0003584, -0.0043430, -0.0003343, -0.0036739, 0.0039846
5: -0.0008736, 0.0072222, -0.0019948, 0.0071910, -0.0080646, 0.0092170
6: -0.0090284, 0.0063551, -0.0102755, 0.0062313, -0.0152598, 0.0166305
7: -0.0175594, 0.0037546, -0.0179409, 0.0047339, -0.0222933, 0.0216955
8: 0.9798525, 0.9929526, 0.9794595, 0.9951038, -0.0152513, 0.0134931
9: -0.0084972, 0.0039893, -0.0103746, 0.0042227, -0.0127199, 0.0143639

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229546, upper bound: 0.0232433
time: 1.58 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229546, upper bound: 0.0233896
time: 1.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0015200, 0.0181953, 0.0015647, 0.0172707, -0.0157507, 0.0166305
1: -0.0045368, 0.0027421, -0.0037936, 0.0027800, -0.0073168, 0.0065358
2: 0.0020283, 0.0135195, 0.0026566, 0.0134948, -0.0114664, 0.0108629
3: -0.0038029, 0.0047802, -0.0038528, 0.0040948, -0.0078977, 0.0086330
4: -0.0039047, -0.0003661, -0.0038464, -0.0001930, -0.0037117, 0.0034774
5: -0.0003983, 0.0072186, 0.0000612, 0.0071924, -0.0075907, 0.0071574
6: -0.0086373, 0.0063410, -0.0088277, 0.0062371, -0.0148745, 0.0151687
7: -0.0169102, 0.0037149, -0.0159850, 0.0048788, -0.0217889, 0.0196999
8: 0.9804956, 0.9928480, 0.9814165, 0.9938838, -0.0133882, 0.0114315
9: -0.0084718, 0.0035579, -0.0095508, 0.0029137, -0.0113855, 0.0131087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229473
time: 2.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230439
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0015200, 0.0181953, 0.0014691, 0.0158447, -0.0143247, 0.0167262
1: -0.0045368, 0.0027421, -0.0035158, 0.0025187, -0.0070554, 0.0062579
2: 0.0020283, 0.0135195, 0.0037143, 0.0135477, -0.0115193, 0.0098051
3: -0.0038029, 0.0047802, -0.0037309, 0.0038546, -0.0076575, 0.0085111
4: -0.0039047, -0.0003661, -0.0037487, -0.0002238, -0.0036809, 0.0033268
5: -0.0003983, 0.0072186, 0.0005785, 0.0072485, -0.0076467, 0.0066401
6: -0.0086373, 0.0063410, -0.0083033, 0.0064595, -0.0150969, 0.0146443
7: -0.0169102, 0.0037149, -0.0159448, 0.0044427, -0.0213528, 0.0196597
8: 0.9804956, 0.9928480, 0.9812160, 0.9931722, -0.0126765, 0.0116320
9: -0.0084718, 0.0035579, -0.0089371, 0.0028752, -0.0112423, 0.0124950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0230333, upper bound: 0.0230071
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230439
time: 1.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0016170, 0.0156379, 0.0013948, 0.0247397, -0.0231227, 0.0142431
1: -0.0033473, 0.0024353, -0.0057236, 0.0045060, -0.0078533, 0.0081589
2: 0.0038793, 0.0134659, -0.0028433, 0.0135887, -0.0097094, 0.0163092
3: -0.0035600, 0.0035757, -0.0043528, 0.0060426, -0.0096025, 0.0079285
4: -0.0036205, -0.0003658, -0.0047612, -0.0001519, -0.0034686, 0.0043953
5: 0.0006764, 0.0071618, -0.0028188, 0.0072919, -0.0066155, 0.0099806
6: -0.0077695, 0.0061156, -0.0116798, 0.0066320, -0.0144015, 0.0177954
7: -0.0151216, 0.0037165, -0.0189300, 0.0065168, -0.0216384, 0.0226465
8: 0.9815459, 0.9926569, 0.9787811, 0.9978203, -0.0162745, 0.0138758
9: -0.0084728, 0.0024216, -0.0132895, 0.0048295, -0.0133023, 0.0157110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229546, upper bound: 0.0233188
time: 1.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229546, upper bound: 0.0233188
time: 2.17 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0014627, 0.0193579, 0.0013948, 0.0247397, -0.0232770, 0.0179631
1: -0.0050950, 0.0028583, -0.0057236, 0.0045060, -0.0096010, 0.0085820
2: 0.0011853, 0.0135512, -0.0028433, 0.0135887, -0.0124034, 0.0163945
3: -0.0039005, 0.0053470, -0.0043528, 0.0060426, -0.0099431, 0.0096998
4: -0.0040476, -0.0003587, -0.0047612, -0.0001519, -0.0038957, 0.0044025
5: -0.0008894, 0.0072522, -0.0028188, 0.0072919, -0.0081814, 0.0100710
6: -0.0090429, 0.0064742, -0.0116798, 0.0066320, -0.0156749, 0.0181540
7: -0.0177883, 0.0037529, -0.0189300, 0.0065168, -0.0243050, 0.0226829
8: 0.9797174, 0.9929560, 0.9787811, 0.9978203, -0.0181029, 0.0141749
9: -0.0084961, 0.0041260, -0.0132895, 0.0048295, -0.0133255, 0.0174155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229546, upper bound: 0.0234650
time: 2.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229546, upper bound: 0.0234650
time: 1.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0014610, 0.0183484, 0.0016808, 0.0163178, -0.0148568, 0.0166676
1: -0.0046425, 0.0028780, -0.0037014, 0.0025343, -0.0071768, 0.0065794
2: 0.0019233, 0.0135521, 0.0033662, 0.0134306, -0.0115073, 0.0101859
3: -0.0041168, 0.0049535, -0.0036526, 0.0039491, -0.0080659, 0.0086061
4: -0.0039685, -0.0000870, -0.0036440, -0.0003466, -0.0036219, 0.0035570
5: -0.0004729, 0.0072532, 0.0004098, 0.0071244, -0.0075973, 0.0068434
6: -0.0092363, 0.0064781, -0.0080043, 0.0059674, -0.0152037, 0.0144824
7: -0.0173342, 0.0051421, -0.0153827, 0.0038147, -0.0211489, 0.0205249
8: 0.9802684, 0.9937229, 0.9816881, 0.9927551, -0.0124868, 0.0120348
9: -0.0093844, 0.0037940, -0.0085356, 0.0025576, -0.0119420, 0.0123296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228558, upper bound: 0.0223474
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228558, upper bound: 0.0223656
time: 1.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0014865, 0.0182202, 0.0017194, 0.0162051, -0.0147186, 0.0165008
1: -0.0045492, 0.0028844, -0.0036331, 0.0025173, -0.0070665, 0.0065175
2: 0.0020257, 0.0135380, 0.0034472, 0.0134093, -0.0113835, 0.0100908
3: -0.0041664, 0.0048318, -0.0036372, 0.0038641, -0.0080306, 0.0084690
4: -0.0039360, -0.0000267, -0.0036078, -0.0003492, -0.0035868, 0.0035810
5: -0.0004262, 0.0072383, 0.0004570, 0.0071018, -0.0075280, 0.0067812
6: -0.0093174, 0.0064190, -0.0079582, 0.0058776, -0.0151951, 0.0143772
7: -0.0170962, 0.0054505, -0.0151639, 0.0038013, -0.0208975, 0.0206144
8: 0.9804052, 0.9939026, 0.9817741, 0.9927368, -0.0123315, 0.0121285
9: -0.0095816, 0.0036626, -0.0085270, 0.0024268, -0.0120083, 0.0121896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228000, upper bound: 0.0223257
time: 2.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228000, upper bound: 0.0223435
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0014412, 0.0159282, 0.0015091, 0.0219892, -0.0205480, 0.0144191
1: -0.0035367, 0.0025689, -0.0055345, 0.0033919, -0.0069286, 0.0081034
2: 0.0036779, 0.0135630, -0.0007980, 0.0135255, -0.0098476, 0.0143611
3: -0.0038700, 0.0038485, -0.0040341, 0.0058204, -0.0096904, 0.0078826
4: -0.0037597, -0.0000895, -0.0043191, -0.0003341, -0.0034256, 0.0042296
5: 0.0005348, 0.0072648, -0.0018378, 0.0072250, -0.0066902, 0.0091026
6: -0.0085080, 0.0065242, -0.0100797, 0.0063663, -0.0148743, 0.0166039
7: -0.0159367, 0.0051298, -0.0182225, 0.0045345, -0.0204712, 0.0233523
8: 0.9811538, 0.9935728, 0.9792914, 0.9945677, -0.0134139, 0.0142814
9: -0.0093765, 0.0029037, -0.0098364, 0.0043818, -0.0137583, 0.0127401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231118, upper bound: 0.0232314
time: 1.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231118, upper bound: 0.0232314
time: 1.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0012839, 0.0198291, 0.0015091, 0.0219892, -0.0207053, 0.0183199
1: -0.0053810, 0.0030531, -0.0055345, 0.0033919, -0.0087729, 0.0085877
2: 0.0008428, 0.0136500, -0.0007980, 0.0135255, -0.0126827, 0.0144481
3: -0.0042689, 0.0057323, -0.0040341, 0.0058204, -0.0100892, 0.0097664
4: -0.0042151, -0.0000691, -0.0043191, -0.0003341, -0.0038742, 0.0042500
5: -0.0010942, 0.0073569, -0.0018378, 0.0072250, -0.0083192, 0.0091948
6: -0.0097211, 0.0068899, -0.0100797, 0.0063663, -0.0160874, 0.0169696
7: -0.0187742, 0.0052338, -0.0182225, 0.0045345, -0.0233087, 0.0234564
8: 0.9790248, 0.9938841, 0.9792914, 0.9945677, -0.0155429, 0.0145928
9: -0.0094430, 0.0047278, -0.0098364, 0.0043818, -0.0138248, 0.0145642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231107, upper bound: 0.0232393
time: 1.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231107, upper bound: 0.0233565
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0013889, 0.0184283, 0.0014958, 0.0178574, -0.0164686, 0.0169325
1: -0.0046922, 0.0028928, -0.0039162, 0.0029263, -0.0076185, 0.0068090
2: 0.0018657, 0.0135920, 0.0022327, 0.0135329, -0.0116672, 0.0113593
3: -0.0041282, 0.0050337, -0.0039162, 0.0042499, -0.0083780, 0.0089499
4: -0.0040273, -0.0000875, -0.0039474, -0.0001688, -0.0038585, 0.0038599
5: -0.0005066, 0.0072955, -0.0001699, 0.0072328, -0.0077394, 0.0074654
6: -0.0092637, 0.0066459, -0.0091160, 0.0063974, -0.0156611, 0.0157619
7: -0.0176694, 0.0051397, -0.0163189, 0.0051320, -0.0228014, 0.0214585
8: 0.9800597, 0.9937285, 0.9811468, 0.9942632, -0.0142035, 0.0125817
9: -0.0093828, 0.0039998, -0.0098767, 0.0031393, -0.0125221, 0.0138766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228558, upper bound: 0.0223857
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228558, upper bound: 0.0224108
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0014132, 0.0183137, 0.0015375, 0.0166386, -0.0152254, 0.0167762
1: -0.0046053, 0.0029009, -0.0038502, 0.0026404, -0.0072457, 0.0067511
2: 0.0019584, 0.0135786, 0.0031492, 0.0135098, -0.0115514, 0.0104293
3: -0.0041791, 0.0049226, -0.0038662, 0.0041673, -0.0083464, 0.0087887
4: -0.0039908, -0.0000273, -0.0037697, -0.0001714, -0.0038194, 0.0037424
5: -0.0004656, 0.0072812, 0.0002513, 0.0072084, -0.0076740, 0.0070299
6: -0.0093487, 0.0065894, -0.0086662, 0.0063004, -0.0156491, 0.0152555
7: -0.0174361, 0.0054478, -0.0160970, 0.0047109, -0.0221470, 0.0215448
8: 0.9801956, 0.9939089, 0.9812951, 0.9934055, -0.0132099, 0.0126138
9: -0.0095798, 0.0038631, -0.0091086, 0.0030031, -0.0125829, 0.0129717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228000, upper bound: 0.0223727
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228000, upper bound: 0.0223962
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0013707, 0.0160002, 0.0013031, 0.0243515, -0.0229808, 0.0146971
1: -0.0035793, 0.0025826, -0.0057979, 0.0043057, -0.0078850, 0.0083805
2: 0.0036261, 0.0136020, -0.0025611, 0.0136394, -0.0100134, 0.0161631
3: -0.0038807, 0.0039270, -0.0043528, 0.0061708, -0.0100515, 0.0082798
4: -0.0038131, -0.0000897, -0.0047602, -0.0001517, -0.0036614, 0.0046705
5: 0.0005053, 0.0073061, -0.0026924, 0.0073457, -0.0068404, 0.0099985
6: -0.0085318, 0.0066881, -0.0115196, 0.0068452, -0.0153770, 0.0182077
7: -0.0162446, 0.0051285, -0.0193342, 0.0062967, -0.0225413, 0.0244626
8: 0.9809965, 0.9935784, 0.9785060, 0.9972782, -0.0162817, 0.0150724
9: -0.0093756, 0.0030871, -0.0126341, 0.0050801, -0.0144558, 0.0157211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231118, upper bound: 0.0233211
time: 2.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231118, upper bound: 0.0233211
time: 1.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0012072, 0.0199321, 0.0013031, 0.0243515, -0.0231443, 0.0186290
1: -0.0054452, 0.0030713, -0.0057979, 0.0043057, -0.0097509, 0.0088692
2: 0.0007674, 0.0136924, -0.0025611, 0.0136394, -0.0128720, 0.0162535
3: -0.0042829, 0.0058271, -0.0043528, 0.0061708, -0.0104537, 0.0101799
4: -0.0042783, -0.0000696, -0.0047602, -0.0001517, -0.0041266, 0.0046906
5: -0.0011382, 0.0074018, -0.0026924, 0.0073457, -0.0084839, 0.0100942
6: -0.0097581, 0.0070680, -0.0115196, 0.0068452, -0.0166033, 0.0185876
7: -0.0191417, 0.0052312, -0.0193342, 0.0062967, -0.0254385, 0.0245654
8: 0.9787950, 0.9938915, 0.9785060, 0.9972782, -0.0184832, 0.0153856
9: -0.0094414, 0.0049518, -0.0126341, 0.0050801, -0.0145215, 0.0175859

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231118, upper bound: 0.0234961
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231118, upper bound: 0.0234961
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0015774, 0.0181404, 0.0016886, 0.0236764, -0.0220990, 0.0164518
1: -0.0045073, 0.0027318, -0.0043027, 0.0050897, -0.0095970, 0.0070344
2: 0.0020673, 0.0134878, -0.0020706, 0.0134263, -0.0113590, 0.0155584
3: -0.0037944, 0.0047217, -0.0040673, 0.0044433, -0.0082378, 0.0087890
4: -0.0038607, -0.0003665, -0.0044844, -0.0002776, -0.0035831, 0.0041179
5: -0.0003749, 0.0071850, -0.0022721, 0.0071199, -0.0074947, 0.0094571
6: -0.0086161, 0.0062078, -0.0114347, 0.0059492, -0.0145653, 0.0176424
7: -0.0166569, 0.0037130, -0.0159437, 0.0069995, -0.0236563, 0.0196567
8: 0.9806543, 0.9928415, 0.9810474, 0.9995902, -0.0189359, 0.0117941
9: -0.0084705, 0.0034106, -0.0156272, 0.0029961, -0.0114666, 0.0190378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234729, upper bound: 0.0233794
time: 1.45 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234962, upper bound: 0.0233800
time: 2.65 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0014842, 0.0200396, 0.0015644, 0.0283563, -0.0268721, 0.0184752
1: -0.0053923, 0.0029144, -0.0060334, 0.0057424, -0.0111347, 0.0089477
2: 0.0006875, 0.0135393, -0.0054724, 0.0134950, -0.0128075, 0.0190117
3: -0.0039476, 0.0056164, -0.0043715, 0.0061964, -0.0101440, 0.0099879
4: -0.0040873, -0.0003551, -0.0050359, -0.0002711, -0.0038162, 0.0046808
5: -0.0011783, 0.0072396, -0.0041816, 0.0071926, -0.0083709, 0.0114212
6: -0.0093110, 0.0064243, -0.0133143, 0.0062380, -0.0155490, 0.0197386
7: -0.0180353, 0.0037714, -0.0185088, 0.0076992, -0.0257345, 0.0222802
8: 0.9793888, 0.9930120, 0.9787133, 1.0013219, -0.0219331, 0.0142986
9: -0.0085079, 0.0043068, -0.0171338, 0.0046443, -0.0131522, 0.0214406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233255, upper bound: 0.0236101
time: 1.40 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233255, upper bound: 0.0238397
time: 1.91 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0015255, 0.0181785, 0.0015180, 0.0259238, -0.0243983, 0.0166606
1: -0.0045298, 0.0027397, -0.0044879, 0.0061129, -0.0106426, 0.0072276
2: 0.0020404, 0.0135165, -0.0037268, 0.0135206, -0.0114802, 0.0172433
3: -0.0038005, 0.0047737, -0.0043797, 0.0046994, -0.0084999, 0.0091534
4: -0.0038972, -0.0003668, -0.0048754, -0.0001005, -0.0037967, 0.0045086
5: -0.0003910, 0.0072154, -0.0030912, 0.0072198, -0.0076108, 0.0103067
6: -0.0086306, 0.0063284, -0.0128523, 0.0063458, -0.0149764, 0.0191807
7: -0.0168479, 0.0037115, -0.0167740, 0.0092478, -0.0260957, 0.0204856
8: 0.9805191, 0.9928451, 0.9804804, 1.0026748, -0.0221558, 0.0123647
9: -0.0084696, 0.0035318, -0.0191186, 0.0034978, -0.0119674, 0.0226504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234729, upper bound: 0.0234741
time: 1.76 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0234962, upper bound: 0.0234743
time: 1.75 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0014283, 0.0200659, 0.0013734, 0.0306145, -0.0291862, 0.0186925
1: -0.0054108, 0.0029228, -0.0062428, 0.0067632, -0.0121739, 0.0091656
2: 0.0006710, 0.0135702, -0.0071420, 0.0136006, -0.0129296, 0.0207122
3: -0.0039540, 0.0056651, -0.0047065, 0.0064818, -0.0104358, 0.0103715
4: -0.0041251, -0.0003554, -0.0054440, -0.0000802, -0.0040449, 0.0050886
5: -0.0011895, 0.0072723, -0.0050012, 0.0073045, -0.0084940, 0.0122736
6: -0.0093217, 0.0065541, -0.0146960, 0.0066819, -0.0160036, 0.0212501
7: -0.0182518, 0.0037698, -0.0194875, 0.0099737, -0.0282255, 0.0232573
8: 0.9792523, 0.9930148, 0.9780244, 1.0044276, -0.0251753, 0.0149904
9: -0.0085069, 0.0044327, -0.0206986, 0.0052501, -0.0137570, 0.0251313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233255, upper bound: 0.0237565
time: 1.54 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233255, upper bound: 0.0239963
time: 1.55 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0013573, 0.0185432, 0.0016327, 0.0233535, -0.0219962, 0.0169105
1: -0.0047633, 0.0029131, -0.0042853, 0.0049696, -0.0097329, 0.0071983
2: 0.0017827, 0.0136095, -0.0018308, 0.0134572, -0.0116745, 0.0154403
3: -0.0041525, 0.0050998, -0.0040554, 0.0044607, -0.0086132, 0.0091552
4: -0.0040469, -0.0000764, -0.0044766, -0.0002771, -0.0037697, 0.0044002
5: -0.0005558, 0.0073139, -0.0021546, 0.0071526, -0.0077084, 0.0094686
6: -0.0093366, 0.0067192, -0.0113068, 0.0060790, -0.0154157, 0.0180261
7: -0.0177505, 0.0051966, -0.0161111, 0.0068460, -0.0245964, 0.0213077
8: 0.9799369, 0.9937780, 0.9809665, 0.9992557, -0.0193188, 0.0128115
9: -0.0094192, 0.0040641, -0.0152533, 0.0030902, -0.0125093, 0.0193174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237012, upper bound: 0.0233849
time: 1.57 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237426, upper bound: 0.0233854
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0012597, 0.0205037, 0.0015069, 0.0279930, -0.0267333, 0.0189968
1: -0.0056828, 0.0031171, -0.0060197, 0.0056086, -0.0112915, 0.0091369
2: 0.0003508, 0.0136634, -0.0052083, 0.0135268, -0.0131759, 0.0188717
3: -0.0043219, 0.0060376, -0.0043629, 0.0062376, -0.0105595, 0.0104005
4: -0.0042873, -0.0000660, -0.0050271, -0.0002710, -0.0040163, 0.0049610
5: -0.0013818, 0.0073711, -0.0040497, 0.0072263, -0.0086081, 0.0114208
6: -0.0099841, 0.0069461, -0.0131658, 0.0063716, -0.0163557, 0.0201119
7: -0.0192196, 0.0052496, -0.0187255, 0.0074966, -0.0267163, 0.0239751
8: 0.9785941, 0.9939346, 0.9786094, 1.0009217, -0.0223277, 0.0153251
9: -0.0094531, 0.0050185, -0.0166716, 0.0047656, -0.0142187, 0.0216901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235009, upper bound: 0.0236169
time: 1.57 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235009, upper bound: 0.0238564
time: 2.06 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0012831, 0.0186252, 0.0014381, 0.0256464, -0.0243633, 0.0171871
1: -0.0048144, 0.0029285, -0.0044899, 0.0060007, -0.0108151, 0.0074184
2: 0.0017236, 0.0136505, -0.0035259, 0.0135648, -0.0118412, 0.0171764
3: -0.0041643, 0.0051850, -0.0043721, 0.0047444, -0.0089087, 0.0095570
4: -0.0041068, -0.0000768, -0.0048858, -0.0001002, -0.0040067, 0.0048089
5: -0.0005905, 0.0073574, -0.0029936, 0.0072666, -0.0078571, 0.0103510
6: -0.0093649, 0.0068917, -0.0127425, 0.0065315, -0.0158964, 0.0196342
7: -0.0180844, 0.0051943, -0.0170543, 0.0090874, -0.0271718, 0.0222486
8: 0.9797275, 0.9937841, 0.9803259, 1.0023599, -0.0226324, 0.0134581
9: -0.0094177, 0.0042683, -0.0187428, 0.0036680, -0.0130857, 0.0230111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237012, upper bound: 0.0234811
time: 1.88 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0237426, upper bound: 0.0234811
time: 1.57 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0011833, 0.0205999, 0.0012905, 0.0303581, -0.0291747, 0.0193094
1: -0.0057437, 0.0031346, -0.0062744, 0.0066443, -0.0123880, 0.0094090
2: 0.0002807, 0.0137056, -0.0069626, 0.0136464, -0.0133657, 0.0206683
3: -0.0043353, 0.0061292, -0.0047058, 0.0065707, -0.0109061, 0.0108350
4: -0.0043507, -0.0000665, -0.0054619, -0.0000803, -0.0042704, 0.0053954
5: -0.0014229, 0.0074159, -0.0049094, 0.0073531, -0.0087760, 0.0123253
6: -0.0100193, 0.0071236, -0.0145817, 0.0068746, -0.0168938, 0.0217053
7: -0.0195764, 0.0052471, -0.0198368, 0.0097833, -0.0293597, 0.0250839
8: 0.9783548, 0.9939417, 0.9778198, 1.0040615, -0.0257066, 0.0161219
9: -0.0094515, 0.0052387, -0.0202569, 0.0054589, -0.0149104, 0.0254956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235009, upper bound: 0.0237639
time: 1.75 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0235009, upper bound: 0.0240263
time: 1.57 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0015774, 0.0181404, 0.0016103, 0.0188797, -0.0173023, 0.0165301
1: -0.0045073, 0.0027318, -0.0040562, 0.0031128, -0.0076201, 0.0067880
2: 0.0020673, 0.0134878, 0.0015003, 0.0134696, -0.0114023, 0.0119875
3: -0.0037944, 0.0047217, -0.0038284, 0.0042309, -0.0080254, 0.0085501
4: -0.0038607, -0.0003665, -0.0039423, -0.0003178, -0.0035430, 0.0035758
5: -0.0003749, 0.0071850, -0.0005782, 0.0071657, -0.0075406, 0.0077632
6: -0.0086161, 0.0062078, -0.0093467, 0.0061313, -0.0147474, 0.0155545
7: -0.0166569, 0.0037130, -0.0159449, 0.0046342, -0.0212911, 0.0196579
8: 0.9806543, 0.9928415, 0.9811572, 0.9944195, -0.0137652, 0.0116844
9: -0.0084705, 0.0034106, -0.0098963, 0.0029771, -0.0114477, 0.0133069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0213756, upper bound: 0.0214271
time: 1.15 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229463, upper bound: 0.0228448
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0014842, 0.0200396, 0.0014813, 0.0233074, -0.0218232, 0.0185583
1: -0.0053923, 0.0029144, -0.0057353, 0.0037364, -0.0091287, 0.0086496
2: 0.0006875, 0.0135393, -0.0017302, 0.0135409, -0.0128534, 0.0152695
3: -0.0039476, 0.0056164, -0.0041320, 0.0059346, -0.0098822, 0.0097484
4: -0.0040873, -0.0003551, -0.0044421, -0.0003122, -0.0037751, 0.0040871
5: -0.0011783, 0.0072396, -0.0023852, 0.0072413, -0.0084196, 0.0096248
6: -0.0093110, 0.0064243, -0.0110651, 0.0064311, -0.0157421, 0.0174894
7: -0.0180353, 0.0037714, -0.0184473, 0.0050708, -0.0231062, 0.0222187
8: 0.9793888, 0.9930120, 0.9789192, 0.9957267, -0.0163379, 0.0140927
9: -0.0085079, 0.0043068, -0.0108534, 0.0045887, -0.0130966, 0.0151601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229681, upper bound: 0.0232279
time: 2.42 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229681, upper bound: 0.0234756
time: 1.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0015255, 0.0181785, 0.0014393, 0.0210302, -0.0195048, 0.0167393
1: -0.0045298, 0.0027397, -0.0042315, 0.0039965, -0.0085262, 0.0069713
2: 0.0020404, 0.0135165, -0.0000846, 0.0135641, -0.0115237, 0.0136011
3: -0.0038005, 0.0047737, -0.0041256, 0.0044824, -0.0082829, 0.0088993
4: -0.0038972, -0.0003668, -0.0042992, -0.0001374, -0.0037598, 0.0039325
5: -0.0003910, 0.0072154, -0.0013612, 0.0072659, -0.0076569, 0.0085766
6: -0.0086306, 0.0063284, -0.0107383, 0.0065287, -0.0151593, 0.0170667
7: -0.0168479, 0.0037115, -0.0167710, 0.0064836, -0.0233314, 0.0204825
8: 0.9805191, 0.9928451, 0.9806131, 0.9970632, -0.0165441, 0.0122321
9: -0.0084696, 0.0035318, -0.0126162, 0.0034738, -0.0119434, 0.0161480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 128

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0229836
time: 2.30 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231244, upper bound: 0.0230734
time: 1.78 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0014283, 0.0200659, 0.0012856, 0.0255368, -0.0241084, 0.0187803
1: -0.0054108, 0.0029228, -0.0059476, 0.0046597, -0.0100705, 0.0088704
2: 0.0006710, 0.0135702, -0.0033740, 0.0136491, -0.0129781, 0.0169441
3: -0.0039540, 0.0056651, -0.0044682, 0.0062250, -0.0101790, 0.0101332
4: -0.0041251, -0.0003554, -0.0048479, -0.0001163, -0.0040088, 0.0044925
5: -0.0011895, 0.0072723, -0.0031876, 0.0073559, -0.0085454, 0.0104600
6: -0.0093217, 0.0065541, -0.0124406, 0.0068858, -0.0162075, 0.0189947
7: -0.0182518, 0.0037698, -0.0194370, 0.0070232, -0.0252750, 0.0232069
8: 0.9792523, 0.9930148, 0.9782242, 0.9985163, -0.0192640, 0.0147905
9: -0.0085069, 0.0044327, -0.0138329, 0.0051984, -0.0137053, 0.0182656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229681, upper bound: 0.0233251
time: 1.22 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0229681, upper bound: 0.0235733
time: 2.06 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0013573, 0.0185432, 0.0015476, 0.0183484, -0.0169911, 0.0169956
1: -0.0047633, 0.0029131, -0.0040136, 0.0029542, -0.0077175, 0.0069267
2: 0.0017827, 0.0136095, 0.0019031, 0.0135043, -0.0117216, 0.0117064
3: -0.0041525, 0.0050998, -0.0038058, 0.0042255, -0.0083779, 0.0089056
4: -0.0040469, -0.0000764, -0.0039191, -0.0003173, -0.0037295, 0.0038427
5: -0.0005558, 0.0073139, -0.0003938, 0.0072025, -0.0077583, 0.0077077
6: -0.0093366, 0.0067192, -0.0091408, 0.0062770, -0.0156136, 0.0158601
7: -0.0177505, 0.0051966, -0.0160960, 0.0044521, -0.0222026, 0.0212925
8: 0.9799369, 0.9937780, 0.9810829, 0.9940116, -0.0140747, 0.0126951
9: -0.0094192, 0.0040641, -0.0095289, 0.0030603, -0.0124795, 0.0135930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0215034, upper bound: 0.0214331
time: 1.28 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232135, upper bound: 0.0228504
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0012597, 0.0205037, 0.0014170, 0.0227853, -0.0215256, 0.0190867
1: -0.0056828, 0.0031171, -0.0057031, 0.0035737, -0.0092566, 0.0088202
2: 0.0003508, 0.0136634, -0.0013406, 0.0135765, -0.0132256, 0.0150040
3: -0.0043219, 0.0060376, -0.0041173, 0.0059581, -0.0102799, 0.0101549
4: -0.0042873, -0.0000660, -0.0044224, -0.0003121, -0.0039752, 0.0043564
5: -0.0013818, 0.0073711, -0.0022043, 0.0072790, -0.0086608, 0.0095754
6: -0.0099841, 0.0069461, -0.0108529, 0.0065805, -0.0165646, 0.0177990
7: -0.0192196, 0.0052496, -0.0186484, 0.0048729, -0.0240926, 0.0238980
8: 0.9785941, 0.9939346, 0.9788374, 0.9952489, -0.0166548, 0.0150971
9: -0.0094531, 0.0050185, -0.0103619, 0.0047041, -0.0141572, 0.0153804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231161, upper bound: 0.0232325
time: 1.70 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231161, upper bound: 0.0234957
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0012831, 0.0186252, 0.0013556, 0.0205487, -0.0192656, 0.0172696
1: -0.0048144, 0.0029285, -0.0042138, 0.0038159, -0.0086303, 0.0071423
2: 0.0017236, 0.0136505, 0.0002763, 0.0136104, -0.0118868, 0.0133742
3: -0.0041643, 0.0051850, -0.0041071, 0.0045076, -0.0086719, 0.0092921
4: -0.0041068, -0.0000768, -0.0042976, -0.0001370, -0.0039699, 0.0042208
5: -0.0005905, 0.0073574, -0.0011958, 0.0073149, -0.0079054, 0.0085532
6: -0.0093649, 0.0068917, -0.0105498, 0.0067231, -0.0160880, 0.0174415
7: -0.0180844, 0.0051943, -0.0170409, 0.0062859, -0.0243703, 0.0222352
8: 0.9797275, 0.9937841, 0.9804706, 0.9966123, -0.0168847, 0.0133135
9: -0.0094177, 0.0042683, -0.0121283, 0.0036372, -0.0130549, 0.0163966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_B2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233011, upper bound: 0.0230484
time: 3.08 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233552, upper bound: 0.0230545
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0011833, 0.0205999, 0.0011968, 0.0250964, -0.0239131, 0.0194031
1: -0.0057437, 0.0031346, -0.0059514, 0.0044970, -0.0102407, 0.0090860
2: 0.0002807, 0.0137056, -0.0030499, 0.0136982, -0.0134175, 0.0167556
3: -0.0043353, 0.0061292, -0.0044601, 0.0062904, -0.0106258, 0.0105893
4: -0.0043507, -0.0000665, -0.0048508, -0.0001164, -0.0042343, 0.0047843
5: -0.0014229, 0.0074159, -0.0030375, 0.0074079, -0.0088309, 0.0104534
6: -0.0100193, 0.0071236, -0.0122671, 0.0070922, -0.0171115, 0.0193908
7: -0.0195764, 0.0052471, -0.0197631, 0.0068237, -0.0264001, 0.0250102
8: 0.9783548, 0.9939417, 0.9780465, 0.9980614, -0.0197065, 0.0158951
9: -0.0094515, 0.0052387, -0.0133074, 0.0053934, -0.0148449, 0.0185461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231161, upper bound: 0.0233291
time: 2.67 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231161, upper bound: 0.0235998
time: 1.45 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0014189, 0.0174732, 0.0017796, 0.0149330, -0.0135141, 0.0156937
1: -0.0042208, 0.0026267, -0.0029647, 0.0022724, -0.0064932, 0.0055914
2: 0.0025513, 0.0135754, 0.0043745, 0.0133760, -0.0108247, 0.0092009
3: -0.0036187, 0.0044685, -0.0032701, 0.0030690, -0.0066877, 0.0077386
4: -0.0039018, -0.0004917, -0.0034373, -0.0005834, -0.0033184, 0.0029457
5: -0.0000922, 0.0072779, 0.0009912, 0.0070666, -0.0071588, 0.0062867
6: -0.0081106, 0.0065761, -0.0069494, 0.0057378, -0.0138484, 0.0135255
7: -0.0167785, 0.0030730, -0.0139952, 0.0026041, -0.0193826, 0.0170683
8: 0.9806482, 0.9923922, 0.9819084, 0.9918678, -0.0112196, 0.0104838
9: -0.0080613, 0.0034867, -0.0077615, 0.0017605, -0.0098219, 0.0112482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0212845, upper bound: 0.0213141
time: 1.40 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0230600, upper bound: 0.0228749
time: 1.37 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0013958, 0.0175714, 0.0017752, 0.0148948, -0.0134990, 0.0157963
1: -0.0042608, 0.0026694, -0.0029343, 0.0023061, -0.0065668, 0.0056038
2: 0.0024824, 0.0135882, 0.0044133, 0.0133784, -0.0108961, 0.0091749
3: -0.0037121, 0.0045123, -0.0034119, 0.0030713, -0.0067834, 0.0079241
4: -0.0039258, -0.0004111, -0.0034491, -0.0004276, -0.0034982, 0.0030380
5: -0.0001358, 0.0072914, 0.0009895, 0.0070691, -0.0072049, 0.0063018
6: -0.0083344, 0.0066297, -0.0073195, 0.0057479, -0.0140824, 0.0139492
7: -0.0169179, 0.0034850, -0.0141144, 0.0034006, -0.0203185, 0.0175994
8: 0.9805515, 0.9926616, 0.9818985, 0.9923787, -0.0118272, 0.0107630
9: -0.0083248, 0.0035736, -0.0082708, 0.0017999, -0.0101247, 0.0118444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0215044, upper bound: 0.0214381
time: 1.23 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231455, upper bound: 0.0229096
time: 2.04 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0014189, 0.0174732, 0.0015714, 0.0152313, -0.0138125, 0.0159018
1: -0.0042208, 0.0026267, -0.0031610, 0.0023966, -0.0066174, 0.0057877
2: 0.0025513, 0.0135754, 0.0041707, 0.0134910, -0.0109398, 0.0094047
3: -0.0036187, 0.0044685, -0.0035767, 0.0033650, -0.0069837, 0.0080452
4: -0.0039018, -0.0004917, -0.0036069, -0.0003017, -0.0036001, 0.0031153
5: -0.0000922, 0.0072779, 0.0008430, 0.0071885, -0.0072807, 0.0064349
6: -0.0081106, 0.0065761, -0.0076939, 0.0062215, -0.0143321, 0.0142700
7: -0.0167785, 0.0030730, -0.0149754, 0.0040446, -0.0208231, 0.0180485
8: 0.9806482, 0.9923922, 0.9814442, 0.9928122, -0.0121641, 0.0109480
9: -0.0080613, 0.0034867, -0.0086826, 0.0023418, -0.0104032, 0.0121693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226734, upper bound: 0.0225976
time: 1.69 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226160, upper bound: 0.0224870
time: 1.59 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0013958, 0.0175714, 0.0015787, 0.0151837, -0.0137879, 0.0159927
1: -0.0042608, 0.0026694, -0.0031186, 0.0024292, -0.0066899, 0.0057880
2: 0.0024824, 0.0135882, 0.0042152, 0.0134870, -0.0110047, 0.0093729
3: -0.0037121, 0.0045123, -0.0037158, 0.0033449, -0.0070570, 0.0082281
4: -0.0039258, -0.0004111, -0.0036026, -0.0001482, -0.0037775, 0.0031916
5: -0.0001358, 0.0072914, 0.0008442, 0.0071842, -0.0073200, 0.0064471
6: -0.0083344, 0.0066297, -0.0080818, 0.0062046, -0.0145390, 0.0147115
7: -0.0169179, 0.0034850, -0.0149615, 0.0048291, -0.0217470, 0.0184466
8: 0.9805515, 0.9926616, 0.9814605, 0.9933137, -0.0127622, 0.0112011
9: -0.0083248, 0.0035736, -0.0091842, 0.0023233, -0.0106481, 0.0127578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0214434, upper bound: 0.0214650
time: 1.16 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231160, upper bound: 0.0230076
time: 1.66 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0014719, 0.0150479, 0.0016211, 0.0186549, -0.0171830, 0.0134268
1: -0.0030834, 0.0023442, -0.0046922, 0.0027757, -0.0058591, 0.0070364
2: 0.0043019, 0.0135461, 0.0017002, 0.0134636, -0.0091617, 0.0118458
3: -0.0034669, 0.0033322, -0.0038322, 0.0048421, -0.0083090, 0.0071644
4: -0.0036585, -0.0003941, -0.0038687, -0.0003626, -0.0032959, 0.0034747
5: 0.0009124, 0.0072468, -0.0005922, 0.0071594, -0.0062470, 0.0078390
6: -0.0075381, 0.0064529, -0.0087730, 0.0061061, -0.0136443, 0.0152259
7: -0.0152423, 0.0035721, -0.0167218, 0.0037331, -0.0189754, 0.0202939
8: 0.9812222, 0.9925228, 0.9804882, 0.9928889, -0.0116667, 0.0120345
9: -0.0083805, 0.0025002, -0.0084834, 0.0034711, -0.0118515, 0.0109836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0230989, upper bound: 0.0231760
time: 1.27 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231345, upper bound: 0.0232575
time: 1.45 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0013338, 0.0187342, 0.0016211, 0.0186549, -0.0173210, 0.0171131
1: -0.0048226, 0.0028038, -0.0046922, 0.0027757, -0.0075983, 0.0074960
2: 0.0016381, 0.0136224, 0.0017002, 0.0134636, -0.0118255, 0.0119222
3: -0.0038384, 0.0050802, -0.0038322, 0.0048421, -0.0086805, 0.0089123
4: -0.0040677, -0.0003842, -0.0038687, -0.0003626, -0.0037051, 0.0034845
5: -0.0006256, 0.0073277, -0.0005922, 0.0071594, -0.0077850, 0.0079199
6: -0.0087945, 0.0067738, -0.0087730, 0.0061061, -0.0149006, 0.0155467
7: -0.0177859, 0.0036224, -0.0167218, 0.0037331, -0.0215190, 0.0203442
8: 0.9797962, 0.9928436, 0.9804882, 0.9928889, -0.0130926, 0.0123554
9: -0.0084126, 0.0041339, -0.0084834, 0.0034711, -0.0118837, 0.0126173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225787, upper bound: 0.0230619
time: 1.29 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225546, upper bound: 0.0229815
time: 1.95 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0014719, 0.0150479, 0.0014160, 0.0190831, -0.0176112, 0.0136319
1: -0.0030834, 0.0023442, -0.0049538, 0.0029644, -0.0060478, 0.0072980
2: 0.0043019, 0.0135461, 0.0013887, 0.0135770, -0.0092751, 0.0121574
3: -0.0034669, 0.0033322, -0.0041981, 0.0052235, -0.0086904, 0.0075303
4: -0.0036585, -0.0003941, -0.0040499, -0.0000696, -0.0035888, 0.0036559
5: 0.0009124, 0.0072468, -0.0007809, 0.0072795, -0.0063671, 0.0080277
6: -0.0075381, 0.0064529, -0.0094841, 0.0065828, -0.0141209, 0.0159370
7: -0.0152423, 0.0035721, -0.0177665, 0.0052312, -0.0204735, 0.0213386
8: 0.9812222, 0.9925228, 0.9797751, 0.9938425, -0.0126203, 0.0127477
9: -0.0083805, 0.0025002, -0.0094413, 0.0041079, -0.0124883, 0.0119415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0230765, upper bound: 0.0233243
time: 1.50 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0233985
time: 2.50 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0013338, 0.0187342, 0.0014160, 0.0190831, -0.0177493, 0.0173181
1: -0.0048226, 0.0028038, -0.0049538, 0.0029644, -0.0077869, 0.0077576
2: 0.0016381, 0.0136224, 0.0013887, 0.0135770, -0.0119389, 0.0122338
3: -0.0038384, 0.0050802, -0.0041981, 0.0052235, -0.0090619, 0.0092783
4: -0.0040677, -0.0003842, -0.0040499, -0.0000696, -0.0039981, 0.0036657
5: -0.0006256, 0.0073277, -0.0007809, 0.0072795, -0.0079051, 0.0081086
6: -0.0087945, 0.0067738, -0.0094841, 0.0065828, -0.0153772, 0.0162579
7: -0.0177859, 0.0036224, -0.0177665, 0.0052312, -0.0230171, 0.0213890
8: 0.9797962, 0.9928436, 0.9797751, 0.9938425, -0.0140463, 0.0130686
9: -0.0084126, 0.0041339, -0.0094413, 0.0041079, -0.0125205, 0.0135752

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225923, upper bound: 0.0230952
time: 1.62 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225206, upper bound: 0.0230729
time: 1.35 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0013777, 0.0175661, 0.0017250, 0.0151415, -0.0137638, 0.0158411
1: -0.0042518, 0.0026755, -0.0030197, 0.0024003, -0.0066521, 0.0056952
2: 0.0024866, 0.0135982, 0.0042538, 0.0134061, -0.0109196, 0.0093444
3: -0.0037281, 0.0045236, -0.0036508, 0.0032011, -0.0069292, 0.0081744
4: -0.0039377, -0.0003959, -0.0035058, -0.0002053, -0.0037325, 0.0031099
5: -0.0001340, 0.0073020, 0.0008488, 0.0070985, -0.0072325, 0.0064532
6: -0.0083862, 0.0066718, -0.0081347, 0.0058645, -0.0142508, 0.0148064
7: -0.0169968, 0.0035627, -0.0144713, 0.0045374, -0.0215342, 0.0180340
8: 0.9805171, 0.9927177, 0.9817867, 0.9931900, -0.0126728, 0.0109311
9: -0.0083745, 0.0036151, -0.0089977, 0.0020048, -0.0103793, 0.0126127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of IS_A2_A1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0208319, upper bound: 0.0208244
time: 1.23 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226876, upper bound: 0.0224991
time: 1.63 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0014264, 0.0174593, 0.0017581, 0.0150075, -0.0135811, 0.0157012
1: -0.0041885, 0.0026593, -0.0029226, 0.0023906, -0.0065791, 0.0055819
2: 0.0025630, 0.0135713, 0.0043584, 0.0133879, -0.0108248, 0.0092128
3: -0.0037135, 0.0044405, -0.0036838, 0.0030750, -0.0067884, 0.0081244
4: -0.0038985, -0.0003984, -0.0034704, -0.0001502, -0.0037484, 0.0030711
5: -0.0000886, 0.0072735, 0.0008960, 0.0070792, -0.0071677, 0.0063775
6: -0.0083397, 0.0065586, -0.0082561, 0.0057878, -0.0141275, 0.0148147
7: -0.0167746, 0.0035501, -0.0142351, 0.0048193, -0.0215938, 0.0177852
8: 0.9806724, 0.9926996, 0.9818603, 0.9933835, -0.0127112, 0.0108392
9: -0.0083664, 0.0034773, -0.0091779, 0.0018749, -0.0102413, 0.0126552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_A1_B2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226509, upper bound: 0.0224693
time: 1.48 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227668, upper bound: 0.0225430
time: 2.09 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0013777, 0.0175661, 0.0014716, 0.0154900, -0.0141123, 0.0160944
1: -0.0042518, 0.0026755, -0.0032425, 0.0025500, -0.0068017, 0.0059180
2: 0.0024866, 0.0135982, 0.0040135, 0.0135462, -0.0110597, 0.0095847
3: -0.0037281, 0.0045236, -0.0039975, 0.0035449, -0.0072730, 0.0085211
4: -0.0039377, -0.0003959, -0.0037067, 0.0001035, -0.0040413, 0.0033108
5: -0.0001340, 0.0073020, 0.0006750, 0.0072470, -0.0073809, 0.0066270
6: -0.0083862, 0.0066718, -0.0090140, 0.0064535, -0.0148398, 0.0156857
7: -0.0169968, 0.0035627, -0.0156133, 0.0061165, -0.0231133, 0.0191760
8: 0.9805171, 0.9927177, 0.9812216, 0.9942386, -0.0137214, 0.0114962
9: -0.0083745, 0.0036151, -0.0100074, 0.0026933, -0.0110677, 0.0136225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226734, upper bound: 0.0226689
time: 1.41 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227716, upper bound: 0.0227154
time: 1.44 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0014264, 0.0174593, 0.0014957, 0.0153750, -0.0139486, 0.0159636
1: -0.0041885, 0.0026593, -0.0031549, 0.0025426, -0.0067312, 0.0058142
2: 0.0025630, 0.0135713, 0.0041061, 0.0135329, -0.0109699, 0.0094651
3: -0.0037135, 0.0044405, -0.0040340, 0.0034342, -0.0071477, 0.0084746
4: -0.0038985, -0.0003984, -0.0036715, 0.0001607, -0.0040593, 0.0032731
5: -0.0000886, 0.0072735, 0.0007138, 0.0072329, -0.0073214, 0.0065597
6: -0.0083397, 0.0065586, -0.0091468, 0.0063976, -0.0147372, 0.0157055
7: -0.0167746, 0.0035501, -0.0153801, 0.0064092, -0.0231838, 0.0189302
8: 0.9806724, 0.9926996, 0.9812753, 0.9944091, -0.0137368, 0.0114242
9: -0.0083664, 0.0034773, -0.0101946, 0.0025656, -0.0109320, 0.0136719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226160, upper bound: 0.0225559
time: 1.34 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227362, upper bound: 0.0226241
time: 1.45 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0014270, 0.0150779, 0.0014536, 0.0190396, -0.0176127, 0.0136243
1: -0.0031017, 0.0023498, -0.0048869, 0.0029056, -0.0060072, 0.0072367
2: 0.0042803, 0.0135709, 0.0014337, 0.0135562, -0.0092759, 0.0121372
3: -0.0034712, 0.0033737, -0.0040707, 0.0051111, -0.0085823, 0.0074444
4: -0.0036917, -0.0003941, -0.0040192, -0.0001789, -0.0035128, 0.0036251
5: 0.0009000, 0.0072731, -0.0007709, 0.0072575, -0.0063575, 0.0080440
6: -0.0075495, 0.0065573, -0.0094529, 0.0064954, -0.0140449, 0.0160103
7: -0.0154161, 0.0035717, -0.0175770, 0.0046722, -0.0200884, 0.0211487
8: 0.9811221, 0.9925259, 0.9799049, 0.9935696, -0.0124475, 0.0126210
9: -0.0083802, 0.0026114, -0.0090839, 0.0039976, -0.0123778, 0.0116953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0230989, upper bound: 0.0232433
time: 1.32 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231345, upper bound: 0.0233251
time: 1.56 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0012808, 0.0187825, 0.0014536, 0.0190396, -0.0177588, 0.0173289
1: -0.0048500, 0.0028135, -0.0048869, 0.0029056, -0.0077555, 0.0077005
2: 0.0016042, 0.0136517, 0.0014337, 0.0135562, -0.0119520, 0.0122181
3: -0.0038469, 0.0051376, -0.0040707, 0.0051111, -0.0089579, 0.0092084
4: -0.0041083, -0.0003832, -0.0040192, -0.0001789, -0.0039293, 0.0036347
5: -0.0006463, 0.0073588, -0.0007709, 0.0072575, -0.0079038, 0.0081296
6: -0.0088179, 0.0068971, -0.0094529, 0.0064954, -0.0153133, 0.0163500
7: -0.0180215, 0.0036276, -0.0175770, 0.0046722, -0.0226937, 0.0212046
8: 0.9796560, 0.9928527, 0.9799049, 0.9935696, -0.0139136, 0.0129479
9: -0.0084159, 0.0042746, -0.0090839, 0.0039976, -0.0124135, 0.0133585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225787, upper bound: 0.0231222
time: 1.34 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225546, upper bound: 0.0230328
time: 1.30 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0014270, 0.0150779, 0.0011888, 0.0195840, -0.0181570, 0.0138891
1: -0.0031017, 0.0023498, -0.0052226, 0.0031298, -0.0062315, 0.0075723
2: 0.0042803, 0.0135709, 0.0010352, 0.0137026, -0.0094223, 0.0125358
3: -0.0034712, 0.0033737, -0.0044830, 0.0055809, -0.0090521, 0.0078566
4: -0.0036917, -0.0003941, -0.0042520, 0.0001384, -0.0038302, 0.0038579
5: 0.0009000, 0.0072731, -0.0010093, 0.0074126, -0.0065126, 0.0082824
6: -0.0075495, 0.0065573, -0.0102554, 0.0071109, -0.0146604, 0.0168127
7: -0.0154161, 0.0035717, -0.0189331, 0.0062951, -0.0217112, 0.0225048
8: 0.9811221, 0.9925259, 0.9789847, 0.9946132, -0.0134912, 0.0135412
9: -0.0083802, 0.0026114, -0.0101216, 0.0048263, -0.0132065, 0.0127330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0230765, upper bound: 0.0234414
time: 1.71 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0231133, upper bound: 0.0235231
time: 2.65 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0012808, 0.0187825, 0.0011888, 0.0195840, -0.0183032, 0.0175937
1: -0.0048500, 0.0028135, -0.0052226, 0.0031298, -0.0079798, 0.0080361
2: 0.0016042, 0.0136517, 0.0010352, 0.0137026, -0.0120984, 0.0126166
3: -0.0038469, 0.0051376, -0.0044830, 0.0055809, -0.0094278, 0.0096206
4: -0.0041083, -0.0003832, -0.0042520, 0.0001384, -0.0042467, 0.0038688
5: -0.0006463, 0.0073588, -0.0010093, 0.0074126, -0.0080589, 0.0083680
6: -0.0088179, 0.0068971, -0.0102554, 0.0071109, -0.0159287, 0.0171524
7: -0.0180215, 0.0036276, -0.0189331, 0.0062951, -0.0243165, 0.0225607
8: 0.9796560, 0.9928527, 0.9789847, 0.9946132, -0.0149572, 0.0138681
9: -0.0084159, 0.0042746, -0.0101216, 0.0048263, -0.0132423, 0.0143962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225465, upper bound: 0.0232703
time: 1.46 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0225206, upper bound: 0.0231642
time: 1.85 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0011798, 0.0178348, 0.0017796, 0.0149330, -0.0137532, 0.0160552
1: -0.0044548, 0.0028003, -0.0029647, 0.0022724, -0.0067272, 0.0057650
2: 0.0022959, 0.0137076, 0.0043745, 0.0133760, -0.0110801, 0.0093331
3: -0.0039744, 0.0048279, -0.0032701, 0.0030690, -0.0070435, 0.0080980
4: -0.0040900, -0.0001963, -0.0034373, -0.0005834, -0.0035067, 0.0032410
5: -0.0002589, 0.0074179, 0.0009912, 0.0070666, -0.0073254, 0.0064267
6: -0.0088628, 0.0071317, -0.0069494, 0.0057378, -0.0146006, 0.0140812
7: -0.0178639, 0.0045833, -0.0139952, 0.0026041, -0.0204680, 0.0185785
8: 0.9799455, 0.9933605, 0.9819084, 0.9918678, -0.0119223, 0.0114521
9: -0.0090270, 0.0041410, -0.0077615, 0.0017605, -0.0107876, 0.0119024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0212579, upper bound: 0.0212483
time: 1.74 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232547, upper bound: 0.0228571
time: 1.42 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0011566, 0.0179360, 0.0017752, 0.0148948, -0.0137382, 0.0161608
1: -0.0044952, 0.0028448, -0.0029343, 0.0023061, -0.0068013, 0.0057791
2: 0.0022252, 0.0137204, 0.0044133, 0.0133784, -0.0111532, 0.0093071
3: -0.0040687, 0.0048718, -0.0034119, 0.0030713, -0.0071400, 0.0082837
4: -0.0041144, -0.0001165, -0.0034491, -0.0004276, -0.0036868, 0.0033326
5: -0.0003041, 0.0074315, 0.0009895, 0.0070691, -0.0073733, 0.0064420
6: -0.0090976, 0.0071857, -0.0073195, 0.0057479, -0.0148455, 0.0145051
7: -0.0180016, 0.0049914, -0.0141144, 0.0034006, -0.0214023, 0.0191057
8: 0.9798461, 0.9936354, 0.9818985, 0.9923787, -0.0125325, 0.0117369
9: -0.0092880, 0.0042284, -0.0082708, 0.0017999, -0.0110879, 0.0124992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0215440, upper bound: 0.0214038
time: 1.10 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233513, upper bound: 0.0228927
time: 1.27 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0011798, 0.0178348, 0.0015714, 0.0152313, -0.0140515, 0.0162634
1: -0.0044548, 0.0028003, -0.0031610, 0.0023966, -0.0068514, 0.0059613
2: 0.0022959, 0.0137076, 0.0041707, 0.0134910, -0.0111952, 0.0095369
3: -0.0039744, 0.0048279, -0.0035767, 0.0033650, -0.0073394, 0.0084046
4: -0.0040900, -0.0001963, -0.0036069, -0.0003017, -0.0036078, 0.0032588
5: -0.0002589, 0.0074179, 0.0008430, 0.0071885, -0.0074473, 0.0065749
6: -0.0088628, 0.0071317, -0.0076939, 0.0062215, -0.0150843, 0.0148257
7: -0.0178639, 0.0045833, -0.0149754, 0.0040446, -0.0215456, 0.0193912
8: 0.9799455, 0.9933605, 0.9814442, 0.9928122, -0.0128667, 0.0119163
9: -0.0090270, 0.0041410, -0.0086826, 0.0023418, -0.0109342, 0.0123571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_A2_B1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0228330, upper bound: 0.0224793
time: 1.98 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227748, upper bound: 0.0224455
time: 2.28 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0011566, 0.0179360, 0.0015787, 0.0151837, -0.0140271, 0.0163573
1: -0.0044952, 0.0028448, -0.0031186, 0.0024292, -0.0069244, 0.0059633
2: 0.0022252, 0.0137204, 0.0042152, 0.0134870, -0.0112618, 0.0095052
3: -0.0040687, 0.0048718, -0.0037158, 0.0033449, -0.0074135, 0.0085877
4: -0.0041144, -0.0001165, -0.0036026, -0.0001482, -0.0037198, 0.0032755
5: -0.0003041, 0.0074315, 0.0008442, 0.0071842, -0.0074883, 0.0065872
6: -0.0090976, 0.0071857, -0.0080818, 0.0062046, -0.0153022, 0.0152675
7: -0.0180016, 0.0049914, -0.0149615, 0.0048291, -0.0220979, 0.0194690
8: 0.9798461, 0.9936354, 0.9814605, 0.9933137, -0.0134676, 0.0121749
9: -0.0092880, 0.0042284, -0.0091842, 0.0023233, -0.0109835, 0.0127274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0215440, upper bound: 0.0214151
time: 1.59 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0233571, upper bound: 0.0229456
time: 1.46 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0012362, 0.0153812, 0.0016211, 0.0186549, -0.0174186, 0.0137601
1: -0.0032909, 0.0024807, -0.0046922, 0.0027757, -0.0060667, 0.0071729
2: 0.0040738, 0.0136764, 0.0017002, 0.0134636, -0.0093898, 0.0119761
3: -0.0037844, 0.0036460, -0.0038322, 0.0048421, -0.0086265, 0.0074782
4: -0.0038384, -0.0001106, -0.0038687, -0.0003626, -0.0034758, 0.0037581
5: 0.0007476, 0.0073849, -0.0005922, 0.0071594, -0.0064118, 0.0079771
6: -0.0083510, 0.0070006, -0.0087730, 0.0061061, -0.0144571, 0.0157736
7: -0.0162432, 0.0050214, -0.0167218, 0.0037331, -0.0199763, 0.0217432
8: 0.9806967, 0.9934936, 0.9804882, 0.9928889, -0.0121921, 0.0130053
9: -0.0093072, 0.0031145, -0.0084834, 0.0034711, -0.0127782, 0.0115979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 214

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232237, upper bound: 0.0231624
time: 2.24 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232599, upper bound: 0.0232415
time: 1.29 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0010883, 0.0191681, 0.0016211, 0.0186549, -0.0175666, 0.0175470
1: -0.0050946, 0.0029994, -0.0046922, 0.0027757, -0.0078703, 0.0076915
2: 0.0013244, 0.0137582, 0.0017002, 0.0134636, -0.0121392, 0.0120579
3: -0.0042114, 0.0054838, -0.0038322, 0.0048421, -0.0090535, 0.0093160
4: -0.0042710, -0.0000892, -0.0038687, -0.0003626, -0.0039084, 0.0037795
5: -0.0008163, 0.0074715, -0.0005922, 0.0071594, -0.0079758, 0.0080637
6: -0.0095269, 0.0073445, -0.0087730, 0.0061061, -0.0156330, 0.0161175
7: -0.0189631, 0.0051311, -0.0167218, 0.0037331, -0.0226962, 0.0218529
8: 0.9790075, 0.9938150, 0.9804882, 0.9928889, -0.0138814, 0.0133268
9: -0.0093773, 0.0048482, -0.0084834, 0.0034711, -0.0128484, 0.0133315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 117

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226463, upper bound: 0.0230535
time: 1.65 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226211, upper bound: 0.0229693
time: 1.67 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0012362, 0.0153812, 0.0014160, 0.0190831, -0.0178468, 0.0139652
1: -0.0032909, 0.0024807, -0.0049538, 0.0029644, -0.0062553, 0.0074344
2: 0.0040738, 0.0136764, 0.0013887, 0.0135770, -0.0095032, 0.0122877
3: -0.0037844, 0.0036460, -0.0041981, 0.0052235, -0.0090079, 0.0078441
4: -0.0038384, -0.0001106, -0.0040499, -0.0000696, -0.0035512, 0.0038190
5: 0.0007476, 0.0073849, -0.0007809, 0.0072795, -0.0065319, 0.0081657
6: -0.0083510, 0.0070006, -0.0094841, 0.0065828, -0.0149338, 0.0164848
7: -0.0162432, 0.0050214, -0.0177665, 0.0052312, -0.0209150, 0.0227879
8: 0.9806967, 0.9934936, 0.9797751, 0.9938425, -0.0131458, 0.0137185
9: -0.0093072, 0.0031145, -0.0094413, 0.0041079, -0.0131799, 0.0119036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 163

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of IS_A2_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232251, upper bound: 0.0232064
time: 1.29 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0232610, upper bound: 0.0233017
time: 1.70 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0010883, 0.0191681, 0.0014160, 0.0190831, -0.0179948, 0.0177521
1: -0.0050946, 0.0029994, -0.0049538, 0.0029644, -0.0080590, 0.0079531
2: 0.0013244, 0.0137582, 0.0013887, 0.0135770, -0.0122526, 0.0123695
3: -0.0042114, 0.0054838, -0.0041981, 0.0052235, -0.0094349, 0.0096819
4: -0.0042710, -0.0000892, -0.0040499, -0.0000696, -0.0039776, 0.0037069
5: -0.0008163, 0.0074715, -0.0007809, 0.0072795, -0.0080959, 0.0082524
6: -0.0095269, 0.0073445, -0.0094841, 0.0065828, -0.0161097, 0.0168286
7: -0.0189631, 0.0051311, -0.0177665, 0.0052312, -0.0235568, 0.0221850
8: 0.9790075, 0.9938150, 0.9797751, 0.9938425, -0.0148351, 0.0140399
9: -0.0093773, 0.0048482, -0.0094413, 0.0041079, -0.0128019, 0.0136946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 117

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0227074, upper bound: 0.0230464
time: 1.24 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226211, upper bound: 0.0230166
time: 1.34 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0011140, 0.0179758, 0.0017250, 0.0151415, -0.0140275, 0.0162508
1: -0.0045135, 0.0028590, -0.0030197, 0.0024003, -0.0069138, 0.0058787
2: 0.0021969, 0.0137440, 0.0042538, 0.0134061, -0.0112092, 0.0094902
3: -0.0040919, 0.0049154, -0.0036508, 0.0032011, -0.0072930, 0.0085661
4: -0.0041491, -0.0001003, -0.0035058, -0.0002053, -0.0039438, 0.0034054
5: -0.0003213, 0.0074565, 0.0008488, 0.0070985, -0.0074199, 0.0066077
6: -0.0091656, 0.0072848, -0.0081347, 0.0058645, -0.0150301, 0.0154194
7: -0.0182215, 0.0050741, -0.0144713, 0.0045374, -0.0227589, 0.0195453
8: 0.9797319, 0.9936966, 0.9817867, 0.9931900, -0.0134581, 0.0119099
9: -0.0093408, 0.0043514, -0.0089977, 0.0020048, -0.0113456, 0.0133491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_A2_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226463, upper bound: 0.0225923
time: 1.53 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0226463, upper bound: 0.0226087
time: 1.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0011579, 0.0178739, 0.0017581, 0.0150075, -0.0138496, 0.0161159
1: -0.0044532, 0.0028423, -0.0029226, 0.0023906, -0.0068438, 0.0057650
2: 0.0022701, 0.0137197, 0.0043584, 0.0133879, -0.0111178, 0.0093613
3: -0.0040767, 0.0048381, -0.0036838, 0.0030750, -0.0071516, 0.0085220
4: -0.0041096, -0.0001031, -0.0034704, -0.0001502, -0.0039594, 0.0033673
5: -0.0002783, 0.0074308, 0.0008960, 0.0070792, -0.0073575, 0.0065348
6: -0.0091225, 0.0071828, -0.0082561, 0.0057878, -0.0149103, 0.0154389
7: -0.0180001, 0.0050602, -0.0142351, 0.0048193, -0.0228194, 0.0192953
8: 0.9798782, 0.9936786, 0.9818603, 0.9933835, -0.0135053, 0.0118182
9: -0.0093320, 0.0042111, -0.0091779, 0.0018749, -0.0112069, 0.0133890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=28, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.53 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.27 + 596.20 = 600.47 seconds
