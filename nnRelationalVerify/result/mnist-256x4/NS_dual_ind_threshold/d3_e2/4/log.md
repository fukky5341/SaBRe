## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.020861730000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0046476, -0.0013784, -0.0046476, -0.0013784, -0.0032693, 0.0032693)
1: (-0.0028081, 0.0054819, -0.0028081, 0.0054819, -0.0082900, 0.0082900)
2: (0.0027189, 0.0212398, 0.0027189, 0.0212398, -0.0185209, 0.0185209)
3: (-0.0025101, 0.0061886, -0.0025101, 0.0061886, -0.0086987, 0.0086987)
4: (0.9890774, 1.0207597, 0.9890774, 1.0207597, -0.0316823, 0.0316823)
5: (-0.0036320, 0.0062800, -0.0036320, 0.0062800, -0.0099120, 0.0099120)
6: (-0.0139154, -0.0062498, -0.0139154, -0.0062498, -0.0076657, 0.0076657)
7: (-0.0105784, -0.0027707, -0.0105784, -0.0027707, -0.0078077, 0.0078077)
8: (-0.0065360, -0.0012396, -0.0065360, -0.0012396, -0.0052964, 0.0052964)
9: (-0.0119650, 0.0183959, -0.0119650, 0.0183959, -0.0303609, 0.0303609)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.13 + 3.51 = 5.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0231797, upper bound: 0.0231797

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222377, upper bound: 0.0215990
time: 2.12 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220711, upper bound: 0.0220711
time: 1.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.85
Output dim: 4, lower bound: -0.0222377, upper bound: 0.0215990
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.85
Output dim: 4, lower bound: -0.0220711, upper bound: 0.0220711

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0045938, -0.0015115, -0.0046476, -0.0013784, -0.0032155, 0.0031362
1: -0.0027695, 0.0051840, -0.0028081, 0.0054819, -0.0082514, 0.0079921
2: 0.0033846, 0.0211534, 0.0027189, 0.0212398, -0.0178551, 0.0184346
3: -0.0024288, 0.0059081, -0.0025101, 0.0061886, -0.0086174, 0.0084181
4: 0.9893184, 1.0196713, 0.9890774, 1.0207597, -0.0314413, 0.0305939
5: -0.0034025, 0.0060683, -0.0036320, 0.0062800, -0.0096825, 0.0097002
6: -0.0136399, -0.0062855, -0.0139154, -0.0062498, -0.0073901, 0.0076299
7: -0.0105433, -0.0031971, -0.0105784, -0.0027707, -0.0077726, 0.0073813
8: -0.0065113, -0.0014300, -0.0065360, -0.0012396, -0.0052717, 0.0051060
9: -0.0110118, 0.0180789, -0.0119650, 0.0183959, -0.0294078, 0.0300439

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215990, upper bound: 0.0215990
time: 2.12 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215990, upper bound: 0.0215990
time: 2.20 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0058403, 0.0026960, -0.0046353, -0.0014448, -0.0043955, 0.0073313
1: -0.0039910, 0.0051766, -0.0027888, 0.0054137, -0.0094048, 0.0079654
2: -0.0035902, 0.0238825, 0.0028712, 0.0211967, -0.0247869, 0.0210113
3: -0.0049986, 0.0059011, -0.0024695, 0.0061244, -0.0111230, 0.0083707
4: 0.9817005, 1.0196445, 0.9891978, 1.0205107, -0.0388101, 0.0304467
5: -0.0106577, 0.0146453, -0.0035175, 0.0062315, -0.0168892, 0.0181628
6: -0.0158097, -0.0051559, -0.0138524, -0.0062676, -0.0095421, 0.0086964
7: -0.0105424, 0.0102827, -0.0105704, -0.0029834, -0.0075590, 0.0208531
8: -0.0072918, 0.0006996, -0.0065237, -0.0012832, -0.0060085, 0.0072233
9: -0.0109884, 0.0280986, -0.0117468, 0.0182378, -0.0292262, 0.0398454

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 129

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0207425, upper bound: 0.0213636
time: 5.84 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0213636, upper bound: 0.0213636
time: 1.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 9.57 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 9.57
Output dim: 4, lower bound: -0.0215990, upper bound: 0.0215990
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 9.57
Output dim: 4, lower bound: -0.0215990, upper bound: 0.0215990
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 9.57
Output dim: 4, lower bound: -0.0207425, upper bound: 0.0213636
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 9.57
Output dim: 4, lower bound: -0.0213636, upper bound: 0.0213636

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0045938, -0.0015115, -0.0045938, -0.0015115, -0.0030824, 0.0030824
1: -0.0027695, 0.0051840, -0.0027695, 0.0051840, -0.0079534, 0.0079534
2: 0.0033846, 0.0211534, 0.0033846, 0.0211534, -0.0177688, 0.0177688
3: -0.0024288, 0.0059081, -0.0024288, 0.0059081, -0.0083368, 0.0083368
4: 0.9893184, 1.0196713, 0.9893184, 1.0196713, -0.0303529, 0.0303529
5: -0.0034025, 0.0060683, -0.0034025, 0.0060683, -0.0094707, 0.0094707
6: -0.0136399, -0.0062855, -0.0136399, -0.0062855, -0.0073544, 0.0073544
7: -0.0105433, -0.0031971, -0.0105433, -0.0031971, -0.0073462, 0.0073462
8: -0.0065113, -0.0014300, -0.0065113, -0.0014300, -0.0050813, 0.0050813
9: -0.0110118, 0.0180789, -0.0110118, 0.0180789, -0.0290908, 0.0290908

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 129

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214643, upper bound: 0.0202935
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214643, upper bound: 0.0209160
time: 1.62 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0045938, -0.0015115, -0.0058403, 0.0026960, -0.0072898, 0.0043288
1: -0.0027695, 0.0051840, -0.0039910, 0.0051766, -0.0079460, 0.0091750
2: 0.0033846, 0.0211534, -0.0035902, 0.0238825, -0.0204979, 0.0247436
3: -0.0024288, 0.0059081, -0.0049986, 0.0059011, -0.0083299, 0.0109067
4: 0.9893184, 1.0196713, 0.9817005, 1.0196445, -0.0303261, 0.0379708
5: -0.0034025, 0.0060683, -0.0106577, 0.0146453, -0.0180477, 0.0167259
6: -0.0136399, -0.0062855, -0.0158097, -0.0051559, -0.0084840, 0.0095242
7: -0.0105433, -0.0031971, -0.0105424, 0.0102827, -0.0208260, 0.0073453
8: -0.0065113, -0.0014300, -0.0072918, 0.0006996, -0.0072110, 0.0058617
9: -0.0110118, 0.0180789, -0.0109884, 0.0280986, -0.0391104, 0.0290673

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 129

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214643, upper bound: 0.0202935
time: 3.38 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0214643, upper bound: 0.0209160
time: 2.58 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0058383, 0.0026908, -0.0045610, -0.0015529, -0.0042855, 0.0072518
1: -0.0039895, 0.0051564, -0.0027574, 0.0050023, -0.0089918, 0.0079138
2: -0.0035703, 0.0238791, 0.0037905, 0.0211266, -0.0246969, 0.0200886
3: -0.0049955, 0.0058821, -0.0024035, 0.0057370, -0.0107325, 0.0082856
4: 0.9817098, 1.0195706, 0.9893934, 1.0190078, -0.0372980, 0.0301772
5: -0.0106488, 0.0146332, -0.0033311, 0.0059392, -0.0165879, 0.0179643
6: -0.0158020, -0.0051573, -0.0134719, -0.0062966, -0.0095054, 0.0083146
7: -0.0105400, 0.0102661, -0.0105218, -0.0033297, -0.0072103, 0.0207880
8: -0.0072908, 0.0006937, -0.0065036, -0.0015461, -0.0057447, 0.0071974
9: -0.0109236, 0.0280862, -0.0104308, 0.0179804, -0.0289040, 0.0385170

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0196320, upper bound: 0.0197902
time: 2.01 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0194269, upper bound: 0.0200267
time: 2.11 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0058367, 0.0026864, -0.0046423, -0.0004615, -0.0053752, 0.0073287
1: -0.0039882, 0.0051539, -0.0030743, 0.0052237, -0.0092119, 0.0082282
2: -0.0035535, 0.0238763, 0.0032959, 0.0218345, -0.0253880, 0.0205804
3: -0.0049928, 0.0058798, -0.0030701, 0.0059455, -0.0109383, 0.0089499
4: 0.9817176, 1.0195615, 0.9874172, 1.0198164, -0.0380988, 0.0321444
5: -0.0106413, 0.0146231, -0.0052131, 0.0072782, -0.0179194, 0.0198362
6: -0.0157955, -0.0051585, -0.0136766, -0.0060036, -0.0097919, 0.0085181
7: -0.0105397, 0.0102522, -0.0105479, 0.0001670, -0.0107067, 0.0208002
8: -0.0072900, 0.0006888, -0.0067061, -0.0014046, -0.0058853, 0.0073949
9: -0.0109158, 0.0280759, -0.0111389, 0.0205794, -0.0314952, 0.0392147

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202084, upper bound: 0.0197902
time: 2.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200261, upper bound: 0.0200261
time: 1.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.82 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.82
Output dim: 4, lower bound: -0.0214643, upper bound: 0.0202935
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.82
Output dim: 4, lower bound: -0.0214643, upper bound: 0.0209160
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.82
Output dim: 4, lower bound: -0.0214643, upper bound: 0.0202935
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.82
Output dim: 4, lower bound: -0.0214643, upper bound: 0.0209160
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 5.82
Output dim: 4, lower bound: -0.0196320, upper bound: 0.0197902
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 5.82
Output dim: 4, lower bound: -0.0194269, upper bound: 0.0200267
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 5.82
Output dim: 4, lower bound: -0.0202084, upper bound: 0.0197902
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 5.82
Output dim: 4, lower bound: -0.0200261, upper bound: 0.0200261

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0045208, -0.0016198, -0.0045898, -0.0015167, -0.0030041, 0.0029700
1: -0.0027380, 0.0047796, -0.0027679, 0.0051617, -0.0078998, 0.0075475
2: 0.0042880, 0.0210832, 0.0034342, 0.0211500, -0.0168621, 0.0176490
3: -0.0023626, 0.0055274, -0.0024256, 0.0058872, -0.0082498, 0.0079530
4: 0.9895145, 1.0181944, 0.9893278, 1.0195903, -0.0300758, 0.0288666
5: -0.0032157, 0.0057809, -0.0033935, 0.0060525, -0.0092682, 0.0091744
6: -0.0132660, -0.0063146, -0.0136194, -0.0062869, -0.0069791, 0.0073048
7: -0.0104956, -0.0035441, -0.0105406, -0.0032138, -0.0072818, 0.0069966
8: -0.0064912, -0.0016884, -0.0065104, -0.0014442, -0.0050470, 0.0048220
9: -0.0097186, 0.0178210, -0.0109409, 0.0180665, -0.0277851, 0.0287619

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202576, upper bound: 0.0201396
time: 2.42 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0206948, upper bound: 0.0201027
time: 2.59 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0046168, -0.0005286, -0.0045896, -0.0015212, -0.0030956, 0.0040611
1: -0.0030548, 0.0049992, -0.0027666, 0.0051607, -0.0082155, 0.0077659
2: 0.0037973, 0.0217910, 0.0034366, 0.0211471, -0.0173498, 0.0183544
3: -0.0030291, 0.0057342, -0.0024228, 0.0058862, -0.0089153, 0.0081570
4: 0.9875388, 1.0189966, 0.9893360, 1.0195863, -0.0320476, 0.0296606
5: -0.0050974, 0.0071216, -0.0033857, 0.0060517, -0.0111491, 0.0105073
6: -0.0134691, -0.0060216, -0.0136184, -0.0062881, -0.0071810, 0.0075968
7: -0.0105215, -0.0000480, -0.0105405, -0.0032283, -0.0072932, 0.0104925
8: -0.0066936, -0.0015481, -0.0065095, -0.0014449, -0.0052487, 0.0049615
9: -0.0104210, 0.0204197, -0.0109374, 0.0180558, -0.0284768, 0.0313571

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202576, upper bound: 0.0207274
time: 2.40 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0206948, upper bound: 0.0206948
time: 1.62 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0045208, -0.0016198, -0.0058383, 0.0026908, -0.0072116, 0.0042185
1: -0.0027380, 0.0047796, -0.0039895, 0.0051564, -0.0078944, 0.0087691
2: 0.0042880, 0.0210832, -0.0035703, 0.0238791, -0.0195912, 0.0246535
3: -0.0023626, 0.0055274, -0.0049955, 0.0058821, -0.0082447, 0.0105229
4: 0.9895145, 1.0181944, 0.9817098, 1.0195706, -0.0300561, 0.0364846
5: -0.0032157, 0.0057809, -0.0106488, 0.0146332, -0.0178489, 0.0164297
6: -0.0132660, -0.0063146, -0.0158020, -0.0051573, -0.0081087, 0.0094874
7: -0.0104956, -0.0035441, -0.0105400, 0.0102661, -0.0207617, 0.0069959
8: -0.0064912, -0.0016884, -0.0072908, 0.0006937, -0.0071850, 0.0056024
9: -0.0097186, 0.0178210, -0.0109236, 0.0280862, -0.0378048, 0.0287446

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200723, upper bound: 0.0192993
time: 1.51 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0204434, upper bound: 0.0191974
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0046168, -0.0005286, -0.0058367, 0.0026864, -0.0073033, 0.0053081
1: -0.0030548, 0.0049992, -0.0039882, 0.0051539, -0.0082087, 0.0089875
2: 0.0037973, 0.0217910, -0.0035535, 0.0238763, -0.0200790, 0.0253445
3: -0.0030291, 0.0057342, -0.0049928, 0.0058798, -0.0089089, 0.0107270
4: 0.9875388, 1.0189966, 0.9817176, 1.0195615, -0.0320228, 0.0372790
5: -0.0050974, 0.0071216, -0.0106413, 0.0146231, -0.0197205, 0.0177629
6: -0.0134691, -0.0060216, -0.0157955, -0.0051585, -0.0083106, 0.0097739
7: -0.0105215, -0.0000480, -0.0105397, 0.0102522, -0.0207737, 0.0104917
8: -0.0066936, -0.0015481, -0.0072900, 0.0006888, -0.0073824, 0.0057419
9: -0.0104210, 0.0204197, -0.0109158, 0.0280759, -0.0384969, 0.0313355

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200723, upper bound: 0.0198728
time: 2.25 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0204433, upper bound: 0.0197852
time: 1.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.93 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.93
Output dim: 4, lower bound: -0.0202576, upper bound: 0.0201396
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.93
Output dim: 4, lower bound: -0.0206948, upper bound: 0.0201027
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.93
Output dim: 4, lower bound: -0.0202576, upper bound: 0.0207274
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.93
Output dim: 4, lower bound: -0.0206948, upper bound: 0.0206948
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.93
Output dim: 4, lower bound: -0.0200723, upper bound: 0.0192993
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.93
Output dim: 4, lower bound: -0.0204434, upper bound: 0.0191974
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.93
Output dim: 4, lower bound: -0.0200723, upper bound: 0.0198728
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.93
Output dim: 4, lower bound: -0.0204433, upper bound: 0.0197852

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.64 + 69.92 = 75.56 seconds
