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
execution time: IAR + RelationalAnalysis = 2.62 + 3.68 = 6.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0231797, upper bound: 0.0231797

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0222377, upper bound: 0.0215990
time: 2.27 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0220711, upper bound: 0.0220711
time: 1.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.25 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.25
Output dim: 4, lower bound: -0.0222377, upper bound: 0.0215990
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.25
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

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 129

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0208796, upper bound: 0.0209160
time: 2.21 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0215498, upper bound: 0.0209160
time: 1.70 seconds

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

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 129

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0213636, upper bound: 0.0207425
time: 1.60 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0213636, upper bound: 0.0213636
time: 1.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.69 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.69
Output dim: 4, lower bound: -0.0208796, upper bound: 0.0209160
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.69
Output dim: 4, lower bound: -0.0215498, upper bound: 0.0209160
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.69
Output dim: 4, lower bound: -0.0213636, upper bound: 0.0207425
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.69
Output dim: 4, lower bound: -0.0213636, upper bound: 0.0213636

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0045898, -0.0015167, -0.0045727, -0.0014863, -0.0031035, 0.0030560
1: -0.0027679, 0.0051617, -0.0027768, 0.0050671, -0.0078350, 0.0079385
2: 0.0034342, 0.0211500, 0.0036458, 0.0211697, -0.0177355, 0.0175042
3: -0.0024256, 0.0058872, -0.0024441, 0.0057980, -0.0082236, 0.0083313
4: 0.9893278, 1.0195903, 0.9892729, 1.0192443, -0.0299165, 0.0303174
5: -0.0033935, 0.0060525, -0.0034458, 0.0059852, -0.0093786, 0.0094983
6: -0.0136194, -0.0062869, -0.0135318, -0.0062787, -0.0073406, 0.0072449
7: -0.0105406, -0.0032138, -0.0105295, -0.0031165, -0.0074241, 0.0073157
8: -0.0065104, -0.0014442, -0.0065160, -0.0015047, -0.0050056, 0.0050718
9: -0.0109409, 0.0180665, -0.0106379, 0.0181388, -0.0290797, 0.0287044

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 145
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0198600, upper bound: 0.0194565
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0197950, upper bound: 0.0197853
time: 2.17 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0045896, -0.0015212, -0.0046729, -0.0003809, -0.0042088, 0.0031517
1: -0.0027666, 0.0051607, -0.0030977, 0.0052873, -0.0080539, 0.0082584
2: 0.0034366, 0.0211471, 0.0031538, 0.0218868, -0.0184502, 0.0179933
3: -0.0024228, 0.0058862, -0.0031193, 0.0060054, -0.0084282, 0.0090055
4: 0.9893360, 1.0195863, 0.9872713, 1.0200489, -0.0307128, 0.0323150
5: -0.0033857, 0.0060517, -0.0053521, 0.0074662, -0.0108519, 0.0114038
6: -0.0136184, -0.0062881, -0.0137355, -0.0059820, -0.0076364, 0.0074473
7: -0.0105405, -0.0032283, -0.0105554, 0.0004252, -0.0109657, 0.0073272
8: -0.0065095, -0.0014449, -0.0067210, -0.0013640, -0.0051455, 0.0052761
9: -0.0109374, 0.0180558, -0.0113423, 0.0207714, -0.0317088, 0.0293981

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 145
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200723, upper bound: 0.0198728
time: 1.94 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0204433, upper bound: 0.0197852
time: 1.81 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0057991, 0.0025875, -0.0046314, -0.0014500, -0.0043491, 0.0072188
1: -0.0039595, 0.0047646, -0.0027873, 0.0053918, -0.0093513, 0.0075519
2: -0.0031720, 0.0238121, 0.0029203, 0.0211933, -0.0243653, 0.0208918
3: -0.0049324, 0.0055133, -0.0024663, 0.0061037, -0.0110361, 0.0079796
4: 0.9818969, 1.0181397, 0.9892071, 1.0204302, -0.0385333, 0.0289326
5: -0.0104706, 0.0143922, -0.0035085, 0.0062159, -0.0166865, 0.0179007
6: -0.0156485, -0.0051851, -0.0138321, -0.0062690, -0.0093795, 0.0086470
7: -0.0104938, 0.0099351, -0.0105678, -0.0030001, -0.0074937, 0.0205029
8: -0.0072716, 0.0005759, -0.0065227, -0.0012973, -0.0059744, 0.0070986
9: -0.0096706, 0.0278402, -0.0116765, 0.0182254, -0.0278960, 0.0395167

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202084, upper bound: 0.0191256
time: 2.56 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200267, upper bound: 0.0194269
time: 1.73 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0062000, 0.0036441, -0.0046310, -0.0014545, -0.0047455, 0.0082751
1: -0.0042663, 0.0049888, -0.0027860, 0.0053899, -0.0096562, 0.0077749
2: -0.0072452, 0.0244975, 0.0029244, 0.0211904, -0.0284356, 0.0215731
3: -0.0055778, 0.0057244, -0.0024636, 0.0061020, -0.0116798, 0.0081880
4: 0.9799837, 1.0189588, 0.9892152, 1.0204237, -0.0404400, 0.0297436
5: -0.0122927, 0.0168576, -0.0035008, 0.0062146, -0.0185073, 0.0203584
6: -0.0172180, -0.0049014, -0.0138304, -0.0062702, -0.0109478, 0.0089290
7: -0.0105202, 0.0133204, -0.0105676, -0.0030144, -0.0075058, 0.0238880
8: -0.0074676, 0.0017814, -0.0065219, -0.0012984, -0.0061692, 0.0083033
9: -0.0103878, 0.0303565, -0.0116707, 0.0182147, -0.0286025, 0.0420272

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 145
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 245
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 145
type: B, layer: 1, pos: 245
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0202084, upper bound: 0.0197902
time: 2.27 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0200261, upper bound: 0.0200261
time: 1.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.54 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 6.54
Output dim: 4, lower bound: -0.0198600, upper bound: 0.0194565
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 6.54
Output dim: 4, lower bound: -0.0197950, upper bound: 0.0197853
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 6.54
Output dim: 4, lower bound: -0.0200723, upper bound: 0.0198728
NS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 6.54
Output dim: 4, lower bound: -0.0204433, upper bound: 0.0197852
NS_A2_A1_A1, status: Status.VERIFIED, split count: 3, time: 6.54
Output dim: 4, lower bound: -0.0202084, upper bound: 0.0191256
NS_A2_A1_A2, status: Status.VERIFIED, split count: 3, time: 6.54
Output dim: 4, lower bound: -0.0200267, upper bound: 0.0194269
NS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 6.54
Output dim: 4, lower bound: -0.0202084, upper bound: 0.0197902
NS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 6.54
Output dim: 4, lower bound: -0.0200261, upper bound: 0.0200261

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 6.30 + 43.06 = 49.36 seconds
