## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0036601199999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042051, -0.0040807, -0.0042051, -0.0040807, -0.0001243, 0.0001243)
1: (-0.0101349, -0.0054782, -0.0101349, -0.0054782, -0.0046566, 0.0046566)
2: (0.9643012, 0.9698893, 0.9643012, 0.9698893, -0.0055881, 0.0055881)
3: (-0.0170025, 0.0242148, -0.0170025, 0.0242148, -0.0382411, 0.0382411)
4: (-0.0025347, 0.0006001, -0.0025347, 0.0006001, -0.0031348, 0.0031348)
5: (0.0147086, 0.0182844, 0.0147086, 0.0182844, -0.0035758, 0.0035758)
6: (0.0021260, 0.0047525, 0.0021260, 0.0047525, -0.0026265, 0.0026265)
7: (-0.0140537, -0.0027273, -0.0140537, -0.0027273, -0.0113264, 0.0113264)
8: (0.0055796, 0.0140540, 0.0055796, 0.0140540, -0.0084744, 0.0084744)
9: (0.0077600, 0.0230022, 0.0077600, 0.0230022, -0.0148966, 0.0148966)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.68 + 3.23 = 5.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0040668, upper bound: 0.0040668

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039668, upper bound: 0.0037433
time: 2.93 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039795, upper bound: 0.0039795
time: 2.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.81 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 5.81
Output dim: 2, lower bound: -0.0039668, upper bound: 0.0037433
NS_B2, status: Status.UNKNOWN, split count: 1, time: 5.81
Output dim: 2, lower bound: -0.0039795, upper bound: 0.0039795

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -0.0042044, -0.0040808, -0.0041985, -0.0040843, -0.0001201, 0.0001177
1: -0.0101121, -0.0054833, -0.0098904, -0.0056137, -0.0044985, 0.0044071
2: 0.9643285, 0.9698833, 0.9645946, 0.9697270, -0.0053985, 0.0052887
3: -0.0168010, 0.0241703, -0.0148382, 0.0230162, -0.0368428, 0.0360523
4: -0.0025313, 0.0005848, -0.0024435, 0.0004355, -0.0029668, 0.0030283
5: 0.0147120, 0.0182394, 0.0148007, 0.0178006, -0.0030886, 0.0034386
6: 0.0022123, 0.0047509, 0.0030525, 0.0047077, -0.0024955, 0.0016984
7: -0.0140422, -0.0028263, -0.0137431, -0.0037903, -0.0102519, 0.0109168
8: 0.0055887, 0.0140126, 0.0058260, 0.0136091, -0.0080203, 0.0081866
9: 0.0077765, 0.0229276, 0.0082033, 0.0222018, -0.0140757, 0.0143718

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037144, upper bound: 0.0036368
time: 2.42 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038715, upper bound: 0.0036373
time: 2.74 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -0.0042051, -0.0040807, -0.0042031, -0.0040817, -0.0001234, 0.0001224
1: -0.0101349, -0.0054782, -0.0100619, -0.0055153, -0.0046196, 0.0045837
2: 0.9643012, 0.9698893, 0.9643887, 0.9698449, -0.0055437, 0.0055006
3: -0.0170025, 0.0242148, -0.0163570, 0.0238871, -0.0379118, 0.0368500
4: -0.0025347, 0.0006001, -0.0025098, 0.0005510, -0.0030857, 0.0031099
5: 0.0147086, 0.0182844, 0.0147338, 0.0181401, -0.0034315, 0.0035507
6: 0.0021260, 0.0047525, 0.0024023, 0.0047403, -0.0026143, 0.0023502
7: -0.0140537, -0.0027273, -0.0139688, -0.0030444, -0.0110094, 0.0112415
8: 0.0055796, 0.0140540, 0.0056470, 0.0139213, -0.0083417, 0.0084071
9: 0.0077600, 0.0230022, 0.0078812, 0.0227635, -0.0145671, 0.0147761

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037434, upper bound: 0.0039668
time: 3.18 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037434, upper bound: 0.0039794
time: 3.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 8.47 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 8.47
Output dim: 2, lower bound: -0.0037144, upper bound: 0.0036368
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 8.47
Output dim: 2, lower bound: -0.0038715, upper bound: 0.0036373
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 8.47
Output dim: 2, lower bound: -0.0037434, upper bound: 0.0039668
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 8.47
Output dim: 2, lower bound: -0.0037434, upper bound: 0.0039794

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -0.0041986, -0.0040838, -0.0041981, -0.0040844, -0.0001142, 0.0001143
1: -0.0098922, -0.0055938, -0.0098731, -0.0056154, -0.0042768, 0.0042793
2: 0.9645925, 0.9697506, 0.9646153, 0.9697247, -0.0051323, 0.0051354
3: -0.0148544, 0.0231924, -0.0146855, 0.0230010, -0.0348608, 0.0348055
4: -0.0024569, 0.0004367, -0.0024424, 0.0004239, -0.0028808, 0.0028791
5: 0.0147872, 0.0178042, 0.0148019, 0.0177664, -0.0029793, 0.0030023
6: 0.0030456, 0.0047143, 0.0031179, 0.0047072, -0.0016616, 0.0015964
7: -0.0137888, -0.0037824, -0.0137392, -0.0038654, -0.0099234, 0.0099568
8: 0.0057898, 0.0136124, 0.0058291, 0.0135776, -0.0077879, 0.0077832
9: 0.0081381, 0.0222078, 0.0082089, 0.0221453, -0.0136484, 0.0136422

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035362, upper bound: 0.0036368
time: 2.19 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035362, upper bound: 0.0036368
time: 2.63 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042021, -0.0040812, -0.0041985, -0.0040843, -0.0001178, 0.0001173
1: -0.0100246, -0.0054979, -0.0098901, -0.0056137, -0.0044109, 0.0043923
2: 0.9644336, 0.9698657, 0.9645948, 0.9697268, -0.0052933, 0.0052709
3: -0.0160264, 0.0240411, -0.0148362, 0.0230159, -0.0353815, 0.0359211
4: -0.0025215, 0.0005259, -0.0024435, 0.0004354, -0.0029568, 0.0029694
5: 0.0147219, 0.0180662, 0.0148007, 0.0178001, -0.0030782, 0.0032655
6: 0.0025438, 0.0047460, 0.0030534, 0.0047077, -0.0021639, 0.0016927
7: -0.0140087, -0.0032067, -0.0137430, -0.0037913, -0.0102174, 0.0105363
8: 0.0056153, 0.0138534, 0.0058261, 0.0136086, -0.0079933, 0.0080273
9: 0.0078243, 0.0226412, 0.0082034, 0.0222011, -0.0140273, 0.0140132

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038699, upper bound: 0.0035291
time: 2.45 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038699, upper bound: 0.0036373
time: 2.69 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -0.0041985, -0.0040843, -0.0042031, -0.0040817, -0.0001168, 0.0001188
1: -0.0098904, -0.0056137, -0.0100619, -0.0055153, -0.0043751, 0.0044483
2: 0.9645946, 0.9697270, 0.9643887, 0.9698449, -0.0052503, 0.0053383
3: -0.0148382, 0.0230162, -0.0163570, 0.0238871, -0.0357679, 0.0364514
4: -0.0024435, 0.0004355, -0.0025098, 0.0005510, -0.0029946, 0.0029453
5: 0.0148007, 0.0178006, 0.0147338, 0.0181401, -0.0033394, 0.0030668
6: 0.0030525, 0.0047077, 0.0024023, 0.0047403, -0.0016878, 0.0023054
7: -0.0137431, -0.0037903, -0.0139688, -0.0030444, -0.0106988, 0.0101785
8: 0.0058260, 0.0136091, 0.0056470, 0.0139213, -0.0080953, 0.0079621
9: 0.0082033, 0.0222018, 0.0078812, 0.0227635, -0.0142138, 0.0139716

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036368, upper bound: 0.0037143
time: 2.51 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036373, upper bound: 0.0038715
time: 2.25 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042031, -0.0040817, -0.0042031, -0.0040817, -0.0001214, 0.0001214
1: -0.0100619, -0.0055153, -0.0100619, -0.0055153, -0.0045467, 0.0045467
2: 0.9643887, 0.9698449, 0.9643887, 0.9698449, -0.0054562, 0.0054562
3: -0.0163570, 0.0238871, -0.0163570, 0.0238871, -0.0365151, 0.0365151
4: -0.0025098, 0.0005510, -0.0025098, 0.0005510, -0.0030608, 0.0030608
5: 0.0147338, 0.0181401, 0.0147338, 0.0181401, -0.0034063, 0.0034063
6: 0.0024023, 0.0047403, 0.0024023, 0.0047403, -0.0023380, 0.0023380
7: -0.0139688, -0.0030444, -0.0139688, -0.0030444, -0.0109244, 0.0109244
8: 0.0056470, 0.0139213, 0.0056470, 0.0139213, -0.0082744, 0.0082744
9: 0.0078812, 0.0227635, 0.0078812, 0.0227635, -0.0144461, 0.0144461

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037068
time: 2.23 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036372, upper bound: 0.0037091
time: 2.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.93 seconds
NS_B1_A1_A1, status: Status.VERIFIED, split count: 3, time: 7.93
Output dim: 2, lower bound: -0.0035362, upper bound: 0.0036368
NS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 7.93
Output dim: 2, lower bound: -0.0035362, upper bound: 0.0036368
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 7.93
Output dim: 2, lower bound: -0.0038699, upper bound: 0.0035291
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 7.93
Output dim: 2, lower bound: -0.0038699, upper bound: 0.0036373
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 7.93
Output dim: 2, lower bound: -0.0036368, upper bound: 0.0037143
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 7.93
Output dim: 2, lower bound: -0.0036373, upper bound: 0.0038715
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 7.93
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037068
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 7.93
Output dim: 2, lower bound: -0.0036372, upper bound: 0.0037091

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042021, -0.0040812, -0.0041928, -0.0040870, -0.0001060, 0.0001116
1: -0.0100246, -0.0054979, -0.0096765, -0.0057132, -0.0039692, 0.0041787
2: 0.9644336, 0.9698657, 0.9648512, 0.9696074, -0.0047632, 0.0050145
3: -0.0160264, 0.0240411, -0.0129456, 0.0221351, -0.0351325, 0.0339733
4: -0.0025215, 0.0005259, -0.0023765, 0.0002916, -0.0028131, 0.0026720
5: 0.0147219, 0.0180662, 0.0148684, 0.0175650, -0.0025889, 0.0031978
6: 0.0025438, 0.0047460, 0.0033632, 0.0046748, -0.0021309, 0.0012592
7: -0.0140087, -0.0032067, -0.0135148, -0.0044233, -0.0087284, 0.0103080
8: 0.0056153, 0.0138534, 0.0060072, 0.0132199, -0.0076046, 0.0072234
9: 0.0078243, 0.0226412, 0.0085291, 0.0215019, -0.0133217, 0.0129920

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036572, upper bound: 0.0035291
time: 2.64 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036572, upper bound: 0.0035291
time: 2.86 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042021, -0.0040812, -0.0041961, -0.0040847, -0.0001041, 0.0001149
1: -0.0100246, -0.0054979, -0.0098004, -0.0056278, -0.0038999, 0.0043025
2: 0.9644336, 0.9698657, 0.9647025, 0.9697099, -0.0046800, 0.0051632
3: -0.0160264, 0.0240411, -0.0140420, 0.0228907, -0.0345189, 0.0344674
4: -0.0025215, 0.0005259, -0.0024340, 0.0003749, -0.0028964, 0.0026254
5: 0.0147219, 0.0180662, 0.0148104, 0.0176493, -0.0025702, 0.0032558
6: 0.0025438, 0.0047460, 0.0033222, 0.0047030, -0.0021592, 0.0012502
7: -0.0140087, -0.0032067, -0.0137106, -0.0041391, -0.0086655, 0.0105039
8: 0.0056153, 0.0138534, 0.0058518, 0.0134453, -0.0078301, 0.0070972
9: 0.0078243, 0.0226412, 0.0082497, 0.0219074, -0.0136629, 0.0127651

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036572, upper bound: 0.0035439
time: 3.13 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036572, upper bound: 0.0035440
time: 3.04 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0041981, -0.0040844, -0.0041972, -0.0040846, -0.0001135, 0.0001128
1: -0.0098731, -0.0056154, -0.0098402, -0.0056239, -0.0042492, 0.0042248
2: 0.9646153, 0.9697247, 0.9646549, 0.9697145, -0.0050992, 0.0050698
3: -0.0146855, 0.0230010, -0.0143939, 0.0229255, -0.0345402, 0.0344639
4: -0.0024424, 0.0004239, -0.0024367, 0.0004017, -0.0028441, 0.0028605
5: 0.0148019, 0.0177664, 0.0148077, 0.0177012, -0.0028994, 0.0029587
6: 0.0031179, 0.0047072, 0.0032427, 0.0047043, -0.0015864, 0.0014644
7: -0.0137392, -0.0038654, -0.0137196, -0.0040086, -0.0097306, 0.0098542
8: 0.0058291, 0.0135776, 0.0058447, 0.0135177, -0.0076885, 0.0077330
9: 0.0082089, 0.0221453, 0.0082368, 0.0220375, -0.0134804, 0.0135500

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037143
time: 2.84 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037143
time: 2.85 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0041985, -0.0040843, -0.0042008, -0.0040821, -0.0001164, 0.0001164
1: -0.0098901, -0.0056137, -0.0099743, -0.0055295, -0.0043606, 0.0043607
2: 0.9645948, 0.9697268, 0.9644938, 0.9698278, -0.0052329, 0.0052330
3: -0.0148362, 0.0230159, -0.0155816, 0.0237608, -0.0356398, 0.0349915
4: -0.0024435, 0.0004354, -0.0025002, 0.0004920, -0.0029356, 0.0029355
5: 0.0148007, 0.0178001, 0.0147435, 0.0179668, -0.0031660, 0.0030566
6: 0.0030534, 0.0047077, 0.0027343, 0.0047356, -0.0016822, 0.0019734
7: -0.0137430, -0.0037913, -0.0139361, -0.0034252, -0.0103178, 0.0101447
8: 0.0058261, 0.0136086, 0.0056729, 0.0137619, -0.0079358, 0.0079357
9: 0.0082034, 0.0222011, 0.0079279, 0.0224767, -0.0138539, 0.0139242

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038699
time: 2.87 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038715
time: 2.18 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0041972, -0.0040846, -0.0042026, -0.0040817, -0.0001154, 0.0001180
1: -0.0098402, -0.0056239, -0.0100445, -0.0055170, -0.0043232, 0.0044206
2: 0.9646549, 0.9697145, 0.9644096, 0.9698428, -0.0051879, 0.0053049
3: -0.0143939, 0.0229255, -0.0162029, 0.0238720, -0.0345330, 0.0353013
4: -0.0024367, 0.0004017, -0.0025086, 0.0005393, -0.0029759, 0.0029103
5: 0.0148077, 0.0177012, 0.0147349, 0.0181057, -0.0032980, 0.0029663
6: 0.0032427, 0.0047043, 0.0024683, 0.0047397, -0.0014970, 0.0022360
7: -0.0137196, -0.0040086, -0.0139649, -0.0031201, -0.0105995, 0.0099563
8: 0.0058447, 0.0135177, 0.0056501, 0.0138896, -0.0080450, 0.0078676
9: 0.0082368, 0.0220375, 0.0078868, 0.0227065, -0.0140267, 0.0137122

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037225, upper bound: 0.0035876
time: 2.91 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037225, upper bound: 0.0037068
time: 2.65 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0042008, -0.0040821, -0.0042031, -0.0040817, -0.0001191, 0.0001210
1: -0.0099743, -0.0055295, -0.0100617, -0.0055153, -0.0044590, 0.0045322
2: 0.9644938, 0.9698278, 0.9643890, 0.9698448, -0.0053510, 0.0054388
3: -0.0155816, 0.0237608, -0.0163550, 0.0238868, -0.0351198, 0.0363869
4: -0.0025002, 0.0004920, -0.0025098, 0.0005509, -0.0030510, 0.0030018
5: 0.0147435, 0.0179668, 0.0147338, 0.0181396, -0.0033962, 0.0032330
6: 0.0027343, 0.0047356, 0.0024032, 0.0047403, -0.0020060, 0.0023324
7: -0.0139361, -0.0034252, -0.0139687, -0.0030454, -0.0108907, 0.0105435
8: 0.0056729, 0.0137619, 0.0056470, 0.0139209, -0.0082480, 0.0081149
9: 0.0079279, 0.0224767, 0.0078813, 0.0227627, -0.0143988, 0.0140938

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0035876
time: 3.32 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0037091
time: 3.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 9.16 seconds
NS_B1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0036572, upper bound: 0.0035291
NS_B1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0036572, upper bound: 0.0035291
NS_B1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0036572, upper bound: 0.0035439
NS_B1_A2_B2_A2, status: Status.VERIFIED, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0036572, upper bound: 0.0035440
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037143
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037143
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038699
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038715
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0037225, upper bound: 0.0035876
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0037225, upper bound: 0.0037068
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0035876
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.16
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0037091

## BFS NS instance: NS_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0041928, -0.0040870, -0.0041972, -0.0040846, -0.0001082, 0.0001006
1: -0.0096765, -0.0057132, -0.0098402, -0.0056239, -0.0040526, 0.0037673
2: 0.9648512, 0.9696074, 0.9646549, 0.9697145, -0.0048633, 0.0045209
3: -0.0129456, 0.0221351, -0.0143939, 0.0229255, -0.0327484, 0.0333453
4: -0.0023765, 0.0002916, -0.0024367, 0.0004017, -0.0025361, 0.0027282
5: 0.0148684, 0.0175650, 0.0148077, 0.0177012, -0.0028328, 0.0024905
6: 0.0033632, 0.0046748, 0.0032427, 0.0047043, -0.0012114, 0.0014320
7: -0.0135148, -0.0044233, -0.0137196, -0.0040086, -0.0095062, 0.0083968
8: 0.0060072, 0.0132199, 0.0058447, 0.0135177, -0.0068559, 0.0073752
9: 0.0085291, 0.0215019, 0.0082368, 0.0220375, -0.0123311, 0.0129002

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_B2_A1_B1_A1_A1

### Relational analysis result of NS_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036452
time: 3.08 seconds

## Relational analysis of NS_B2_A1_B1_A1_A2

### Relational analysis result of NS_B2_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036313
time: 2.19 seconds

## BFS NS instance: NS_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0041961, -0.0040847, -0.0041972, -0.0040846, -0.0001115, 0.0001034
1: -0.0098004, -0.0056278, -0.0098402, -0.0056239, -0.0041765, 0.0038736
2: 0.9647025, 0.9697099, 0.9646549, 0.9697145, -0.0050119, 0.0046484
3: -0.0140420, 0.0228907, -0.0143939, 0.0229255, -0.0339625, 0.0342860
4: -0.0024340, 0.0003749, -0.0024367, 0.0004017, -0.0026077, 0.0028116
5: 0.0148104, 0.0176493, 0.0148077, 0.0177012, -0.0028909, 0.0025925
6: 0.0033222, 0.0047030, 0.0032427, 0.0047043, -0.0012610, 0.0014603
7: -0.0137106, -0.0041391, -0.0137196, -0.0040086, -0.0097020, 0.0087407
8: 0.0058518, 0.0134453, 0.0058447, 0.0135177, -0.0070494, 0.0076007
9: 0.0082497, 0.0219074, 0.0082368, 0.0220375, -0.0126790, 0.0133199

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_B2_A1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036452
time: 2.00 seconds

## Relational analysis of NS_B2_A1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036313
time: 2.55 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0041928, -0.0040870, -0.0042008, -0.0040821, -0.0001107, 0.0001046
1: -0.0096765, -0.0057132, -0.0099743, -0.0055295, -0.0041470, 0.0039156
2: 0.9648512, 0.9696074, 0.9644938, 0.9698278, -0.0049766, 0.0046989
3: -0.0129456, 0.0221351, -0.0155816, 0.0237608, -0.0336920, 0.0346581
4: -0.0023765, 0.0002916, -0.0025002, 0.0004920, -0.0026359, 0.0027917
5: 0.0148684, 0.0175650, 0.0147435, 0.0179668, -0.0030983, 0.0025686
6: 0.0033632, 0.0046748, 0.0027343, 0.0047356, -0.0012494, 0.0019405
7: -0.0135148, -0.0044233, -0.0139361, -0.0034252, -0.0100895, 0.0086601
8: 0.0060072, 0.0132199, 0.0056729, 0.0137619, -0.0071258, 0.0075470
9: 0.0085291, 0.0215019, 0.0079279, 0.0224767, -0.0128165, 0.0132187

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_B2_A1_B2_A1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0037985
time: 2.53 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2

### Relational analysis result of NS_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0037816
time: 2.90 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0041961, -0.0040847, -0.0042008, -0.0040821, -0.0001140, 0.0001029
1: -0.0098004, -0.0056278, -0.0099743, -0.0055295, -0.0042709, 0.0038523
2: 0.9647025, 0.9697099, 0.9644938, 0.9698278, -0.0051252, 0.0046230
3: -0.0140420, 0.0228907, -0.0155816, 0.0237608, -0.0341879, 0.0340983
4: -0.0024340, 0.0003749, -0.0025002, 0.0004920, -0.0025934, 0.0028751
5: 0.0148104, 0.0176493, 0.0147435, 0.0179668, -0.0031564, 0.0025503
6: 0.0033222, 0.0047030, 0.0027343, 0.0047356, -0.0012405, 0.0019688
7: -0.0137106, -0.0041391, -0.0139361, -0.0034252, -0.0102854, 0.0085984
8: 0.0058518, 0.0134453, 0.0056729, 0.0137619, -0.0070108, 0.0077724
9: 0.0082497, 0.0219074, 0.0079279, 0.0224767, -0.0126095, 0.0135597

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_B2_A1_B2_A2_A1

### Relational analysis result of NS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0036649
time: 2.59 seconds

## Relational analysis of NS_B2_A1_B2_A2_A2

### Relational analysis result of NS_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0036508
time: 2.54 seconds

## BFS NS instance: NS_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0041972, -0.0040846, -0.0041972, -0.0040846, -0.0001126, 0.0001126
1: -0.0098402, -0.0056239, -0.0098402, -0.0056239, -0.0042163, 0.0042163
2: 0.9646549, 0.9697145, 0.9646549, 0.9697145, -0.0050595, 0.0050595
3: -0.0143939, 0.0229255, -0.0143939, 0.0229255, -0.0334950, 0.0334950
4: -0.0024367, 0.0004017, -0.0024367, 0.0004017, -0.0028384, 0.0028384
5: 0.0148077, 0.0177012, 0.0148077, 0.0177012, -0.0028936, 0.0028936
6: 0.0032427, 0.0047043, 0.0032427, 0.0047043, -0.0014616, 0.0014616
7: -0.0137196, -0.0040086, -0.0137196, -0.0040086, -0.0097110, 0.0097110
8: 0.0058447, 0.0135177, 0.0058447, 0.0135177, -0.0076730, 0.0076730
9: 0.0082368, 0.0220375, 0.0082368, 0.0220375, -0.0133562, 0.0133562

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_B2_A2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0035157
time: 3.13 seconds

## Relational analysis of NS_B2_A2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0035135
time: 3.67 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0041972, -0.0040846, -0.0042008, -0.0040821, -0.0001151, 0.0001162
1: -0.0098402, -0.0056239, -0.0099743, -0.0055295, -0.0043106, 0.0043504
2: 0.9646549, 0.9697145, 0.9644938, 0.9698278, -0.0051728, 0.0052207
3: -0.0143939, 0.0229255, -0.0155816, 0.0237608, -0.0344220, 0.0347380
4: -0.0024367, 0.0004017, -0.0025002, 0.0004920, -0.0029287, 0.0029019
5: 0.0148077, 0.0177012, 0.0147435, 0.0179668, -0.0031591, 0.0029578
6: 0.0032427, 0.0047043, 0.0027343, 0.0047356, -0.0014928, 0.0019701
7: -0.0137196, -0.0040086, -0.0139361, -0.0034252, -0.0102944, 0.0099275
8: 0.0058447, 0.0135177, 0.0056729, 0.0137619, -0.0079172, 0.0078448
9: 0.0082368, 0.0220375, 0.0079279, 0.0224767, -0.0138015, 0.0136712

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0036253
time: 3.14 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0036242
time: 3.24 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042008, -0.0040821, -0.0041972, -0.0040846, -0.0001162, 0.0001151
1: -0.0099743, -0.0055295, -0.0098402, -0.0056239, -0.0043504, 0.0043106
2: 0.9644938, 0.9698278, 0.9646549, 0.9697145, -0.0052207, 0.0051728
3: -0.0155816, 0.0237608, -0.0143939, 0.0229255, -0.0347380, 0.0344220
4: -0.0025002, 0.0004920, -0.0024367, 0.0004017, -0.0029019, 0.0029287
5: 0.0147435, 0.0179668, 0.0148077, 0.0177012, -0.0029578, 0.0031591
6: 0.0027343, 0.0047356, 0.0032427, 0.0047043, -0.0019701, 0.0014928
7: -0.0139361, -0.0034252, -0.0137196, -0.0040086, -0.0099275, 0.0102944
8: 0.0056729, 0.0137619, 0.0058447, 0.0135177, -0.0078448, 0.0079172
9: 0.0079279, 0.0224767, 0.0082368, 0.0220375, -0.0136712, 0.0138015

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_B2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0035071
time: 9.78 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035071
time: 3.25 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042008, -0.0040821, -0.0042008, -0.0040821, -0.0001187, 0.0001187
1: -0.0099743, -0.0055295, -0.0099743, -0.0055295, -0.0044448, 0.0044448
2: 0.9644938, 0.9698278, 0.9644938, 0.9698278, -0.0053340, 0.0053340
3: -0.0155816, 0.0237608, -0.0155816, 0.0237608, -0.0349928, 0.0349928
4: -0.0025002, 0.0004920, -0.0025002, 0.0004920, -0.0029922, 0.0029922
5: 0.0147435, 0.0179668, 0.0147435, 0.0179668, -0.0032233, 0.0032233
6: 0.0027343, 0.0047356, 0.0027343, 0.0047356, -0.0020013, 0.0020013
7: -0.0139361, -0.0034252, -0.0139361, -0.0034252, -0.0105109, 0.0105109
8: 0.0056729, 0.0137619, 0.0056729, 0.0137619, -0.0080890, 0.0080890
9: 0.0079279, 0.0224767, 0.0079279, 0.0224767, -0.0140473, 0.0140473

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035356
time: 3.08 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035333
time: 2.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 8.31 seconds
NS_B2_A1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036452
NS_B2_A1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036313
NS_B2_A1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036452
NS_B2_A1_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036313
NS_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0037985
NS_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0037816
NS_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0036649
NS_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0036508
NS_B2_A2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0035157
NS_B2_A2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0035135
NS_B2_A2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0036253
NS_B2_A2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0036242
NS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0038103, upper bound: 0.0035071
NS_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035071
NS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035356
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.31
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035333

## BFS NS instance: NS_B2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0041926, -0.0040894, -0.0042008, -0.0040821, -0.0001105, 0.0001015
1: -0.0096675, -0.0058039, -0.0099743, -0.0055295, -0.0041380, 0.0038023
2: 0.9648620, 0.9694986, 0.9644938, 0.9698278, -0.0049657, 0.0045629
3: -0.0128660, 0.0213324, -0.0155816, 0.0237608, -0.0336156, 0.0336553
4: -0.0023155, 0.0002855, -0.0025002, 0.0004920, -0.0025597, 0.0027857
5: 0.0149301, 0.0175589, 0.0147435, 0.0179668, -0.0030366, 0.0025634
6: 0.0033661, 0.0046448, 0.0027343, 0.0047356, -0.0012468, 0.0019105
7: -0.0133067, -0.0044439, -0.0139361, -0.0034252, -0.0098815, 0.0086424
8: 0.0061722, 0.0132036, 0.0056729, 0.0137619, -0.0069197, 0.0075306
9: 0.0088259, 0.0214725, 0.0079279, 0.0224767, -0.0124457, 0.0131894

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_B2_A1_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0032689, upper bound: 0.0037025
time: 2.70 seconds

## Relational analysis of NS_B2_A1_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032419, upper bound: 0.0036092
time: 2.62 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0041954, -0.0040894, -0.0042007, -0.0040826, -0.0001129, 0.0001046
1: -0.0097747, -0.0058027, -0.0099728, -0.0055478, -0.0042270, 0.0039166
2: 0.9647334, 0.9694999, 0.9644958, 0.9698060, -0.0050725, 0.0047000
3: -0.0138146, 0.0213429, -0.0155677, 0.0235995, -0.0344696, 0.0346667
4: -0.0023163, 0.0003577, -0.0024879, 0.0004910, -0.0026366, 0.0028456
5: 0.0149293, 0.0176318, 0.0147559, 0.0179636, -0.0030343, 0.0026385
6: 0.0033307, 0.0046452, 0.0027402, 0.0047295, -0.0012834, 0.0019049
7: -0.0133094, -0.0041981, -0.0138943, -0.0034320, -0.0098774, 0.0088958
8: 0.0061701, 0.0133986, 0.0057061, 0.0137590, -0.0071276, 0.0076925
9: 0.0088221, 0.0218233, 0.0079876, 0.0224716, -0.0128197, 0.0134902

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_B2_A1_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0032641, upper bound: 0.0036795
time: 3.19 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032303, upper bound: 0.0035941
time: 3.01 seconds

## BFS NS instance: NS_B2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0041959, -0.0040871, -0.0042008, -0.0040821, -0.0001138, 0.0000999
1: -0.0097914, -0.0057188, -0.0099743, -0.0055295, -0.0042618, 0.0037400
2: 0.9647134, 0.9696007, 0.9644938, 0.9698278, -0.0051144, 0.0044881
3: -0.0139621, 0.0220859, -0.0155816, 0.0237608, -0.0341114, 0.0331035
4: -0.0023728, 0.0003689, -0.0025002, 0.0004920, -0.0025177, 0.0028690
5: 0.0148722, 0.0176432, 0.0147435, 0.0179668, -0.0030945, 0.0025451
6: 0.0033252, 0.0046729, 0.0027343, 0.0047356, -0.0012379, 0.0019387
7: -0.0135020, -0.0041599, -0.0139361, -0.0034252, -0.0100768, 0.0085806
8: 0.0060173, 0.0134289, 0.0056729, 0.0137619, -0.0068062, 0.0077560
9: 0.0085473, 0.0218778, 0.0079279, 0.0224767, -0.0122416, 0.0135306

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_B2_A1_B2_A2_A1_A1

### Relational analysis result of NS_B2_A1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033923, upper bound: 0.0035490
time: 3.32 seconds

## Relational analysis of NS_B2_A1_B2_A2_A1_A2

### Relational analysis result of NS_B2_A1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0033789, upper bound: 0.0035030
time: 2.83 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0042008, -0.0040821, -0.0041969, -0.0040869, -0.0001138, 0.0001149
1: -0.0099743, -0.0055295, -0.0098314, -0.0057112, -0.0042631, 0.0043018
2: 0.9644938, 0.9698278, 0.9646655, 0.9696097, -0.0051159, 0.0051623
3: -0.0155816, 0.0237608, -0.0143161, 0.0221527, -0.0339865, 0.0343456
4: -0.0025002, 0.0004920, -0.0023779, 0.0003958, -0.0028960, 0.0028699
5: 0.0147435, 0.0179668, 0.0148671, 0.0176838, -0.0029404, 0.0030997
6: 0.0027343, 0.0047356, 0.0032760, 0.0046754, -0.0019412, 0.0014595
7: -0.0139361, -0.0034252, -0.0135193, -0.0040468, -0.0098893, 0.0100941
8: 0.0056729, 0.0137619, 0.0060036, 0.0135017, -0.0078288, 0.0077583
9: 0.0079279, 0.0224767, 0.0085226, 0.0220087, -0.0136424, 0.0135153

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035051
time: 3.43 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035071
time: 2.77 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0042007, -0.0040826, -0.0041995, -0.0040870, -0.0001137, 0.0001169
1: -0.0099728, -0.0055478, -0.0099265, -0.0057136, -0.0042592, 0.0043787
2: 0.9644958, 0.9698060, 0.9645513, 0.9696069, -0.0051111, 0.0052546
3: -0.0155677, 0.0235995, -0.0151578, 0.0221317, -0.0343044, 0.0351544
4: -0.0024879, 0.0004910, -0.0023763, 0.0004598, -0.0029477, 0.0028673
5: 0.0147559, 0.0179636, 0.0148687, 0.0178720, -0.0031161, 0.0030949
6: 0.0027402, 0.0047295, 0.0029157, 0.0046747, -0.0019344, 0.0018138
7: -0.0138943, -0.0034320, -0.0135139, -0.0036334, -0.0102609, 0.0100818
8: 0.0057061, 0.0137590, 0.0060079, 0.0136748, -0.0079687, 0.0077512
9: 0.0079876, 0.0224716, 0.0085304, 0.0223200, -0.0139070, 0.0135383

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_B2_A2_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036076, upper bound: 0.0033400
time: 2.43 seconds

## Relational analysis of NS_B2_A2_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036061, upper bound: 0.0032955
time: 2.87 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0042008, -0.0040821, -0.0001184, 0.0001164
1: -0.0099656, -0.0056155, -0.0099743, -0.0055295, -0.0044360, 0.0043588
2: 0.9645044, 0.9697246, 0.9644938, 0.9698278, -0.0053234, 0.0052308
3: -0.0155038, 0.0229998, -0.0155816, 0.0237608, -0.0349170, 0.0342437
4: -0.0024423, 0.0004861, -0.0025002, 0.0004920, -0.0029343, 0.0029863
5: 0.0148020, 0.0179494, 0.0147435, 0.0179668, -0.0031648, 0.0032059
6: 0.0027676, 0.0047071, 0.0027343, 0.0047356, -0.0019680, 0.0019728
7: -0.0137389, -0.0034634, -0.0139361, -0.0034252, -0.0103136, 0.0104726
8: 0.0058294, 0.0137459, 0.0056729, 0.0137619, -0.0079325, 0.0080730
9: 0.0082093, 0.0224479, 0.0079279, 0.0224767, -0.0137656, 0.0140185

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037982, upper bound: 0.0035313
time: 2.56 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037982, upper bound: 0.0035313
time: 2.60 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0042007, -0.0040826, -0.0001201, 0.0001162
1: -0.0100444, -0.0056220, -0.0099728, -0.0055478, -0.0044966, 0.0043508
2: 0.9644098, 0.9697168, 0.9644958, 0.9698060, -0.0053962, 0.0052210
3: -0.0162017, 0.0229426, -0.0155677, 0.0235995, -0.0356448, 0.0345104
4: -0.0024379, 0.0005392, -0.0024879, 0.0004910, -0.0029289, 0.0030271
5: 0.0148064, 0.0181054, 0.0147559, 0.0179636, -0.0031573, 0.0033495
6: 0.0024688, 0.0047050, 0.0027402, 0.0047295, -0.0022607, 0.0019648
7: -0.0137240, -0.0031207, -0.0138943, -0.0034320, -0.0102920, 0.0107736
8: 0.0058411, 0.0138894, 0.0057061, 0.0137590, -0.0079179, 0.0081833
9: 0.0082305, 0.0227060, 0.0079876, 0.0224716, -0.0137735, 0.0142288

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036941, upper bound: 0.0033864
time: 2.73 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033667
time: 3.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 8.30 seconds
NS_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0032689, upper bound: 0.0037025
NS_B2_A1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0032419, upper bound: 0.0036092
NS_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0032641, upper bound: 0.0036795
NS_B2_A1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0032303, upper bound: 0.0035941
NS_B2_A1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0033923, upper bound: 0.0035490
NS_B2_A1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0033789, upper bound: 0.0035030
NS_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035051
NS_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035071
NS_B2_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0036076, upper bound: 0.0033400
NS_B2_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0036061, upper bound: 0.0032955
NS_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0037982, upper bound: 0.0035313
NS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0037982, upper bound: 0.0035313
NS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0036941, upper bound: 0.0033864
NS_B2_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 8.30
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033667

## BFS NS instance: NS_B2_A1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0041925, -0.0040909, -0.0042008, -0.0040821, -0.0001104, 0.0000994
1: -0.0096635, -0.0058594, -0.0099743, -0.0055295, -0.0041339, 0.0037239
2: 0.9648669, 0.9694319, 0.9644938, 0.9698278, -0.0049609, 0.0044688
3: -0.0128300, 0.0208410, -0.0155816, 0.0237608, -0.0335820, 0.0329613
4: -0.0022781, 0.0002828, -0.0025002, 0.0004920, -0.0025069, 0.0027829
5: 0.0149679, 0.0175561, 0.0147435, 0.0179668, -0.0029988, 0.0025608
6: 0.0033675, 0.0046264, 0.0027343, 0.0047356, -0.0012456, 0.0018921
7: -0.0131794, -0.0044532, -0.0139361, -0.0034252, -0.0097542, 0.0086337
8: 0.0062733, 0.0131962, 0.0056729, 0.0137619, -0.0067770, 0.0075232
9: 0.0090077, 0.0214592, 0.0079279, 0.0224767, -0.0121891, 0.0131763

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_B2_A1_B2_A1_A1_A1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031701, upper bound: 0.0036334
time: 2.71 seconds

## Relational analysis of NS_B2_A1_B2_A1_A1_A1_A2

### Relational analysis result of NS_B2_A1_B2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031974, upper bound: 0.0036334
time: 2.56 seconds

## BFS NS instance: NS_B2_A1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0041953, -0.0040909, -0.0042007, -0.0040826, -0.0001128, 0.0001026
1: -0.0097706, -0.0058585, -0.0099728, -0.0055478, -0.0042228, 0.0038433
2: 0.9647384, 0.9694330, 0.9644958, 0.9698060, -0.0050676, 0.0046122
3: -0.0137780, 0.0208491, -0.0155677, 0.0235995, -0.0344352, 0.0340185
4: -0.0022787, 0.0003549, -0.0024879, 0.0004910, -0.0025873, 0.0028428
5: 0.0149673, 0.0176290, 0.0147559, 0.0179636, -0.0029964, 0.0026359
6: 0.0033320, 0.0046267, 0.0027402, 0.0047295, -0.0012821, 0.0018865
7: -0.0131815, -0.0042076, -0.0138943, -0.0034320, -0.0097494, 0.0088869
8: 0.0062716, 0.0133911, 0.0057061, 0.0137590, -0.0069943, 0.0076850
9: 0.0090046, 0.0218097, 0.0079876, 0.0224716, -0.0125800, 0.0134768

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_B2_A1_B2_A1_A2_A1_A1

### Relational analysis result of NS_B2_A1_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031639, upper bound: 0.0036072
time: 2.72 seconds

## Relational analysis of NS_B2_A1_B2_A1_A2_A1_A2

### Relational analysis result of NS_B2_A1_B2_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031937, upper bound: 0.0036072
time: 2.50 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041969, -0.0040869, -0.0001136, 0.0001126
1: -0.0099656, -0.0056155, -0.0098314, -0.0057112, -0.0042543, 0.0042159
2: 0.9645044, 0.9697246, 0.9646655, 0.9696097, -0.0051054, 0.0050591
3: -0.0155038, 0.0229998, -0.0143161, 0.0221527, -0.0339109, 0.0335975
4: -0.0024423, 0.0004861, -0.0023779, 0.0003958, -0.0028381, 0.0028640
5: 0.0148020, 0.0179494, 0.0148671, 0.0176838, -0.0028819, 0.0030823
6: 0.0027676, 0.0047071, 0.0032760, 0.0046754, -0.0019079, 0.0014311
7: -0.0137389, -0.0034634, -0.0135193, -0.0040468, -0.0096921, 0.0100559
8: 0.0058294, 0.0137459, 0.0060036, 0.0135017, -0.0076723, 0.0077423
9: 0.0082093, 0.0224479, 0.0085226, 0.0220087, -0.0133617, 0.0134865

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 155

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_B2_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037097, upper bound: 0.0033259
time: 2.58 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036201, upper bound: 0.0033009
time: 2.46 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0041969, -0.0040869, -0.0001157, 0.0001124
1: -0.0100444, -0.0056220, -0.0098314, -0.0057112, -0.0043332, 0.0042094
2: 0.9644098, 0.9697168, 0.9646655, 0.9696097, -0.0052000, 0.0050513
3: -0.0162017, 0.0229426, -0.0143161, 0.0221527, -0.0347492, 0.0335417
4: -0.0024379, 0.0005392, -0.0023779, 0.0003958, -0.0028337, 0.0029171
5: 0.0148064, 0.0181054, 0.0148671, 0.0176838, -0.0028775, 0.0032383
6: 0.0024688, 0.0047050, 0.0032760, 0.0046754, -0.0022066, 0.0014289
7: -0.0137240, -0.0031207, -0.0135193, -0.0040468, -0.0096773, 0.0103987
8: 0.0058411, 0.0138894, 0.0060036, 0.0135017, -0.0076606, 0.0078858
9: 0.0082305, 0.0227060, 0.0085226, 0.0220087, -0.0133403, 0.0137528

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_B2_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037097, upper bound: 0.0033260
time: 2.70 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036201, upper bound: 0.0033009
time: 2.63 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0042005, -0.0040844, -0.0001162, 0.0001162
1: -0.0099656, -0.0056155, -0.0099656, -0.0056155, -0.0043501, 0.0043501
2: 0.9645044, 0.9697246, 0.9645044, 0.9697246, -0.0052202, 0.0052202
3: -0.0155038, 0.0229998, -0.0155038, 0.0229998, -0.0341678, 0.0341678
4: -0.0024423, 0.0004861, -0.0024423, 0.0004861, -0.0029284, 0.0029284
5: 0.0148020, 0.0179494, 0.0148020, 0.0179494, -0.0031474, 0.0031474
6: 0.0027676, 0.0047071, 0.0027676, 0.0047071, -0.0019395, 0.0019395
7: -0.0137389, -0.0034634, -0.0137389, -0.0034634, -0.0102754, 0.0102754
8: 0.0058294, 0.0137459, 0.0058294, 0.0137459, -0.0079165, 0.0079165
9: 0.0082093, 0.0224479, 0.0082093, 0.0224479, -0.0137368, 0.0137368

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036208, upper bound: 0.0034040
time: 2.95 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033741
time: 2.44 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0042026, -0.0040845, -0.0001160, 0.0001183
1: -0.0099656, -0.0056155, -0.0100444, -0.0056220, -0.0043436, 0.0044289
2: 0.9645044, 0.9697246, 0.9644098, 0.9697168, -0.0052124, 0.0053148
3: -0.0155038, 0.0229998, -0.0162017, 0.0229426, -0.0341189, 0.0350557
4: -0.0024423, 0.0004861, -0.0024379, 0.0005392, -0.0029815, 0.0029241
5: 0.0148020, 0.0179494, 0.0148064, 0.0181054, -0.0033034, 0.0031430
6: 0.0027676, 0.0047071, 0.0024688, 0.0047050, -0.0019374, 0.0022383
7: -0.0137389, -0.0034634, -0.0137240, -0.0031207, -0.0106182, 0.0102606
8: 0.0058294, 0.0137459, 0.0058411, 0.0138894, -0.0080600, 0.0079048
9: 0.0082093, 0.0224479, 0.0082305, 0.0227060, -0.0140066, 0.0137166

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036208, upper bound: 0.0034040
time: 3.30 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033742
time: 3.39 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0042006, -0.0040839, -0.0001187, 0.0001161
1: -0.0100444, -0.0056220, -0.0099693, -0.0055978, -0.0044466, 0.0043473
2: 0.9644098, 0.9697168, 0.9644999, 0.9697458, -0.0053360, 0.0052169
3: -0.0162017, 0.0229426, -0.0155367, 0.0231569, -0.0351665, 0.0344801
4: -0.0024379, 0.0005392, -0.0024542, 0.0004886, -0.0029266, 0.0029934
5: 0.0148064, 0.0181054, 0.0147899, 0.0179567, -0.0031504, 0.0033155
6: 0.0024688, 0.0047050, 0.0027535, 0.0047130, -0.0022441, 0.0019515
7: -0.0137240, -0.0031207, -0.0137796, -0.0034473, -0.0102768, 0.0106589
8: 0.0058411, 0.0138894, 0.0057971, 0.0137527, -0.0079115, 0.0080923
9: 0.0082305, 0.0227060, 0.0081513, 0.0224601, -0.0137620, 0.0140611

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035824, upper bound: 0.0033183
time: 3.79 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036209, upper bound: 0.0033182
time: 2.51 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 9.14 seconds
NS_B2_A1_B2_A1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0031701, upper bound: 0.0036334
NS_B2_A1_B2_A1_A1_A1_A2, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0031974, upper bound: 0.0036334
NS_B2_A1_B2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0031639, upper bound: 0.0036072
NS_B2_A1_B2_A1_A2_A1_A2, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0031937, upper bound: 0.0036072
NS_B2_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0037097, upper bound: 0.0033259
NS_B2_A2_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0036201, upper bound: 0.0033009
NS_B2_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0037097, upper bound: 0.0033260
NS_B2_A2_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0036201, upper bound: 0.0033009
NS_B2_A2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0036208, upper bound: 0.0034040
NS_B2_A2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033741
NS_B2_A2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0036208, upper bound: 0.0034040
NS_B2_A2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033742
NS_B2_A2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0035824, upper bound: 0.0033183
NS_B2_A2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 9.14
Output dim: 2, lower bound: -0.0036209, upper bound: 0.0033182

## BFS NS instance: NS_B2_A2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041968, -0.0040883, -0.0001122, 0.0001125
1: -0.0099656, -0.0056155, -0.0098276, -0.0057642, -0.0042014, 0.0042121
2: 0.9645044, 0.9697246, 0.9646698, 0.9695463, -0.0050419, 0.0050548
3: -0.0155038, 0.0229998, -0.0142830, 0.0216840, -0.0334167, 0.0335649
4: -0.0024423, 0.0004861, -0.0023422, 0.0003933, -0.0028356, 0.0028284
5: 0.0148020, 0.0179494, 0.0149031, 0.0176764, -0.0028745, 0.0030463
6: 0.0027676, 0.0047071, 0.0032902, 0.0046579, -0.0018904, 0.0014169
7: -0.0137389, -0.0034634, -0.0133979, -0.0040630, -0.0096758, 0.0099344
8: 0.0058294, 0.0137459, 0.0060999, 0.0134949, -0.0076655, 0.0076460
9: 0.0082093, 0.0224479, 0.0086959, 0.0219965, -0.0133494, 0.0133114

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: B, layer: 1, pos: 155

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_B2_A2_A2_B1_B1_A1_B1_B1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036413, upper bound: 0.0032508
time: 3.22 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A1_B1_B2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036414, upper bound: 0.0032775
time: 2.96 seconds

## BFS NS instance: NS_B2_A2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0041968, -0.0040883, -0.0001143, 0.0001123
1: -0.0100444, -0.0056220, -0.0098276, -0.0057642, -0.0042802, 0.0042057
2: 0.9644098, 0.9697168, 0.9646698, 0.9695463, -0.0051365, 0.0050470
3: -0.0162017, 0.0229426, -0.0142830, 0.0216840, -0.0342550, 0.0335091
4: -0.0024379, 0.0005392, -0.0023422, 0.0003933, -0.0028312, 0.0028814
5: 0.0148064, 0.0181054, 0.0149031, 0.0176764, -0.0028701, 0.0032023
6: 0.0024688, 0.0047050, 0.0032902, 0.0046579, -0.0021891, 0.0014148
7: -0.0137240, -0.0031207, -0.0133979, -0.0040630, -0.0096610, 0.0102772
8: 0.0058411, 0.0138894, 0.0060999, 0.0134949, -0.0076538, 0.0077894
9: 0.0082305, 0.0227060, 0.0086959, 0.0219965, -0.0133280, 0.0135777

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_B2_A2_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035876, upper bound: 0.0032554
time: 2.90 seconds

## Relational analysis of NS_B2_A2_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036377, upper bound: 0.0032554
time: 2.83 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 7.85 seconds
NS_B2_A2_A2_B1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 7.85
Output dim: 2, lower bound: -0.0036413, upper bound: 0.0032508
NS_B2_A2_A2_B1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 7.85
Output dim: 2, lower bound: -0.0036414, upper bound: 0.0032775
NS_B2_A2_A2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 7.85
Output dim: 2, lower bound: -0.0035876, upper bound: 0.0032554
NS_B2_A2_A2_B1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 7.85
Output dim: 2, lower bound: -0.0036377, upper bound: 0.0032554

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.91 + 302.22 = 308.13 seconds
