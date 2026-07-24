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
execution time: IAR + RelationalAnalysis = 2.41 + 3.29 = 5.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0040668, upper bound: 0.0040668

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037434, upper bound: 0.0039668
time: 2.73 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039795, upper bound: 0.0039794
time: 2.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.62 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.62
Output dim: 2, lower bound: -0.0037434, upper bound: 0.0039668
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.62
Output dim: 2, lower bound: -0.0039795, upper bound: 0.0039794

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0041985, -0.0040843, -0.0042044, -0.0040808, -0.0001177, 0.0001201
1: -0.0098904, -0.0056137, -0.0101121, -0.0054833, -0.0044071, 0.0044985
2: 0.9645946, 0.9697270, 0.9643285, 0.9698833, -0.0052887, 0.0053985
3: -0.0148382, 0.0230162, -0.0168010, 0.0241703, -0.0360524, 0.0368428
4: -0.0024435, 0.0004355, -0.0025313, 0.0005848, -0.0030283, 0.0029668
5: 0.0148007, 0.0178006, 0.0147120, 0.0182394, -0.0034386, 0.0030886
6: 0.0030525, 0.0047077, 0.0022123, 0.0047509, -0.0016984, 0.0024955
7: -0.0137431, -0.0037903, -0.0140422, -0.0028263, -0.0109168, 0.0102519
8: 0.0058260, 0.0136091, 0.0055887, 0.0140126, -0.0081866, 0.0080203
9: 0.0082033, 0.0222018, 0.0077765, 0.0229276, -0.0143718, 0.0140757

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036368, upper bound: 0.0037143
time: 6.62 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036373, upper bound: 0.0038716
time: 2.38 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0042031, -0.0040817, -0.0042051, -0.0040807, -0.0001224, 0.0001234
1: -0.0100619, -0.0055153, -0.0101349, -0.0054782, -0.0045837, 0.0046196
2: 0.9643887, 0.9698449, 0.9643012, 0.9698893, -0.0055006, 0.0055437
3: -0.0163570, 0.0238871, -0.0170025, 0.0242148, -0.0368500, 0.0379118
4: -0.0025098, 0.0005510, -0.0025347, 0.0006001, -0.0031099, 0.0030857
5: 0.0147338, 0.0181401, 0.0147086, 0.0182844, -0.0035507, 0.0034315
6: 0.0024023, 0.0047403, 0.0021260, 0.0047525, -0.0023502, 0.0026143
7: -0.0139688, -0.0030444, -0.0140537, -0.0027273, -0.0112415, 0.0110094
8: 0.0056470, 0.0139213, 0.0055796, 0.0140540, -0.0084071, 0.0083417
9: 0.0078812, 0.0227635, 0.0077600, 0.0230022, -0.0147761, 0.0145671

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039668, upper bound: 0.0037434
time: 2.89 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0039668, upper bound: 0.0039795
time: 2.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.68 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.68
Output dim: 2, lower bound: -0.0036368, upper bound: 0.0037143
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.68
Output dim: 2, lower bound: -0.0036373, upper bound: 0.0038716
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.68
Output dim: 2, lower bound: -0.0039668, upper bound: 0.0037434
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.68
Output dim: 2, lower bound: -0.0039668, upper bound: 0.0039795

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0041981, -0.0040844, -0.0041986, -0.0040838, -0.0001143, 0.0001142
1: -0.0098731, -0.0056154, -0.0098922, -0.0055938, -0.0042793, 0.0042768
2: 0.9646153, 0.9697247, 0.9645925, 0.9697506, -0.0051354, 0.0051323
3: -0.0146855, 0.0230010, -0.0148544, 0.0231924, -0.0348055, 0.0348608
4: -0.0024424, 0.0004239, -0.0024569, 0.0004367, -0.0028791, 0.0028808
5: 0.0148019, 0.0177664, 0.0147872, 0.0178042, -0.0030023, 0.0029793
6: 0.0031179, 0.0047072, 0.0030456, 0.0047143, -0.0015964, 0.0016616
7: -0.0137392, -0.0038654, -0.0137888, -0.0037824, -0.0099568, 0.0099234
8: 0.0058291, 0.0135776, 0.0057898, 0.0136124, -0.0077832, 0.0077879
9: 0.0082089, 0.0221453, 0.0081381, 0.0222078, -0.0136422, 0.0136484

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037143
time: 2.29 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037143
time: 2.27 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0041985, -0.0040843, -0.0042021, -0.0040812, -0.0001173, 0.0001178
1: -0.0098901, -0.0056137, -0.0100246, -0.0054979, -0.0043923, 0.0044109
2: 0.9645948, 0.9697268, 0.9644336, 0.9698657, -0.0052709, 0.0052933
3: -0.0148362, 0.0230159, -0.0160264, 0.0240411, -0.0359211, 0.0353815
4: -0.0024435, 0.0004354, -0.0025215, 0.0005259, -0.0029694, 0.0029568
5: 0.0148007, 0.0178001, 0.0147219, 0.0180662, -0.0032655, 0.0030782
6: 0.0030534, 0.0047077, 0.0025438, 0.0047460, -0.0016927, 0.0021639
7: -0.0137430, -0.0037913, -0.0140087, -0.0032067, -0.0105363, 0.0102174
8: 0.0058261, 0.0136086, 0.0056153, 0.0138534, -0.0080273, 0.0079933
9: 0.0082034, 0.0222011, 0.0078243, 0.0226412, -0.0140132, 0.0140273

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038699
time: 3.32 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038715
time: 2.25 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042031, -0.0040817, -0.0041985, -0.0040843, -0.0001188, 0.0001168
1: -0.0100619, -0.0055153, -0.0098904, -0.0056137, -0.0044483, 0.0043751
2: 0.9643887, 0.9698449, 0.9645946, 0.9697270, -0.0053383, 0.0052503
3: -0.0163570, 0.0238871, -0.0148382, 0.0230162, -0.0364514, 0.0357679
4: -0.0025098, 0.0005510, -0.0024435, 0.0004355, -0.0029453, 0.0029946
5: 0.0147338, 0.0181401, 0.0148007, 0.0178006, -0.0030668, 0.0033394
6: 0.0024023, 0.0047403, 0.0030525, 0.0047077, -0.0023054, 0.0016878
7: -0.0139688, -0.0030444, -0.0137431, -0.0037903, -0.0101785, 0.0106988
8: 0.0056470, 0.0139213, 0.0058260, 0.0136091, -0.0079621, 0.0080953
9: 0.0078812, 0.0227635, 0.0082033, 0.0222018, -0.0139716, 0.0142138

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037143, upper bound: 0.0036368
time: 3.04 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038715, upper bound: 0.0036373
time: 2.78 seconds

## BFS NS instance: NS_A2_B2

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

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 103

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037143, upper bound: 0.0037068
time: 2.72 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038716, upper bound: 0.0037090
time: 2.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 8.02 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.02
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037143
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.02
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037143
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.02
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038699
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.02
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038715
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.02
Output dim: 2, lower bound: -0.0037143, upper bound: 0.0036368
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.02
Output dim: 2, lower bound: -0.0038715, upper bound: 0.0036373
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.02
Output dim: 2, lower bound: -0.0037143, upper bound: 0.0037068
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.02
Output dim: 2, lower bound: -0.0038716, upper bound: 0.0037090

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0041928, -0.0040870, -0.0041986, -0.0040838, -0.0001090, 0.0001019
1: -0.0096765, -0.0057132, -0.0098922, -0.0055938, -0.0040828, 0.0038150
2: 0.9648512, 0.9696074, 0.9645925, 0.9697506, -0.0048994, 0.0045782
3: -0.0129456, 0.0221351, -0.0148544, 0.0231924, -0.0330137, 0.0337677
4: -0.0023765, 0.0002916, -0.0024569, 0.0004367, -0.0025682, 0.0027485
5: 0.0148684, 0.0175650, 0.0147872, 0.0178042, -0.0029357, 0.0025093
6: 0.0033632, 0.0046748, 0.0030456, 0.0047143, -0.0012205, 0.0016292
7: -0.0135148, -0.0044233, -0.0137888, -0.0037824, -0.0097324, 0.0084600
8: 0.0060072, 0.0132199, 0.0057898, 0.0136124, -0.0069428, 0.0074301
9: 0.0085291, 0.0215019, 0.0081381, 0.0222078, -0.0124872, 0.0129986

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035384, upper bound: 0.0035362
time: 2.04 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035384, upper bound: 0.0037143
time: 2.80 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0041961, -0.0040847, -0.0041986, -0.0040838, -0.0001123, 0.0001047
1: -0.0098004, -0.0056278, -0.0098922, -0.0055938, -0.0042066, 0.0039213
2: 0.9647025, 0.9697099, 0.9645925, 0.9697506, -0.0050481, 0.0047057
3: -0.0140420, 0.0228907, -0.0148544, 0.0231924, -0.0342279, 0.0347084
4: -0.0024340, 0.0003749, -0.0024569, 0.0004367, -0.0026398, 0.0028319
5: 0.0148104, 0.0176493, 0.0147872, 0.0178042, -0.0029938, 0.0026113
6: 0.0033222, 0.0047030, 0.0030456, 0.0047143, -0.0012701, 0.0016574
7: -0.0137106, -0.0041391, -0.0137888, -0.0037824, -0.0099282, 0.0088039
8: 0.0058518, 0.0134453, 0.0057898, 0.0136124, -0.0071362, 0.0076555
9: 0.0082497, 0.0219074, 0.0081381, 0.0222078, -0.0128351, 0.0134183

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035384, upper bound: 0.0035362
time: 2.24 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035384, upper bound: 0.0037143
time: 2.44 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0041928, -0.0040870, -0.0042021, -0.0040812, -0.0001116, 0.0001060
1: -0.0096765, -0.0057132, -0.0100246, -0.0054979, -0.0041787, 0.0039692
2: 0.9648512, 0.9696074, 0.9644336, 0.9698657, -0.0050145, 0.0047632
3: -0.0129456, 0.0221351, -0.0160264, 0.0240411, -0.0339733, 0.0351325
4: -0.0023765, 0.0002916, -0.0025215, 0.0005259, -0.0026720, 0.0028131
5: 0.0148684, 0.0175650, 0.0147219, 0.0180662, -0.0031978, 0.0025889
6: 0.0033632, 0.0046748, 0.0025438, 0.0047460, -0.0012592, 0.0021309
7: -0.0135148, -0.0044233, -0.0140087, -0.0032067, -0.0103080, 0.0087284
8: 0.0060072, 0.0132199, 0.0056153, 0.0138534, -0.0072234, 0.0076046
9: 0.0085291, 0.0215019, 0.0078243, 0.0226412, -0.0129920, 0.0133217

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0036576
time: 3.26 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038699
time: 2.95 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0041961, -0.0040847, -0.0042021, -0.0040812, -0.0001149, 0.0001041
1: -0.0098004, -0.0056278, -0.0100246, -0.0054979, -0.0043025, 0.0038999
2: 0.9647025, 0.9697099, 0.9644336, 0.9698657, -0.0051632, 0.0046800
3: -0.0140420, 0.0228907, -0.0160264, 0.0240411, -0.0344674, 0.0345189
4: -0.0024340, 0.0003749, -0.0025215, 0.0005259, -0.0026254, 0.0028964
5: 0.0148104, 0.0176493, 0.0147219, 0.0180662, -0.0032558, 0.0025702
6: 0.0033222, 0.0047030, 0.0025438, 0.0047460, -0.0012502, 0.0021592
7: -0.0137106, -0.0041391, -0.0140087, -0.0032067, -0.0105039, 0.0086655
8: 0.0058518, 0.0134453, 0.0056153, 0.0138534, -0.0070972, 0.0078301
9: 0.0082497, 0.0219074, 0.0078243, 0.0226412, -0.0127651, 0.0136629

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0035511
time: 2.67 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037381
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0041972, -0.0040846, -0.0041981, -0.0040844, -0.0001128, 0.0001135
1: -0.0098402, -0.0056239, -0.0098731, -0.0056154, -0.0042248, 0.0042492
2: 0.9646549, 0.9697145, 0.9646153, 0.9697247, -0.0050698, 0.0050992
3: -0.0143939, 0.0229255, -0.0146855, 0.0230010, -0.0344639, 0.0345402
4: -0.0024367, 0.0004017, -0.0024424, 0.0004239, -0.0028605, 0.0028441
5: 0.0148077, 0.0177012, 0.0148019, 0.0177664, -0.0029587, 0.0028994
6: 0.0032427, 0.0047043, 0.0031179, 0.0047072, -0.0014644, 0.0015864
7: -0.0137196, -0.0040086, -0.0137392, -0.0038654, -0.0098542, 0.0097306
8: 0.0058447, 0.0135177, 0.0058291, 0.0135776, -0.0077330, 0.0076885
9: 0.0082368, 0.0220375, 0.0082089, 0.0221453, -0.0135500, 0.0134804

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037143, upper bound: 0.0035291
time: 2.75 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037143, upper bound: 0.0036367
time: 3.07 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042008, -0.0040821, -0.0041985, -0.0040843, -0.0001164, 0.0001164
1: -0.0099743, -0.0055295, -0.0098901, -0.0056137, -0.0043607, 0.0043606
2: 0.9644938, 0.9698278, 0.9645948, 0.9697268, -0.0052330, 0.0052329
3: -0.0155816, 0.0237608, -0.0148362, 0.0230159, -0.0349915, 0.0356398
4: -0.0025002, 0.0004920, -0.0024435, 0.0004354, -0.0029355, 0.0029356
5: 0.0147435, 0.0179668, 0.0148007, 0.0178001, -0.0030566, 0.0031660
6: 0.0027343, 0.0047356, 0.0030534, 0.0047077, -0.0019734, 0.0016822
7: -0.0139361, -0.0034252, -0.0137430, -0.0037913, -0.0101447, 0.0103178
8: 0.0056729, 0.0137619, 0.0058261, 0.0136086, -0.0079357, 0.0079358
9: 0.0079279, 0.0224767, 0.0082034, 0.0222011, -0.0139242, 0.0138539

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038700, upper bound: 0.0035291
time: 2.72 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038700, upper bound: 0.0036373
time: 2.88 seconds

## BFS NS instance: NS_A2_B2_A1

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

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037225, upper bound: 0.0035876
time: 3.04 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037225, upper bound: 0.0037068
time: 2.74 seconds

## BFS NS instance: NS_A2_B2_A2

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

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 103

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0035876
time: 3.30 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0037091
time: 3.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 9.01 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0035384, upper bound: 0.0035362
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0035384, upper bound: 0.0037143
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0035384, upper bound: 0.0035362
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0035384, upper bound: 0.0037143
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0036576
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0038699
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0035511
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0035291, upper bound: 0.0037381
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0037143, upper bound: 0.0035291
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0037143, upper bound: 0.0036367
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0038700, upper bound: 0.0035291
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0038700, upper bound: 0.0036373
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0037225, upper bound: 0.0035876
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0037225, upper bound: 0.0037068
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0035876
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.01
Output dim: 2, lower bound: -0.0038812, upper bound: 0.0037091

## BFS NS instance: NS_A1_B1_A1_B2

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

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036642
time: 2.44 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036544
time: 3.05 seconds

## BFS NS instance: NS_A1_B1_A2_B2

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

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035552, upper bound: 0.0036452
time: 2.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035552, upper bound: 0.0036313
time: 2.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2

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

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0037985
time: 2.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0037815
time: 2.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2

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

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0035559, upper bound: 0.0036648
time: 6.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035559, upper bound: 0.0036507
time: 3.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0041972, -0.0040846, -0.0041928, -0.0040870, -0.0001006, 0.0001082
1: -0.0098402, -0.0056239, -0.0096765, -0.0057132, -0.0037673, 0.0040526
2: 0.9646549, 0.9697145, 0.9648512, 0.9696074, -0.0045209, 0.0048633
3: -0.0143939, 0.0229255, -0.0129456, 0.0221351, -0.0333453, 0.0327484
4: -0.0024367, 0.0004017, -0.0023765, 0.0002916, -0.0027282, 0.0025361
5: 0.0148077, 0.0177012, 0.0148684, 0.0175650, -0.0024905, 0.0028328
6: 0.0032427, 0.0047043, 0.0033632, 0.0046748, -0.0014320, 0.0012114
7: -0.0137196, -0.0040086, -0.0135148, -0.0044233, -0.0083968, 0.0095062
8: 0.0058447, 0.0135177, 0.0060072, 0.0132199, -0.0073752, 0.0068559
9: 0.0082368, 0.0220375, 0.0085291, 0.0215019, -0.0129002, 0.0123311

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036313, upper bound: 0.0034592
time: 2.89 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036313, upper bound: 0.0034580
time: 3.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0041972, -0.0040846, -0.0041961, -0.0040847, -0.0001034, 0.0001115
1: -0.0098402, -0.0056239, -0.0098004, -0.0056278, -0.0038736, 0.0041765
2: 0.9646549, 0.9697145, 0.9647025, 0.9697099, -0.0046484, 0.0050119
3: -0.0143939, 0.0229255, -0.0140420, 0.0228907, -0.0342860, 0.0339625
4: -0.0024367, 0.0004017, -0.0024340, 0.0003749, -0.0028116, 0.0026077
5: 0.0148077, 0.0177012, 0.0148104, 0.0176493, -0.0025925, 0.0028909
6: 0.0032427, 0.0047043, 0.0033222, 0.0047030, -0.0014603, 0.0012610
7: -0.0137196, -0.0040086, -0.0137106, -0.0041391, -0.0087407, 0.0097020
8: 0.0058447, 0.0135177, 0.0058518, 0.0134453, -0.0076007, 0.0070494
9: 0.0082368, 0.0220375, 0.0082497, 0.0219074, -0.0133199, 0.0126789

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036313, upper bound: 0.0035554
time: 2.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036313, upper bound: 0.0035552
time: 2.76 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042008, -0.0040821, -0.0041928, -0.0040870, -0.0001046, 0.0001107
1: -0.0099743, -0.0055295, -0.0096765, -0.0057132, -0.0039156, 0.0041470
2: 0.9644938, 0.9698278, 0.9648512, 0.9696074, -0.0046989, 0.0049766
3: -0.0155816, 0.0237608, -0.0129456, 0.0221351, -0.0346581, 0.0336920
4: -0.0025002, 0.0004920, -0.0023765, 0.0002916, -0.0027917, 0.0026359
5: 0.0147435, 0.0179668, 0.0148684, 0.0175650, -0.0025686, 0.0030983
6: 0.0027343, 0.0047356, 0.0033632, 0.0046748, -0.0019405, 0.0012494
7: -0.0139361, -0.0034252, -0.0135148, -0.0044233, -0.0086601, 0.0100895
8: 0.0056729, 0.0137619, 0.0060072, 0.0132199, -0.0075470, 0.0071258
9: 0.0079279, 0.0224767, 0.0085291, 0.0215019, -0.0132187, 0.0128165

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034508
time: 3.44 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037816, upper bound: 0.0034476
time: 2.97 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042008, -0.0040821, -0.0041961, -0.0040847, -0.0001029, 0.0001140
1: -0.0099743, -0.0055295, -0.0098004, -0.0056278, -0.0038523, 0.0042709
2: 0.9644938, 0.9698278, 0.9647025, 0.9697099, -0.0046230, 0.0051252
3: -0.0155816, 0.0237608, -0.0140420, 0.0228907, -0.0340983, 0.0341879
4: -0.0025002, 0.0004920, -0.0024340, 0.0003749, -0.0028751, 0.0025934
5: 0.0147435, 0.0179668, 0.0148104, 0.0176493, -0.0025503, 0.0031564
6: 0.0027343, 0.0047356, 0.0033222, 0.0047030, -0.0019688, 0.0012405
7: -0.0139361, -0.0034252, -0.0137106, -0.0041391, -0.0085984, 0.0102854
8: 0.0056729, 0.0137619, 0.0058518, 0.0134453, -0.0077724, 0.0070108
9: 0.0079279, 0.0224767, 0.0082497, 0.0219074, -0.0135597, 0.0126095

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034646
time: 2.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034616
time: 3.38 seconds

## BFS NS instance: NS_A2_B2_A1_B1

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

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0035157
time: 3.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0035135
time: 3.59 seconds

## BFS NS instance: NS_A2_B2_A1_B2

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

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 181

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0036253
time: 3.06 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0036242
time: 3.10 seconds

## BFS NS instance: NS_A2_B2_A2_B1

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

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035095
time: 2.41 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035071
time: 2.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2

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

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 156

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035356
time: 3.03 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035333
time: 2.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.82 seconds
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036642
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0034580, upper bound: 0.0036544
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0035552, upper bound: 0.0036452
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0035552, upper bound: 0.0036313
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0037985
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0034477, upper bound: 0.0037815
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0035559, upper bound: 0.0036648
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0035559, upper bound: 0.0036507
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0036313, upper bound: 0.0034592
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0036313, upper bound: 0.0034580
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0036313, upper bound: 0.0035554
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0036313, upper bound: 0.0035552
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034508
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0037816, upper bound: 0.0034476
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034646
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034616
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0035157
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0035135
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0036253
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0036405, upper bound: 0.0036242
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035095
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035071
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035356
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.82
Output dim: 2, lower bound: -0.0037958, upper bound: 0.0035333

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0041926, -0.0040894, -0.0041972, -0.0040846, -0.0001080, 0.0000976
1: -0.0096675, -0.0058039, -0.0098402, -0.0056239, -0.0040436, 0.0036540
2: 0.9648620, 0.9694986, 0.9646549, 0.9697145, -0.0048524, 0.0043849
3: -0.0128660, 0.0213324, -0.0143939, 0.0229255, -0.0326720, 0.0323426
4: -0.0023155, 0.0002855, -0.0024367, 0.0004017, -0.0024598, 0.0027222
5: 0.0149301, 0.0175589, 0.0148077, 0.0177012, -0.0027711, 0.0024853
6: 0.0033661, 0.0046448, 0.0032427, 0.0047043, -0.0012088, 0.0014020
7: -0.0133067, -0.0044439, -0.0137196, -0.0040086, -0.0092982, 0.0083792
8: 0.0061722, 0.0132036, 0.0058447, 0.0135177, -0.0066498, 0.0073589
9: 0.0088259, 0.0214725, 0.0082368, 0.0220375, -0.0119603, 0.0128710

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034564, upper bound: 0.0036544
time: 3.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0034564, upper bound: 0.0036544
time: 2.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

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

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0034469, upper bound: 0.0037815
time: 3.13 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0034469, upper bound: 0.0037816
time: 3.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

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

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032765, upper bound: 0.0035957
time: 2.48 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032303, upper bound: 0.0035941
time: 2.51 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

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

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035546, upper bound: 0.0036508
time: 3.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035546, upper bound: 0.0036507
time: 2.73 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041928, -0.0040870, -0.0001043, 0.0001085
1: -0.0099656, -0.0056155, -0.0096765, -0.0057132, -0.0039077, 0.0040610
2: 0.9645044, 0.9697246, 0.9648512, 0.9696074, -0.0046894, 0.0048734
3: -0.0155038, 0.0229998, -0.0129456, 0.0221351, -0.0345880, 0.0329497
4: -0.0024423, 0.0004861, -0.0023765, 0.0002916, -0.0027339, 0.0026306
5: 0.0148020, 0.0179494, 0.0148684, 0.0175650, -0.0024960, 0.0030809
6: 0.0027676, 0.0047071, 0.0033632, 0.0046748, -0.0019072, 0.0012140
7: -0.0137389, -0.0034634, -0.0135148, -0.0044233, -0.0084152, 0.0100513
8: 0.0058294, 0.0137459, 0.0060072, 0.0132199, -0.0073905, 0.0071114
9: 0.0082093, 0.0224479, 0.0085291, 0.0215019, -0.0129375, 0.0127906

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034469
time: 2.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034469
time: 2.53 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0041928, -0.0040874, -0.0001062, 0.0001082
1: -0.0100444, -0.0056220, -0.0096749, -0.0057307, -0.0039766, 0.0040530
2: 0.9644098, 0.9697168, 0.9648532, 0.9695864, -0.0047721, 0.0048636
3: -0.0162017, 0.0229426, -0.0129313, 0.0219800, -0.0351982, 0.0332895
4: -0.0024379, 0.0005392, -0.0023647, 0.0002905, -0.0027284, 0.0026770
5: 0.0148064, 0.0181054, 0.0148804, 0.0175639, -0.0025777, 0.0032250
6: 0.0024688, 0.0047050, 0.0033637, 0.0046690, -0.0022001, 0.0012538
7: -0.0137240, -0.0031207, -0.0134746, -0.0044270, -0.0086906, 0.0103539
8: 0.0058411, 0.0138894, 0.0060391, 0.0132170, -0.0073758, 0.0072369
9: 0.0082305, 0.0227060, 0.0085865, 0.0214966, -0.0129523, 0.0130163

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037171, upper bound: 0.0033559
time: 3.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037187, upper bound: 0.0033843
time: 3.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041961, -0.0040847, -0.0001027, 0.0001118
1: -0.0099656, -0.0056155, -0.0098004, -0.0056278, -0.0038444, 0.0041849
2: 0.9645044, 0.9697246, 0.9647025, 0.9697099, -0.0046135, 0.0050220
3: -0.0155038, 0.0229998, -0.0140420, 0.0228907, -0.0340284, 0.0334496
4: -0.0024423, 0.0004861, -0.0024340, 0.0003749, -0.0028172, 0.0025881
5: 0.0148020, 0.0179494, 0.0148104, 0.0176493, -0.0024803, 0.0031390
6: 0.0027676, 0.0047071, 0.0033222, 0.0047030, -0.0019355, 0.0012064
7: -0.0137389, -0.0034634, -0.0137106, -0.0041391, -0.0083622, 0.0102472
8: 0.0058294, 0.0137459, 0.0058518, 0.0134453, -0.0076159, 0.0069964
9: 0.0082093, 0.0224479, 0.0082497, 0.0219074, -0.0132790, 0.0125837

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037830, upper bound: 0.0034608
time: 2.57 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037830, upper bound: 0.0034607
time: 2.51 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0041961, -0.0040852, -0.0001049, 0.0001115
1: -0.0100444, -0.0056220, -0.0097988, -0.0056457, -0.0039282, 0.0041768
2: 0.9644098, 0.9697168, 0.9647045, 0.9696884, -0.0047141, 0.0050123
3: -0.0162017, 0.0229426, -0.0140278, 0.0227326, -0.0347702, 0.0337854
4: -0.0024379, 0.0005392, -0.0024220, 0.0003739, -0.0028118, 0.0026445
5: 0.0148064, 0.0181054, 0.0148225, 0.0176482, -0.0025639, 0.0032829
6: 0.0024688, 0.0047050, 0.0033227, 0.0046971, -0.0022283, 0.0012471
7: -0.0137240, -0.0031207, -0.0136696, -0.0041428, -0.0086442, 0.0105489
8: 0.0058411, 0.0138894, 0.0058843, 0.0134424, -0.0076013, 0.0071489
9: 0.0082305, 0.0227060, 0.0083081, 0.0219021, -0.0132937, 0.0128580

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037830, upper bound: 0.0034616
time: 3.37 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037830, upper bound: 0.0034616
time: 3.35 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041972, -0.0040846, -0.0001159, 0.0001128
1: -0.0099656, -0.0056155, -0.0098402, -0.0056239, -0.0043417, 0.0042247
2: 0.9645044, 0.9697246, 0.9646549, 0.9697145, -0.0052101, 0.0050697
3: -0.0155038, 0.0229998, -0.0143939, 0.0229255, -0.0346624, 0.0336738
4: -0.0024423, 0.0004861, -0.0024367, 0.0004017, -0.0028440, 0.0029228
5: 0.0148020, 0.0179494, 0.0148077, 0.0177012, -0.0028993, 0.0031417
6: 0.0027676, 0.0047071, 0.0032427, 0.0047043, -0.0019368, 0.0014644
7: -0.0137389, -0.0034634, -0.0137196, -0.0040086, -0.0097303, 0.0102562
8: 0.0058294, 0.0137459, 0.0058447, 0.0135177, -0.0076883, 0.0079012
9: 0.0082093, 0.0224479, 0.0082368, 0.0220375, -0.0133905, 0.0137727

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035051
time: 3.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035051
time: 2.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0041971, -0.0040851, -0.0001176, 0.0001126
1: -0.0100444, -0.0056220, -0.0098386, -0.0056417, -0.0044027, 0.0042166
2: 0.9644098, 0.9697168, 0.9646568, 0.9696932, -0.0052834, 0.0050600
3: -0.0162017, 0.0229426, -0.0143799, 0.0227681, -0.0353433, 0.0339989
4: -0.0024379, 0.0005392, -0.0024247, 0.0004006, -0.0028386, 0.0029639
5: 0.0148064, 0.0181054, 0.0148198, 0.0176981, -0.0028917, 0.0032856
6: 0.0024688, 0.0047050, 0.0032487, 0.0046984, -0.0022296, 0.0014562
7: -0.0137240, -0.0031207, -0.0136788, -0.0040155, -0.0097086, 0.0105581
8: 0.0058411, 0.0138894, 0.0058770, 0.0135148, -0.0076737, 0.0080123
9: 0.0082305, 0.0227060, 0.0082950, 0.0220323, -0.0134034, 0.0139808

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037271, upper bound: 0.0034091
time: 3.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037287, upper bound: 0.0034396
time: 2.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

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

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 156

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037982, upper bound: 0.0035313
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0037982, upper bound: 0.0035313
time: 2.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

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

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0036941, upper bound: 0.0033864
time: 2.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033667
time: 3.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.96 seconds
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0034564, upper bound: 0.0036544
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0034564, upper bound: 0.0036544
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0034469, upper bound: 0.0037815
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0034469, upper bound: 0.0037816
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0032765, upper bound: 0.0035957
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0032303, upper bound: 0.0035941
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0035546, upper bound: 0.0036508
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0035546, upper bound: 0.0036507
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034469
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037815, upper bound: 0.0034469
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037171, upper bound: 0.0033559
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037187, upper bound: 0.0033843
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037830, upper bound: 0.0034608
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037830, upper bound: 0.0034607
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037830, upper bound: 0.0034616
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037830, upper bound: 0.0034616
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035051
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037957, upper bound: 0.0035051
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037271, upper bound: 0.0034091
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037287, upper bound: 0.0034396
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037982, upper bound: 0.0035313
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0037982, upper bound: 0.0035313
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0036941, upper bound: 0.0033864
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.96
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033667

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0041926, -0.0040894, -0.0042005, -0.0040844, -0.0001082, 0.0001013
1: -0.0096675, -0.0058039, -0.0099656, -0.0056155, -0.0040520, 0.0037944
2: 0.9648620, 0.9694986, 0.9645044, 0.9697246, -0.0048625, 0.0045534
3: -0.0128660, 0.0213324, -0.0155038, 0.0229998, -0.0328733, 0.0335853
4: -0.0023155, 0.0002855, -0.0024423, 0.0004861, -0.0025544, 0.0027278
5: 0.0149301, 0.0175589, 0.0148020, 0.0179494, -0.0030192, 0.0024907
6: 0.0033661, 0.0046448, 0.0027676, 0.0047071, -0.0012115, 0.0018772
7: -0.0133067, -0.0044439, -0.0137389, -0.0034634, -0.0098433, 0.0083975
8: 0.0061722, 0.0132036, 0.0058294, 0.0137459, -0.0069053, 0.0073742
9: 0.0088259, 0.0214725, 0.0082093, 0.0224479, -0.0124198, 0.0129082

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0033568, upper bound: 0.0037373
time: 3.39 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0033841, upper bound: 0.0037378
time: 3.09 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0041926, -0.0040894, -0.0042026, -0.0040845, -0.0001080, 0.0001037
1: -0.0096675, -0.0058039, -0.0100444, -0.0056220, -0.0040456, 0.0038839
2: 0.9648620, 0.9694986, 0.9644098, 0.9697168, -0.0048547, 0.0046609
3: -0.0128660, 0.0213324, -0.0162017, 0.0229426, -0.0328025, 0.0343780
4: -0.0023155, 0.0002855, -0.0024379, 0.0005392, -0.0026146, 0.0027234
5: 0.0149301, 0.0175589, 0.0148064, 0.0181054, -0.0031752, 0.0024841
6: 0.0033661, 0.0046448, 0.0024688, 0.0047050, -0.0012083, 0.0021759
7: -0.0133067, -0.0044439, -0.0137240, -0.0031207, -0.0101861, 0.0083752
8: 0.0061722, 0.0132036, 0.0058411, 0.0138894, -0.0070683, 0.0073624
9: 0.0088259, 0.0214725, 0.0082305, 0.0227060, -0.0127129, 0.0128858

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0033568, upper bound: 0.0037373
time: 2.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0033841, upper bound: 0.0037378
time: 2.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041926, -0.0040894, -0.0001013, 0.0001082
1: -0.0099656, -0.0056155, -0.0096675, -0.0058039, -0.0037944, 0.0040520
2: 0.9645044, 0.9697246, 0.9648620, 0.9694986, -0.0045534, 0.0048625
3: -0.0155038, 0.0229998, -0.0128660, 0.0213324, -0.0335853, 0.0328733
4: -0.0024423, 0.0004861, -0.0023155, 0.0002855, -0.0027278, 0.0025544
5: 0.0148020, 0.0179494, 0.0149301, 0.0175589, -0.0024907, 0.0030192
6: 0.0027676, 0.0047071, 0.0033661, 0.0046448, -0.0018772, 0.0012115
7: -0.0137389, -0.0034634, -0.0133067, -0.0044439, -0.0083975, 0.0098433
8: 0.0058294, 0.0137459, 0.0061722, 0.0132036, -0.0073742, 0.0069053
9: 0.0082093, 0.0224479, 0.0088259, 0.0214725, -0.0129082, 0.0124198

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035957, upper bound: 0.0032837
time: 2.57 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035941, upper bound: 0.0032434
time: 2.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041954, -0.0040894, -0.0001015, 0.0001111
1: -0.0099656, -0.0056155, -0.0097747, -0.0058027, -0.0038010, 0.0041592
2: 0.9645044, 0.9697246, 0.9647334, 0.9694999, -0.0045614, 0.0049912
3: -0.0155038, 0.0229998, -0.0138146, 0.0213429, -0.0336438, 0.0338892
4: -0.0024423, 0.0004861, -0.0023163, 0.0003577, -0.0027999, 0.0025588
5: 0.0148020, 0.0179494, 0.0149293, 0.0176318, -0.0025820, 0.0030200
6: 0.0027676, 0.0047071, 0.0033307, 0.0046452, -0.0018776, 0.0012559
7: -0.0137389, -0.0034634, -0.0133094, -0.0041981, -0.0087050, 0.0098460
8: 0.0058294, 0.0137459, 0.0061701, 0.0133986, -0.0075692, 0.0069173
9: 0.0082093, 0.0224479, 0.0088221, 0.0218233, -0.0132688, 0.0124414

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035957, upper bound: 0.0032837
time: 2.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035941, upper bound: 0.0032434
time: 3.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042025, -0.0040846, -0.0041901, -0.0040886, -0.0001051, 0.0001056
1: -0.0100395, -0.0056231, -0.0095761, -0.0057744, -0.0039366, 0.0039530
2: 0.9644156, 0.9697155, 0.9649718, 0.9695340, -0.0047241, 0.0047437
3: -0.0161587, 0.0229327, -0.0120566, 0.0215930, -0.0348443, 0.0324054
4: -0.0024372, 0.0005359, -0.0023353, 0.0002239, -0.0026611, 0.0026501
5: 0.0148071, 0.0180958, 0.0149101, 0.0174967, -0.0025177, 0.0031857
6: 0.0024872, 0.0047046, 0.0033964, 0.0046545, -0.0021673, 0.0012246
7: -0.0137215, -0.0031418, -0.0133743, -0.0046537, -0.0084883, 0.0102325
8: 0.0058432, 0.0138805, 0.0061186, 0.0130371, -0.0071939, 0.0071641
9: 0.0082342, 0.0226901, 0.0087295, 0.0211732, -0.0126268, 0.0128854

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035254, upper bound: 0.0031842
time: 2.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035229, upper bound: 0.0031354
time: 2.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040846, -0.0041908, -0.0040879, -0.0001056, 0.0001062
1: -0.0100415, -0.0056227, -0.0096016, -0.0057496, -0.0039543, 0.0039789
2: 0.9644133, 0.9697161, 0.9649412, 0.9695637, -0.0047453, 0.0047749
3: -0.0161759, 0.0229364, -0.0122819, 0.0218131, -0.0350008, 0.0325339
4: -0.0024375, 0.0005372, -0.0023520, 0.0002411, -0.0026786, 0.0026620
5: 0.0148068, 0.0180996, 0.0148932, 0.0175140, -0.0025248, 0.0032064
6: 0.0024799, 0.0047047, 0.0033880, 0.0046627, -0.0021829, 0.0012281
7: -0.0137224, -0.0031333, -0.0134313, -0.0045953, -0.0085125, 0.0102980
8: 0.0058424, 0.0138841, 0.0060734, 0.0130835, -0.0072410, 0.0071963
9: 0.0082328, 0.0226965, 0.0086482, 0.0212565, -0.0126972, 0.0129433

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035254, upper bound: 0.0032045
time: 3.10 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035229, upper bound: 0.0031557
time: 2.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041959, -0.0040871, -0.0000997, 0.0001115
1: -0.0099656, -0.0056155, -0.0097914, -0.0057188, -0.0037321, 0.0041759
2: 0.9645044, 0.9697246, 0.9647134, 0.9696007, -0.0044786, 0.0050112
3: -0.0155038, 0.0229998, -0.0139621, 0.0220859, -0.0330336, 0.0333732
4: -0.0024423, 0.0004861, -0.0023728, 0.0003689, -0.0028112, 0.0025124
5: 0.0148020, 0.0179494, 0.0148722, 0.0176432, -0.0024750, 0.0030771
6: 0.0027676, 0.0047071, 0.0033252, 0.0046729, -0.0019054, 0.0012038
7: -0.0137389, -0.0034634, -0.0135020, -0.0041599, -0.0083444, 0.0100386
8: 0.0058294, 0.0137459, 0.0060173, 0.0134289, -0.0075995, 0.0067918
9: 0.0082093, 0.0224479, 0.0085473, 0.0218778, -0.0132499, 0.0122158

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036073, upper bound: 0.0033317
time: 2.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036059, upper bound: 0.0032993
time: 2.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041983, -0.0040871, -0.0000998, 0.0001140
1: -0.0099656, -0.0056155, -0.0098833, -0.0057194, -0.0037382, 0.0042678
2: 0.9645044, 0.9697246, 0.9646031, 0.9695999, -0.0044861, 0.0051215
3: -0.0155038, 0.0229998, -0.0147758, 0.0220801, -0.0330883, 0.0342933
4: -0.0024423, 0.0004861, -0.0023724, 0.0004308, -0.0028731, 0.0025166
5: 0.0148020, 0.0179494, 0.0148727, 0.0177057, -0.0025654, 0.0030767
6: 0.0027676, 0.0047071, 0.0032947, 0.0046727, -0.0019052, 0.0012478
7: -0.0137389, -0.0034634, -0.0135005, -0.0039490, -0.0086493, 0.0100371
8: 0.0058294, 0.0137459, 0.0060185, 0.0135962, -0.0077668, 0.0068031
9: 0.0082093, 0.0224479, 0.0085494, 0.0221787, -0.0135625, 0.0122360

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036073, upper bound: 0.0033317
time: 2.55 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036059, upper bound: 0.0032993
time: 2.44 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0041959, -0.0040871, -0.0001025, 0.0001113
1: -0.0100444, -0.0056220, -0.0097914, -0.0057188, -0.0038368, 0.0041694
2: 0.9644098, 0.9697168, 0.9647134, 0.9696007, -0.0046044, 0.0050034
3: -0.0162017, 0.0229426, -0.0139621, 0.0220859, -0.0339610, 0.0332901
4: -0.0024379, 0.0005392, -0.0023728, 0.0003689, -0.0028068, 0.0025829
5: 0.0148064, 0.0181054, 0.0148722, 0.0176432, -0.0024674, 0.0032332
6: 0.0024688, 0.0047050, 0.0033252, 0.0046729, -0.0022041, 0.0012001
7: -0.0137240, -0.0031207, -0.0135020, -0.0041599, -0.0083188, 0.0103813
8: 0.0058411, 0.0138894, 0.0060173, 0.0134289, -0.0075878, 0.0069825
9: 0.0082305, 0.0227060, 0.0085473, 0.0218778, -0.0132274, 0.0125588

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036072, upper bound: 0.0033230
time: 2.39 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036059, upper bound: 0.0032884
time: 3.00 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040845, -0.0041983, -0.0040871, -0.0001032, 0.0001138
1: -0.0100444, -0.0056220, -0.0098833, -0.0057194, -0.0038652, 0.0042613
2: 0.9644098, 0.9697168, 0.9646031, 0.9695999, -0.0046384, 0.0051137
3: -0.0162017, 0.0229426, -0.0147758, 0.0220801, -0.0342119, 0.0342123
4: -0.0024379, 0.0005392, -0.0023724, 0.0004308, -0.0028687, 0.0026020
5: 0.0148064, 0.0181054, 0.0148727, 0.0177057, -0.0025706, 0.0032327
6: 0.0024688, 0.0047050, 0.0032947, 0.0046727, -0.0022039, 0.0012503
7: -0.0137240, -0.0031207, -0.0135005, -0.0039490, -0.0086666, 0.0103798
8: 0.0058411, 0.0138894, 0.0060185, 0.0135962, -0.0077551, 0.0070341
9: 0.0082305, 0.0227060, 0.0085494, 0.0221787, -0.0135371, 0.0126515

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036072, upper bound: 0.0033230
time: 2.28 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036059, upper bound: 0.0032884
time: 2.90 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

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

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036076, upper bound: 0.0033424
time: 2.87 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036061, upper bound: 0.0033023
time: 3.10 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0042005, -0.0040844, -0.0041995, -0.0040870, -0.0001135, 0.0001151
1: -0.0099656, -0.0056155, -0.0099265, -0.0057136, -0.0042520, 0.0043110
2: 0.9645044, 0.9697246, 0.9645513, 0.9696069, -0.0051025, 0.0051733
3: -0.0155038, 0.0229998, -0.0151578, 0.0221317, -0.0338746, 0.0345680
4: -0.0024423, 0.0004861, -0.0023763, 0.0004598, -0.0029021, 0.0028624
5: 0.0148020, 0.0179494, 0.0148687, 0.0178720, -0.0030700, 0.0030807
6: 0.0027676, 0.0047071, 0.0029157, 0.0046747, -0.0019071, 0.0017914
7: -0.0137389, -0.0034634, -0.0135139, -0.0036334, -0.0101055, 0.0100505
8: 0.0058294, 0.0137459, 0.0060079, 0.0136748, -0.0078454, 0.0077380
9: 0.0082093, 0.0224479, 0.0085304, 0.0223200, -0.0136858, 0.0134805

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036076, upper bound: 0.0033424
time: 3.26 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036061, upper bound: 0.0033023
time: 2.26 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0042025, -0.0040846, -0.0041945, -0.0040863, -0.0001051, 0.0001099
1: -0.0100395, -0.0056231, -0.0097397, -0.0056891, -0.0039376, 0.0041167
2: 0.9644156, 0.9697155, 0.9647753, 0.9696364, -0.0047252, 0.0049402
3: -0.0161587, 0.0229327, -0.0135050, 0.0223489, -0.0348526, 0.0331135
4: -0.0024372, 0.0005359, -0.0023928, 0.0003341, -0.0027713, 0.0026507
5: 0.0148071, 0.0180958, 0.0148520, 0.0176080, -0.0025671, 0.0032438
6: 0.0024872, 0.0047046, 0.0033423, 0.0046828, -0.0021956, 0.0012486
7: -0.0137215, -0.0031418, -0.0135702, -0.0042783, -0.0086550, 0.0104284
8: 0.0058432, 0.0138805, 0.0059632, 0.0133349, -0.0074917, 0.0071658
9: 0.0082342, 0.0226901, 0.0084501, 0.0217088, -0.0130765, 0.0128885

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035379, upper bound: 0.0032474
time: 2.86 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035352, upper bound: 0.0032003
time: 2.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0042026, -0.0040846, -0.0041952, -0.0040856, -0.0001056, 0.0001107
1: -0.0100415, -0.0056227, -0.0097670, -0.0056598, -0.0039556, 0.0041444
2: 0.9644133, 0.9697161, 0.9647427, 0.9696715, -0.0047469, 0.0049734
3: -0.0161759, 0.0229364, -0.0137466, 0.0226074, -0.0350120, 0.0331673
4: -0.0024375, 0.0005372, -0.0024125, 0.0003525, -0.0027900, 0.0026629
5: 0.0148068, 0.0180996, 0.0148321, 0.0176266, -0.0025647, 0.0032675
6: 0.0024799, 0.0047047, 0.0033332, 0.0046924, -0.0022126, 0.0012475
7: -0.0137224, -0.0031333, -0.0136372, -0.0042157, -0.0086469, 0.0105038
8: 0.0058424, 0.0138841, 0.0059101, 0.0133846, -0.0075422, 0.0071986
9: 0.0082328, 0.0226965, 0.0083544, 0.0217981, -0.0131511, 0.0129474

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035379, upper bound: 0.0032691
time: 2.89 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0035352, upper bound: 0.0032220
time: 2.35 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

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

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036208, upper bound: 0.0034040
time: 2.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033741
time: 2.46 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

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

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036208, upper bound: 0.0034040
time: 3.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033742
time: 3.09 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

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

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033666
time: 3.27 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033666
time: 2.33 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 7.74 seconds
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0033568, upper bound: 0.0037373
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0033841, upper bound: 0.0037378
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0033568, upper bound: 0.0037373
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0033841, upper bound: 0.0037378
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035957, upper bound: 0.0032837
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035941, upper bound: 0.0032434
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035957, upper bound: 0.0032837
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035941, upper bound: 0.0032434
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035254, upper bound: 0.0031842
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035229, upper bound: 0.0031354
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035254, upper bound: 0.0032045
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035229, upper bound: 0.0031557
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036073, upper bound: 0.0033317
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036059, upper bound: 0.0032993
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036073, upper bound: 0.0033317
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036059, upper bound: 0.0032993
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036072, upper bound: 0.0033230
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036059, upper bound: 0.0032884
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036072, upper bound: 0.0033230
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036059, upper bound: 0.0032884
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036076, upper bound: 0.0033424
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036061, upper bound: 0.0033023
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036076, upper bound: 0.0033424
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036061, upper bound: 0.0033023
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035379, upper bound: 0.0032474
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035352, upper bound: 0.0032003
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035379, upper bound: 0.0032691
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0035352, upper bound: 0.0032220
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036208, upper bound: 0.0034040
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033741
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036208, upper bound: 0.0034040
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033742
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033666
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 7.74
Output dim: 2, lower bound: -0.0036190, upper bound: 0.0033666

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0041899, -0.0040906, -0.0042004, -0.0040844, -0.0001055, 0.0001001
1: -0.0095689, -0.0058500, -0.0099606, -0.0056166, -0.0039523, 0.0037489
2: 0.9649804, 0.9694433, 0.9645104, 0.9697232, -0.0047428, 0.0044989
3: -0.0119926, 0.0209241, -0.0154598, 0.0229900, -0.0319919, 0.0331830
4: -0.0022844, 0.0002191, -0.0024415, 0.0004828, -0.0025238, 0.0026606
5: 0.0149615, 0.0174918, 0.0148027, 0.0179395, -0.0029780, 0.0024309
6: 0.0033988, 0.0046295, 0.0027864, 0.0047067, -0.0011824, 0.0018431
7: -0.0132009, -0.0046703, -0.0137363, -0.0034850, -0.0097159, 0.0081957
8: 0.0062562, 0.0130240, 0.0058314, 0.0137369, -0.0068226, 0.0071926
9: 0.0089769, 0.0211495, 0.0082130, 0.0224317, -0.0122711, 0.0125834

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032254, upper bound: 0.0035519
time: 2.38 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031817, upper bound: 0.0035502
time: 2.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0041906, -0.0040899, -0.0042005, -0.0040844, -0.0001062, 0.0001008
1: -0.0095942, -0.0058213, -0.0099627, -0.0056162, -0.0039780, 0.0037736
2: 0.9649500, 0.9694778, 0.9645078, 0.9697238, -0.0047737, 0.0045285
3: -0.0122168, 0.0211784, -0.0154786, 0.0229937, -0.0321241, 0.0334010
4: -0.0023038, 0.0002361, -0.0024418, 0.0004842, -0.0025403, 0.0026780
5: 0.0149420, 0.0175090, 0.0148024, 0.0179437, -0.0030018, 0.0024392
6: 0.0033904, 0.0046390, 0.0027784, 0.0047069, -0.0011864, 0.0018607
7: -0.0132668, -0.0046122, -0.0137373, -0.0034758, -0.0097910, 0.0082237
8: 0.0062039, 0.0130701, 0.0058307, 0.0137407, -0.0068674, 0.0072394
9: 0.0088829, 0.0212324, 0.0082116, 0.0224386, -0.0123517, 0.0126533

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032423, upper bound: 0.0035519
time: 2.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031994, upper bound: 0.0035502
time: 2.88 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0041899, -0.0040906, -0.0042025, -0.0040846, -0.0001054, 0.0001025
1: -0.0095689, -0.0058500, -0.0100395, -0.0056231, -0.0039458, 0.0038389
2: 0.9649804, 0.9694433, 0.9644156, 0.9697155, -0.0047351, 0.0046068
3: -0.0119926, 0.0209241, -0.0161587, 0.0229327, -0.0319210, 0.0339792
4: -0.0022844, 0.0002191, -0.0024372, 0.0005359, -0.0025843, 0.0026563
5: 0.0149615, 0.0174918, 0.0148071, 0.0180958, -0.0031343, 0.0024243
6: 0.0033988, 0.0046295, 0.0024872, 0.0047046, -0.0011792, 0.0021423
7: -0.0132009, -0.0046703, -0.0137215, -0.0031418, -0.0100592, 0.0081734
8: 0.0062562, 0.0130240, 0.0058432, 0.0138805, -0.0069863, 0.0071808
9: 0.0089769, 0.0211495, 0.0082342, 0.0226901, -0.0125655, 0.0125610

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031924, upper bound: 0.0035409
time: 2.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031480, upper bound: 0.0035391
time: 2.56 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0041906, -0.0040899, -0.0042026, -0.0040846, -0.0001060, 0.0001032
1: -0.0095942, -0.0058213, -0.0100415, -0.0056227, -0.0039715, 0.0038629
2: 0.9649500, 0.9694778, 0.9644133, 0.9697161, -0.0047660, 0.0046356
3: -0.0122168, 0.0211784, -0.0161759, 0.0229364, -0.0320589, 0.0341915
4: -0.0023038, 0.0002361, -0.0024375, 0.0005372, -0.0026005, 0.0026736
5: 0.0149420, 0.0175090, 0.0148068, 0.0180996, -0.0031576, 0.0024338
6: 0.0033904, 0.0046390, 0.0024799, 0.0047047, -0.0011838, 0.0021592
7: -0.0132668, -0.0046122, -0.0137224, -0.0031333, -0.0101335, 0.0082057
8: 0.0062039, 0.0130701, 0.0058424, 0.0138841, -0.0070299, 0.0072277
9: 0.0088829, 0.0212324, 0.0082328, 0.0226965, -0.0126440, 0.0126321

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0032100, upper bound: 0.0035409
time: 2.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0031480, upper bound: 0.0035392
time: 3.35 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 8.28 seconds
NS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 8.28
Output dim: 2, lower bound: -0.0032254, upper bound: 0.0035519
NS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 8.28
Output dim: 2, lower bound: -0.0031817, upper bound: 0.0035502
NS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 8.28
Output dim: 2, lower bound: -0.0032423, upper bound: 0.0035519
NS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 8.28
Output dim: 2, lower bound: -0.0031994, upper bound: 0.0035502
NS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 8.28
Output dim: 2, lower bound: -0.0031924, upper bound: 0.0035409
NS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 8.28
Output dim: 2, lower bound: -0.0031480, upper bound: 0.0035391
NS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 8.28
Output dim: 2, lower bound: -0.0032100, upper bound: 0.0035409
NS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 8.28
Output dim: 2, lower bound: -0.0031480, upper bound: 0.0035392

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.70 + 473.32 = 479.02 seconds
