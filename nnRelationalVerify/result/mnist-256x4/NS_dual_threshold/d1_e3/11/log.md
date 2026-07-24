## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01340901


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0059459, 0.0009457, -0.0059459, 0.0009457, -0.0068916, 0.0068916)
1: (-0.0016795, 0.0135519, -0.0016795, 0.0135519, -0.0152314, 0.0152314)
2: (0.0083163, 0.0233715, 0.0083163, 0.0233715, -0.0150552, 0.0150552)
3: (-0.0076262, -0.0024626, -0.0076262, -0.0024626, -0.0051636, 0.0051636)
4: (-0.0060095, 0.0030900, -0.0060095, 0.0030900, -0.0090996, 0.0090996)
5: (-0.0018541, 0.0101220, -0.0018541, 0.0101220, -0.0119761, 0.0119761)
6: (-0.0057942, 0.0013492, -0.0057942, 0.0013492, -0.0070595, 0.0070595)
7: (-0.0105739, 0.0000491, -0.0105739, 0.0000491, -0.0106231, 0.0106231)
8: (-0.0068949, -0.0000789, -0.0068949, -0.0000789, -0.0068160, 0.0068160)
9: (0.9801895, 1.0025480, 0.9801895, 1.0025480, -0.0223585, 0.0223585)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 2.54 = 4.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0148989, upper bound: 0.0148989

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146094, upper bound: 0.0145063
time: 1.69 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146094, upper bound: 0.0146094
time: 2.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.25 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 4.25
Output dim: 9, lower bound: -0.0146094, upper bound: 0.0145063
NS_B2, status: Status.UNKNOWN, split count: 1, time: 4.25
Output dim: 9, lower bound: -0.0146094, upper bound: 0.0146094

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -0.0059263, 0.0009079, -0.0058004, 0.0006841, -0.0066104, 0.0067083
1: -0.0016379, 0.0135157, -0.0013758, 0.0132853, -0.0149232, 0.0148916
2: 0.0083453, 0.0232974, 0.0085352, 0.0228277, -0.0144824, 0.0147622
3: -0.0075750, -0.0024715, -0.0072512, -0.0025268, -0.0050482, 0.0047798
4: -0.0059589, 0.0030592, -0.0056355, 0.0028655, -0.0088244, 0.0086947
5: -0.0017811, 0.0100809, -0.0013221, 0.0098169, -0.0115980, 0.0114030
6: -0.0057578, 0.0013149, -0.0055290, 0.0010924, -0.0067648, 0.0067500
7: -0.0105284, -0.0000168, -0.0102420, -0.0004327, -0.0100957, 0.0102252
8: -0.0068202, -0.0000843, -0.0063495, -0.0001200, -0.0067002, 0.0062651
9: 0.9802593, 1.0024542, 0.9807101, 1.0018632, -0.0216039, 0.0217441

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0140798
time: 2.37 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0141977
time: 2.03 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -0.0058910, 0.0008413, -0.0060925, 0.0013062, -0.0071972, 0.0069339
1: -0.0015619, 0.0134477, -0.0017528, 0.0135603, -0.0151222, 0.0152005
2: 0.0083929, 0.0231589, 0.0083981, 0.0237417, -0.0153488, 0.0147608
3: -0.0074840, -0.0024851, -0.0081334, -0.0024903, -0.0049937, 0.0056483
4: -0.0058907, 0.0030139, -0.0056885, 0.0035566, -0.0094473, 0.0087024
5: -0.0016601, 0.0100185, -0.0027876, 0.0098145, -0.0114746, 0.0128061
6: -0.0056971, 0.0012693, -0.0064150, 0.0011414, -0.0067814, 0.0076749
7: -0.0104578, -0.0001245, -0.0103390, 0.0008715, -0.0113292, 0.0102145
8: -0.0066878, -0.0000933, -0.0076653, -0.0001128, -0.0065750, 0.0075721
9: 0.9804090, 1.0023034, 0.9807625, 1.0034755, -0.0230666, 0.0215408

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141872, upper bound: 0.0142860
time: 1.31 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0142860
time: 1.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.61 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 4.61
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0140798
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 4.61
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0141977
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.61
Output dim: 9, lower bound: -0.0141872, upper bound: 0.0142860
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.61
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0142860

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -0.0058966, 0.0008803, -0.0056854, 0.0006197, -0.0065163, 0.0065656
1: -0.0016114, 0.0134369, -0.0012773, 0.0129777, -0.0145891, 0.0147142
2: 0.0083704, 0.0231637, 0.0086429, 0.0222897, -0.0139193, 0.0145208
3: -0.0075095, -0.0024728, -0.0069919, -0.0025329, -0.0049766, 0.0045190
4: -0.0058811, 0.0030298, -0.0053194, 0.0027542, -0.0086352, 0.0083491
5: -0.0017018, 0.0099992, -0.0010110, 0.0094836, -0.0111854, 0.0110102
6: -0.0057122, 0.0012724, -0.0053496, 0.0009187, -0.0065330, 0.0065158
7: -0.0104664, -0.0000848, -0.0099857, -0.0006972, -0.0097692, 0.0099008
8: -0.0067289, -0.0000984, -0.0059824, -0.0001782, -0.0065507, 0.0058839
9: 0.9804308, 1.0023699, 0.9814277, 1.0015398, -0.0211090, 0.0209422

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_B1_B1_B1

### Relational analysis result of NS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0138065
time: 2.16 seconds

## Relational analysis of NS_B1_B1_B2

### Relational analysis result of NS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0138580
time: 2.42 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -0.0058884, 0.0008768, -0.0057285, 0.0006646, -0.0065530, 0.0066053
1: -0.0016080, 0.0134027, -0.0014376, 0.0130042, -0.0146121, 0.0148402
2: 0.0083743, 0.0231238, 0.0084961, 0.0224397, -0.0140654, 0.0146277
3: -0.0074996, -0.0024730, -0.0071409, -0.0025281, -0.0049715, 0.0046679
4: -0.0058759, 0.0030260, -0.0053748, 0.0030445, -0.0089204, 0.0084008
5: -0.0016918, 0.0099965, -0.0014634, 0.0095468, -0.0112386, 0.0114599
6: -0.0057066, 0.0012680, -0.0056039, 0.0009494, -0.0065635, 0.0067938
7: -0.0104557, -0.0000939, -0.0100245, -0.0004004, -0.0100553, 0.0099307
8: -0.0067170, -0.0000999, -0.0062409, -0.0001667, -0.0065503, 0.0061411
9: 0.9804657, 1.0023589, 0.9813340, 1.0021896, -0.0217239, 0.0210249

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_B1_B2_B1

### Relational analysis result of NS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139281
time: 2.93 seconds

## Relational analysis of NS_B1_B2_B2

### Relational analysis result of NS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139904
time: 1.82 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057686, 0.0007510, -0.0060646, 0.0012399, -0.0070084, 0.0068157
1: -0.0014517, 0.0131272, -0.0017234, 0.0134828, -0.0149345, 0.0148507
2: 0.0084981, 0.0226027, 0.0084235, 0.0236128, -0.0151147, 0.0141792
3: -0.0072076, -0.0024911, -0.0080714, -0.0024918, -0.0047158, 0.0055803
4: -0.0055679, 0.0028889, -0.0056098, 0.0035282, -0.0090961, 0.0084987
5: -0.0013223, 0.0096800, -0.0027063, 0.0097340, -0.0110563, 0.0123864
6: -0.0055020, 0.0010955, -0.0063705, 0.0011005, -0.0065327, 0.0074435
7: -0.0102026, -0.0004117, -0.0102786, 0.0008020, -0.0110046, 0.0098669
8: -0.0062966, -0.0001514, -0.0075755, -0.0001269, -0.0061697, 0.0074241
9: 0.9811249, 1.0019456, 0.9809347, 1.0033902, -0.0222653, 0.0210109

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0140978
time: 2.71 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0140978
time: 2.03 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -0.0058174, 0.0008144, -0.0060567, 0.0012293, -0.0070467, 0.0068711
1: -0.0016219, 0.0131712, -0.0017189, 0.0134513, -0.0150732, 0.0148901
2: 0.0083510, 0.0227761, 0.0084278, 0.0235694, -0.0152183, 0.0143483
3: -0.0073684, -0.0024866, -0.0080611, -0.0024921, -0.0048764, 0.0055745
4: -0.0056342, 0.0031736, -0.0056054, 0.0035241, -0.0091583, 0.0087790
5: -0.0017802, 0.0097506, -0.0026940, 0.0097272, -0.0115073, 0.0124447
6: -0.0057601, 0.0011262, -0.0063639, 0.0010929, -0.0068160, 0.0074785
7: -0.0102386, -0.0001099, -0.0102636, 0.0007911, -0.0110298, 0.0101538
8: -0.0065747, -0.0001392, -0.0075612, -0.0001285, -0.0064462, 0.0074220
9: 0.9810387, 1.0026062, 0.9809697, 1.0033771, -0.0223383, 0.0216364

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0140978
time: 2.05 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0140978
time: 2.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.22 seconds
NS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0138065
NS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0138580
NS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139281
NS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139904
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0140978
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0140978
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0140978
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 6.22
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0140978

## BFS NS instance: NS_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0058499, 0.0008172, -0.0055048, 0.0004810, -0.0063309, 0.0063220
1: -0.0015397, 0.0133433, -0.0010231, 0.0126230, -0.0141627, 0.0143664
2: 0.0084108, 0.0229597, 0.0088106, 0.0215174, -0.0131066, 0.0141490
3: -0.0073873, -0.0024886, -0.0065157, -0.0025965, -0.0047908, 0.0040271
4: -0.0058021, 0.0029801, -0.0050177, 0.0025498, -0.0083520, 0.0079978
5: -0.0015512, 0.0099347, -0.0004196, 0.0092337, -0.0107849, 0.0103543
6: -0.0056313, 0.0012211, -0.0050328, 0.0007224, -0.0062361, 0.0061271
7: -0.0103850, -0.0002099, -0.0096715, -0.0011866, -0.0091984, 0.0094616
8: -0.0065445, -0.0001062, -0.0052640, -0.0002102, -0.0063343, 0.0051578
9: 0.9805779, 1.0022082, 0.9819962, 1.0009048, -0.0203269, 0.0202120

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_B1_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0138065
time: 1.99 seconds

## Relational analysis of NS_B1_B1_B1_A2

### Relational analysis result of NS_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0138065
time: 2.34 seconds

## BFS NS instance: NS_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0058563, 0.0008206, -0.0056225, 0.0005084, -0.0063647, 0.0064431
1: -0.0015430, 0.0133576, -0.0011817, 0.0127804, -0.0143234, 0.0145392
2: 0.0084138, 0.0229888, 0.0087071, 0.0219365, -0.0135228, 0.0142818
3: -0.0074043, -0.0024885, -0.0068535, -0.0025865, -0.0048178, 0.0043650
4: -0.0058083, 0.0029804, -0.0050734, 0.0028022, -0.0086105, 0.0080538
5: -0.0015654, 0.0099383, -0.0009428, 0.0092737, -0.0108391, 0.0108810
6: -0.0056409, 0.0012242, -0.0053476, 0.0007586, -0.0062877, 0.0064643
7: -0.0103915, -0.0002007, -0.0097281, -0.0007302, -0.0096613, 0.0095274
8: -0.0065719, -0.0001069, -0.0057627, -0.0002027, -0.0063693, 0.0056558
9: 0.9805660, 1.0022160, 0.9818777, 1.0014769, -0.0209109, 0.0203382

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_B1_B2_A1

### Relational analysis result of NS_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0138580
time: 1.61 seconds

## Relational analysis of NS_B1_B1_B2_A2

### Relational analysis result of NS_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0138580
time: 1.83 seconds

## BFS NS instance: NS_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0058418, 0.0008143, -0.0055479, 0.0005018, -0.0063436, 0.0063622
1: -0.0015363, 0.0133085, -0.0011752, 0.0126435, -0.0141798, 0.0144837
2: 0.0084145, 0.0229191, 0.0086616, 0.0216721, -0.0132576, 0.0142574
3: -0.0073773, -0.0024888, -0.0066647, -0.0025921, -0.0047852, 0.0041759
4: -0.0057970, 0.0029765, -0.0050620, 0.0028451, -0.0086421, 0.0080385
5: -0.0015411, 0.0099312, -0.0008715, 0.0092885, -0.0108296, 0.0108027
6: -0.0056257, 0.0012171, -0.0052894, 0.0007509, -0.0062636, 0.0064056
7: -0.0103742, -0.0002189, -0.0096999, -0.0008926, -0.0094816, 0.0094810
8: -0.0065322, -0.0001076, -0.0055225, -0.0001994, -0.0063328, 0.0054149
9: 0.9806128, 1.0021975, 0.9819055, 1.0015521, -0.0209393, 0.0202920

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_B2_B1_A1

### Relational analysis result of NS_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0139281
time: 2.09 seconds

## Relational analysis of NS_B1_B2_B1_A2

### Relational analysis result of NS_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0139281
time: 2.05 seconds

## BFS NS instance: NS_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0058483, 0.0008177, -0.0056659, 0.0005455, -0.0063938, 0.0064835
1: -0.0015397, 0.0133229, -0.0013467, 0.0127925, -0.0143322, 0.0146696
2: 0.0084173, 0.0229491, 0.0085493, 0.0220862, -0.0136689, 0.0143998
3: -0.0073947, -0.0024886, -0.0070035, -0.0025819, -0.0048128, 0.0045148
4: -0.0058039, 0.0029770, -0.0051275, 0.0031021, -0.0089060, 0.0081044
5: -0.0015558, 0.0099361, -0.0014043, 0.0093433, -0.0108990, 0.0113404
6: -0.0056355, 0.0012200, -0.0056009, 0.0007861, -0.0063190, 0.0067458
7: -0.0103822, -0.0002094, -0.0097721, -0.0004168, -0.0099654, 0.0095627
8: -0.0065602, -0.0001083, -0.0060244, -0.0001911, -0.0063690, 0.0059161
9: 0.9806012, 1.0022055, 0.9817917, 1.0021466, -0.0215454, 0.0204138

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B1_B2_B2_A1

### Relational analysis result of NS_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0139904
time: 3.42 seconds

## Relational analysis of NS_B1_B2_B2_A2

### Relational analysis result of NS_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0139904
time: 1.92 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0055753, 0.0005966, -0.0060177, 0.0011054, -0.0066807, 0.0066143
1: -0.0011759, 0.0127451, -0.0016448, 0.0133876, -0.0145635, 0.0143899
2: 0.0086600, 0.0217678, 0.0084639, 0.0234122, -0.0147522, 0.0133039
3: -0.0066980, -0.0025522, -0.0079496, -0.0025073, -0.0041907, 0.0053974
4: -0.0052591, 0.0026837, -0.0055293, 0.0034787, -0.0087377, 0.0082130
5: -0.0007015, 0.0094276, -0.0025514, 0.0096684, -0.0103698, 0.0119790
6: -0.0051665, 0.0008916, -0.0062891, 0.0010485, -0.0061231, 0.0071469
7: -0.0098847, -0.0009215, -0.0101955, 0.0006718, -0.0105565, 0.0092740
8: -0.0055250, -0.0001826, -0.0073896, -0.0001349, -0.0053902, 0.0072070
9: 0.9817110, 1.0012872, 0.9810820, 1.0032212, -0.0215102, 0.0202052

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140978
time: 3.09 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139904
time: 2.09 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0057066, 0.0006347, -0.0060240, 0.0011230, -0.0068295, 0.0066587
1: -0.0013549, 0.0129355, -0.0016478, 0.0134014, -0.0147563, 0.0145833
2: 0.0085610, 0.0222540, 0.0084672, 0.0234415, -0.0148805, 0.0137868
3: -0.0070720, -0.0025433, -0.0079665, -0.0025080, -0.0045640, 0.0054232
4: -0.0053260, 0.0029314, -0.0055394, 0.0034791, -0.0088052, 0.0084709
5: -0.0012516, 0.0094760, -0.0025677, 0.0096749, -0.0109265, 0.0120438
6: -0.0054998, 0.0009376, -0.0062988, 0.0010532, -0.0064815, 0.0071996
7: -0.0099484, -0.0004473, -0.0102049, 0.0006827, -0.0106310, 0.0097575
8: -0.0060808, -0.0001752, -0.0074169, -0.0001357, -0.0059451, 0.0072417
9: 0.9815733, 1.0018798, 0.9810708, 1.0032290, -0.0216557, 0.0208090

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0140978
time: 4.06 seconds

## Relational analysis of NS_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0139904
time: 2.38 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0056227, 0.0006207, -0.0060100, 0.0010954, -0.0067182, 0.0066306
1: -0.0013369, 0.0127810, -0.0016404, 0.0133558, -0.0146927, 0.0144214
2: 0.0085126, 0.0219409, 0.0084681, 0.0233687, -0.0148561, 0.0134729
3: -0.0068560, -0.0025481, -0.0079393, -0.0025075, -0.0043485, 0.0053913
4: -0.0053180, 0.0029724, -0.0055255, 0.0034746, -0.0087926, 0.0084979
5: -0.0011569, 0.0094886, -0.0025391, 0.0096614, -0.0108183, 0.0120277
6: -0.0054263, 0.0009208, -0.0062825, 0.0010411, -0.0064056, 0.0071801
7: -0.0099098, -0.0006236, -0.0101806, 0.0006609, -0.0105707, 0.0095570
8: -0.0057986, -0.0001711, -0.0073753, -0.0001364, -0.0056622, 0.0072042
9: 0.9816355, 1.0019429, 0.9811172, 1.0032083, -0.0215728, 0.0208257

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0140978
time: 2.30 seconds

## Relational analysis of NS_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0139904
time: 2.13 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0057559, 0.0006915, -0.0060164, 0.0011133, -0.0068692, 0.0067079
1: -0.0015301, 0.0129674, -0.0016434, 0.0133696, -0.0148997, 0.0146108
2: 0.0084037, 0.0224301, 0.0084713, 0.0233984, -0.0149947, 0.0139589
3: -0.0072355, -0.0025390, -0.0079566, -0.0025083, -0.0047272, 0.0054177
4: -0.0053871, 0.0032301, -0.0055341, 0.0034752, -0.0088623, 0.0087641
5: -0.0017219, 0.0095511, -0.0025560, 0.0096681, -0.0113900, 0.0121071
6: -0.0057579, 0.0009650, -0.0062925, 0.0010456, -0.0067696, 0.0072344
7: -0.0099950, -0.0001284, -0.0101904, 0.0006722, -0.0106673, 0.0100620
8: -0.0063641, -0.0001629, -0.0074032, -0.0001372, -0.0062269, 0.0072403
9: 0.9814852, 1.0025589, 0.9811047, 1.0032166, -0.0217314, 0.0214542

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_B2_A2_A2_A1

### Relational analysis result of NS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0140978
time: 2.38 seconds

## Relational analysis of NS_B2_A2_A2_A2

### Relational analysis result of NS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0139904
time: 1.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.82 seconds
NS_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0138065
NS_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0138065
NS_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0138580
NS_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0138580
NS_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0139281
NS_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0139281
NS_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0139904
NS_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0140053, upper bound: 0.0139904
NS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140978
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139904
NS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0140978
NS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0139904
NS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0140978
NS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0139904
NS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0140978
NS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.82
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0139904

## BFS NS instance: NS_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0057249, 0.0006158, -0.0055048, 0.0004810, -0.0062059, 0.0061206
1: -0.0012848, 0.0131167, -0.0010231, 0.0126230, -0.0139077, 0.0141398
2: 0.0086025, 0.0224938, 0.0088106, 0.0215174, -0.0129149, 0.0136831
3: -0.0070659, -0.0025443, -0.0065157, -0.0025965, -0.0044694, 0.0039713
4: -0.0054806, 0.0027890, -0.0050177, 0.0025498, -0.0080304, 0.0078067
5: -0.0010974, 0.0096718, -0.0004196, 0.0092337, -0.0103310, 0.0100914
6: -0.0054053, 0.0010000, -0.0050328, 0.0007224, -0.0060046, 0.0059074
7: -0.0100987, -0.0006217, -0.0096715, -0.0011866, -0.0089120, 0.0090498
8: -0.0060769, -0.0001421, -0.0052640, -0.0002102, -0.0058667, 0.0051218
9: 0.9810304, 1.0016257, 0.9819962, 1.0009048, -0.0198744, 0.0196294

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B1_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138065
time: 3.36 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138065
time: 2.42 seconds

## BFS NS instance: NS_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0060177, 0.0011054, -0.0055048, 0.0004810, -0.0064987, 0.0066102
1: -0.0016448, 0.0133876, -0.0010231, 0.0126230, -0.0142678, 0.0144107
2: 0.0084639, 0.0234122, 0.0088106, 0.0215174, -0.0130535, 0.0146015
3: -0.0079496, -0.0025073, -0.0065157, -0.0025965, -0.0053531, 0.0040084
4: -0.0055293, 0.0034787, -0.0050177, 0.0025498, -0.0080791, 0.0084964
5: -0.0025514, 0.0096684, -0.0004196, 0.0092337, -0.0117850, 0.0100879
6: -0.0062891, 0.0010485, -0.0050328, 0.0007224, -0.0069631, 0.0060146
7: -0.0101955, 0.0006718, -0.0096715, -0.0011866, -0.0090089, 0.0103432
8: -0.0073896, -0.0001349, -0.0052640, -0.0002102, -0.0071794, 0.0051291
9: 0.9810820, 1.0032212, 0.9819962, 1.0009048, -0.0198228, 0.0212249

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B1_B1_B1_A2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138065
time: 2.42 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138065
time: 2.43 seconds

## BFS NS instance: NS_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057310, 0.0006166, -0.0056225, 0.0005084, -0.0062394, 0.0062391
1: -0.0012872, 0.0131307, -0.0011817, 0.0127804, -0.0140676, 0.0143123
2: 0.0086047, 0.0225215, 0.0087071, 0.0219365, -0.0133318, 0.0138144
3: -0.0070824, -0.0025442, -0.0068535, -0.0025865, -0.0044959, 0.0043093
4: -0.0054843, 0.0027899, -0.0050734, 0.0028022, -0.0082864, 0.0078633
5: -0.0011116, 0.0096749, -0.0009428, 0.0092737, -0.0103854, 0.0106177
6: -0.0054148, 0.0010016, -0.0053476, 0.0007586, -0.0060560, 0.0062457
7: -0.0101034, -0.0006121, -0.0097281, -0.0007302, -0.0093732, 0.0091160
8: -0.0061036, -0.0001428, -0.0057627, -0.0002027, -0.0059009, 0.0056199
9: 0.9810185, 1.0016336, 0.9818777, 1.0014769, -0.0204584, 0.0197559

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B1_B1_B2_A1_A1

### Relational analysis result of NS_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138580
time: 4.09 seconds

## Relational analysis of NS_B1_B1_B2_A1_A2

### Relational analysis result of NS_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138580
time: 1.95 seconds

## BFS NS instance: NS_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0060240, 0.0011230, -0.0056225, 0.0005084, -0.0065324, 0.0067455
1: -0.0016478, 0.0134014, -0.0011817, 0.0127804, -0.0144281, 0.0145831
2: 0.0084672, 0.0234415, 0.0087071, 0.0219365, -0.0134694, 0.0147344
3: -0.0079665, -0.0025080, -0.0068535, -0.0025865, -0.0053800, 0.0043455
4: -0.0055394, 0.0034791, -0.0050734, 0.0028022, -0.0083416, 0.0085525
5: -0.0025677, 0.0096749, -0.0009428, 0.0092737, -0.0118414, 0.0106177
6: -0.0062988, 0.0010532, -0.0053476, 0.0007586, -0.0070149, 0.0063516
7: -0.0102049, 0.0006827, -0.0097281, -0.0007302, -0.0094747, 0.0104108
8: -0.0074169, -0.0001357, -0.0057627, -0.0002027, -0.0072142, 0.0056270
9: 0.9810708, 1.0032290, 0.9818777, 1.0014769, -0.0204061, 0.0213513

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B1_B1_B2_A2_A1

### Relational analysis result of NS_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138580
time: 2.30 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2

### Relational analysis result of NS_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138580
time: 3.18 seconds

## BFS NS instance: NS_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0057175, 0.0006140, -0.0055479, 0.0005018, -0.0062193, 0.0061619
1: -0.0012819, 0.0130803, -0.0011752, 0.0126435, -0.0139254, 0.0142555
2: 0.0086071, 0.0224532, 0.0086616, 0.0216721, -0.0130650, 0.0137916
3: -0.0070570, -0.0025445, -0.0066647, -0.0025921, -0.0044650, 0.0041202
4: -0.0054716, 0.0027852, -0.0050620, 0.0028451, -0.0083167, 0.0078472
5: -0.0010876, 0.0096657, -0.0008715, 0.0092885, -0.0103761, 0.0105372
6: -0.0054001, 0.0009944, -0.0052894, 0.0007509, -0.0060326, 0.0061853
7: -0.0100876, -0.0006304, -0.0096999, -0.0008926, -0.0091950, 0.0090695
8: -0.0060654, -0.0001437, -0.0055225, -0.0001994, -0.0058660, 0.0053788
9: 0.9810593, 1.0016147, 0.9819055, 1.0015521, -0.0204928, 0.0197092

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_B2_B1_A1_B1

### Relational analysis result of NS_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135636, upper bound: 0.0131686
time: 2.45 seconds

## Relational analysis of NS_B1_B2_B1_A1_B2

### Relational analysis result of NS_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137730, upper bound: 0.0136881
time: 2.28 seconds

## BFS NS instance: NS_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0060100, 0.0010954, -0.0055479, 0.0005018, -0.0065118, 0.0066433
1: -0.0016404, 0.0133558, -0.0011752, 0.0126435, -0.0142839, 0.0145310
2: 0.0084681, 0.0233687, 0.0086616, 0.0216721, -0.0132040, 0.0147071
3: -0.0079393, -0.0025075, -0.0066647, -0.0025921, -0.0053472, 0.0041571
4: -0.0055255, 0.0034746, -0.0050620, 0.0028451, -0.0083706, 0.0085366
5: -0.0025391, 0.0096614, -0.0008715, 0.0092885, -0.0118276, 0.0105329
6: -0.0062825, 0.0010411, -0.0052894, 0.0007509, -0.0069895, 0.0062881
7: -0.0101806, 0.0006609, -0.0096999, -0.0008926, -0.0092880, 0.0103609
8: -0.0073753, -0.0001364, -0.0055225, -0.0001994, -0.0071759, 0.0053861
9: 0.9811172, 1.0032083, 0.9819055, 1.0015521, -0.0204349, 0.0213028

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_B2_B1_A2_B1

### Relational analysis result of NS_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135636, upper bound: 0.0131686
time: 2.83 seconds

## Relational analysis of NS_B1_B2_B1_A2_B2

### Relational analysis result of NS_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137730, upper bound: 0.0136881
time: 2.30 seconds

## BFS NS instance: NS_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0057238, 0.0006147, -0.0056659, 0.0005455, -0.0062693, 0.0062806
1: -0.0012844, 0.0130944, -0.0013467, 0.0127925, -0.0140769, 0.0144411
2: 0.0086089, 0.0224819, 0.0085493, 0.0220862, -0.0134773, 0.0139326
3: -0.0070740, -0.0025443, -0.0070035, -0.0025819, -0.0044920, 0.0044591
4: -0.0054756, 0.0027865, -0.0051275, 0.0031021, -0.0085777, 0.0079139
5: -0.0011025, 0.0096690, -0.0014043, 0.0093433, -0.0104457, 0.0110733
6: -0.0054099, 0.0009962, -0.0056009, 0.0007861, -0.0060878, 0.0065254
7: -0.0100947, -0.0006203, -0.0097721, -0.0004168, -0.0096779, 0.0091518
8: -0.0060926, -0.0001443, -0.0060244, -0.0001911, -0.0059015, 0.0058801
9: 0.9810460, 1.0016235, 0.9817917, 1.0021466, -0.0211006, 0.0198318

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_B2_B2_A1_B1

### Relational analysis result of NS_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135636, upper bound: 0.0132262
time: 1.57 seconds

## Relational analysis of NS_B1_B2_B2_A1_B2

### Relational analysis result of NS_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137730, upper bound: 0.0137575
time: 2.70 seconds

## BFS NS instance: NS_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0060164, 0.0011133, -0.0056659, 0.0005455, -0.0065620, 0.0067791
1: -0.0016434, 0.0133696, -0.0013467, 0.0127925, -0.0144359, 0.0147163
2: 0.0084713, 0.0233984, 0.0085493, 0.0220862, -0.0136149, 0.0148490
3: -0.0079566, -0.0025083, -0.0070035, -0.0025819, -0.0053747, 0.0044952
4: -0.0055341, 0.0034752, -0.0051275, 0.0031021, -0.0086362, 0.0086026
5: -0.0025560, 0.0096681, -0.0014043, 0.0093433, -0.0118993, 0.0110724
6: -0.0062925, 0.0010456, -0.0056009, 0.0007861, -0.0070451, 0.0066293
7: -0.0101904, 0.0006722, -0.0097721, -0.0004168, -0.0097736, 0.0104444
8: -0.0074032, -0.0001372, -0.0060244, -0.0001911, -0.0072121, 0.0058872
9: 0.9811047, 1.0032166, 0.9817917, 1.0021466, -0.0210419, 0.0214249

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_B2_B2_A2_B1

### Relational analysis result of NS_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135636, upper bound: 0.0132262
time: 1.79 seconds

## Relational analysis of NS_B1_B2_B2_A2_B2

### Relational analysis result of NS_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137730, upper bound: 0.0137575
time: 2.28 seconds

## BFS NS instance: NS_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0055048, 0.0004810, -0.0060177, 0.0011054, -0.0066102, 0.0064987
1: -0.0010231, 0.0126230, -0.0016448, 0.0133876, -0.0144107, 0.0142678
2: 0.0088106, 0.0215174, 0.0084639, 0.0234122, -0.0146015, 0.0130535
3: -0.0065157, -0.0025965, -0.0079496, -0.0025073, -0.0040084, 0.0053531
4: -0.0050177, 0.0025498, -0.0055293, 0.0034787, -0.0084964, 0.0080791
5: -0.0004196, 0.0092337, -0.0025514, 0.0096684, -0.0100879, 0.0117850
6: -0.0050328, 0.0007224, -0.0062891, 0.0010485, -0.0060146, 0.0069631
7: -0.0096715, -0.0011866, -0.0101955, 0.0006718, -0.0103432, 0.0090089
8: -0.0052640, -0.0002102, -0.0073896, -0.0001349, -0.0051291, 0.0071794
9: 0.9819962, 1.0009048, 0.9810820, 1.0032212, -0.0212249, 0.0198228

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139893
time: 1.94 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140978
time: 2.00 seconds

## BFS NS instance: NS_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0058241, 0.0006755, -0.0060177, 0.0011054, -0.0069295, 0.0066932
1: -0.0013586, 0.0129461, -0.0016448, 0.0133876, -0.0147462, 0.0145910
2: 0.0086668, 0.0225649, 0.0084639, 0.0234122, -0.0147454, 0.0141010
3: -0.0074656, -0.0025583, -0.0079496, -0.0025073, -0.0049583, 0.0053913
4: -0.0050568, 0.0032515, -0.0055293, 0.0034787, -0.0085355, 0.0087808
5: -0.0019103, 0.0092243, -0.0025514, 0.0096684, -0.0115787, 0.0117756
6: -0.0059464, 0.0007666, -0.0062891, 0.0010485, -0.0067823, 0.0068511
7: -0.0097656, 0.0001241, -0.0101955, 0.0006718, -0.0104373, 0.0103196
8: -0.0066687, -0.0002028, -0.0073896, -0.0001349, -0.0065339, 0.0071868
9: 0.9820247, 1.0025110, 0.9810820, 1.0032212, -0.0211964, 0.0214290

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0138670
time: 2.08 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139904
time: 1.87 seconds

## BFS NS instance: NS_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0056225, 0.0005084, -0.0060240, 0.0011230, -0.0067455, 0.0065324
1: -0.0011817, 0.0127804, -0.0016478, 0.0134014, -0.0145831, 0.0144281
2: 0.0087071, 0.0219365, 0.0084672, 0.0234415, -0.0147344, 0.0134694
3: -0.0068535, -0.0025865, -0.0079665, -0.0025080, -0.0043455, 0.0053800
4: -0.0050734, 0.0028022, -0.0055394, 0.0034791, -0.0085525, 0.0083416
5: -0.0009428, 0.0092737, -0.0025677, 0.0096749, -0.0106177, 0.0118414
6: -0.0053476, 0.0007586, -0.0062988, 0.0010532, -0.0063516, 0.0070149
7: -0.0097281, -0.0007302, -0.0102049, 0.0006827, -0.0104108, 0.0094747
8: -0.0057627, -0.0002027, -0.0074169, -0.0001357, -0.0056270, 0.0072142
9: 0.9818777, 1.0014769, 0.9810708, 1.0032290, -0.0213513, 0.0204061

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A1_A2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0139893
time: 1.88 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0140978
time: 2.06 seconds

## BFS NS instance: NS_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0059143, 0.0008830, -0.0060240, 0.0011230, -0.0070373, 0.0069070
1: -0.0015336, 0.0130480, -0.0016478, 0.0134014, -0.0149350, 0.0146957
2: 0.0085650, 0.0228767, 0.0084672, 0.0234415, -0.0148765, 0.0144095
3: -0.0077382, -0.0025513, -0.0079665, -0.0025080, -0.0052302, 0.0054152
4: -0.0051312, 0.0035171, -0.0055394, 0.0034791, -0.0086104, 0.0090566
5: -0.0024369, 0.0092820, -0.0025677, 0.0096749, -0.0121118, 0.0118497
6: -0.0062407, 0.0008181, -0.0062988, 0.0010532, -0.0070779, 0.0068865
7: -0.0098417, 0.0006024, -0.0102049, 0.0006827, -0.0105244, 0.0108073
8: -0.0070713, -0.0001960, -0.0074169, -0.0001357, -0.0069356, 0.0072209
9: 0.9819208, 1.0031018, 0.9810708, 1.0032290, -0.0213082, 0.0220310

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A1_A2_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0138670
time: 2.81 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0139904
time: 2.56 seconds

## BFS NS instance: NS_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0055497, 0.0005021, -0.0060100, 0.0010954, -0.0066451, 0.0065120
1: -0.0011765, 0.0126482, -0.0016404, 0.0133558, -0.0145323, 0.0142886
2: 0.0086616, 0.0216806, 0.0084681, 0.0233687, -0.0147072, 0.0132126
3: -0.0066690, -0.0025921, -0.0079393, -0.0025075, -0.0041615, 0.0053472
4: -0.0050620, 0.0028453, -0.0055255, 0.0034746, -0.0085366, 0.0083708
5: -0.0008751, 0.0092888, -0.0025391, 0.0096614, -0.0105366, 0.0118279
6: -0.0052916, 0.0007509, -0.0062825, 0.0010411, -0.0062908, 0.0069796
7: -0.0097009, -0.0008902, -0.0101806, 0.0006609, -0.0103618, 0.0092905
8: -0.0055292, -0.0001994, -0.0073753, -0.0001364, -0.0053928, 0.0071760
9: 0.9819016, 1.0015547, 0.9811172, 1.0032083, -0.0213067, 0.0204375

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A1_A1_A1

### Relational analysis result of NS_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131686, upper bound: 0.0136509
time: 2.00 seconds

## Relational analysis of NS_B2_A2_A1_A1_A2

### Relational analysis result of NS_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136881, upper bound: 0.0138702
time: 2.04 seconds

## BFS NS instance: NS_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0058714, 0.0008140, -0.0060100, 0.0010954, -0.0069668, 0.0068240
1: -0.0015337, 0.0129864, -0.0016404, 0.0133558, -0.0148895, 0.0146267
2: 0.0085205, 0.0227349, 0.0084681, 0.0233687, -0.0148483, 0.0142668
3: -0.0076280, -0.0025551, -0.0079393, -0.0025075, -0.0051205, 0.0053843
4: -0.0051036, 0.0035584, -0.0055255, 0.0034746, -0.0085782, 0.0090839
5: -0.0024018, 0.0092754, -0.0025391, 0.0096614, -0.0120632, 0.0118145
6: -0.0062002, 0.0007863, -0.0062825, 0.0010411, -0.0070685, 0.0068606
7: -0.0097791, 0.0004561, -0.0101806, 0.0006609, -0.0104401, 0.0106368
8: -0.0069514, -0.0001921, -0.0073753, -0.0001364, -0.0068149, 0.0071832
9: 0.9819590, 1.0031875, 0.9811172, 1.0032083, -0.0212493, 0.0220703

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A1_A2_A1

### Relational analysis result of NS_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131686, upper bound: 0.0135461
time: 1.34 seconds

## Relational analysis of NS_B2_A2_A1_A2_A2

### Relational analysis result of NS_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136881, upper bound: 0.0137575
time: 1.80 seconds

## BFS NS instance: NS_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0056688, 0.0005473, -0.0060164, 0.0011133, -0.0067821, 0.0065637
1: -0.0013490, 0.0128004, -0.0016434, 0.0133696, -0.0147186, 0.0144438
2: 0.0085492, 0.0221008, 0.0084713, 0.0233984, -0.0148491, 0.0136295
3: -0.0070110, -0.0025819, -0.0079566, -0.0025083, -0.0045027, 0.0053747
4: -0.0051275, 0.0031024, -0.0055341, 0.0034752, -0.0086027, 0.0086364
5: -0.0014110, 0.0093438, -0.0025560, 0.0096681, -0.0110791, 0.0118998
6: -0.0056046, 0.0007861, -0.0062925, 0.0010456, -0.0066338, 0.0070343
7: -0.0097736, -0.0004121, -0.0101904, 0.0006722, -0.0104459, 0.0097783
8: -0.0060363, -0.0001911, -0.0074032, -0.0001372, -0.0058991, 0.0072121
9: 0.9817853, 1.0021515, 0.9811047, 1.0032166, -0.0214313, 0.0210468

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132262, upper bound: 0.0136509
time: 1.97 seconds

## Relational analysis of NS_B2_A2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137575, upper bound: 0.0138702
time: 2.27 seconds

## BFS NS instance: NS_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0059632, 0.0010810, -0.0060164, 0.0011133, -0.0070764, 0.0070974
1: -0.0017205, 0.0130860, -0.0016434, 0.0133696, -0.0150901, 0.0147294
2: 0.0084129, 0.0230463, 0.0084713, 0.0233984, -0.0149855, 0.0145750
3: -0.0079041, -0.0025482, -0.0079566, -0.0025083, -0.0053958, 0.0054084
4: -0.0051849, 0.0038253, -0.0055341, 0.0034752, -0.0086600, 0.0093594
5: -0.0029275, 0.0093436, -0.0025560, 0.0096681, -0.0125957, 0.0118996
6: -0.0065055, 0.0008341, -0.0062925, 0.0010456, -0.0073710, 0.0069183
7: -0.0098716, 0.0009549, -0.0101904, 0.0006722, -0.0105438, 0.0111453
8: -0.0073575, -0.0001842, -0.0074032, -0.0001372, -0.0072203, 0.0072190
9: 0.9818428, 1.0038506, 0.9811047, 1.0032166, -0.0213739, 0.0227458

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A2_A2_A1

### Relational analysis result of NS_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132262, upper bound: 0.0135461
time: 2.67 seconds

## Relational analysis of NS_B2_A2_A2_A2_A2

### Relational analysis result of NS_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137575, upper bound: 0.0137575
time: 2.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.27 seconds
NS_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138065
NS_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138065
NS_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138065
NS_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138065
NS_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138580
NS_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138580
NS_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138580
NS_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138792, upper bound: 0.0138580
NS_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0135636, upper bound: 0.0131686
NS_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0137730, upper bound: 0.0136881
NS_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0135636, upper bound: 0.0131686
NS_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0137730, upper bound: 0.0136881
NS_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0135636, upper bound: 0.0132262
NS_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0137730, upper bound: 0.0137575
NS_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0135636, upper bound: 0.0132262
NS_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0137730, upper bound: 0.0137575
NS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139893
NS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140978
NS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0138670
NS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139904
NS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0139893
NS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0140978
NS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0138670
NS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0139904
NS_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0131686, upper bound: 0.0136509
NS_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0136881, upper bound: 0.0138702
NS_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0131686, upper bound: 0.0135461
NS_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0136881, upper bound: 0.0137575
NS_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0132262, upper bound: 0.0136509
NS_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0137575, upper bound: 0.0138702
NS_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0132262, upper bound: 0.0135461
NS_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 9, lower bound: -0.0137575, upper bound: 0.0137575

## BFS NS instance: NS_B1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0056382, 0.0005820, -0.0055048, 0.0004810, -0.0061192, 0.0060868
1: -0.0012104, 0.0128840, -0.0010231, 0.0126230, -0.0138334, 0.0139070
2: 0.0086848, 0.0220835, 0.0088106, 0.0215174, -0.0128326, 0.0132729
3: -0.0068677, -0.0025491, -0.0065157, -0.0025965, -0.0042713, 0.0039666
4: -0.0052428, 0.0027033, -0.0050177, 0.0025498, -0.0077926, 0.0077211
5: -0.0008587, 0.0094211, -0.0004196, 0.0092337, -0.0100924, 0.0098407
6: -0.0052678, 0.0008684, -0.0050328, 0.0007224, -0.0058606, 0.0057695
7: -0.0099057, -0.0008228, -0.0096715, -0.0011866, -0.0087191, 0.0088487
8: -0.0057946, -0.0001862, -0.0052640, -0.0002102, -0.0055844, 0.0050778
9: 0.9815757, 1.0013775, 0.9819962, 1.0009048, -0.0193291, 0.0193812

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_B1_B1_A1_A1_B1

### Relational analysis result of NS_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134614, upper bound: 0.0131101
time: 3.16 seconds

## Relational analysis of NS_B1_B1_B1_A1_A1_B2

### Relational analysis result of NS_B1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136548, upper bound: 0.0135892
time: 1.92 seconds

## BFS NS instance: NS_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0056812, 0.0006146, -0.0055048, 0.0004810, -0.0061622, 0.0061194
1: -0.0013689, 0.0129086, -0.0010231, 0.0126230, -0.0139919, 0.0139317
2: 0.0085377, 0.0222333, 0.0088106, 0.0215174, -0.0129798, 0.0134227
3: -0.0070163, -0.0025443, -0.0065157, -0.0025965, -0.0044198, 0.0039714
4: -0.0052961, 0.0029948, -0.0050177, 0.0025498, -0.0078459, 0.0080126
5: -0.0013103, 0.0094818, -0.0004196, 0.0092337, -0.0105440, 0.0099013
6: -0.0055228, 0.0008987, -0.0050328, 0.0007224, -0.0061358, 0.0058242
7: -0.0099425, -0.0005273, -0.0096715, -0.0011866, -0.0087558, 0.0091441
8: -0.0060525, -0.0001749, -0.0052640, -0.0002102, -0.0058423, 0.0050891
9: 0.9814811, 1.0020263, 0.9819962, 1.0009048, -0.0194237, 0.0200301

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_B1_B1_A1_A2_B1

### Relational analysis result of NS_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134614, upper bound: 0.0131101
time: 2.88 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2_B2

### Relational analysis result of NS_B1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136548, upper bound: 0.0135892
time: 2.11 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0059344, 0.0009223, -0.0055048, 0.0004810, -0.0064154, 0.0064271
1: -0.0015557, 0.0131546, -0.0010231, 0.0126230, -0.0141787, 0.0141777
2: 0.0085452, 0.0230116, 0.0088106, 0.0215174, -0.0129722, 0.0142009
3: -0.0077575, -0.0025124, -0.0065157, -0.0025965, -0.0051610, 0.0040032
4: -0.0052913, 0.0033956, -0.0050177, 0.0025498, -0.0078412, 0.0084133
5: -0.0023041, 0.0094154, -0.0004196, 0.0092337, -0.0115377, 0.0098350
6: -0.0061533, 0.0009197, -0.0050328, 0.0007224, -0.0068212, 0.0058825
7: -0.0100057, 0.0004631, -0.0096715, -0.0011866, -0.0088191, 0.0101345
8: -0.0071104, -0.0001789, -0.0052640, -0.0002102, -0.0069002, 0.0050850
9: 0.9816216, 1.0029659, 0.9819962, 1.0009048, -0.0192832, 0.0209697

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_B1_B1_A2_A1_A1

### Relational analysis result of NS_B1_B1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132158, upper bound: 0.0133561
time: 1.75 seconds

## Relational analysis of NS_B1_B1_B1_A2_A1_A2

### Relational analysis result of NS_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0135645
time: 1.33 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0059774, 0.0010995, -0.0055048, 0.0004810, -0.0064584, 0.0066043
1: -0.0017346, 0.0131852, -0.0010231, 0.0126230, -0.0143576, 0.0142082
2: 0.0084015, 0.0231545, 0.0088106, 0.0215174, -0.0131159, 0.0143438
3: -0.0079086, -0.0025089, -0.0065157, -0.0025965, -0.0053121, 0.0040067
4: -0.0053449, 0.0036994, -0.0050177, 0.0025498, -0.0078947, 0.0087171
5: -0.0027814, 0.0094740, -0.0004196, 0.0092337, -0.0120151, 0.0098935
6: -0.0063971, 0.0009404, -0.0050328, 0.0007224, -0.0070857, 0.0059229
7: -0.0100253, 0.0007832, -0.0096715, -0.0011866, -0.0088387, 0.0104547
8: -0.0073746, -0.0001679, -0.0052640, -0.0002102, -0.0071644, 0.0050961
9: 0.9815614, 1.0036356, 0.9819962, 1.0009048, -0.0193434, 0.0216394

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_B1_B1_A2_A2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132158, upper bound: 0.0133561
time: 2.33 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0135645
time: 2.23 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0056446, 0.0005808, -0.0056225, 0.0005084, -0.0061530, 0.0062033
1: -0.0012134, 0.0128983, -0.0011817, 0.0127804, -0.0139937, 0.0140800
2: 0.0086871, 0.0221116, 0.0087071, 0.0219365, -0.0132494, 0.0134045
3: -0.0068848, -0.0025488, -0.0068535, -0.0025865, -0.0042983, 0.0043047
4: -0.0052448, 0.0027042, -0.0050734, 0.0028022, -0.0080470, 0.0077776
5: -0.0008734, 0.0094214, -0.0009428, 0.0092737, -0.0101471, 0.0103641
6: -0.0052775, 0.0008707, -0.0053476, 0.0007586, -0.0059123, 0.0061066
7: -0.0099087, -0.0008138, -0.0097281, -0.0007302, -0.0091785, 0.0089142
8: -0.0058220, -0.0001868, -0.0057627, -0.0002027, -0.0056193, 0.0055759
9: 0.9815599, 1.0013860, 0.9818777, 1.0014769, -0.0199170, 0.0195083

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_B1_B2_A1_A1_B1

### Relational analysis result of NS_B1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134614, upper bound: 0.0131506
time: 2.23 seconds

## Relational analysis of NS_B1_B1_B2_A1_A1_B2

### Relational analysis result of NS_B1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136548, upper bound: 0.0136413
time: 1.91 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0056875, 0.0006158, -0.0056225, 0.0005084, -0.0061960, 0.0062383
1: -0.0013720, 0.0129230, -0.0011817, 0.0127804, -0.0141523, 0.0141047
2: 0.0085391, 0.0222631, 0.0087071, 0.0219365, -0.0133974, 0.0135560
3: -0.0070337, -0.0025441, -0.0068535, -0.0025865, -0.0044471, 0.0043094
4: -0.0053017, 0.0029964, -0.0050734, 0.0028022, -0.0081039, 0.0080698
5: -0.0013264, 0.0094871, -0.0009428, 0.0092737, -0.0106001, 0.0104298
6: -0.0055327, 0.0009007, -0.0053476, 0.0007586, -0.0061879, 0.0061596
7: -0.0099502, -0.0005168, -0.0097281, -0.0007302, -0.0092200, 0.0092113
8: -0.0060803, -0.0001754, -0.0057627, -0.0002027, -0.0058776, 0.0055873
9: 0.9814692, 1.0020361, 0.9818777, 1.0014769, -0.0200077, 0.0201584

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_B1_B2_A1_A2_B1

### Relational analysis result of NS_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134614, upper bound: 0.0131506
time: 2.14 seconds

## Relational analysis of NS_B1_B1_B2_A1_A2_B2

### Relational analysis result of NS_B1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136548, upper bound: 0.0136413
time: 1.91 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0059411, 0.0009383, -0.0056225, 0.0005084, -0.0064495, 0.0065608
1: -0.0015594, 0.0131688, -0.0011817, 0.0127804, -0.0143398, 0.0143505
2: 0.0085488, 0.0230421, 0.0087071, 0.0219365, -0.0133878, 0.0143351
3: -0.0077754, -0.0025131, -0.0068535, -0.0025865, -0.0051889, 0.0043404
4: -0.0052996, 0.0033964, -0.0050734, 0.0028022, -0.0081018, 0.0084698
5: -0.0023217, 0.0094213, -0.0009428, 0.0092737, -0.0115954, 0.0103641
6: -0.0061638, 0.0009252, -0.0053476, 0.0007586, -0.0068737, 0.0062184
7: -0.0100139, 0.0004748, -0.0097281, -0.0007302, -0.0092837, 0.0102029
8: -0.0071393, -0.0001798, -0.0057627, -0.0002027, -0.0069366, 0.0055829
9: 0.9816075, 1.0029752, 0.9818777, 1.0014769, -0.0198694, 0.0210975

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_B1_B2_A2_A1_A1

### Relational analysis result of NS_B1_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132158, upper bound: 0.0134051
time: 1.52 seconds

## Relational analysis of NS_B1_B1_B2_A2_A1_A2

### Relational analysis result of NS_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0136182
time: 2.19 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0059841, 0.0011180, -0.0056225, 0.0005084, -0.0064925, 0.0067405
1: -0.0017380, 0.0131994, -0.0011817, 0.0127804, -0.0145184, 0.0143810
2: 0.0084042, 0.0231850, 0.0087071, 0.0219365, -0.0135323, 0.0144779
3: -0.0079266, -0.0025096, -0.0068535, -0.0025865, -0.0053401, 0.0043439
4: -0.0053539, 0.0037004, -0.0050734, 0.0028022, -0.0081561, 0.0087738
5: -0.0027994, 0.0094830, -0.0009428, 0.0092737, -0.0120731, 0.0104258
6: -0.0064075, 0.0009453, -0.0053476, 0.0007586, -0.0071380, 0.0062610
7: -0.0100376, 0.0007961, -0.0097281, -0.0007302, -0.0093074, 0.0105242
8: -0.0074036, -0.0001686, -0.0057627, -0.0002027, -0.0072009, 0.0055941
9: 0.9815442, 1.0036454, 0.9818777, 1.0014769, -0.0199327, 0.0217677

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_B1_B2_A2_A2_A1

### Relational analysis result of NS_B1_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132158, upper bound: 0.0134051
time: 2.67 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2_A2

### Relational analysis result of NS_B1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0136182
time: 2.05 seconds

## BFS NS instance: NS_B1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055964, 0.0005836, -0.0051855, 0.0004596, -0.0060560, 0.0057690
1: -0.0011840, 0.0128218, -0.0008601, 0.0118428, -0.0130267, 0.0136819
2: 0.0086281, 0.0219382, 0.0087560, 0.0200849, -0.0114568, 0.0131822
3: -0.0067532, -0.0025451, -0.0057712, -0.0025944, -0.0041588, 0.0032261
4: -0.0054150, 0.0027436, -0.0048797, 0.0027442, -0.0081591, 0.0076233
5: -0.0008267, 0.0095816, -0.0001331, 0.0090185, -0.0098452, 0.0097147
6: -0.0052520, 0.0009795, -0.0048833, 0.0007066, -0.0058251, 0.0057871
7: -0.0100327, -0.0008403, -0.0095259, -0.0014611, -0.0085716, 0.0086856
8: -0.0056326, -0.0001628, -0.0042489, -0.0002530, -0.0053796, 0.0040862
9: 0.9813545, 1.0013838, 0.9828619, 1.0008541, -0.0194996, 0.0185219

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B2_B1_A1_B1_A1

### Relational analysis result of NS_B1_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130154, upper bound: 0.0127201
time: 1.47 seconds

## Relational analysis of NS_B1_B2_B1_A1_B1_A2

### Relational analysis result of NS_B1_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131467, upper bound: 0.0127201
time: 3.07 seconds

## BFS NS instance: NS_B1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056917, 0.0006033, -0.0054079, 0.0004855, -0.0061772, 0.0060112
1: -0.0012605, 0.0130305, -0.0010626, 0.0123717, -0.0136322, 0.0140931
2: 0.0086123, 0.0223475, 0.0086892, 0.0210953, -0.0124830, 0.0136584
3: -0.0069923, -0.0025446, -0.0063124, -0.0025925, -0.0043998, 0.0037678
4: -0.0054582, 0.0027745, -0.0049854, 0.0027865, -0.0082446, 0.0077599
5: -0.0010304, 0.0096488, -0.0005575, 0.0091939, -0.0102243, 0.0102063
6: -0.0053670, 0.0009908, -0.0051202, 0.0007299, -0.0059751, 0.0059894
7: -0.0100748, -0.0006781, -0.0096283, -0.0011450, -0.0089298, 0.0089502
8: -0.0059714, -0.0001473, -0.0050108, -0.0002191, -0.0057523, 0.0048635
9: 0.9811141, 1.0015626, 0.9822112, 1.0012690, -0.0201549, 0.0193514

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B2_B1_A1_B2_A1

### Relational analysis result of NS_B1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132516, upper bound: 0.0133131
time: 2.01 seconds

## Relational analysis of NS_B1_B2_B1_A1_B2_A2

### Relational analysis result of NS_B1_B2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0133131
time: 1.68 seconds

## BFS NS instance: NS_B1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0058922, 0.0008404, -0.0051855, 0.0004596, -0.0063518, 0.0060259
1: -0.0015267, 0.0130999, -0.0008601, 0.0118428, -0.0133694, 0.0139600
2: 0.0084891, 0.0228645, 0.0087560, 0.0200849, -0.0115958, 0.0141085
3: -0.0076481, -0.0025085, -0.0057712, -0.0025944, -0.0050537, 0.0032627
4: -0.0054679, 0.0034340, -0.0048797, 0.0027442, -0.0082120, 0.0083137
5: -0.0022694, 0.0095799, -0.0001331, 0.0090185, -0.0112879, 0.0097131
6: -0.0061380, 0.0010263, -0.0048833, 0.0007066, -0.0067873, 0.0058944
7: -0.0101275, 0.0004366, -0.0095259, -0.0014611, -0.0086664, 0.0099626
8: -0.0069570, -0.0001557, -0.0042489, -0.0002530, -0.0067040, 0.0040932
9: 0.9814114, 1.0029653, 0.9828619, 1.0008541, -0.0194427, 0.0201035

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B2_B1_A2_B1_A1

### Relational analysis result of NS_B1_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130391, upper bound: 0.0126517
time: 1.97 seconds

## Relational analysis of NS_B1_B2_B1_A2_B1_A2

### Relational analysis result of NS_B1_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131775, upper bound: 0.0126517
time: 1.98 seconds

## BFS NS instance: NS_B1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0059841, 0.0010313, -0.0054079, 0.0004855, -0.0064696, 0.0064392
1: -0.0016135, 0.0133054, -0.0010626, 0.0123717, -0.0139852, 0.0143681
2: 0.0084733, 0.0232621, 0.0086892, 0.0210953, -0.0126220, 0.0145729
3: -0.0078732, -0.0025077, -0.0063124, -0.0025925, -0.0052807, 0.0038047
4: -0.0055123, 0.0034639, -0.0049854, 0.0027865, -0.0082988, 0.0084494
5: -0.0024762, 0.0096433, -0.0005575, 0.0091939, -0.0116700, 0.0102008
6: -0.0062486, 0.0010375, -0.0051202, 0.0007299, -0.0069299, 0.0060922
7: -0.0101677, 0.0006084, -0.0096283, -0.0011450, -0.0090227, 0.0102367
8: -0.0072793, -0.0001400, -0.0050108, -0.0002191, -0.0070602, 0.0048708
9: 0.9811733, 1.0031507, 0.9822112, 1.0012690, -0.0200956, 0.0209395

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B2_B1_A2_B2_A1

### Relational analysis result of NS_B1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0132808
time: 3.53 seconds

## Relational analysis of NS_B1_B2_B1_A2_B2_A2

### Relational analysis result of NS_B1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0132808
time: 2.02 seconds

## BFS NS instance: NS_B1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056028, 0.0005816, -0.0052950, 0.0004744, -0.0060773, 0.0058766
1: -0.0011867, 0.0128361, -0.0009806, 0.0119715, -0.0131582, 0.0138167
2: 0.0086299, 0.0219670, 0.0086525, 0.0204726, -0.0118427, 0.0133145
3: -0.0067705, -0.0025449, -0.0060985, -0.0025862, -0.0041843, 0.0035535
4: -0.0054189, 0.0027448, -0.0049469, 0.0030077, -0.0084266, 0.0076917
5: -0.0008420, 0.0095855, -0.0006439, 0.0090716, -0.0099136, 0.0102294
6: -0.0052618, 0.0009814, -0.0051912, 0.0007441, -0.0058816, 0.0061321
7: -0.0100397, -0.0008309, -0.0096003, -0.0010088, -0.0090309, 0.0087694
8: -0.0056603, -0.0001634, -0.0047264, -0.0002446, -0.0054157, 0.0045630
9: 0.9813416, 1.0013932, 0.9827595, 1.0014209, -0.0200793, 0.0186337

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B2_B2_A1_B1_A1

### Relational analysis result of NS_B1_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130154, upper bound: 0.0127642
time: 1.28 seconds

## Relational analysis of NS_B1_B2_B2_A1_B1_A2

### Relational analysis result of NS_B1_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131467, upper bound: 0.0127642
time: 3.84 seconds

## BFS NS instance: NS_B1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056979, 0.0006027, -0.0055246, 0.0005072, -0.0062051, 0.0061274
1: -0.0012630, 0.0130445, -0.0012234, 0.0125104, -0.0137734, 0.0142680
2: 0.0086141, 0.0223765, 0.0085775, 0.0214961, -0.0128820, 0.0137990
3: -0.0070092, -0.0025445, -0.0066449, -0.0025825, -0.0044267, 0.0041004
4: -0.0054624, 0.0027757, -0.0050549, 0.0030434, -0.0085058, 0.0078306
5: -0.0010453, 0.0096519, -0.0010737, 0.0092492, -0.0102945, 0.0107256
6: -0.0053768, 0.0009927, -0.0054265, 0.0007652, -0.0060313, 0.0063225
7: -0.0100819, -0.0006689, -0.0097008, -0.0006888, -0.0093931, 0.0090319
8: -0.0059985, -0.0001479, -0.0054999, -0.0002109, -0.0057876, 0.0053520
9: 0.9811009, 1.0015715, 0.9820943, 1.0018504, -0.0207494, 0.0194772

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B2_B2_A1_B2_A1

### Relational analysis result of NS_B1_B2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132516, upper bound: 0.0133793
time: 2.00 seconds

## Relational analysis of NS_B1_B2_B2_A1_B2_A2

### Relational analysis result of NS_B1_B2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0133793
time: 2.30 seconds

## BFS NS instance: NS_B1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0058986, 0.0008519, -0.0052950, 0.0004744, -0.0063730, 0.0061469
1: -0.0015293, 0.0131135, -0.0009806, 0.0119715, -0.0135008, 0.0140942
2: 0.0084923, 0.0228939, 0.0086525, 0.0204726, -0.0119803, 0.0142414
3: -0.0076654, -0.0025093, -0.0060985, -0.0025862, -0.0050792, 0.0035892
4: -0.0054769, 0.0034346, -0.0049469, 0.0030077, -0.0084846, 0.0083815
5: -0.0022855, 0.0095865, -0.0006439, 0.0090716, -0.0113570, 0.0102304
6: -0.0061480, 0.0010311, -0.0051912, 0.0007441, -0.0068441, 0.0062223
7: -0.0101368, 0.0004478, -0.0096003, -0.0010088, -0.0091280, 0.0100481
8: -0.0069848, -0.0001565, -0.0047264, -0.0002446, -0.0067402, 0.0045699
9: 0.9814016, 1.0029742, 0.9827595, 1.0014209, -0.0200193, 0.0202147

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B2_B2_A2_B1_A1

### Relational analysis result of NS_B1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130391, upper bound: 0.0126893
time: 1.80 seconds

## Relational analysis of NS_B1_B2_B2_A2_B1_A2

### Relational analysis result of NS_B1_B2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131775, upper bound: 0.0126893
time: 1.84 seconds

## BFS NS instance: NS_B1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0059905, 0.0010484, -0.0055246, 0.0005072, -0.0064977, 0.0065730
1: -0.0016166, 0.0133192, -0.0012234, 0.0125104, -0.0141270, 0.0145426
2: 0.0084765, 0.0232918, 0.0085775, 0.0214961, -0.0130196, 0.0147143
3: -0.0078905, -0.0025085, -0.0066449, -0.0025825, -0.0053081, 0.0041364
4: -0.0055210, 0.0034645, -0.0050549, 0.0030434, -0.0085644, 0.0085195
5: -0.0024931, 0.0096504, -0.0010737, 0.0092492, -0.0117423, 0.0107242
6: -0.0062585, 0.0010421, -0.0054265, 0.0007652, -0.0069866, 0.0064263
7: -0.0101778, 0.0006196, -0.0097008, -0.0006888, -0.0094890, 0.0103204
8: -0.0073072, -0.0001408, -0.0054999, -0.0002109, -0.0070963, 0.0053590
9: 0.9811599, 1.0031590, 0.9820943, 1.0018504, -0.0206905, 0.0210647

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B2_B2_A2_B2_A1

### Relational analysis result of NS_B1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0133405
time: 2.35 seconds

## Relational analysis of NS_B1_B2_B2_A2_B2_A2

### Relational analysis result of NS_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0133405
time: 2.25 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055048, 0.0004810, -0.0059344, 0.0009223, -0.0064271, 0.0064154
1: -0.0010231, 0.0126230, -0.0015557, 0.0131546, -0.0141777, 0.0141787
2: 0.0088106, 0.0215174, 0.0085452, 0.0230116, -0.0142009, 0.0129722
3: -0.0065157, -0.0025965, -0.0077575, -0.0025124, -0.0040032, 0.0051610
4: -0.0050177, 0.0025498, -0.0052913, 0.0033956, -0.0084133, 0.0078412
5: -0.0004196, 0.0092337, -0.0023041, 0.0094154, -0.0098350, 0.0115377
6: -0.0050328, 0.0007224, -0.0061533, 0.0009197, -0.0058825, 0.0068212
7: -0.0096715, -0.0011866, -0.0100057, 0.0004631, -0.0101345, 0.0088191
8: -0.0052640, -0.0002102, -0.0071104, -0.0001789, -0.0050850, 0.0069002
9: 0.9819962, 1.0009048, 0.9816216, 1.0029659, -0.0209697, 0.0192832

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A1_A1_B1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133561, upper bound: 0.0132158
time: 2.39 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135645, upper bound: 0.0137469
time: 1.97 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055048, 0.0004810, -0.0059774, 0.0010995, -0.0066043, 0.0064584
1: -0.0010231, 0.0126230, -0.0017346, 0.0131852, -0.0142082, 0.0143576
2: 0.0088106, 0.0215174, 0.0084015, 0.0231545, -0.0143438, 0.0131159
3: -0.0065157, -0.0025965, -0.0079086, -0.0025089, -0.0040067, 0.0053121
4: -0.0050177, 0.0025498, -0.0053449, 0.0036994, -0.0087171, 0.0078947
5: -0.0004196, 0.0092337, -0.0027814, 0.0094740, -0.0098935, 0.0120151
6: -0.0050328, 0.0007224, -0.0063971, 0.0009404, -0.0059229, 0.0070857
7: -0.0096715, -0.0011866, -0.0100253, 0.0007832, -0.0104547, 0.0088387
8: -0.0052640, -0.0002102, -0.0073746, -0.0001679, -0.0050961, 0.0071644
9: 0.9819962, 1.0009048, 0.9815614, 1.0036356, -0.0216394, 0.0193434

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133561, upper bound: 0.0133127
time: 2.22 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135645, upper bound: 0.0138702
time: 1.92 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0058241, 0.0006755, -0.0059344, 0.0009223, -0.0067465, 0.0066099
1: -0.0013586, 0.0129461, -0.0015557, 0.0131546, -0.0145132, 0.0145018
2: 0.0086668, 0.0225649, 0.0085452, 0.0230116, -0.0143447, 0.0140197
3: -0.0074656, -0.0025583, -0.0077575, -0.0025124, -0.0049532, 0.0051991
4: -0.0050568, 0.0032515, -0.0052913, 0.0033956, -0.0084524, 0.0085429
5: -0.0019103, 0.0092243, -0.0023041, 0.0094154, -0.0113257, 0.0115284
6: -0.0059464, 0.0007666, -0.0061533, 0.0009197, -0.0066449, 0.0067088
7: -0.0097656, 0.0001241, -0.0100057, 0.0004631, -0.0102286, 0.0101298
8: -0.0066687, -0.0002028, -0.0071104, -0.0001789, -0.0064898, 0.0069076
9: 0.9820247, 1.0025110, 0.9816216, 1.0029659, -0.0209412, 0.0208895

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131611, upper bound: 0.0134205
time: 2.02 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136707, upper bound: 0.0136303
time: 2.31 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0058241, 0.0006755, -0.0059774, 0.0010995, -0.0069236, 0.0066529
1: -0.0013586, 0.0129461, -0.0017346, 0.0131852, -0.0145438, 0.0146807
2: 0.0086668, 0.0225649, 0.0084015, 0.0231545, -0.0144876, 0.0141634
3: -0.0074656, -0.0025583, -0.0079086, -0.0025089, -0.0049567, 0.0053502
4: -0.0050568, 0.0032515, -0.0053449, 0.0036994, -0.0087562, 0.0085964
5: -0.0019103, 0.0092243, -0.0027814, 0.0094740, -0.0113843, 0.0120057
6: -0.0059464, 0.0007666, -0.0063971, 0.0009404, -0.0067010, 0.0069817
7: -0.0097656, 0.0001241, -0.0100253, 0.0007832, -0.0105487, 0.0101494
8: -0.0066687, -0.0002028, -0.0073746, -0.0001679, -0.0065008, 0.0071718
9: 0.9820247, 1.0025110, 0.9815614, 1.0036356, -0.0216109, 0.0209497

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131611, upper bound: 0.0135461
time: 2.21 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136707, upper bound: 0.0137575
time: 1.93 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056225, 0.0005084, -0.0059411, 0.0009383, -0.0065608, 0.0064495
1: -0.0011817, 0.0127804, -0.0015594, 0.0131688, -0.0143505, 0.0143398
2: 0.0087071, 0.0219365, 0.0085488, 0.0230421, -0.0143351, 0.0133878
3: -0.0068535, -0.0025865, -0.0077754, -0.0025131, -0.0043404, 0.0051889
4: -0.0050734, 0.0028022, -0.0052996, 0.0033964, -0.0084698, 0.0081018
5: -0.0009428, 0.0092737, -0.0023217, 0.0094213, -0.0103641, 0.0115954
6: -0.0053476, 0.0007586, -0.0061638, 0.0009252, -0.0062184, 0.0068737
7: -0.0097281, -0.0007302, -0.0100139, 0.0004748, -0.0102029, 0.0092837
8: -0.0057627, -0.0002027, -0.0071393, -0.0001798, -0.0055829, 0.0069366
9: 0.9818777, 1.0014769, 0.9816075, 1.0029752, -0.0210975, 0.0198694

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A2_A1_B1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134051, upper bound: 0.0132158
time: 1.33 seconds

## Relational analysis of NS_B2_A1_A2_A1_B1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136182, upper bound: 0.0137469
time: 1.63 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056225, 0.0005084, -0.0059841, 0.0011180, -0.0067405, 0.0064925
1: -0.0011817, 0.0127804, -0.0017380, 0.0131994, -0.0143810, 0.0145184
2: 0.0087071, 0.0219365, 0.0084042, 0.0231850, -0.0144779, 0.0135323
3: -0.0068535, -0.0025865, -0.0079266, -0.0025096, -0.0043439, 0.0053401
4: -0.0050734, 0.0028022, -0.0053539, 0.0037004, -0.0087738, 0.0081561
5: -0.0009428, 0.0092737, -0.0027994, 0.0094830, -0.0104258, 0.0120731
6: -0.0053476, 0.0007586, -0.0064075, 0.0009453, -0.0062610, 0.0071380
7: -0.0097281, -0.0007302, -0.0100376, 0.0007961, -0.0105242, 0.0093074
8: -0.0057627, -0.0002027, -0.0074036, -0.0001686, -0.0055941, 0.0072009
9: 0.9818777, 1.0014769, 0.9815442, 1.0036454, -0.0217677, 0.0199327

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A2_A1_B2_B1

### Relational analysis result of NS_B2_A1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134051, upper bound: 0.0133127
time: 2.68 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136182, upper bound: 0.0138702
time: 1.65 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0059143, 0.0008830, -0.0059411, 0.0009383, -0.0068527, 0.0068241
1: -0.0015336, 0.0130480, -0.0015594, 0.0131688, -0.0147024, 0.0146074
2: 0.0085650, 0.0228767, 0.0085488, 0.0230421, -0.0144771, 0.0143280
3: -0.0077382, -0.0025513, -0.0077754, -0.0025131, -0.0052252, 0.0052241
4: -0.0051312, 0.0035171, -0.0052996, 0.0033964, -0.0085276, 0.0088167
5: -0.0024369, 0.0092820, -0.0023217, 0.0094213, -0.0118583, 0.0116036
6: -0.0062407, 0.0008181, -0.0061638, 0.0009252, -0.0069406, 0.0067452
7: -0.0098417, 0.0006024, -0.0100139, 0.0004748, -0.0103165, 0.0106163
8: -0.0070713, -0.0001960, -0.0071393, -0.0001798, -0.0068915, 0.0069433
9: 0.9819208, 1.0031018, 0.9816075, 1.0029752, -0.0210544, 0.0214943

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A2_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132042, upper bound: 0.0134205
time: 2.36 seconds

## Relational analysis of NS_B2_A1_A2_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137317, upper bound: 0.0136303
time: 1.78 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0059143, 0.0008830, -0.0059841, 0.0011180, -0.0070323, 0.0068671
1: -0.0015336, 0.0130480, -0.0017380, 0.0131994, -0.0147329, 0.0147860
2: 0.0085650, 0.0228767, 0.0084042, 0.0231850, -0.0146200, 0.0144725
3: -0.0077382, -0.0025513, -0.0079266, -0.0025096, -0.0052287, 0.0053753
4: -0.0051312, 0.0035171, -0.0053539, 0.0037004, -0.0088317, 0.0088710
5: -0.0024369, 0.0092820, -0.0027994, 0.0094830, -0.0119200, 0.0120813
6: -0.0062407, 0.0008181, -0.0064075, 0.0009453, -0.0069950, 0.0070158
7: -0.0098417, 0.0006024, -0.0100376, 0.0007961, -0.0106378, 0.0106400
8: -0.0070713, -0.0001960, -0.0074036, -0.0001686, -0.0069027, 0.0072076
9: 0.9819208, 1.0031018, 0.9815442, 1.0036454, -0.0217246, 0.0215576

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_A2_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132042, upper bound: 0.0135461
time: 1.89 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137317, upper bound: 0.0137575
time: 2.25 seconds

## BFS NS instance: NS_B2_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0051855, 0.0004596, -0.0058922, 0.0008404, -0.0060259, 0.0063518
1: -0.0008601, 0.0118428, -0.0015267, 0.0130999, -0.0139600, 0.0133694
2: 0.0087560, 0.0200849, 0.0084891, 0.0228645, -0.0141085, 0.0115958
3: -0.0057712, -0.0025944, -0.0076481, -0.0025085, -0.0032627, 0.0050537
4: -0.0048797, 0.0027442, -0.0054679, 0.0034340, -0.0083137, 0.0082120
5: -0.0001331, 0.0090185, -0.0022694, 0.0095799, -0.0097131, 0.0112879
6: -0.0048833, 0.0007066, -0.0061380, 0.0010263, -0.0058944, 0.0067873
7: -0.0095259, -0.0014611, -0.0101275, 0.0004366, -0.0099626, 0.0086664
8: -0.0042489, -0.0002530, -0.0069570, -0.0001557, -0.0040932, 0.0067040
9: 0.9828619, 1.0008541, 0.9814114, 1.0029653, -0.0201035, 0.0194427

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126517, upper bound: 0.0130391
time: 1.99 seconds

## Relational analysis of NS_B2_A2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126517, upper bound: 0.0131775
time: 2.55 seconds

## BFS NS instance: NS_B2_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0054079, 0.0004855, -0.0059841, 0.0010313, -0.0064392, 0.0064696
1: -0.0010626, 0.0123717, -0.0016135, 0.0133054, -0.0143681, 0.0139852
2: 0.0086892, 0.0210953, 0.0084733, 0.0232621, -0.0145729, 0.0126220
3: -0.0063124, -0.0025925, -0.0078732, -0.0025077, -0.0038047, 0.0052807
4: -0.0049854, 0.0027865, -0.0055123, 0.0034639, -0.0084494, 0.0082988
5: -0.0005575, 0.0091939, -0.0024762, 0.0096433, -0.0102008, 0.0116700
6: -0.0051202, 0.0007299, -0.0062486, 0.0010375, -0.0060922, 0.0069299
7: -0.0096283, -0.0011450, -0.0101677, 0.0006084, -0.0102367, 0.0090227
8: -0.0050108, -0.0002191, -0.0072793, -0.0001400, -0.0048708, 0.0070602
9: 0.9822112, 1.0012690, 0.9811733, 1.0031507, -0.0209395, 0.0200956

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0133125
time: 2.40 seconds

## Relational analysis of NS_B2_A2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0134412
time: 3.55 seconds

## BFS NS instance: NS_B2_A2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0055207, 0.0004951, -0.0058922, 0.0008404, -0.0063611, 0.0063873
1: -0.0011612, 0.0121854, -0.0015267, 0.0130999, -0.0142611, 0.0137121
2: 0.0086171, 0.0211763, 0.0084891, 0.0228645, -0.0142474, 0.0126872
3: -0.0067662, -0.0025555, -0.0076481, -0.0025085, -0.0042577, 0.0050926
4: -0.0049189, 0.0034642, -0.0054679, 0.0034340, -0.0083529, 0.0089320
5: -0.0016510, 0.0090060, -0.0022694, 0.0095799, -0.0112310, 0.0112753
6: -0.0058009, 0.0007415, -0.0061380, 0.0010263, -0.0066535, 0.0066718
7: -0.0096047, -0.0001645, -0.0101275, 0.0004366, -0.0100413, 0.0099630
8: -0.0057099, -0.0002438, -0.0069570, -0.0001557, -0.0055542, 0.0067132
9: 0.9829198, 1.0024698, 0.9814114, 1.0029653, -0.0200455, 0.0210584

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A1_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0129625
time: 2.70 seconds

## Relational analysis of NS_B2_A2_A1_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0130910
time: 4.51 seconds

## BFS NS instance: NS_B2_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0057268, 0.0006035, -0.0059841, 0.0010313, -0.0067581, 0.0065876
1: -0.0013941, 0.0127008, -0.0016135, 0.0133054, -0.0146995, 0.0143143
2: 0.0085479, 0.0221357, 0.0084733, 0.0232621, -0.0147142, 0.0136623
3: -0.0072589, -0.0025560, -0.0078732, -0.0025077, -0.0047511, 0.0053172
4: -0.0050276, 0.0035011, -0.0055123, 0.0034639, -0.0084915, 0.0090134
5: -0.0020546, 0.0091790, -0.0024762, 0.0096433, -0.0116979, 0.0116551
6: -0.0060211, 0.0007659, -0.0062486, 0.0010375, -0.0068639, 0.0068156
7: -0.0097036, 0.0001611, -0.0101677, 0.0006084, -0.0103120, 0.0103288
8: -0.0064156, -0.0002117, -0.0072793, -0.0001400, -0.0062756, 0.0070676
9: 0.9822744, 1.0028697, 0.9811733, 1.0031507, -0.0208763, 0.0216964

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A2_A1_A2_A2_B1

### Relational analysis result of NS_B2_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137739, upper bound: 0.0136182
time: 2.24 seconds

## Relational analysis of NS_B2_A2_A1_A2_A2_B2

### Relational analysis result of NS_B2_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137739, upper bound: 0.0136182
time: 2.10 seconds

## BFS NS instance: NS_B2_A2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0052950, 0.0004744, -0.0058986, 0.0008519, -0.0061469, 0.0063730
1: -0.0009806, 0.0119715, -0.0015293, 0.0131135, -0.0140942, 0.0135008
2: 0.0086525, 0.0204726, 0.0084923, 0.0228939, -0.0142414, 0.0119803
3: -0.0060985, -0.0025862, -0.0076654, -0.0025093, -0.0035892, 0.0050792
4: -0.0049469, 0.0030077, -0.0054769, 0.0034346, -0.0083815, 0.0084846
5: -0.0006439, 0.0090716, -0.0022855, 0.0095865, -0.0102304, 0.0113570
6: -0.0051912, 0.0007441, -0.0061480, 0.0010311, -0.0062223, 0.0068441
7: -0.0096003, -0.0010088, -0.0101368, 0.0004478, -0.0100481, 0.0091280
8: -0.0047264, -0.0002446, -0.0069848, -0.0001565, -0.0045699, 0.0067402
9: 0.9827595, 1.0014209, 0.9814016, 1.0029742, -0.0202147, 0.0200193

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A2_A1_A1_B1

### Relational analysis result of NS_B2_A2_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126893, upper bound: 0.0130391
time: 2.50 seconds

## Relational analysis of NS_B2_A2_A2_A1_A1_B2

### Relational analysis result of NS_B2_A2_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126893, upper bound: 0.0131775
time: 1.32 seconds

## BFS NS instance: NS_B2_A2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0055246, 0.0005072, -0.0059905, 0.0010484, -0.0065730, 0.0064977
1: -0.0012234, 0.0125104, -0.0016166, 0.0133192, -0.0145426, 0.0141270
2: 0.0085775, 0.0214961, 0.0084765, 0.0232918, -0.0147143, 0.0130196
3: -0.0066449, -0.0025825, -0.0078905, -0.0025085, -0.0041364, 0.0053081
4: -0.0050549, 0.0030434, -0.0055210, 0.0034645, -0.0085195, 0.0085644
5: -0.0010737, 0.0092492, -0.0024931, 0.0096504, -0.0107242, 0.0117423
6: -0.0054265, 0.0007652, -0.0062585, 0.0010421, -0.0064263, 0.0069866
7: -0.0097008, -0.0006888, -0.0101778, 0.0006196, -0.0103204, 0.0094890
8: -0.0054999, -0.0002109, -0.0073072, -0.0001408, -0.0053590, 0.0070963
9: 0.9820943, 1.0018504, 0.9811599, 1.0031590, -0.0210647, 0.0206905

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A2_A1_A2_B1

### Relational analysis result of NS_B2_A2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0133125
time: 1.69 seconds

## Relational analysis of NS_B2_A2_A2_A1_A2_B2

### Relational analysis result of NS_B2_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0134412
time: 2.77 seconds

## BFS NS instance: NS_B2_A2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0056137, 0.0005344, -0.0058986, 0.0008519, -0.0064656, 0.0064331
1: -0.0013269, 0.0122856, -0.0015293, 0.0131135, -0.0144404, 0.0138148
2: 0.0085095, 0.0214888, 0.0084923, 0.0228939, -0.0143843, 0.0129965
3: -0.0070492, -0.0025483, -0.0076654, -0.0025093, -0.0045399, 0.0051171
4: -0.0050043, 0.0037451, -0.0054769, 0.0034346, -0.0084389, 0.0092220
5: -0.0021682, 0.0090723, -0.0022855, 0.0095865, -0.0117547, 0.0113578
6: -0.0061139, 0.0007916, -0.0061480, 0.0010311, -0.0069648, 0.0067317
7: -0.0096998, 0.0003223, -0.0101368, 0.0004478, -0.0101476, 0.0104591
8: -0.0061192, -0.0002346, -0.0069848, -0.0001565, -0.0059627, 0.0067502
9: 0.9828117, 1.0031171, 0.9814016, 1.0029742, -0.0201624, 0.0217155

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A2_A2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133127, upper bound: 0.0134051
time: 2.26 seconds

## Relational analysis of NS_B2_A2_A2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133127, upper bound: 0.0134051
time: 2.89 seconds

## BFS NS instance: NS_B2_A2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0058191, 0.0007532, -0.0059905, 0.0010484, -0.0068675, 0.0067437
1: -0.0015673, 0.0127987, -0.0016166, 0.0133192, -0.0148866, 0.0144153
2: 0.0084409, 0.0224448, 0.0084765, 0.0232918, -0.0148508, 0.0139682
3: -0.0075349, -0.0025492, -0.0078905, -0.0025085, -0.0050264, 0.0053413
4: -0.0051090, 0.0037681, -0.0055210, 0.0034645, -0.0085736, 0.0092891
5: -0.0025735, 0.0092466, -0.0024931, 0.0096504, -0.0122239, 0.0117398
6: -0.0063263, 0.0008134, -0.0062585, 0.0010421, -0.0071643, 0.0068568
7: -0.0097970, 0.0006536, -0.0101778, 0.0006196, -0.0104166, 0.0108314
8: -0.0068215, -0.0002039, -0.0073072, -0.0001408, -0.0066806, 0.0071033
9: 0.9821495, 1.0035193, 0.9811599, 1.0031590, -0.0210095, 0.0223594

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A2_A2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138702, upper bound: 0.0136182
time: 1.44 seconds

## Relational analysis of NS_B2_A2_A2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138702, upper bound: 0.0136182
time: 2.21 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.13 seconds
NS_B1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0134614, upper bound: 0.0131101
NS_B1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0136548, upper bound: 0.0135892
NS_B1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0134614, upper bound: 0.0131101
NS_B1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0136548, upper bound: 0.0135892
NS_B1_B1_B1_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132158, upper bound: 0.0133561
NS_B1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0135645
NS_B1_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132158, upper bound: 0.0133561
NS_B1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0135645
NS_B1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0134614, upper bound: 0.0131506
NS_B1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0136548, upper bound: 0.0136413
NS_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0134614, upper bound: 0.0131506
NS_B1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0136548, upper bound: 0.0136413
NS_B1_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132158, upper bound: 0.0134051
NS_B1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0136182
NS_B1_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132158, upper bound: 0.0134051
NS_B1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0136182
NS_B1_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0130154, upper bound: 0.0127201
NS_B1_B2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0131467, upper bound: 0.0127201
NS_B1_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132516, upper bound: 0.0133131
NS_B1_B2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0133131
NS_B1_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0130391, upper bound: 0.0126517
NS_B1_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0131775, upper bound: 0.0126517
NS_B1_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0132808
NS_B1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0132808
NS_B1_B2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0130154, upper bound: 0.0127642
NS_B1_B2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0131467, upper bound: 0.0127642
NS_B1_B2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132516, upper bound: 0.0133793
NS_B1_B2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0133793
NS_B1_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0130391, upper bound: 0.0126893
NS_B1_B2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0131775, upper bound: 0.0126893
NS_B1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0133405
NS_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0133405
NS_B2_A1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133561, upper bound: 0.0132158
NS_B2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0135645, upper bound: 0.0137469
NS_B2_A1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133561, upper bound: 0.0133127
NS_B2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0135645, upper bound: 0.0138702
NS_B2_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0131611, upper bound: 0.0134205
NS_B2_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0136707, upper bound: 0.0136303
NS_B2_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0131611, upper bound: 0.0135461
NS_B2_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0136707, upper bound: 0.0137575
NS_B2_A1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0134051, upper bound: 0.0132158
NS_B2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0136182, upper bound: 0.0137469
NS_B2_A1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0134051, upper bound: 0.0133127
NS_B2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0136182, upper bound: 0.0138702
NS_B2_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132042, upper bound: 0.0134205
NS_B2_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0137317, upper bound: 0.0136303
NS_B2_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132042, upper bound: 0.0135461
NS_B2_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0137317, upper bound: 0.0137575
NS_B2_A2_A1_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0126517, upper bound: 0.0130391
NS_B2_A2_A1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0126517, upper bound: 0.0131775
NS_B2_A2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0133125
NS_B2_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0134412
NS_B2_A2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0129625
NS_B2_A2_A1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0130910
NS_B2_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0137739, upper bound: 0.0136182
NS_B2_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0137739, upper bound: 0.0136182
NS_B2_A2_A2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0126893, upper bound: 0.0130391
NS_B2_A2_A2_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0126893, upper bound: 0.0131775
NS_B2_A2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0133125
NS_B2_A2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0134412
NS_B2_A2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133127, upper bound: 0.0134051
NS_B2_A2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0133127, upper bound: 0.0134051
NS_B2_A2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0138702, upper bound: 0.0136182
NS_B2_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.13
Output dim: 9, lower bound: -0.0138702, upper bound: 0.0136182

## BFS NS instance: NS_B1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055176, 0.0005645, -0.0051444, 0.0004507, -0.0059683, 0.0057089
1: -0.0011144, 0.0126296, -0.0007391, 0.0118297, -0.0129441, 0.0133687
2: 0.0087059, 0.0215668, 0.0088845, 0.0199461, -0.0112403, 0.0126823
3: -0.0065633, -0.0025496, -0.0056253, -0.0025983, -0.0039650, 0.0030757
4: -0.0051858, 0.0026611, -0.0048368, 0.0024425, -0.0076283, 0.0074980
5: -0.0005965, 0.0093375, 0.0003234, 0.0089647, -0.0095612, 0.0090141
6: -0.0051214, 0.0008536, -0.0046251, 0.0006787, -0.0056542, 0.0053643
7: -0.0098508, -0.0010319, -0.0094977, -0.0017488, -0.0081020, 0.0084658
8: -0.0053604, -0.0002050, -0.0039882, -0.0002613, -0.0050991, 0.0037832
9: 0.9818687, 1.0011463, 0.9829327, 1.0002489, -0.0183802, 0.0182136

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129166, upper bound: 0.0126288
time: 2.30 seconds

## Relational analysis of NS_B1_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_B1_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130228, upper bound: 0.0126288
time: 2.54 seconds

## BFS NS instance: NS_B1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056123, 0.0005774, -0.0053637, 0.0004686, -0.0060809, 0.0059412
1: -0.0011890, 0.0128320, -0.0009171, 0.0123398, -0.0135288, 0.0137491
2: 0.0086901, 0.0219755, 0.0088386, 0.0209259, -0.0122359, 0.0131368
3: -0.0068022, -0.0025492, -0.0061610, -0.0025968, -0.0042054, 0.0036118
4: -0.0052294, 0.0026925, -0.0049413, 0.0024897, -0.0077191, 0.0076337
5: -0.0008008, 0.0094039, -0.0001043, 0.0091365, -0.0099372, 0.0095082
6: -0.0052350, 0.0008647, -0.0048615, 0.0007007, -0.0058027, 0.0055705
7: -0.0098927, -0.0008704, -0.0095982, -0.0014414, -0.0084514, 0.0087278
8: -0.0056994, -0.0001898, -0.0047485, -0.0002300, -0.0054693, 0.0045587
9: 0.9816315, 1.0013248, 0.9823045, 1.0006214, -0.0189900, 0.0190203

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_B1_B1_B1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131354, upper bound: 0.0131863
time: 2.11 seconds

## Relational analysis of NS_B1_B1_B1_A1_A1_B2_A2

### Relational analysis result of NS_B1_B1_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132376, upper bound: 0.0131863
time: 5.03 seconds

## BFS NS instance: NS_B1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0055610, 0.0005850, -0.0051444, 0.0004507, -0.0060117, 0.0057294
1: -0.0012690, 0.0126510, -0.0007391, 0.0118297, -0.0130988, 0.0133901
2: 0.0085582, 0.0217166, 0.0088845, 0.0199461, -0.0113879, 0.0128321
3: -0.0067140, -0.0025449, -0.0056253, -0.0025983, -0.0041157, 0.0030804
4: -0.0052386, 0.0029539, -0.0048368, 0.0024425, -0.0076811, 0.0077908
5: -0.0010489, 0.0093963, 0.0003234, 0.0089647, -0.0100136, 0.0090729
6: -0.0053786, 0.0008838, -0.0046251, 0.0006787, -0.0059318, 0.0054222
7: -0.0098872, -0.0007353, -0.0094977, -0.0017488, -0.0081384, 0.0087624
8: -0.0056220, -0.0001940, -0.0039882, -0.0002613, -0.0053607, 0.0037942
9: 0.9817790, 1.0017978, 0.9829327, 1.0002489, -0.0184699, 0.0188651

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_B1_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130154, upper bound: 0.0126155
time: 2.47 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_B1_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131467, upper bound: 0.0126155
time: 2.36 seconds

## BFS NS instance: NS_B1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056555, 0.0006047, -0.0053637, 0.0004686, -0.0061240, 0.0059685
1: -0.0013469, 0.0128585, -0.0009171, 0.0123398, -0.0136867, 0.0137756
2: 0.0085428, 0.0221280, 0.0088386, 0.0209259, -0.0123832, 0.0132894
3: -0.0069510, -0.0025444, -0.0061610, -0.0025968, -0.0043542, 0.0036166
4: -0.0052827, 0.0029842, -0.0049413, 0.0024897, -0.0077725, 0.0079255
5: -0.0012519, 0.0094652, -0.0001043, 0.0091365, -0.0103884, 0.0095695
6: -0.0054905, 0.0008951, -0.0048615, 0.0007007, -0.0060786, 0.0056251
7: -0.0099297, -0.0005763, -0.0095982, -0.0014414, -0.0084884, 0.0090219
8: -0.0059577, -0.0001785, -0.0047485, -0.0002300, -0.0057277, 0.0045700
9: 0.9815362, 1.0019742, 0.9823045, 1.0006214, -0.0190852, 0.0196698

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_B1_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132516, upper bound: 0.0131779
time: 2.03 seconds

## Relational analysis of NS_B1_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_B1_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0131779
time: 1.69 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0057894, 0.0006903, -0.0054795, 0.0004785, -0.0062679, 0.0061698
1: -0.0014144, 0.0128646, -0.0010038, 0.0125724, -0.0139868, 0.0138683
2: 0.0085745, 0.0224053, 0.0088156, 0.0214112, -0.0128366, 0.0135897
3: -0.0073870, -0.0025135, -0.0064517, -0.0025965, -0.0047905, 0.0039382
4: -0.0052152, 0.0033358, -0.0050042, 0.0025392, -0.0077543, 0.0083400
5: -0.0019593, 0.0093162, -0.0003632, 0.0092164, -0.0111756, 0.0096794
6: -0.0059693, 0.0009000, -0.0050021, 0.0007186, -0.0066115, 0.0058268
7: -0.0099326, 0.0001700, -0.0096585, -0.0012324, -0.0087002, 0.0098285
8: -0.0065710, -0.0001987, -0.0051711, -0.0002138, -0.0063572, 0.0049724
9: 0.9819455, 1.0026394, 0.9820525, 1.0008541, -0.0189087, 0.0205869

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B1_A2_A1_A2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132152, upper bound: 0.0131599
time: 1.92 seconds

## Relational analysis of NS_B1_B1_B1_A2_A1_A2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0131599
time: 2.16 seconds

## BFS NS instance: NS_B1_B1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0058326, 0.0007948, -0.0054795, 0.0004785, -0.0063111, 0.0062744
1: -0.0015848, 0.0129005, -0.0010038, 0.0125724, -0.0141573, 0.0139043
2: 0.0084305, 0.0225591, 0.0088156, 0.0214112, -0.0129807, 0.0137434
3: -0.0075378, -0.0025100, -0.0064517, -0.0025965, -0.0049413, 0.0039417
4: -0.0052702, 0.0036410, -0.0050042, 0.0025392, -0.0078093, 0.0086452
5: -0.0024321, 0.0093786, -0.0003632, 0.0092164, -0.0116485, 0.0097419
6: -0.0062150, 0.0009201, -0.0050021, 0.0007186, -0.0068788, 0.0058669
7: -0.0099534, 0.0004858, -0.0096585, -0.0012324, -0.0087210, 0.0101442
8: -0.0068362, -0.0001875, -0.0051711, -0.0002138, -0.0066225, 0.0049836
9: 0.9818661, 1.0033122, 0.9820525, 1.0008541, -0.0189881, 0.0212597

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B1_A2_A2_A2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0131510
time: 3.84 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2_A2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131510
time: 2.08 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055239, 0.0005626, -0.0052537, 0.0004637, -0.0059876, 0.0058163
1: -0.0011173, 0.0126440, -0.0008467, 0.0119663, -0.0130836, 0.0134907
2: 0.0087082, 0.0215945, 0.0087960, 0.0203312, -0.0116230, 0.0127985
3: -0.0065803, -0.0025494, -0.0059509, -0.0025905, -0.0039898, 0.0034015
4: -0.0051879, 0.0026619, -0.0048947, 0.0027022, -0.0078901, 0.0075566
5: -0.0006110, 0.0093382, -0.0001741, 0.0090090, -0.0096200, 0.0095123
6: -0.0051308, 0.0008560, -0.0049392, 0.0007155, -0.0057060, 0.0057085
7: -0.0098538, -0.0010232, -0.0095574, -0.0013180, -0.0085359, 0.0085342
8: -0.0053877, -0.0002056, -0.0044629, -0.0002539, -0.0051338, 0.0042573
9: 0.9818539, 1.0011548, 0.9828345, 1.0007819, -0.0189280, 0.0183203

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B2_A1_A1_B1_A1

### Relational analysis result of NS_B1_B1_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129166, upper bound: 0.0126589
time: 1.81 seconds

## Relational analysis of NS_B1_B1_B2_A1_A1_B1_A2

### Relational analysis result of NS_B1_B1_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130228, upper bound: 0.0126589
time: 1.93 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056187, 0.0005759, -0.0054807, 0.0004879, -0.0061066, 0.0060566
1: -0.0011919, 0.0128464, -0.0010638, 0.0124907, -0.0136826, 0.0139102
2: 0.0086924, 0.0220037, 0.0087357, 0.0213354, -0.0126430, 0.0132680
3: -0.0068194, -0.0025489, -0.0064949, -0.0025870, -0.0042324, 0.0039460
4: -0.0052314, 0.0026933, -0.0049965, 0.0027420, -0.0079734, 0.0076898
5: -0.0008156, 0.0094043, -0.0006193, 0.0091780, -0.0099936, 0.0100236
6: -0.0052447, 0.0008671, -0.0051744, 0.0007379, -0.0058551, 0.0059035
7: -0.0098959, -0.0008614, -0.0096529, -0.0009959, -0.0089000, 0.0087915
8: -0.0057269, -0.0001904, -0.0052392, -0.0002226, -0.0055043, 0.0050488
9: 0.9816156, 1.0013334, 0.9821915, 1.0011854, -0.0195698, 0.0191419

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B2_A1_A1_B2_A1

### Relational analysis result of NS_B1_B1_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131354, upper bound: 0.0132376
time: 1.76 seconds

## Relational analysis of NS_B1_B1_B2_A1_A1_B2_A2

### Relational analysis result of NS_B1_B1_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132376, upper bound: 0.0132376
time: 2.30 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0055674, 0.0005831, -0.0052537, 0.0004637, -0.0060311, 0.0058368
1: -0.0012724, 0.0126656, -0.0008467, 0.0119663, -0.0132386, 0.0135122
2: 0.0085597, 0.0217461, 0.0087960, 0.0203312, -0.0117715, 0.0129501
3: -0.0067316, -0.0025447, -0.0059509, -0.0025905, -0.0041411, 0.0034062
4: -0.0052443, 0.0029557, -0.0048947, 0.0027022, -0.0079465, 0.0078503
5: -0.0010648, 0.0094012, -0.0001741, 0.0090090, -0.0100738, 0.0095753
6: -0.0053884, 0.0008860, -0.0049392, 0.0007155, -0.0059842, 0.0057662
7: -0.0098945, -0.0007256, -0.0095574, -0.0013180, -0.0085765, 0.0088319
8: -0.0056500, -0.0001945, -0.0044629, -0.0002539, -0.0053962, 0.0042684
9: 0.9817686, 1.0018082, 0.9828345, 1.0007819, -0.0190133, 0.0189737

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_A1

### Relational analysis result of NS_B1_B1_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130154, upper bound: 0.0126364
time: 2.56 seconds

## Relational analysis of NS_B1_B1_B2_A1_A2_B1_A2

### Relational analysis result of NS_B1_B1_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131467, upper bound: 0.0126364
time: 1.89 seconds

## BFS NS instance: NS_B1_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056617, 0.0006043, -0.0054807, 0.0004879, -0.0061497, 0.0060850
1: -0.0013499, 0.0128728, -0.0010638, 0.0124907, -0.0138406, 0.0139366
2: 0.0085442, 0.0221575, 0.0087357, 0.0213354, -0.0127912, 0.0134218
3: -0.0069682, -0.0025442, -0.0064949, -0.0025870, -0.0043811, 0.0039507
4: -0.0052885, 0.0029858, -0.0049965, 0.0027420, -0.0080305, 0.0079823
5: -0.0012676, 0.0094707, -0.0006193, 0.0091780, -0.0104456, 0.0100900
6: -0.0055002, 0.0008970, -0.0051744, 0.0007379, -0.0061312, 0.0059565
7: -0.0099376, -0.0005660, -0.0096529, -0.0009959, -0.0089417, 0.0090868
8: -0.0059852, -0.0001791, -0.0052392, -0.0002226, -0.0057626, 0.0050602
9: 0.9815230, 1.0019839, 0.9821915, 1.0011854, -0.0196624, 0.0197924

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_A1

### Relational analysis result of NS_B1_B1_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132516, upper bound: 0.0132208
time: 1.96 seconds

## Relational analysis of NS_B1_B1_B2_A1_A2_B2_A2

### Relational analysis result of NS_B1_B1_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0132208
time: 1.60 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0057960, 0.0006964, -0.0055970, 0.0005034, -0.0062994, 0.0062934
1: -0.0014177, 0.0128788, -0.0011603, 0.0127288, -0.0141464, 0.0140392
2: 0.0085780, 0.0224353, 0.0087122, 0.0218286, -0.0132506, 0.0137231
3: -0.0074047, -0.0025142, -0.0067890, -0.0025866, -0.0048181, 0.0042748
4: -0.0052240, 0.0033367, -0.0050598, 0.0027915, -0.0080155, 0.0083965
5: -0.0019759, 0.0093225, -0.0008844, 0.0092566, -0.0112324, 0.0102069
6: -0.0059795, 0.0009054, -0.0053157, 0.0007549, -0.0066638, 0.0061634
7: -0.0099404, 0.0001812, -0.0097149, -0.0007791, -0.0091612, 0.0098962
8: -0.0065996, -0.0001996, -0.0056681, -0.0002063, -0.0063933, 0.0054685
9: 0.9819301, 1.0026491, 0.9819338, 1.0014248, -0.0194947, 0.0207152

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B2_A2_A1_A2_B1

### Relational analysis result of NS_B1_B1_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0131117
time: 2.15 seconds

## Relational analysis of NS_B1_B1_B2_A2_A1_A2_B2

### Relational analysis result of NS_B1_B1_B2_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0132092
time: 2.24 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0058392, 0.0008043, -0.0055970, 0.0005034, -0.0063426, 0.0064013
1: -0.0015879, 0.0129146, -0.0011603, 0.0127288, -0.0143166, 0.0140749
2: 0.0084332, 0.0225888, 0.0087122, 0.0218286, -0.0133954, 0.0138766
3: -0.0075553, -0.0025107, -0.0067890, -0.0025866, -0.0049687, 0.0042782
4: -0.0052789, 0.0036420, -0.0050598, 0.0027915, -0.0080704, 0.0087018
5: -0.0024488, 0.0093872, -0.0008844, 0.0092566, -0.0117054, 0.0102715
6: -0.0062250, 0.0009252, -0.0053157, 0.0007549, -0.0069308, 0.0062059
7: -0.0099662, 0.0004972, -0.0097149, -0.0007791, -0.0091870, 0.0102121
8: -0.0068645, -0.0001882, -0.0056681, -0.0002063, -0.0066582, 0.0054798
9: 0.9818462, 1.0033220, 0.9819338, 1.0014248, -0.0195786, 0.0213882

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B1_B1_B2_A2_A2_A2_A1

### Relational analysis result of NS_B1_B1_B2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0131959
time: 2.28 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2_A2_A2

### Relational analysis result of NS_B1_B1_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131959
time: 1.47 seconds

## BFS NS instance: NS_B1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0056759, 0.0007518, -0.0052987, 0.0004749, -0.0061507, 0.0060505
1: -0.0017400, 0.0126836, -0.0009782, 0.0121446, -0.0138845, 0.0136618
2: 0.0081242, 0.0219555, 0.0087067, 0.0206359, -0.0125117, 0.0132488
3: -0.0070631, -0.0024746, -0.0060331, -0.0025933, -0.0044698, 0.0035585
4: -0.0053372, 0.0035027, -0.0049282, 0.0027503, -0.0080875, 0.0084309
5: -0.0018383, 0.0094077, -0.0003131, 0.0091226, -0.0109609, 0.0097208
6: -0.0058314, 0.0009852, -0.0049758, 0.0007115, -0.0064167, 0.0058789
7: -0.0100226, 0.0001116, -0.0095802, -0.0013269, -0.0086957, 0.0096918
8: -0.0060683, -0.0001675, -0.0045965, -0.0002329, -0.0058353, 0.0044291
9: 0.9818722, 1.0030588, 0.9824350, 1.0010599, -0.0191877, 0.0206238

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B1_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_B1_B2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0132808
time: 2.17 seconds

## Relational analysis of NS_B1_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_B1_B2_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0131510
time: 2.05 seconds

## BFS NS instance: NS_B1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0056831, 0.0007515, -0.0054170, 0.0004930, -0.0061761, 0.0061684
1: -0.0017451, 0.0126988, -0.0011331, 0.0122850, -0.0140301, 0.0138318
2: 0.0081259, 0.0219867, 0.0085951, 0.0210454, -0.0129194, 0.0133916
3: -0.0070825, -0.0024756, -0.0063730, -0.0025834, -0.0044991, 0.0038974
4: -0.0053460, 0.0035052, -0.0049983, 0.0030081, -0.0083541, 0.0085035
5: -0.0018570, 0.0094148, -0.0008297, 0.0091756, -0.0110326, 0.0102445
6: -0.0058429, 0.0009890, -0.0052829, 0.0007460, -0.0064745, 0.0062152
7: -0.0100323, 0.0001247, -0.0096513, -0.0008715, -0.0091607, 0.0097760
8: -0.0060991, -0.0001680, -0.0050954, -0.0002246, -0.0058745, 0.0049274
9: 0.9818591, 1.0030731, 0.9823198, 1.0016394, -0.0197803, 0.0207533

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_B1_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_B1_B2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0133405
time: 1.48 seconds

## Relational analysis of NS_B1_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_B1_B2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0131961
time: 2.65 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0054795, 0.0004785, -0.0057894, 0.0006903, -0.0061698, 0.0062679
1: -0.0010038, 0.0125724, -0.0014144, 0.0128646, -0.0138683, 0.0139868
2: 0.0088156, 0.0214112, 0.0085745, 0.0224053, -0.0135897, 0.0128366
3: -0.0064517, -0.0025965, -0.0073870, -0.0025135, -0.0039382, 0.0047905
4: -0.0050042, 0.0025392, -0.0052152, 0.0033358, -0.0083400, 0.0077543
5: -0.0003632, 0.0092164, -0.0019593, 0.0093162, -0.0096794, 0.0111756
6: -0.0050021, 0.0007186, -0.0059693, 0.0009000, -0.0058268, 0.0066115
7: -0.0096585, -0.0012324, -0.0099326, 0.0001700, -0.0098285, 0.0087002
8: -0.0051711, -0.0002138, -0.0065710, -0.0001987, -0.0049724, 0.0063572
9: 0.9820525, 1.0008541, 0.9819455, 1.0026394, -0.0205869, 0.0189087

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A1_A1_B1_B2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131599, upper bound: 0.0132152
time: 1.37 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_B2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131599, upper bound: 0.0133170
time: 2.17 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0054795, 0.0004785, -0.0058326, 0.0007948, -0.0062744, 0.0063111
1: -0.0010038, 0.0125724, -0.0015848, 0.0129005, -0.0139043, 0.0141573
2: 0.0088156, 0.0214112, 0.0084305, 0.0225591, -0.0137434, 0.0129807
3: -0.0064517, -0.0025965, -0.0075378, -0.0025100, -0.0039417, 0.0049413
4: -0.0050042, 0.0025392, -0.0052702, 0.0036410, -0.0086452, 0.0078093
5: -0.0003632, 0.0092164, -0.0024321, 0.0093786, -0.0097419, 0.0116485
6: -0.0050021, 0.0007186, -0.0062150, 0.0009201, -0.0058669, 0.0068788
7: -0.0096585, -0.0012324, -0.0099534, 0.0004858, -0.0101442, 0.0087210
8: -0.0051711, -0.0002138, -0.0068362, -0.0001875, -0.0049836, 0.0066225
9: 0.9820525, 1.0008541, 0.9818661, 1.0033122, -0.0212597, 0.0189881

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131510, upper bound: 0.0133125
time: 1.87 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131510, upper bound: 0.0134412
time: 2.12 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054781, 0.0004739, -0.0058171, 0.0007281, -0.0062063, 0.0062911
1: -0.0010157, 0.0121644, -0.0014464, 0.0129038, -0.0139195, 0.0136108
2: 0.0087466, 0.0210331, 0.0085663, 0.0225091, -0.0137625, 0.0124668
3: -0.0066141, -0.0025585, -0.0074662, -0.0025134, -0.0041007, 0.0049077
4: -0.0048708, 0.0031491, -0.0052342, 0.0033545, -0.0082252, 0.0083832
5: -0.0011672, 0.0089536, -0.0020372, 0.0093321, -0.0104993, 0.0109908
6: -0.0055461, 0.0007214, -0.0060095, 0.0009046, -0.0062180, 0.0065074
7: -0.0095962, -0.0004803, -0.0099520, 0.0002395, -0.0098357, 0.0094718
8: -0.0054383, -0.0002523, -0.0066925, -0.0001978, -0.0052405, 0.0064402
9: 0.9829475, 1.0018072, 0.9819103, 1.0027215, -0.0197740, 0.0198969

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A1_A2_B1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126513, upper bound: 0.0128664
time: 1.73 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126513, upper bound: 0.0129719
time: 1.79 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0056823, 0.0005313, -0.0059084, 0.0008675, -0.0065499, 0.0064397
1: -0.0012270, 0.0126629, -0.0015299, 0.0131031, -0.0143301, 0.0141928
2: 0.0086947, 0.0219729, 0.0085504, 0.0229036, -0.0142089, 0.0134225
3: -0.0071040, -0.0025592, -0.0076913, -0.0025126, -0.0045914, 0.0051321
4: -0.0049811, 0.0031931, -0.0052780, 0.0033849, -0.0083661, 0.0084711
5: -0.0015742, 0.0091246, -0.0022413, 0.0093979, -0.0109721, 0.0113659
6: -0.0057691, 0.0007463, -0.0061199, 0.0009162, -0.0064421, 0.0066504
7: -0.0096926, -0.0001597, -0.0099928, 0.0004106, -0.0101031, 0.0098331
8: -0.0061444, -0.0002226, -0.0070143, -0.0001825, -0.0059619, 0.0067917
9: 0.9823431, 1.0021955, 0.9816803, 1.0029082, -0.0205652, 0.0205151

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A1_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132625, upper bound: 0.0131117
time: 2.39 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132625, upper bound: 0.0132092
time: 1.59 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054781, 0.0004739, -0.0058604, 0.0008466, -0.0063247, 0.0063344
1: -0.0010157, 0.0121644, -0.0016184, 0.0129305, -0.0139462, 0.0137829
2: 0.0087466, 0.0210331, 0.0084222, 0.0226526, -0.0139060, 0.0126109
3: -0.0066141, -0.0025585, -0.0076187, -0.0025099, -0.0041042, 0.0050602
4: -0.0048708, 0.0031491, -0.0052877, 0.0036601, -0.0085309, 0.0084368
5: -0.0011672, 0.0089536, -0.0025122, 0.0093908, -0.0105580, 0.0114658
6: -0.0055461, 0.0007214, -0.0062570, 0.0009256, -0.0062787, 0.0067852
7: -0.0095962, -0.0004803, -0.0099721, 0.0005571, -0.0101532, 0.0094918
8: -0.0054383, -0.0002523, -0.0069593, -0.0001871, -0.0052513, 0.0067071
9: 0.9829475, 1.0018072, 0.9818573, 1.0033958, -0.0204483, 0.0199499

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A1_A2_B2_A1_B1

### Relational analysis result of NS_B2_A1_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126370, upper bound: 0.0129625
time: 2.30 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_A1_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126370, upper bound: 0.0130910
time: 2.04 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0056823, 0.0005313, -0.0059514, 0.0010357, -0.0067180, 0.0064827
1: -0.0012270, 0.0126629, -0.0017074, 0.0131347, -0.0143617, 0.0143703
2: 0.0086947, 0.0219729, 0.0084067, 0.0230479, -0.0143532, 0.0135663
3: -0.0071040, -0.0025592, -0.0078427, -0.0025091, -0.0045949, 0.0052835
4: -0.0049811, 0.0031931, -0.0053315, 0.0036890, -0.0086702, 0.0085246
5: -0.0015742, 0.0091246, -0.0027192, 0.0094570, -0.0110312, 0.0118438
6: -0.0057691, 0.0007463, -0.0063642, 0.0009368, -0.0064981, 0.0069231
7: -0.0096926, -0.0001597, -0.0100122, 0.0007301, -0.0104227, 0.0098525
8: -0.0061444, -0.0002226, -0.0072787, -0.0001715, -0.0059730, 0.0070562
9: 0.9823431, 1.0021955, 0.9816164, 1.0035785, -0.0212355, 0.0205790

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A1_A2_B2_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132527, upper bound: 0.0132218
time: 2.22 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132527, upper bound: 0.0133405
time: 1.95 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0055970, 0.0005034, -0.0057960, 0.0006964, -0.0062934, 0.0062994
1: -0.0011603, 0.0127288, -0.0014177, 0.0128788, -0.0140392, 0.0141464
2: 0.0087122, 0.0218286, 0.0085780, 0.0224353, -0.0137231, 0.0132506
3: -0.0067890, -0.0025866, -0.0074047, -0.0025142, -0.0042748, 0.0048181
4: -0.0050598, 0.0027915, -0.0052240, 0.0033367, -0.0083965, 0.0080155
5: -0.0008844, 0.0092566, -0.0019759, 0.0093225, -0.0102069, 0.0112324
6: -0.0053157, 0.0007549, -0.0059795, 0.0009054, -0.0061634, 0.0066638
7: -0.0097149, -0.0007791, -0.0099404, 0.0001812, -0.0098962, 0.0091612
8: -0.0056681, -0.0002063, -0.0065996, -0.0001996, -0.0054685, 0.0063933
9: 0.9819338, 1.0014248, 0.9819301, 1.0026491, -0.0207152, 0.0194947

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A2_A1_B1_B2_A1

### Relational analysis result of NS_B2_A1_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131117, upper bound: 0.0133170
time: 2.19 seconds

## Relational analysis of NS_B2_A1_A2_A1_B1_B2_A2

### Relational analysis result of NS_B2_A1_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132092, upper bound: 0.0133170
time: 2.24 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0055970, 0.0005034, -0.0058392, 0.0008043, -0.0064013, 0.0063426
1: -0.0011603, 0.0127288, -0.0015879, 0.0129146, -0.0140749, 0.0143166
2: 0.0087122, 0.0218286, 0.0084332, 0.0225888, -0.0138766, 0.0133954
3: -0.0067890, -0.0025866, -0.0075553, -0.0025107, -0.0042782, 0.0049687
4: -0.0050598, 0.0027915, -0.0052789, 0.0036420, -0.0087018, 0.0080704
5: -0.0008844, 0.0092566, -0.0024488, 0.0093872, -0.0102715, 0.0117054
6: -0.0053157, 0.0007549, -0.0062250, 0.0009252, -0.0062059, 0.0069308
7: -0.0097149, -0.0007791, -0.0099662, 0.0004972, -0.0102121, 0.0091870
8: -0.0056681, -0.0002063, -0.0068645, -0.0001882, -0.0054798, 0.0066582
9: 0.9819338, 1.0014248, 0.9818462, 1.0033220, -0.0213882, 0.0195786

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131959, upper bound: 0.0133125
time: 2.02 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131959, upper bound: 0.0134412
time: 1.93 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055694, 0.0004934, -0.0058237, 0.0007360, -0.0063054, 0.0063171
1: -0.0011606, 0.0122679, -0.0014495, 0.0129179, -0.0140784, 0.0137174
2: 0.0086557, 0.0213440, 0.0085699, 0.0225388, -0.0138831, 0.0127741
3: -0.0068923, -0.0025515, -0.0074838, -0.0025140, -0.0043782, 0.0049324
4: -0.0049492, 0.0034268, -0.0052426, 0.0033552, -0.0083044, 0.0086695
5: -0.0016907, 0.0090109, -0.0020535, 0.0093384, -0.0110291, 0.0110645
6: -0.0058506, 0.0007743, -0.0060197, 0.0009105, -0.0065284, 0.0065608
7: -0.0096711, -0.0000167, -0.0099603, 0.0002506, -0.0099217, 0.0099435
8: -0.0058404, -0.0002449, -0.0067208, -0.0001987, -0.0056417, 0.0064759
9: 0.9828515, 1.0024089, 0.9818947, 1.0027311, -0.0198796, 0.0205141

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A1_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126790, upper bound: 0.0128664
time: 1.69 seconds

## Relational analysis of NS_B2_A1_A2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A1_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126790, upper bound: 0.0129719
time: 1.91 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057736, 0.0006355, -0.0059151, 0.0008813, -0.0066549, 0.0065506
1: -0.0013924, 0.0127645, -0.0015331, 0.0131174, -0.0145097, 0.0142976
2: 0.0085932, 0.0222836, 0.0085539, 0.0229343, -0.0143411, 0.0137296
3: -0.0073780, -0.0025523, -0.0077094, -0.0025132, -0.0048647, 0.0051571
4: -0.0050571, 0.0034591, -0.0052863, 0.0033858, -0.0084429, 0.0087454
5: -0.0020956, 0.0091835, -0.0022591, 0.0094038, -0.0114994, 0.0114426
6: -0.0060654, 0.0007974, -0.0061304, 0.0009217, -0.0067397, 0.0067050
7: -0.0097660, 0.0003108, -0.0100010, 0.0004225, -0.0101884, 0.0103119
8: -0.0065482, -0.0002156, -0.0070434, -0.0001834, -0.0063648, 0.0068278
9: 0.9822422, 1.0027851, 0.9816657, 1.0029179, -0.0206757, 0.0211194

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0131117
time: 1.64 seconds

## Relational analysis of NS_B2_A1_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0132092
time: 2.88 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055694, 0.0004934, -0.0058672, 0.0008581, -0.0064276, 0.0063605
1: -0.0011606, 0.0122679, -0.0016216, 0.0129446, -0.0141052, 0.0138895
2: 0.0086557, 0.0213440, 0.0084249, 0.0226827, -0.0140269, 0.0129191
3: -0.0068923, -0.0025515, -0.0076368, -0.0025106, -0.0043817, 0.0050853
4: -0.0049492, 0.0034268, -0.0052970, 0.0036613, -0.0086105, 0.0087238
5: -0.0016907, 0.0090109, -0.0025294, 0.0093994, -0.0110901, 0.0115404
6: -0.0058506, 0.0007743, -0.0062675, 0.0009304, -0.0065863, 0.0068387
7: -0.0096711, -0.0000167, -0.0099840, 0.0005696, -0.0102407, 0.0099673
8: -0.0058404, -0.0002449, -0.0069881, -0.0001878, -0.0056526, 0.0067432
9: 0.9828515, 1.0024089, 0.9818401, 1.0034059, -0.0205544, 0.0205688

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126608, upper bound: 0.0129625
time: 1.53 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A1_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126608, upper bound: 0.0130910
time: 1.92 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057736, 0.0006355, -0.0059581, 0.0010534, -0.0068270, 0.0065936
1: -0.0013924, 0.0127645, -0.0017108, 0.0131489, -0.0145413, 0.0144753
2: 0.0085932, 0.0222836, 0.0084094, 0.0230784, -0.0144852, 0.0138742
3: -0.0073780, -0.0025523, -0.0078606, -0.0025098, -0.0048682, 0.0053083
4: -0.0050571, 0.0034591, -0.0053404, 0.0036901, -0.0087472, 0.0087995
5: -0.0020956, 0.0091835, -0.0027372, 0.0094658, -0.0115614, 0.0119206
6: -0.0060654, 0.0007974, -0.0063745, 0.0009417, -0.0067939, 0.0069773
7: -0.0097660, 0.0003108, -0.0100247, 0.0007428, -0.0105087, 0.0103355
8: -0.0065482, -0.0002156, -0.0073077, -0.0001722, -0.0063760, 0.0070920
9: 0.9822422, 1.0027851, 0.9815986, 1.0035884, -0.0213463, 0.0211865

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A1_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0132218
time: 2.08 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0133405
time: 2.06 seconds

## BFS NS instance: NS_B2_A2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0052987, 0.0004749, -0.0056759, 0.0007518, -0.0060505, 0.0061507
1: -0.0009782, 0.0121446, -0.0017400, 0.0126836, -0.0136618, 0.0138845
2: 0.0087067, 0.0206359, 0.0081242, 0.0219555, -0.0132488, 0.0125117
3: -0.0060331, -0.0025933, -0.0070631, -0.0024746, -0.0035585, 0.0044698
4: -0.0049282, 0.0027503, -0.0053372, 0.0035027, -0.0084309, 0.0080875
5: -0.0003131, 0.0091226, -0.0018383, 0.0094077, -0.0097208, 0.0109609
6: -0.0049758, 0.0007115, -0.0058314, 0.0009852, -0.0058789, 0.0064167
7: -0.0095802, -0.0013269, -0.0100226, 0.0001116, -0.0096918, 0.0086957
8: -0.0045965, -0.0002329, -0.0060683, -0.0001675, -0.0044291, 0.0058353
9: 0.9824350, 1.0010599, 0.9818722, 1.0030588, -0.0206238, 0.0191877

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A2_A1_A1_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0133010
time: 2.49 seconds

## Relational analysis of NS_B2_A2_A1_A1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0133010
time: 2.03 seconds

## BFS NS instance: NS_B2_A2_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0057268, 0.0006035, -0.0059084, 0.0008675, -0.0065943, 0.0065120
1: -0.0013941, 0.0127008, -0.0015299, 0.0131031, -0.0144972, 0.0142306
2: 0.0085479, 0.0221357, 0.0085504, 0.0229036, -0.0143556, 0.0135852
3: -0.0072589, -0.0025560, -0.0076913, -0.0025126, -0.0047462, 0.0051353
4: -0.0050276, 0.0035011, -0.0052780, 0.0033849, -0.0084125, 0.0087792
5: -0.0020546, 0.0091790, -0.0022413, 0.0093979, -0.0114525, 0.0114203
6: -0.0060211, 0.0007659, -0.0061199, 0.0009162, -0.0067240, 0.0067009
7: -0.0097036, 0.0001611, -0.0099928, 0.0004106, -0.0101142, 0.0101539
8: -0.0064156, -0.0002117, -0.0070143, -0.0001825, -0.0062331, 0.0068026
9: 0.9822744, 1.0028697, 0.9816803, 1.0029082, -0.0206338, 0.0211894

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A1_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131008
time: 1.93 seconds

## Relational analysis of NS_B2_A2_A1_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131959
time: 2.77 seconds

## BFS NS instance: NS_B2_A2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057268, 0.0006035, -0.0059514, 0.0010357, -0.0067625, 0.0065550
1: -0.0013941, 0.0127008, -0.0017074, 0.0131347, -0.0145289, 0.0144082
2: 0.0085479, 0.0221357, 0.0084067, 0.0230479, -0.0145000, 0.0137290
3: -0.0072589, -0.0025560, -0.0078427, -0.0025091, -0.0047497, 0.0052867
4: -0.0050276, 0.0035011, -0.0053315, 0.0036890, -0.0087166, 0.0088327
5: -0.0020546, 0.0091790, -0.0027192, 0.0094570, -0.0115116, 0.0118981
6: -0.0060211, 0.0007659, -0.0063642, 0.0009368, -0.0067151, 0.0069103
7: -0.0097036, 0.0001611, -0.0100122, 0.0007301, -0.0104337, 0.0101734
8: -0.0064156, -0.0002117, -0.0072787, -0.0001715, -0.0062442, 0.0070670
9: 0.9822744, 1.0028697, 0.9816164, 1.0035785, -0.0213041, 0.0212533

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A1_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131008
time: 3.53 seconds

## Relational analysis of NS_B2_A2_A1_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131961
time: 2.66 seconds

## BFS NS instance: NS_B2_A2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0054170, 0.0004930, -0.0056831, 0.0007515, -0.0061684, 0.0061761
1: -0.0011331, 0.0122850, -0.0017451, 0.0126988, -0.0138318, 0.0140301
2: 0.0085951, 0.0210454, 0.0081259, 0.0219867, -0.0133916, 0.0129194
3: -0.0063730, -0.0025834, -0.0070825, -0.0024756, -0.0038974, 0.0044991
4: -0.0049983, 0.0030081, -0.0053460, 0.0035052, -0.0085035, 0.0083541
5: -0.0008297, 0.0091756, -0.0018570, 0.0094148, -0.0102445, 0.0110326
6: -0.0052829, 0.0007460, -0.0058429, 0.0009890, -0.0062152, 0.0064745
7: -0.0096513, -0.0008715, -0.0100323, 0.0001247, -0.0097760, 0.0091607
8: -0.0050954, -0.0002246, -0.0060991, -0.0001680, -0.0049274, 0.0058745
9: 0.9823198, 1.0016394, 0.9818591, 1.0030731, -0.0207533, 0.0197803

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_B2_A2_A2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0133010
time: 2.37 seconds

## Relational analysis of NS_B2_A2_A2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0133010
time: 2.44 seconds

## BFS NS instance: NS_B2_A2_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0058191, 0.0007532, -0.0059151, 0.0008813, -0.0067005, 0.0066683
1: -0.0015673, 0.0127987, -0.0015331, 0.0131174, -0.0146847, 0.0143317
2: 0.0084409, 0.0224448, 0.0085539, 0.0229343, -0.0144933, 0.0138908
3: -0.0075349, -0.0025492, -0.0077094, -0.0025132, -0.0050217, 0.0051601
4: -0.0051090, 0.0037681, -0.0052863, 0.0033858, -0.0084948, 0.0090545
5: -0.0025735, 0.0092466, -0.0022591, 0.0094038, -0.0119773, 0.0115057
6: -0.0063263, 0.0008134, -0.0061304, 0.0009217, -0.0070244, 0.0067392
7: -0.0097970, 0.0006536, -0.0100010, 0.0004225, -0.0102195, 0.0106547
8: -0.0068215, -0.0002039, -0.0070434, -0.0001834, -0.0066381, 0.0068395
9: 0.9821495, 1.0035193, 0.9816657, 1.0029179, -0.0207683, 0.0218536

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0131959
time: 2.21 seconds

## Relational analysis of NS_B2_A2_A2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131959
time: 2.73 seconds

## BFS NS instance: NS_B2_A2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0058191, 0.0007532, -0.0059581, 0.0010534, -0.0068725, 0.0067113
1: -0.0015673, 0.0127987, -0.0017108, 0.0131489, -0.0147162, 0.0145095
2: 0.0084409, 0.0224448, 0.0084094, 0.0230784, -0.0146374, 0.0140354
3: -0.0075349, -0.0025492, -0.0078606, -0.0025098, -0.0050252, 0.0053114
4: -0.0051090, 0.0037681, -0.0053404, 0.0036901, -0.0087991, 0.0091086
5: -0.0025735, 0.0092466, -0.0027372, 0.0094658, -0.0120393, 0.0119838
6: -0.0063263, 0.0008134, -0.0063745, 0.0009417, -0.0070157, 0.0069565
7: -0.0097970, 0.0006536, -0.0100247, 0.0007428, -0.0105398, 0.0106783
8: -0.0068215, -0.0002039, -0.0073077, -0.0001722, -0.0066493, 0.0071037
9: 0.9821495, 1.0035193, 0.9815986, 1.0035884, -0.0214389, 0.0219207

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_B2_A2_A2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131008
time: 2.02 seconds

## Relational analysis of NS_B2_A2_A2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131961
time: 2.16 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.08 seconds
NS_B1_B1_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0129166, upper bound: 0.0126288
NS_B1_B1_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0130228, upper bound: 0.0126288
NS_B1_B1_B1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131354, upper bound: 0.0131863
NS_B1_B1_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132376, upper bound: 0.0131863
NS_B1_B1_B1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0130154, upper bound: 0.0126155
NS_B1_B1_B1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131467, upper bound: 0.0126155
NS_B1_B1_B1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132516, upper bound: 0.0131779
NS_B1_B1_B1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0131779
NS_B1_B1_B1_A2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132152, upper bound: 0.0131599
NS_B1_B1_B1_A2_A1_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0131599
NS_B1_B1_B1_A2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0131510
NS_B1_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131510
NS_B1_B1_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0129166, upper bound: 0.0126589
NS_B1_B1_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0130228, upper bound: 0.0126589
NS_B1_B1_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131354, upper bound: 0.0132376
NS_B1_B1_B2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132376, upper bound: 0.0132376
NS_B1_B1_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0130154, upper bound: 0.0126364
NS_B1_B1_B2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131467, upper bound: 0.0126364
NS_B1_B1_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132516, upper bound: 0.0132208
NS_B1_B1_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0132208
NS_B1_B1_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0131117
NS_B1_B1_B2_A2_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0132092
NS_B1_B1_B2_A2_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0131959
NS_B1_B1_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131959
NS_B1_B2_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0132808
NS_B1_B2_B1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0131510
NS_B1_B2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0133405
NS_B1_B2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0131961
NS_B2_A1_A1_A1_B1_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131599, upper bound: 0.0132152
NS_B2_A1_A1_A1_B1_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131599, upper bound: 0.0133170
NS_B2_A1_A1_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131510, upper bound: 0.0133125
NS_B2_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131510, upper bound: 0.0134412
NS_B2_A1_A1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0126513, upper bound: 0.0128664
NS_B2_A1_A1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0126513, upper bound: 0.0129719
NS_B2_A1_A1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132625, upper bound: 0.0131117
NS_B2_A1_A1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132625, upper bound: 0.0132092
NS_B2_A1_A1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0126370, upper bound: 0.0129625
NS_B2_A1_A1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0126370, upper bound: 0.0130910
NS_B2_A1_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132527, upper bound: 0.0132218
NS_B2_A1_A1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132527, upper bound: 0.0133405
NS_B2_A1_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131117, upper bound: 0.0133170
NS_B2_A1_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132092, upper bound: 0.0133170
NS_B2_A1_A2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131959, upper bound: 0.0133125
NS_B2_A1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0131959, upper bound: 0.0134412
NS_B2_A1_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0126790, upper bound: 0.0128664
NS_B2_A1_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0126790, upper bound: 0.0129719
NS_B2_A1_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0131117
NS_B2_A1_A2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133170, upper bound: 0.0132092
NS_B2_A1_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0126608, upper bound: 0.0129625
NS_B2_A1_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0126608, upper bound: 0.0130910
NS_B2_A1_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0132218
NS_B2_A1_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133010, upper bound: 0.0133405
NS_B2_A2_A1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0133010
NS_B2_A2_A1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0133010
NS_B2_A2_A1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131008
NS_B2_A2_A1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131959
NS_B2_A2_A1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131008
NS_B2_A2_A1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131961
NS_B2_A2_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0133010
NS_B2_A2_A2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0133010
NS_B2_A2_A2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0133125, upper bound: 0.0131959
NS_B2_A2_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131959
NS_B2_A2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131008
NS_B2_A2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 6.08
Output dim: 9, lower bound: -0.0134412, upper bound: 0.0131961

## BFS NS instance: NS_B1_B1_B1_A2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0055244, 0.0007357, -0.0053687, 0.0004668, -0.0059913, 0.0061044
1: -0.0017274, 0.0122728, -0.0009219, 0.0123390, -0.0140664, 0.0131947
2: 0.0080623, 0.0212536, 0.0088327, 0.0209385, -0.0128762, 0.0124209
3: -0.0067270, -0.0024767, -0.0061680, -0.0025974, -0.0041297, 0.0036913
4: -0.0050976, 0.0036812, -0.0049469, 0.0025037, -0.0076013, 0.0086281
5: -0.0017881, 0.0091512, -0.0001196, 0.0091406, -0.0109287, 0.0092708
6: -0.0058066, 0.0008686, -0.0048564, 0.0006997, -0.0063783, 0.0056492
7: -0.0098105, -0.0000008, -0.0096099, -0.0014158, -0.0083947, 0.0096091
8: -0.0056228, -0.0002144, -0.0047507, -0.0002286, -0.0053943, 0.0045362
9: 0.9825424, 1.0032214, 0.9822834, 1.0006462, -0.0181038, 0.0209380

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_B1_B1_B1_A2_A2_A2_A2_A1

### Relational analysis result of NS_B1_B1_B1_A2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133843, upper bound: 0.0131510
time: 1.95 seconds

## Relational analysis of NS_B1_B1_B1_A2_A2_A2_A2_A2

### Relational analysis result of NS_B1_B1_B1_A2_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133843, upper bound: 0.0131510
time: 2.25 seconds

## BFS NS instance: NS_B1_B1_B2_A2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0055316, 0.0007342, -0.0054893, 0.0004868, -0.0060185, 0.0062235
1: -0.0017326, 0.0122879, -0.0010725, 0.0124981, -0.0142306, 0.0133603
2: 0.0080642, 0.0212849, 0.0087296, 0.0213632, -0.0132990, 0.0125553
3: -0.0067464, -0.0024778, -0.0065139, -0.0025875, -0.0041589, 0.0040360
4: -0.0051062, 0.0036837, -0.0050012, 0.0027563, -0.0078625, 0.0086848
5: -0.0018065, 0.0091589, -0.0006440, 0.0091808, -0.0109873, 0.0098029
6: -0.0058178, 0.0008730, -0.0051722, 0.0007357, -0.0064312, 0.0059933
7: -0.0098212, 0.0000116, -0.0096652, -0.0009631, -0.0088581, 0.0096768
8: -0.0056536, -0.0002150, -0.0052583, -0.0002209, -0.0054327, 0.0050433
9: 0.9825284, 1.0032361, 0.9821637, 1.0012181, -0.0186896, 0.0210723

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 216
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_B1_B1_B2_A2_A2_A2_A2_A1

### Relational analysis result of NS_B1_B1_B2_A2_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131959
time: 2.27 seconds

## Relational analysis of NS_B1_B1_B2_A2_A2_A2_A2_A2

### Relational analysis result of NS_B1_B1_B2_A2_A2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131510
time: 2.44 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0053687, 0.0004668, -0.0055244, 0.0007357, -0.0061044, 0.0059913
1: -0.0009219, 0.0123390, -0.0017274, 0.0122728, -0.0131947, 0.0140664
2: 0.0088327, 0.0209385, 0.0080623, 0.0212536, -0.0124209, 0.0128762
3: -0.0061680, -0.0025974, -0.0067270, -0.0024767, -0.0036913, 0.0041297
4: -0.0049469, 0.0025037, -0.0050976, 0.0036812, -0.0086281, 0.0076013
5: -0.0001196, 0.0091406, -0.0017881, 0.0091512, -0.0092708, 0.0109287
6: -0.0048564, 0.0006997, -0.0058066, 0.0008686, -0.0056492, 0.0063783
7: -0.0096099, -0.0014158, -0.0098105, -0.0000008, -0.0096091, 0.0083947
8: -0.0047507, -0.0002286, -0.0056228, -0.0002144, -0.0045362, 0.0053943
9: 0.9822834, 1.0006462, 0.9825424, 1.0032214, -0.0209380, 0.0181038

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131510, upper bound: 0.0133843
time: 2.01 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_B2_B2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131510, upper bound: 0.0134412
time: 2.42 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0054893, 0.0004868, -0.0055316, 0.0007342, -0.0062235, 0.0060185
1: -0.0010725, 0.0124981, -0.0017326, 0.0122879, -0.0133603, 0.0142306
2: 0.0087296, 0.0213632, 0.0080642, 0.0212849, -0.0125553, 0.0132990
3: -0.0065139, -0.0025875, -0.0067464, -0.0024778, -0.0040360, 0.0041589
4: -0.0050012, 0.0027563, -0.0051062, 0.0036837, -0.0086848, 0.0078625
5: -0.0006440, 0.0091808, -0.0018065, 0.0091589, -0.0098029, 0.0109873
6: -0.0051722, 0.0007357, -0.0058178, 0.0008730, -0.0059933, 0.0064312
7: -0.0096652, -0.0009631, -0.0098212, 0.0000116, -0.0096768, 0.0088581
8: -0.0052583, -0.0002209, -0.0056536, -0.0002150, -0.0050433, 0.0054327
9: 0.9821637, 1.0012181, 0.9825284, 1.0032361, -0.0210723, 0.0186896

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: A, layer: 1, pos: 237
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 216
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131959, upper bound: 0.0133669
time: 3.36 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2_B2_B2_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131959, upper bound: 0.0133669
time: 2.03 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.01 + 597.82 = 601.83 seconds
