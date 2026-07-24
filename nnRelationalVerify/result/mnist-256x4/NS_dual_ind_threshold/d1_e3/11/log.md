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
execution time: IAR + RelationalAnalysis = 1.51 + 2.52 = 4.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0148989, upper bound: 0.0148989

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 184

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145063, upper bound: 0.0146094
time: 1.92 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146094, upper bound: 0.0146094
time: 1.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.87 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.87
Output dim: 9, lower bound: -0.0145063, upper bound: 0.0146094
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.87
Output dim: 9, lower bound: -0.0146094, upper bound: 0.0146094

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0058004, 0.0006841, -0.0059263, 0.0009079, -0.0067083, 0.0066104
1: -0.0013758, 0.0132853, -0.0016379, 0.0135157, -0.0148916, 0.0149232
2: 0.0085352, 0.0228277, 0.0083453, 0.0232974, -0.0147622, 0.0144824
3: -0.0072512, -0.0025268, -0.0075750, -0.0024715, -0.0047798, 0.0050482
4: -0.0056355, 0.0028655, -0.0059589, 0.0030592, -0.0086947, 0.0088244
5: -0.0013221, 0.0098169, -0.0017811, 0.0100809, -0.0114030, 0.0115980
6: -0.0055290, 0.0010924, -0.0057578, 0.0013149, -0.0067500, 0.0067648
7: -0.0102420, -0.0004327, -0.0105284, -0.0000168, -0.0102252, 0.0100957
8: -0.0063495, -0.0001200, -0.0068202, -0.0000843, -0.0062651, 0.0067002
9: 0.9807101, 1.0018632, 0.9802593, 1.0024542, -0.0217441, 0.0216039

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141977, upper bound: 0.0141872
time: 2.13 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141976, upper bound: 0.0142860
time: 2.15 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0060925, 0.0013062, -0.0058910, 0.0008413, -0.0069339, 0.0071972
1: -0.0017528, 0.0135603, -0.0015619, 0.0134477, -0.0152005, 0.0151222
2: 0.0083981, 0.0237417, 0.0083929, 0.0231589, -0.0147608, 0.0153488
3: -0.0081334, -0.0024903, -0.0074840, -0.0024851, -0.0056483, 0.0049937
4: -0.0056885, 0.0035566, -0.0058907, 0.0030139, -0.0087024, 0.0094473
5: -0.0027876, 0.0098145, -0.0016601, 0.0100185, -0.0128061, 0.0114746
6: -0.0064150, 0.0011414, -0.0056971, 0.0012693, -0.0076749, 0.0067814
7: -0.0103390, 0.0008715, -0.0104578, -0.0001245, -0.0102145, 0.0113292
8: -0.0076653, -0.0001128, -0.0066878, -0.0000933, -0.0075721, 0.0065750
9: 0.9807625, 1.0034755, 0.9804090, 1.0023034, -0.0215408, 0.0230666

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0141872
time: 3.14 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0142860
time: 2.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.94 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 9, lower bound: -0.0141977, upper bound: 0.0141872
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 9, lower bound: -0.0141976, upper bound: 0.0142860
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0141872
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 9, lower bound: -0.0142860, upper bound: 0.0142860

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057724, 0.0006654, -0.0058040, 0.0008080, -0.0065804, 0.0064694
1: -0.0013525, 0.0132106, -0.0015259, 0.0131926, -0.0145451, 0.0147365
2: 0.0085608, 0.0226981, 0.0084505, 0.0227427, -0.0141818, 0.0142475
3: -0.0071893, -0.0025282, -0.0073000, -0.0024775, -0.0047119, 0.0047718
4: -0.0055586, 0.0028391, -0.0056399, 0.0029349, -0.0084936, 0.0084790
5: -0.0012485, 0.0097361, -0.0014453, 0.0097452, -0.0109937, 0.0111815
6: -0.0054866, 0.0010507, -0.0055643, 0.0011392, -0.0065192, 0.0065178
7: -0.0101798, -0.0004956, -0.0102717, -0.0003023, -0.0098775, 0.0097760
8: -0.0062632, -0.0001342, -0.0064323, -0.0001426, -0.0061206, 0.0062981
9: 0.9808833, 1.0017868, 0.9809736, 1.0020976, -0.0212143, 0.0208132

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140798, upper bound: 0.0141872
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140798, upper bound: 0.0141872
time: 2.00 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057649, 0.0006631, -0.0058532, 0.0008789, -0.0066438, 0.0065163
1: -0.0013495, 0.0131750, -0.0016979, 0.0132381, -0.0145875, 0.0148728
2: 0.0085655, 0.0226583, 0.0083028, 0.0229173, -0.0143518, 0.0143555
3: -0.0071805, -0.0025284, -0.0074618, -0.0024729, -0.0047075, 0.0049334
4: -0.0055498, 0.0028353, -0.0057041, 0.0032180, -0.0087677, 0.0085394
5: -0.0012387, 0.0097301, -0.0019024, 0.0098146, -0.0110533, 0.0116324
6: -0.0054815, 0.0010450, -0.0058215, 0.0011725, -0.0065588, 0.0068000
7: -0.0101694, -0.0005044, -0.0103112, -0.0000005, -0.0101689, 0.0098068
8: -0.0062517, -0.0001358, -0.0067112, -0.0001299, -0.0061219, 0.0065755
9: 0.9809116, 1.0017759, 0.9808804, 1.0027585, -0.0218469, 0.0208955

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0140978
time: 1.45 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0140978
time: 1.99 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0060646, 0.0012399, -0.0057686, 0.0007510, -0.0068157, 0.0070084
1: -0.0017234, 0.0134828, -0.0014517, 0.0131272, -0.0148507, 0.0149345
2: 0.0084235, 0.0236128, 0.0084981, 0.0226027, -0.0141792, 0.0151147
3: -0.0080714, -0.0024918, -0.0072076, -0.0024911, -0.0055803, 0.0047158
4: -0.0056098, 0.0035282, -0.0055679, 0.0028889, -0.0084987, 0.0090961
5: -0.0027063, 0.0097340, -0.0013223, 0.0096800, -0.0123864, 0.0110563
6: -0.0063705, 0.0011005, -0.0055020, 0.0010955, -0.0074435, 0.0065327
7: -0.0102786, 0.0008020, -0.0102026, -0.0004117, -0.0098669, 0.0110046
8: -0.0075755, -0.0001269, -0.0062966, -0.0001514, -0.0074241, 0.0061697
9: 0.9809347, 1.0033902, 0.9811249, 1.0019456, -0.0210109, 0.0222653

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0139752
time: 2.23 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139752
time: 3.90 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0060567, 0.0012293, -0.0058174, 0.0008144, -0.0068711, 0.0070467
1: -0.0017189, 0.0134513, -0.0016219, 0.0131712, -0.0148901, 0.0150732
2: 0.0084278, 0.0235694, 0.0083510, 0.0227761, -0.0143483, 0.0152183
3: -0.0080611, -0.0024921, -0.0073684, -0.0024866, -0.0055745, 0.0048764
4: -0.0056054, 0.0035241, -0.0056342, 0.0031736, -0.0087790, 0.0091583
5: -0.0026940, 0.0097272, -0.0017802, 0.0097506, -0.0124447, 0.0115073
6: -0.0063639, 0.0010929, -0.0057601, 0.0011262, -0.0074785, 0.0068160
7: -0.0102636, 0.0007911, -0.0102386, -0.0001099, -0.0101538, 0.0110298
8: -0.0075612, -0.0001285, -0.0065747, -0.0001392, -0.0074220, 0.0064462
9: 0.9809697, 1.0033771, 0.9810387, 1.0026062, -0.0216364, 0.0223383

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0140978
time: 2.06 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0140978
time: 2.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.86 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.86
Output dim: 9, lower bound: -0.0140798, upper bound: 0.0141872
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.86
Output dim: 9, lower bound: -0.0140798, upper bound: 0.0141872
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.86
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0140978
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.86
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0140978
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.86
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0139752
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.86
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139752
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.86
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0140978
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.86
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0140978

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0056854, 0.0006197, -0.0058040, 0.0008080, -0.0064934, 0.0064237
1: -0.0012773, 0.0129777, -0.0015259, 0.0131926, -0.0144699, 0.0145036
2: 0.0086429, 0.0222897, 0.0084505, 0.0227427, -0.0140997, 0.0138392
3: -0.0069919, -0.0025329, -0.0073000, -0.0024775, -0.0045144, 0.0047671
4: -0.0053194, 0.0027542, -0.0056399, 0.0029349, -0.0082543, 0.0083941
5: -0.0010110, 0.0094836, -0.0014453, 0.0097452, -0.0107562, 0.0109289
6: -0.0053496, 0.0009187, -0.0055643, 0.0011392, -0.0063757, 0.0063787
7: -0.0099857, -0.0006972, -0.0102717, -0.0003023, -0.0096834, 0.0095745
8: -0.0059824, -0.0001782, -0.0064323, -0.0001426, -0.0058398, 0.0062541
9: 0.9814277, 1.0015398, 0.9809736, 1.0020976, -0.0206699, 0.0205662

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140901, upper bound: 0.0140932
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140901, upper bound: 0.0141872
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057285, 0.0006646, -0.0058040, 0.0008080, -0.0065366, 0.0064686
1: -0.0014376, 0.0130042, -0.0015259, 0.0131926, -0.0146302, 0.0145301
2: 0.0084961, 0.0224397, 0.0084505, 0.0227427, -0.0142465, 0.0139891
3: -0.0071409, -0.0025281, -0.0073000, -0.0024775, -0.0046635, 0.0047719
4: -0.0053748, 0.0030445, -0.0056399, 0.0029349, -0.0083097, 0.0086844
5: -0.0014634, 0.0095468, -0.0014453, 0.0097452, -0.0112086, 0.0109922
6: -0.0056039, 0.0009494, -0.0055643, 0.0011392, -0.0066491, 0.0064328
7: -0.0100245, -0.0004004, -0.0102717, -0.0003023, -0.0097223, 0.0098713
8: -0.0062409, -0.0001667, -0.0064323, -0.0001426, -0.0060984, 0.0062656
9: 0.9813340, 1.0021896, 0.9809736, 1.0020976, -0.0207636, 0.0212160

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140901, upper bound: 0.0140932
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140901, upper bound: 0.0141872
time: 1.31 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055795, 0.0005002, -0.0058062, 0.0008155, -0.0063950, 0.0063064
1: -0.0010877, 0.0128052, -0.0016249, 0.0131444, -0.0142321, 0.0144301
2: 0.0087328, 0.0218711, 0.0083431, 0.0227109, -0.0139782, 0.0135279
3: -0.0066962, -0.0025920, -0.0073377, -0.0024888, -0.0042075, 0.0047457
4: -0.0052404, 0.0026327, -0.0056250, 0.0031687, -0.0084091, 0.0082577
5: -0.0006416, 0.0094728, -0.0017501, 0.0097497, -0.0103913, 0.0112229
6: -0.0051602, 0.0008479, -0.0057408, 0.0011209, -0.0061661, 0.0064976
7: -0.0098460, -0.0010004, -0.0102290, -0.0001266, -0.0097193, 0.0092286
8: -0.0055216, -0.0001680, -0.0065231, -0.0001377, -0.0053839, 0.0063551
9: 0.9814906, 1.0011365, 0.9810279, 1.0025949, -0.0211043, 0.0201086

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0140053
time: 2.36 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0140978
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057013, 0.0005454, -0.0058129, 0.0008192, -0.0065205, 0.0063583
1: -0.0012537, 0.0129708, -0.0016288, 0.0131594, -0.0144131, 0.0145996
2: 0.0086280, 0.0223058, 0.0083457, 0.0227421, -0.0141141, 0.0139602
3: -0.0070424, -0.0025820, -0.0073558, -0.0024885, -0.0045539, 0.0047738
4: -0.0053024, 0.0028850, -0.0056320, 0.0031697, -0.0084721, 0.0085170
5: -0.0011765, 0.0095224, -0.0017659, 0.0097549, -0.0109314, 0.0112883
6: -0.0054789, 0.0008839, -0.0057508, 0.0011250, -0.0065086, 0.0065534
7: -0.0099156, -0.0005304, -0.0102378, -0.0001165, -0.0097991, 0.0097074
8: -0.0060359, -0.0001596, -0.0065520, -0.0001384, -0.0058976, 0.0063924
9: 0.9813670, 1.0017171, 0.9810133, 1.0026038, -0.0212368, 0.0207038

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0140053
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0140978
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0059015, 0.0007985, -0.0057214, 0.0007045, -0.0066060, 0.0065199
1: -0.0014373, 0.0131646, -0.0013824, 0.0130348, -0.0144720, 0.0145470
2: 0.0085850, 0.0229456, 0.0085386, 0.0223963, -0.0138113, 0.0144070
3: -0.0076467, -0.0025532, -0.0070835, -0.0025067, -0.0051400, 0.0045304
4: -0.0052914, 0.0033338, -0.0054901, 0.0028384, -0.0081298, 0.0088239
5: -0.0021465, 0.0094724, -0.0011701, 0.0096162, -0.0117626, 0.0106425
6: -0.0060759, 0.0008946, -0.0054200, 0.0010437, -0.0070785, 0.0062354
7: -0.0099487, 0.0003244, -0.0101217, -0.0005371, -0.0094116, 0.0104461
8: -0.0069304, -0.0001588, -0.0061087, -0.0001592, -0.0067712, 0.0059500
9: 0.9814926, 1.0027567, 0.9812723, 1.0017828, -0.0202902, 0.0214844

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0138580
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0138580
time: 1.96 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0059935, 0.0010659, -0.0057276, 0.0007046, -0.0066981, 0.0067935
1: -0.0016215, 0.0132721, -0.0013855, 0.0130487, -0.0146702, 0.0146575
2: 0.0084838, 0.0232649, 0.0085415, 0.0224238, -0.0139399, 0.0147234
3: -0.0079236, -0.0025463, -0.0071000, -0.0025066, -0.0054170, 0.0045538
4: -0.0053695, 0.0035978, -0.0054961, 0.0028386, -0.0082081, 0.0090939
5: -0.0026749, 0.0095310, -0.0011836, 0.0096194, -0.0122944, 0.0107146
6: -0.0063694, 0.0009427, -0.0054292, 0.0010485, -0.0073723, 0.0062905
7: -0.0100323, 0.0008058, -0.0101272, -0.0005292, -0.0095031, 0.0109330
8: -0.0073398, -0.0001514, -0.0061353, -0.0001598, -0.0071799, 0.0059838
9: 0.9813914, 1.0033524, 0.9812583, 1.0017900, -0.0203986, 0.0220941

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0138580
time: 2.11 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0138580
time: 1.91 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0058954, 0.0007930, -0.0057700, 0.0007546, -0.0066500, 0.0065629
1: -0.0014341, 0.0131330, -0.0015499, 0.0130768, -0.0145110, 0.0146828
2: 0.0085897, 0.0229065, 0.0083914, 0.0225693, -0.0139795, 0.0145151
3: -0.0076393, -0.0025534, -0.0072436, -0.0025023, -0.0051371, 0.0046902
4: -0.0052853, 0.0033302, -0.0055550, 0.0031244, -0.0084097, 0.0088851
5: -0.0021374, 0.0094642, -0.0016273, 0.0096854, -0.0118228, 0.0110916
6: -0.0060710, 0.0008872, -0.0056789, 0.0010748, -0.0071162, 0.0065173
7: -0.0099325, 0.0003158, -0.0101551, -0.0002364, -0.0096961, 0.0104709
8: -0.0069207, -0.0001605, -0.0063857, -0.0001471, -0.0067736, 0.0062253
9: 0.9815274, 1.0027462, 0.9811879, 1.0024425, -0.0209150, 0.0215583

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0139904
time: 1.99 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0139904
time: 2.66 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0059868, 0.0010568, -0.0057763, 0.0007575, -0.0067443, 0.0068332
1: -0.0016174, 0.0132374, -0.0015533, 0.0130909, -0.0147083, 0.0147907
2: 0.0084881, 0.0232218, 0.0083941, 0.0225988, -0.0141107, 0.0148277
3: -0.0079145, -0.0025465, -0.0072608, -0.0025022, -0.0054123, 0.0047143
4: -0.0053641, 0.0035944, -0.0055618, 0.0031250, -0.0084891, 0.0091562
5: -0.0026647, 0.0095248, -0.0016422, 0.0096919, -0.0123566, 0.0111670
6: -0.0063640, 0.0009356, -0.0056885, 0.0010785, -0.0074077, 0.0065741
7: -0.0100208, 0.0007964, -0.0101668, -0.0002271, -0.0097938, 0.0109632
8: -0.0073276, -0.0001528, -0.0064133, -0.0001477, -0.0071798, 0.0062604
9: 0.9814240, 1.0033411, 0.9811716, 1.0024503, -0.0210263, 0.0221695

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139904
time: 2.42 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139904
time: 2.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.28 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140901, upper bound: 0.0140932
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140901, upper bound: 0.0141872
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140901, upper bound: 0.0140932
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140901, upper bound: 0.0141872
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0140053
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0140978
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0140053
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0140978
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0138580
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0138580
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0138580
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0138580
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0139904
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140154, upper bound: 0.0139904
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139904
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.28
Output dim: 9, lower bound: -0.0140978, upper bound: 0.0139904

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0056854, 0.0006197, -0.0056854, 0.0006197, -0.0063051, 0.0063051
1: -0.0012773, 0.0129777, -0.0012773, 0.0129777, -0.0142550, 0.0142550
2: 0.0086429, 0.0222897, 0.0086429, 0.0222897, -0.0136468, 0.0136468
3: -0.0069919, -0.0025329, -0.0069919, -0.0025329, -0.0044590, 0.0044590
4: -0.0053194, 0.0027542, -0.0053194, 0.0027542, -0.0080736, 0.0080736
5: -0.0010110, 0.0094836, -0.0010110, 0.0094836, -0.0104946, 0.0104946
6: -0.0053496, 0.0009187, -0.0053496, 0.0009187, -0.0061585, 0.0061585
7: -0.0099857, -0.0006972, -0.0099857, -0.0006972, -0.0092885, 0.0092885
8: -0.0059824, -0.0001782, -0.0059824, -0.0001782, -0.0058042, 0.0058042
9: 0.9814277, 1.0015398, 0.9814277, 1.0015398, -0.0201121, 0.0201121

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138248, upper bound: 0.0138872
time: 2.18 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138750, upper bound: 0.0138872
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0056854, 0.0006197, -0.0059811, 0.0010451, -0.0067305, 0.0066009
1: -0.0012773, 0.0129777, -0.0016335, 0.0132486, -0.0145259, 0.0146112
2: 0.0086429, 0.0222897, 0.0085048, 0.0232134, -0.0145705, 0.0137849
3: -0.0069919, -0.0025329, -0.0078800, -0.0024969, -0.0044950, 0.0053471
4: -0.0053194, 0.0027542, -0.0053705, 0.0034449, -0.0087643, 0.0081246
5: -0.0010110, 0.0094836, -0.0024594, 0.0094799, -0.0104909, 0.0119430
6: -0.0053496, 0.0009187, -0.0062352, 0.0009716, -0.0062702, 0.0071193
7: -0.0099857, -0.0006972, -0.0100878, 0.0005931, -0.0105788, 0.0093906
8: -0.0059824, -0.0001782, -0.0072974, -0.0001711, -0.0058113, 0.0071192
9: 0.9814277, 1.0015398, 0.9814736, 1.0031344, -0.0217066, 0.0200662

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138248, upper bound: 0.0140011
time: 1.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138750, upper bound: 0.0140011
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0057285, 0.0006646, -0.0056854, 0.0006197, -0.0063483, 0.0063500
1: -0.0014376, 0.0130042, -0.0012773, 0.0129777, -0.0144153, 0.0142814
2: 0.0084961, 0.0224397, 0.0086429, 0.0222897, -0.0137936, 0.0137967
3: -0.0071409, -0.0025281, -0.0069919, -0.0025329, -0.0046081, 0.0044638
4: -0.0053748, 0.0030445, -0.0053194, 0.0027542, -0.0081289, 0.0083639
5: -0.0014634, 0.0095468, -0.0010110, 0.0094836, -0.0109470, 0.0105579
6: -0.0056039, 0.0009494, -0.0053496, 0.0009187, -0.0064338, 0.0062127
7: -0.0100245, -0.0004004, -0.0099857, -0.0006972, -0.0093274, 0.0095853
8: -0.0062409, -0.0001667, -0.0059824, -0.0001782, -0.0060628, 0.0058156
9: 0.9813340, 1.0021896, 0.9814277, 1.0015398, -0.0202058, 0.0207619

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0138693
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0138693
time: 1.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057285, 0.0006646, -0.0059811, 0.0010451, -0.0067736, 0.0066457
1: -0.0014376, 0.0130042, -0.0016335, 0.0132486, -0.0146862, 0.0146377
2: 0.0084961, 0.0224397, 0.0085048, 0.0232134, -0.0147173, 0.0139349
3: -0.0071409, -0.0025281, -0.0078800, -0.0024969, -0.0046440, 0.0053519
4: -0.0053748, 0.0030445, -0.0053705, 0.0034449, -0.0088196, 0.0084150
5: -0.0014634, 0.0095468, -0.0024594, 0.0094799, -0.0109433, 0.0120062
6: -0.0056039, 0.0009494, -0.0062352, 0.0009716, -0.0065455, 0.0071735
7: -0.0100245, -0.0004004, -0.0100878, 0.0005931, -0.0106177, 0.0096874
8: -0.0062409, -0.0001667, -0.0072974, -0.0001711, -0.0060699, 0.0071307
9: 0.9813340, 1.0021896, 0.9814736, 1.0031344, -0.0218003, 0.0207160

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 71

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0139752
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0139752
time: 2.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055795, 0.0005002, -0.0056830, 0.0006155, -0.0061950, 0.0061832
1: -0.0010877, 0.0128052, -0.0013703, 0.0129133, -0.0140010, 0.0141755
2: 0.0087328, 0.0218711, 0.0085376, 0.0222418, -0.0135090, 0.0133335
3: -0.0066962, -0.0025920, -0.0070207, -0.0025443, -0.0041520, 0.0044286
4: -0.0052404, 0.0026327, -0.0052961, 0.0029950, -0.0082354, 0.0079288
5: -0.0006416, 0.0094728, -0.0013141, 0.0094821, -0.0101237, 0.0107869
6: -0.0051602, 0.0008479, -0.0055250, 0.0008987, -0.0059303, 0.0062752
7: -0.0098460, -0.0010004, -0.0099434, -0.0005247, -0.0093213, 0.0089429
8: -0.0055216, -0.0001680, -0.0060595, -0.0001748, -0.0053468, 0.0058915
9: 0.9814906, 1.0011365, 0.9814771, 1.0020293, -0.0205387, 0.0196594

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140053
time: 2.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0138693
time: 3.13 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055795, 0.0005002, -0.0059808, 0.0011084, -0.0066880, 0.0064810
1: -0.0010877, 0.0128052, -0.0017378, 0.0131949, -0.0142826, 0.0145430
2: 0.0087328, 0.0218711, 0.0084014, 0.0231720, -0.0144392, 0.0134697
3: -0.0066962, -0.0025920, -0.0079170, -0.0025089, -0.0041873, 0.0053250
4: -0.0052404, 0.0026327, -0.0053450, 0.0036997, -0.0089401, 0.0079777
5: -0.0006416, 0.0094728, -0.0027893, 0.0094747, -0.0101163, 0.0122621
6: -0.0051602, 0.0008479, -0.0064013, 0.0009404, -0.0060332, 0.0072273
7: -0.0098460, -0.0010004, -0.0100276, 0.0007887, -0.0106347, 0.0090272
8: -0.0055216, -0.0001680, -0.0073880, -0.0001678, -0.0053538, 0.0072201
9: 0.9814906, 1.0011365, 0.9815520, 1.0036416, -0.0221510, 0.0195845

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140978
time: 2.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139752
time: 2.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0057013, 0.0005454, -0.0056893, 0.0006168, -0.0063180, 0.0062347
1: -0.0012537, 0.0129708, -0.0013733, 0.0129277, -0.0141814, 0.0143441
2: 0.0086280, 0.0223058, 0.0085391, 0.0222716, -0.0136435, 0.0137668
3: -0.0070424, -0.0025820, -0.0070381, -0.0025441, -0.0044984, 0.0044561
4: -0.0053024, 0.0028850, -0.0053017, 0.0029966, -0.0082991, 0.0081867
5: -0.0011765, 0.0095224, -0.0013302, 0.0094874, -0.0106639, 0.0108526
6: -0.0054789, 0.0008839, -0.0055349, 0.0009007, -0.0062742, 0.0063306
7: -0.0099156, -0.0005304, -0.0099511, -0.0005141, -0.0094015, 0.0094208
8: -0.0060359, -0.0001596, -0.0060873, -0.0001754, -0.0058605, 0.0059276
9: 0.9813670, 1.0017171, 0.9814653, 1.0020390, -0.0206720, 0.0202518

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0140053
time: 2.26 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0138693
time: 2.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057013, 0.0005454, -0.0059875, 0.0011270, -0.0068283, 0.0065329
1: -0.0012537, 0.0129708, -0.0017412, 0.0132091, -0.0144628, 0.0147120
2: 0.0086280, 0.0223058, 0.0084041, 0.0232025, -0.0145745, 0.0139017
3: -0.0070424, -0.0025820, -0.0079351, -0.0025096, -0.0045329, 0.0053531
4: -0.0053024, 0.0028850, -0.0053540, 0.0037008, -0.0090032, 0.0082389
5: -0.0011765, 0.0095224, -0.0028073, 0.0094838, -0.0106603, 0.0123297
6: -0.0054789, 0.0008839, -0.0064117, 0.0009453, -0.0063782, 0.0072830
7: -0.0099156, -0.0005304, -0.0100399, 0.0008016, -0.0107172, 0.0095095
8: -0.0060359, -0.0001596, -0.0074170, -0.0001686, -0.0058674, 0.0072574
9: 0.9813670, 1.0017171, 0.9815348, 1.0036515, -0.0222845, 0.0201823

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0140978
time: 1.94 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0139752
time: 2.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0059015, 0.0007985, -0.0056382, 0.0005820, -0.0064836, 0.0064367
1: -0.0014373, 0.0131646, -0.0012104, 0.0128840, -0.0143212, 0.0143750
2: 0.0085850, 0.0229456, 0.0086848, 0.0220835, -0.0134985, 0.0142608
3: -0.0076467, -0.0025532, -0.0068677, -0.0025491, -0.0050976, 0.0043146
4: -0.0052914, 0.0033338, -0.0052428, 0.0027033, -0.0079948, 0.0085766
5: -0.0021465, 0.0094724, -0.0008587, 0.0094211, -0.0115676, 0.0103311
6: -0.0060759, 0.0008946, -0.0052678, 0.0008684, -0.0068963, 0.0061100
7: -0.0099487, 0.0003244, -0.0099057, -0.0008228, -0.0091259, 0.0102300
8: -0.0069304, -0.0001588, -0.0057946, -0.0001862, -0.0067442, 0.0056358
9: 0.9814926, 1.0027567, 0.9815757, 1.0013775, -0.0198849, 0.0211810

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139196, upper bound: 0.0138580
time: 1.87 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139196, upper bound: 0.0138580
time: 2.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0059015, 0.0007985, -0.0059344, 0.0009223, -0.0068239, 0.0067329
1: -0.0014373, 0.0131646, -0.0015557, 0.0131546, -0.0145919, 0.0147203
2: 0.0085850, 0.0229456, 0.0085452, 0.0230116, -0.0144266, 0.0144004
3: -0.0076467, -0.0025532, -0.0077575, -0.0025124, -0.0051342, 0.0052043
4: -0.0052914, 0.0033338, -0.0052913, 0.0033956, -0.0086870, 0.0086252
5: -0.0021465, 0.0094724, -0.0023041, 0.0094154, -0.0115618, 0.0117765
6: -0.0060759, 0.0008946, -0.0061533, 0.0009197, -0.0067801, 0.0068436
7: -0.0099487, 0.0003244, -0.0100057, 0.0004631, -0.0104118, 0.0103300
8: -0.0069304, -0.0001588, -0.0071104, -0.0001789, -0.0067515, 0.0069516
9: 0.9814926, 1.0027567, 0.9816216, 1.0029659, -0.0214733, 0.0211352

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139196, upper bound: 0.0138580
time: 2.04 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139196, upper bound: 0.0138580
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0059935, 0.0010659, -0.0056446, 0.0005808, -0.0065743, 0.0067105
1: -0.0016215, 0.0132721, -0.0012134, 0.0128983, -0.0145199, 0.0144854
2: 0.0084838, 0.0232649, 0.0086871, 0.0221116, -0.0136278, 0.0145777
3: -0.0079236, -0.0025463, -0.0068848, -0.0025488, -0.0053748, 0.0043385
4: -0.0053695, 0.0035978, -0.0052448, 0.0027042, -0.0080737, 0.0088426
5: -0.0026749, 0.0095310, -0.0008734, 0.0094214, -0.0120963, 0.0104043
6: -0.0063694, 0.0009427, -0.0052775, 0.0008707, -0.0071882, 0.0061584
7: -0.0100323, 0.0008058, -0.0099087, -0.0008138, -0.0092184, 0.0107145
8: -0.0073398, -0.0001514, -0.0058220, -0.0001868, -0.0071530, 0.0056705
9: 0.9813914, 1.0033524, 0.9815599, 1.0013860, -0.0199946, 0.0217925

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139893, upper bound: 0.0138580
time: 2.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139893, upper bound: 0.0138580
time: 2.37 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0059935, 0.0010659, -0.0059411, 0.0009383, -0.0069318, 0.0070070
1: -0.0016215, 0.0132721, -0.0015594, 0.0131688, -0.0147904, 0.0148315
2: 0.0084838, 0.0232649, 0.0085488, 0.0230421, -0.0145583, 0.0147161
3: -0.0079236, -0.0025463, -0.0077754, -0.0025131, -0.0054105, 0.0052292
4: -0.0053695, 0.0035978, -0.0052996, 0.0033964, -0.0087659, 0.0088974
5: -0.0026749, 0.0095310, -0.0023217, 0.0094213, -0.0120963, 0.0118527
6: -0.0063694, 0.0009427, -0.0061638, 0.0009252, -0.0070765, 0.0068999
7: -0.0100323, 0.0008058, -0.0100139, 0.0004748, -0.0105071, 0.0108197
8: -0.0073398, -0.0001514, -0.0071393, -0.0001798, -0.0071599, 0.0069878
9: 0.9813914, 1.0033524, 0.9816075, 1.0029752, -0.0215838, 0.0217449

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139893, upper bound: 0.0138580
time: 1.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139893, upper bound: 0.0138580
time: 2.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0058954, 0.0007930, -0.0056830, 0.0006155, -0.0065109, 0.0064759
1: -0.0014341, 0.0131330, -0.0013703, 0.0129133, -0.0143474, 0.0145033
2: 0.0085897, 0.0229065, 0.0085376, 0.0222418, -0.0136521, 0.0143689
3: -0.0076393, -0.0025534, -0.0070207, -0.0025443, -0.0050951, 0.0044672
4: -0.0052853, 0.0033302, -0.0052961, 0.0029950, -0.0082804, 0.0086262
5: -0.0021374, 0.0094642, -0.0013141, 0.0094821, -0.0116195, 0.0107783
6: -0.0060710, 0.0008872, -0.0055250, 0.0008987, -0.0069174, 0.0063839
7: -0.0099325, 0.0003158, -0.0099434, -0.0005247, -0.0094079, 0.0102592
8: -0.0069207, -0.0001605, -0.0060595, -0.0001748, -0.0067459, 0.0058990
9: 0.9815274, 1.0027462, 0.9814771, 1.0020293, -0.0205019, 0.0212691

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0139904
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0138580
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0058954, 0.0007930, -0.0059808, 0.0011084, -0.0070038, 0.0067738
1: -0.0014341, 0.0131330, -0.0017378, 0.0131949, -0.0146290, 0.0148708
2: 0.0085897, 0.0229065, 0.0084014, 0.0231720, -0.0145822, 0.0145051
3: -0.0076393, -0.0025534, -0.0079170, -0.0025089, -0.0051304, 0.0053636
4: -0.0052853, 0.0033302, -0.0053450, 0.0036997, -0.0089851, 0.0086751
5: -0.0021374, 0.0094642, -0.0027893, 0.0094747, -0.0116121, 0.0122535
6: -0.0060710, 0.0008872, -0.0064013, 0.0009404, -0.0067963, 0.0071230
7: -0.0099325, 0.0003158, -0.0100276, 0.0007887, -0.0107212, 0.0103434
8: -0.0069207, -0.0001605, -0.0073880, -0.0001678, -0.0067529, 0.0072276
9: 0.9815274, 1.0027462, 0.9815520, 1.0036416, -0.0221142, 0.0211942

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0139904
time: 2.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0138580
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0059868, 0.0010568, -0.0056893, 0.0006168, -0.0066035, 0.0067462
1: -0.0016174, 0.0132374, -0.0013733, 0.0129277, -0.0145451, 0.0146107
2: 0.0084881, 0.0232218, 0.0085391, 0.0222716, -0.0137835, 0.0146827
3: -0.0079145, -0.0025465, -0.0070381, -0.0025441, -0.0053705, 0.0044915
4: -0.0053641, 0.0035944, -0.0053017, 0.0029966, -0.0083607, 0.0088961
5: -0.0026647, 0.0095248, -0.0013302, 0.0094874, -0.0121521, 0.0108550
6: -0.0063640, 0.0009356, -0.0055349, 0.0009007, -0.0072089, 0.0064357
7: -0.0100208, 0.0007964, -0.0099511, -0.0005141, -0.0095067, 0.0107475
8: -0.0073276, -0.0001528, -0.0060873, -0.0001754, -0.0071522, 0.0059344
9: 0.9814240, 1.0033411, 0.9814653, 1.0020390, -0.0206149, 0.0218758

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0139904
time: 2.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0138580
time: 2.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0059868, 0.0010568, -0.0059875, 0.0011270, -0.0071137, 0.0070444
1: -0.0016174, 0.0132374, -0.0017412, 0.0132091, -0.0148265, 0.0149786
2: 0.0084881, 0.0232218, 0.0084041, 0.0232025, -0.0147144, 0.0148177
3: -0.0079145, -0.0025465, -0.0079351, -0.0025096, -0.0054050, 0.0053885
4: -0.0053641, 0.0035944, -0.0053540, 0.0037008, -0.0090649, 0.0089484
5: -0.0026647, 0.0095248, -0.0028073, 0.0094838, -0.0121485, 0.0123321
6: -0.0063640, 0.0009356, -0.0064117, 0.0009453, -0.0070905, 0.0071801
7: -0.0100208, 0.0007964, -0.0100399, 0.0008016, -0.0108224, 0.0108362
8: -0.0073276, -0.0001528, -0.0074170, -0.0001686, -0.0071590, 0.0072642
9: 0.9814240, 1.0033411, 0.9815348, 1.0036515, -0.0222275, 0.0218062

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0139904
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0138580
time: 1.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.08 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138248, upper bound: 0.0138872
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138750, upper bound: 0.0138872
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138248, upper bound: 0.0140011
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138750, upper bound: 0.0140011
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0138693
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0138693
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139281, upper bound: 0.0139752
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139904, upper bound: 0.0139752
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140053
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0138693
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140978
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139752
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0140053
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0138693
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0140978
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0138580, upper bound: 0.0139752
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139196, upper bound: 0.0138580
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139196, upper bound: 0.0138580
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139196, upper bound: 0.0138580
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139196, upper bound: 0.0138580
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139893, upper bound: 0.0138580
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139893, upper bound: 0.0138580
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139893, upper bound: 0.0138580
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139893, upper bound: 0.0138580
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0139904
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0138580
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0139904
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0138580
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0139904
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0138580
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0139904
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.08
Output dim: 9, lower bound: -0.0139752, upper bound: 0.0138580

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055048, 0.0004810, -0.0056382, 0.0005820, -0.0060868, 0.0061192
1: -0.0010231, 0.0126230, -0.0012104, 0.0128840, -0.0139070, 0.0138334
2: 0.0088106, 0.0215174, 0.0086848, 0.0220835, -0.0132729, 0.0128326
3: -0.0065157, -0.0025965, -0.0068677, -0.0025491, -0.0039666, 0.0042713
4: -0.0050177, 0.0025498, -0.0052428, 0.0027033, -0.0077211, 0.0077926
5: -0.0004196, 0.0092337, -0.0008587, 0.0094211, -0.0098407, 0.0100924
6: -0.0050328, 0.0007224, -0.0052678, 0.0008684, -0.0057695, 0.0058606
7: -0.0096715, -0.0011866, -0.0099057, -0.0008228, -0.0088487, 0.0087191
8: -0.0052640, -0.0002102, -0.0057946, -0.0001862, -0.0050778, 0.0055844
9: 0.9819962, 1.0009048, 0.9815757, 1.0013775, -0.0193812, 0.0193291

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134338, upper bound: 0.0133968
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134338, upper bound: 0.0134852
time: 2.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0056225, 0.0005084, -0.0056446, 0.0005808, -0.0062033, 0.0061530
1: -0.0011817, 0.0127804, -0.0012134, 0.0128983, -0.0140800, 0.0139937
2: 0.0087071, 0.0219365, 0.0086871, 0.0221116, -0.0134045, 0.0132494
3: -0.0068535, -0.0025865, -0.0068848, -0.0025488, -0.0043047, 0.0042983
4: -0.0050734, 0.0028022, -0.0052448, 0.0027042, -0.0077776, 0.0080470
5: -0.0009428, 0.0092737, -0.0008734, 0.0094214, -0.0103641, 0.0101471
6: -0.0053476, 0.0007586, -0.0052775, 0.0008707, -0.0061066, 0.0059123
7: -0.0097281, -0.0007302, -0.0099087, -0.0008138, -0.0089142, 0.0091785
8: -0.0057627, -0.0002027, -0.0058220, -0.0001868, -0.0055759, 0.0056193
9: 0.9818777, 1.0014769, 0.9815599, 1.0013860, -0.0195083, 0.0199170

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134852, upper bound: 0.0133968
time: 2.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134852, upper bound: 0.0134852
time: 1.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

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

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133774, upper bound: 0.0132328
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135829, upper bound: 0.0137576
time: 2.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

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

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134272, upper bound: 0.0132328
time: 1.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136373, upper bound: 0.0137576
time: 2.00 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055479, 0.0005018, -0.0056382, 0.0005820, -0.0061299, 0.0061400
1: -0.0011752, 0.0126435, -0.0012104, 0.0128840, -0.0140592, 0.0138540
2: 0.0086616, 0.0216721, 0.0086848, 0.0220835, -0.0134219, 0.0129873
3: -0.0066647, -0.0025921, -0.0068677, -0.0025491, -0.0041156, 0.0042757
4: -0.0050620, 0.0028451, -0.0052428, 0.0027033, -0.0077653, 0.0080879
5: -0.0008715, 0.0092885, -0.0008587, 0.0094211, -0.0102926, 0.0101472
6: -0.0052894, 0.0007509, -0.0052678, 0.0008684, -0.0060454, 0.0059151
7: -0.0096999, -0.0008926, -0.0099057, -0.0008228, -0.0088771, 0.0090131
8: -0.0055225, -0.0001994, -0.0057946, -0.0001862, -0.0053363, 0.0055952
9: 0.9819055, 1.0015521, 0.9815757, 1.0013775, -0.0194719, 0.0199764

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135541, upper bound: 0.0133848
time: 1.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135541, upper bound: 0.0134681
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0056659, 0.0005455, -0.0056446, 0.0005808, -0.0062467, 0.0061901
1: -0.0013467, 0.0127925, -0.0012134, 0.0128983, -0.0142450, 0.0140059
2: 0.0085493, 0.0220862, 0.0086871, 0.0221116, -0.0135623, 0.0133991
3: -0.0070035, -0.0025819, -0.0068848, -0.0025488, -0.0044547, 0.0043029
4: -0.0051275, 0.0031021, -0.0052448, 0.0027042, -0.0078316, 0.0083469
5: -0.0014043, 0.0093433, -0.0008734, 0.0094214, -0.0108257, 0.0102166
6: -0.0056009, 0.0007861, -0.0052775, 0.0008707, -0.0063850, 0.0059646
7: -0.0097721, -0.0004168, -0.0099087, -0.0008138, -0.0089583, 0.0094919
8: -0.0060244, -0.0001911, -0.0058220, -0.0001868, -0.0058376, 0.0056308
9: 0.9817917, 1.0021466, 0.9815599, 1.0013860, -0.0195944, 0.0205867

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136221, upper bound: 0.0133847
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136221, upper bound: 0.0134681
time: 2.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055479, 0.0005018, -0.0059344, 0.0009223, -0.0064703, 0.0064362
1: -0.0011752, 0.0126435, -0.0015557, 0.0131546, -0.0143298, 0.0141992
2: 0.0086616, 0.0216721, 0.0085452, 0.0230116, -0.0143499, 0.0131269
3: -0.0066647, -0.0025921, -0.0077575, -0.0025124, -0.0041523, 0.0051654
4: -0.0050620, 0.0028451, -0.0052913, 0.0033956, -0.0084576, 0.0081365
5: -0.0008715, 0.0092885, -0.0023041, 0.0094154, -0.0102869, 0.0115925
6: -0.0052894, 0.0007509, -0.0061533, 0.0009197, -0.0061584, 0.0068757
7: -0.0096999, -0.0008926, -0.0100057, 0.0004631, -0.0101630, 0.0091131
8: -0.0055225, -0.0001994, -0.0071104, -0.0001789, -0.0053436, 0.0069110
9: 0.9819055, 1.0015521, 0.9816216, 1.0029659, -0.0210604, 0.0199305

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134798, upper bound: 0.0132042
time: 2.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136881, upper bound: 0.0137317
time: 2.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0056659, 0.0005455, -0.0059411, 0.0009383, -0.0066042, 0.0064866
1: -0.0013467, 0.0127925, -0.0015594, 0.0131688, -0.0145155, 0.0143519
2: 0.0085493, 0.0220862, 0.0085488, 0.0230421, -0.0144928, 0.0135374
3: -0.0070035, -0.0025819, -0.0077754, -0.0025131, -0.0044904, 0.0051935
4: -0.0051275, 0.0031021, -0.0052996, 0.0033964, -0.0085238, 0.0084017
5: -0.0014043, 0.0093433, -0.0023217, 0.0094213, -0.0108257, 0.0116649
6: -0.0056009, 0.0007861, -0.0061638, 0.0009252, -0.0064968, 0.0069260
7: -0.0097721, -0.0004168, -0.0100139, 0.0004748, -0.0102469, 0.0095971
8: -0.0060244, -0.0001911, -0.0071393, -0.0001798, -0.0058446, 0.0069482
9: 0.9817917, 1.0021466, 0.9816075, 1.0029752, -0.0211836, 0.0205391

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135461, upper bound: 0.0132042
time: 2.38 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137575, upper bound: 0.0137317
time: 2.07 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0055048, 0.0004810, -0.0056830, 0.0006155, -0.0061203, 0.0061640
1: -0.0010231, 0.0126230, -0.0013703, 0.0129133, -0.0139364, 0.0139932
2: 0.0088106, 0.0215174, 0.0085376, 0.0222418, -0.0134312, 0.0129799
3: -0.0065157, -0.0025965, -0.0070207, -0.0025443, -0.0039714, 0.0044242
4: -0.0050177, 0.0025498, -0.0052961, 0.0029950, -0.0080128, 0.0078459
5: -0.0004196, 0.0092337, -0.0013141, 0.0094821, -0.0099017, 0.0105478
6: -0.0050328, 0.0007224, -0.0055250, 0.0008987, -0.0058138, 0.0061385
7: -0.0096715, -0.0011866, -0.0099434, -0.0005247, -0.0091468, 0.0087568
8: -0.0052640, -0.0002102, -0.0060595, -0.0001748, -0.0050891, 0.0058493
9: 0.9819962, 1.0009048, 0.9814771, 1.0020293, -0.0200331, 0.0194277

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134244, upper bound: 0.0135090
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134244, upper bound: 0.0136221
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0055479, 0.0005018, -0.0056830, 0.0006155, -0.0061634, 0.0061848
1: -0.0011752, 0.0126435, -0.0013703, 0.0129133, -0.0140885, 0.0140138
2: 0.0086616, 0.0216721, 0.0085376, 0.0222418, -0.0135802, 0.0131345
3: -0.0066647, -0.0025921, -0.0070207, -0.0025443, -0.0041204, 0.0044286
4: -0.0050620, 0.0028451, -0.0052961, 0.0029950, -0.0080570, 0.0081412
5: -0.0008715, 0.0092885, -0.0013141, 0.0094821, -0.0103536, 0.0106025
6: -0.0052894, 0.0007509, -0.0055250, 0.0008987, -0.0060323, 0.0061275
7: -0.0096999, -0.0008926, -0.0099434, -0.0005247, -0.0091753, 0.0090508
8: -0.0055225, -0.0001994, -0.0060595, -0.0001748, -0.0053476, 0.0058601
9: 0.9819055, 1.0015521, 0.9814771, 1.0020293, -0.0201238, 0.0200750

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134244, upper bound: 0.0133848
time: 2.50 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134244, upper bound: 0.0134688
time: 2.18 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0055048, 0.0004810, -0.0059808, 0.0011084, -0.0066132, 0.0064618
1: -0.0010231, 0.0126230, -0.0017378, 0.0131949, -0.0142180, 0.0143608
2: 0.0088106, 0.0215174, 0.0084014, 0.0231720, -0.0143613, 0.0131161
3: -0.0065157, -0.0025965, -0.0079170, -0.0025089, -0.0040067, 0.0053205
4: -0.0050177, 0.0025498, -0.0053450, 0.0036997, -0.0087175, 0.0078948
5: -0.0004196, 0.0092337, -0.0027893, 0.0094747, -0.0098943, 0.0120230
6: -0.0050328, 0.0007224, -0.0064013, 0.0009404, -0.0059145, 0.0070907
7: -0.0096715, -0.0011866, -0.0100276, 0.0007887, -0.0104602, 0.0088410
8: -0.0052640, -0.0002102, -0.0073880, -0.0001678, -0.0050961, 0.0071778
9: 0.9819962, 1.0009048, 0.9815520, 1.0036416, -0.0216454, 0.0193528

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140304
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140978
time: 2.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0055479, 0.0005018, -0.0059808, 0.0011084, -0.0066563, 0.0064826
1: -0.0011752, 0.0126435, -0.0017378, 0.0131949, -0.0143701, 0.0143813
2: 0.0086616, 0.0216721, 0.0084014, 0.0231720, -0.0145103, 0.0132707
3: -0.0066647, -0.0025921, -0.0079170, -0.0025089, -0.0041558, 0.0053249
4: -0.0050620, 0.0028451, -0.0053450, 0.0036997, -0.0087617, 0.0081901
5: -0.0008715, 0.0092885, -0.0027893, 0.0094747, -0.0103462, 0.0120778
6: -0.0052894, 0.0007509, -0.0064013, 0.0009404, -0.0061352, 0.0070836
7: -0.0096999, -0.0008926, -0.0100276, 0.0007887, -0.0104886, 0.0091350
8: -0.0055225, -0.0001994, -0.0073880, -0.0001678, -0.0053547, 0.0071886
9: 0.9819055, 1.0015521, 0.9815520, 1.0036416, -0.0217361, 0.0200001

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139202
time: 4.38 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139752
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0056225, 0.0005084, -0.0056893, 0.0006168, -0.0062393, 0.0061977
1: -0.0011817, 0.0127804, -0.0013733, 0.0129277, -0.0141094, 0.0141537
2: 0.0087071, 0.0219365, 0.0085391, 0.0222716, -0.0135645, 0.0133975
3: -0.0068535, -0.0025865, -0.0070381, -0.0025441, -0.0043094, 0.0044515
4: -0.0050734, 0.0028022, -0.0053017, 0.0029966, -0.0080700, 0.0081039
5: -0.0009428, 0.0092737, -0.0013302, 0.0094874, -0.0104301, 0.0106039
6: -0.0053476, 0.0007586, -0.0055349, 0.0009007, -0.0061492, 0.0061906
7: -0.0097281, -0.0007302, -0.0099511, -0.0005141, -0.0092140, 0.0092209
8: -0.0057627, -0.0002027, -0.0060873, -0.0001754, -0.0055873, 0.0058846
9: 0.9818777, 1.0014769, 0.9814653, 1.0020390, -0.0201612, 0.0200116

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134681, upper bound: 0.0135090
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134681, upper bound: 0.0136221
time: 3.10 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0056659, 0.0005455, -0.0056893, 0.0006168, -0.0062826, 0.0062349
1: -0.0013467, 0.0127925, -0.0013733, 0.0129277, -0.0142744, 0.0141658
2: 0.0085493, 0.0220862, 0.0085391, 0.0222716, -0.0137222, 0.0135471
3: -0.0070035, -0.0025819, -0.0070381, -0.0025441, -0.0044594, 0.0044561
4: -0.0051275, 0.0031021, -0.0053017, 0.0029966, -0.0081241, 0.0084038
5: -0.0014043, 0.0093433, -0.0013302, 0.0094874, -0.0108917, 0.0106734
6: -0.0056009, 0.0007861, -0.0055349, 0.0009007, -0.0063747, 0.0061831
7: -0.0097721, -0.0004168, -0.0099511, -0.0005141, -0.0092580, 0.0095343
8: -0.0060244, -0.0001911, -0.0060873, -0.0001754, -0.0058490, 0.0058961
9: 0.9817917, 1.0021466, 0.9814653, 1.0020390, -0.0202473, 0.0206813

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134681, upper bound: 0.0133848
time: 2.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134681, upper bound: 0.0134688
time: 2.38 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0056225, 0.0005084, -0.0059875, 0.0011270, -0.0067495, 0.0064959
1: -0.0011817, 0.0127804, -0.0017412, 0.0132091, -0.0143908, 0.0145216
2: 0.0087071, 0.0219365, 0.0084041, 0.0232025, -0.0144954, 0.0135325
3: -0.0068535, -0.0025865, -0.0079351, -0.0025096, -0.0043439, 0.0053486
4: -0.0050734, 0.0028022, -0.0053540, 0.0037008, -0.0087742, 0.0081562
5: -0.0009428, 0.0092737, -0.0028073, 0.0094838, -0.0104265, 0.0120810
6: -0.0053476, 0.0007586, -0.0064117, 0.0009453, -0.0062534, 0.0071430
7: -0.0097281, -0.0007302, -0.0100399, 0.0008016, -0.0105297, 0.0093097
8: -0.0057627, -0.0002027, -0.0074170, -0.0001686, -0.0055941, 0.0072143
9: 0.9818777, 1.0014769, 0.9815348, 1.0036515, -0.0217738, 0.0199420

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134051, upper bound: 0.0133127
time: 2.49 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136182, upper bound: 0.0138702
time: 2.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0056659, 0.0005455, -0.0059875, 0.0011270, -0.0067928, 0.0065330
1: -0.0013467, 0.0127925, -0.0017412, 0.0132091, -0.0145558, 0.0145338
2: 0.0085493, 0.0220862, 0.0084041, 0.0232025, -0.0146532, 0.0136821
3: -0.0070035, -0.0025819, -0.0079351, -0.0025096, -0.0044939, 0.0053532
4: -0.0051275, 0.0031021, -0.0053540, 0.0037008, -0.0088282, 0.0084561
5: -0.0014043, 0.0093433, -0.0028073, 0.0094838, -0.0108881, 0.0121506
6: -0.0056009, 0.0007861, -0.0064117, 0.0009453, -0.0064787, 0.0071396
7: -0.0097721, -0.0004168, -0.0100399, 0.0008016, -0.0105737, 0.0096231
8: -0.0060244, -0.0001911, -0.0074170, -0.0001686, -0.0058559, 0.0072259
9: 0.9817917, 1.0021466, 0.9815348, 1.0036515, -0.0218598, 0.0206118

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0134051, upper bound: 0.0132042
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136182, upper bound: 0.0137317
time: 1.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0058241, 0.0006755, -0.0056382, 0.0005820, -0.0064062, 0.0063137
1: -0.0013586, 0.0129461, -0.0012104, 0.0128840, -0.0142426, 0.0141566
2: 0.0086668, 0.0225649, 0.0086848, 0.0220835, -0.0134167, 0.0138801
3: -0.0074656, -0.0025583, -0.0068677, -0.0025491, -0.0049165, 0.0043094
4: -0.0050568, 0.0032515, -0.0052428, 0.0027033, -0.0077602, 0.0084943
5: -0.0019103, 0.0092243, -0.0008587, 0.0094211, -0.0113314, 0.0100830
6: -0.0059464, 0.0007666, -0.0052678, 0.0008684, -0.0067614, 0.0059795
7: -0.0097656, 0.0001241, -0.0099057, -0.0008228, -0.0089428, 0.0100298
8: -0.0066687, -0.0002028, -0.0057946, -0.0001862, -0.0064825, 0.0055918
9: 0.9820247, 1.0025110, 0.9815757, 1.0013775, -0.0193527, 0.0209354

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135037, upper bound: 0.0133607
time: 2.05 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135037, upper bound: 0.0134427
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0058680, 0.0008071, -0.0056382, 0.0005820, -0.0064501, 0.0064453
1: -0.0015309, 0.0129769, -0.0012104, 0.0128840, -0.0144148, 0.0141874
2: 0.0085206, 0.0227174, 0.0086848, 0.0220835, -0.0135630, 0.0140326
3: -0.0076197, -0.0025551, -0.0068677, -0.0025491, -0.0050707, 0.0043127
4: -0.0051035, 0.0035581, -0.0052428, 0.0027033, -0.0078069, 0.0088009
5: -0.0023940, 0.0092746, -0.0008587, 0.0094211, -0.0118151, 0.0101333
6: -0.0061961, 0.0007863, -0.0052678, 0.0008684, -0.0070311, 0.0060150
7: -0.0097768, 0.0004508, -0.0099057, -0.0008228, -0.0089540, 0.0103565
8: -0.0069383, -0.0001921, -0.0057946, -0.0001862, -0.0067521, 0.0056025
9: 0.9819684, 1.0031818, 0.9815757, 1.0013775, -0.0194091, 0.0216061

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135037, upper bound: 0.0133607
time: 2.14 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135037, upper bound: 0.0134427
time: 2.14 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134653, upper bound: 0.0131037
time: 3.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136769, upper bound: 0.0136182
time: 2.58 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0058680, 0.0008071, -0.0059344, 0.0009223, -0.0067904, 0.0067415
1: -0.0015309, 0.0129769, -0.0015557, 0.0131546, -0.0146854, 0.0145326
2: 0.0085206, 0.0227174, 0.0085452, 0.0230116, -0.0144910, 0.0141722
3: -0.0076197, -0.0025551, -0.0077575, -0.0025124, -0.0051073, 0.0052024
4: -0.0051035, 0.0035581, -0.0052913, 0.0033956, -0.0084991, 0.0088495
5: -0.0023940, 0.0092746, -0.0023041, 0.0094154, -0.0118094, 0.0115787
6: -0.0061961, 0.0007863, -0.0061533, 0.0009197, -0.0069236, 0.0067602
7: -0.0097768, 0.0004508, -0.0100057, 0.0004631, -0.0102399, 0.0104565
8: -0.0069383, -0.0001921, -0.0071104, -0.0001789, -0.0067594, 0.0069182
9: 0.9819684, 1.0031818, 0.9816216, 1.0029659, -0.0209975, 0.0215603

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134653, upper bound: 0.0131037
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136769, upper bound: 0.0136182
time: 2.39 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0059139, 0.0008819, -0.0056446, 0.0005808, -0.0064947, 0.0065265
1: -0.0015332, 0.0130467, -0.0012134, 0.0128983, -0.0144315, 0.0142600
2: 0.0085650, 0.0228744, 0.0086871, 0.0221116, -0.0135466, 0.0141872
3: -0.0077371, -0.0025513, -0.0068848, -0.0025488, -0.0051884, 0.0043335
4: -0.0051312, 0.0035171, -0.0052448, 0.0027042, -0.0078354, 0.0087619
5: -0.0024359, 0.0092818, -0.0008734, 0.0094214, -0.0118573, 0.0101552
6: -0.0062402, 0.0008181, -0.0052775, 0.0008707, -0.0070527, 0.0060257
7: -0.0098414, 0.0006017, -0.0099087, -0.0008138, -0.0090275, 0.0105104
8: -0.0070695, -0.0001960, -0.0058220, -0.0001868, -0.0068827, 0.0056260
9: 0.9819222, 1.0031011, 0.9815599, 1.0013860, -0.0194638, 0.0215412

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135622, upper bound: 0.0133607
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135622, upper bound: 0.0134427
time: 1.99 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0059585, 0.0010684, -0.0056446, 0.0005808, -0.0065393, 0.0067130
1: -0.0017161, 0.0130735, -0.0012134, 0.0128983, -0.0146144, 0.0142868
2: 0.0084131, 0.0230229, 0.0086871, 0.0221116, -0.0136985, 0.0143358
3: -0.0078928, -0.0025482, -0.0068848, -0.0025488, -0.0053441, 0.0043366
4: -0.0051848, 0.0038250, -0.0052448, 0.0027042, -0.0078889, 0.0090697
5: -0.0029168, 0.0093426, -0.0008734, 0.0094214, -0.0123381, 0.0102159
6: -0.0064998, 0.0008341, -0.0052775, 0.0008707, -0.0073276, 0.0060672
7: -0.0098685, 0.0009475, -0.0099087, -0.0008138, -0.0090547, 0.0108563
8: -0.0073397, -0.0001843, -0.0058220, -0.0001868, -0.0071529, 0.0056377
9: 0.9818552, 1.0038422, 0.9815599, 1.0013860, -0.0195309, 0.0222824

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135622, upper bound: 0.0133607
time: 3.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135622, upper bound: 0.0134427
time: 2.77 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0059139, 0.0008819, -0.0059411, 0.0009383, -0.0068522, 0.0068230
1: -0.0015332, 0.0130467, -0.0015594, 0.0131688, -0.0147020, 0.0146061
2: 0.0085650, 0.0228744, 0.0085488, 0.0230421, -0.0144771, 0.0143256
3: -0.0077371, -0.0025513, -0.0077754, -0.0025131, -0.0052241, 0.0052241
4: -0.0051312, 0.0035171, -0.0052996, 0.0033964, -0.0085276, 0.0088167
5: -0.0024359, 0.0092818, -0.0023217, 0.0094213, -0.0118573, 0.0116035
6: -0.0062402, 0.0008181, -0.0061638, 0.0009252, -0.0069399, 0.0067633
7: -0.0098414, 0.0006017, -0.0100139, 0.0004748, -0.0103162, 0.0106156
8: -0.0070695, -0.0001960, -0.0071393, -0.0001798, -0.0068897, 0.0069433
9: 0.9819222, 1.0031011, 0.9816075, 1.0029752, -0.0210530, 0.0214936

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135317, upper bound: 0.0131037
time: 2.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0136182
time: 2.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0059585, 0.0010684, -0.0059411, 0.0009383, -0.0068969, 0.0070095
1: -0.0017161, 0.0130735, -0.0015594, 0.0131688, -0.0148849, 0.0146329
2: 0.0084131, 0.0230229, 0.0085488, 0.0230421, -0.0146291, 0.0144741
3: -0.0078928, -0.0025482, -0.0077754, -0.0025131, -0.0053798, 0.0052272
4: -0.0051848, 0.0038250, -0.0052996, 0.0033964, -0.0085812, 0.0091246
5: -0.0029168, 0.0093426, -0.0023217, 0.0094213, -0.0123381, 0.0116642
6: -0.0064998, 0.0008341, -0.0061638, 0.0009252, -0.0072241, 0.0068162
7: -0.0098685, 0.0009475, -0.0100139, 0.0004748, -0.0103433, 0.0109615
8: -0.0073397, -0.0001843, -0.0071393, -0.0001798, -0.0071599, 0.0069550
9: 0.9818552, 1.0038422, 0.9816075, 1.0029752, -0.0211201, 0.0222347

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135317, upper bound: 0.0131037
time: 2.51 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0136182
time: 3.06 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0058241, 0.0006755, -0.0056830, 0.0006155, -0.0064396, 0.0063585
1: -0.0013586, 0.0129461, -0.0013703, 0.0129133, -0.0142719, 0.0143164
2: 0.0086668, 0.0225649, 0.0085376, 0.0222418, -0.0135750, 0.0140273
3: -0.0074656, -0.0025583, -0.0070207, -0.0025443, -0.0049213, 0.0044623
4: -0.0050568, 0.0032515, -0.0052961, 0.0029950, -0.0080518, 0.0085476
5: -0.0019103, 0.0092243, -0.0013141, 0.0094821, -0.0113924, 0.0105384
6: -0.0059464, 0.0007666, -0.0055250, 0.0008987, -0.0068049, 0.0062574
7: -0.0097656, 0.0001241, -0.0099434, -0.0005247, -0.0092409, 0.0100675
8: -0.0066687, -0.0002028, -0.0060595, -0.0001748, -0.0064939, 0.0058567
9: 0.9820247, 1.0025110, 0.9814771, 1.0020293, -0.0200046, 0.0210339

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135022, upper bound: 0.0134771
time: 1.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135022, upper bound: 0.0135830
time: 1.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0058680, 0.0008071, -0.0056830, 0.0006155, -0.0064835, 0.0064900
1: -0.0015309, 0.0129769, -0.0013703, 0.0129133, -0.0144442, 0.0143472
2: 0.0085206, 0.0227174, 0.0085376, 0.0222418, -0.0137213, 0.0141798
3: -0.0076197, -0.0025551, -0.0070207, -0.0025443, -0.0050755, 0.0044656
4: -0.0051035, 0.0035581, -0.0052961, 0.0029950, -0.0080985, 0.0088542
5: -0.0023940, 0.0092746, -0.0013141, 0.0094821, -0.0118761, 0.0105887
6: -0.0061961, 0.0007863, -0.0055250, 0.0008987, -0.0070200, 0.0062324
7: -0.0097768, 0.0004508, -0.0099434, -0.0005247, -0.0092521, 0.0103942
8: -0.0069383, -0.0001921, -0.0060595, -0.0001748, -0.0067635, 0.0058673
9: 0.9819684, 1.0031818, 0.9814771, 1.0020293, -0.0200609, 0.0217047

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135022, upper bound: 0.0133607
time: 2.24 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135022, upper bound: 0.0134427
time: 2.22 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0058241, 0.0006755, -0.0059808, 0.0011084, -0.0069326, 0.0066563
1: -0.0013586, 0.0129461, -0.0017378, 0.0131949, -0.0145535, 0.0146840
2: 0.0086668, 0.0225649, 0.0084014, 0.0231720, -0.0145051, 0.0141635
3: -0.0074656, -0.0025583, -0.0079170, -0.0025089, -0.0049567, 0.0053587
4: -0.0050568, 0.0032515, -0.0053450, 0.0036997, -0.0087565, 0.0085965
5: -0.0019103, 0.0092243, -0.0027893, 0.0094747, -0.0113851, 0.0120136
6: -0.0059464, 0.0007666, -0.0064013, 0.0009404, -0.0066852, 0.0069867
7: -0.0097656, 0.0001241, -0.0100276, 0.0007887, -0.0105542, 0.0101517
8: -0.0066687, -0.0002028, -0.0073880, -0.0001678, -0.0065009, 0.0071852
9: 0.9820247, 1.0025110, 0.9815520, 1.0036416, -0.0216169, 0.0209590

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0139406
time: 2.55 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0139904
time: 2.94 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0058680, 0.0008071, -0.0059808, 0.0011084, -0.0069765, 0.0067879
1: -0.0015309, 0.0129769, -0.0017378, 0.0131949, -0.0147258, 0.0147148
2: 0.0085206, 0.0227174, 0.0084014, 0.0231720, -0.0146514, 0.0143160
3: -0.0076197, -0.0025551, -0.0079170, -0.0025089, -0.0051108, 0.0053620
4: -0.0051035, 0.0035581, -0.0053450, 0.0036997, -0.0088032, 0.0089031
5: -0.0023940, 0.0092746, -0.0027893, 0.0094747, -0.0118687, 0.0120639
6: -0.0061961, 0.0007863, -0.0064013, 0.0009404, -0.0069053, 0.0069748
7: -0.0097768, 0.0004508, -0.0100276, 0.0007887, -0.0105655, 0.0104784
8: -0.0069383, -0.0001921, -0.0073880, -0.0001678, -0.0067705, 0.0071959
9: 0.9819684, 1.0031818, 0.9815520, 1.0036416, -0.0216732, 0.0216298

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 71

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0138075
time: 2.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0138580
time: 2.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0059139, 0.0008819, -0.0056893, 0.0006168, -0.0065306, 0.0065712
1: -0.0015332, 0.0130467, -0.0013733, 0.0129277, -0.0144609, 0.0144200
2: 0.0085650, 0.0228744, 0.0085391, 0.0222716, -0.0137065, 0.0143353
3: -0.0077371, -0.0025513, -0.0070381, -0.0025441, -0.0051931, 0.0044867
4: -0.0051312, 0.0035171, -0.0053017, 0.0029966, -0.0081279, 0.0088188
5: -0.0024359, 0.0092818, -0.0013302, 0.0094874, -0.0119233, 0.0106120
6: -0.0062402, 0.0008181, -0.0055349, 0.0009007, -0.0070938, 0.0063040
7: -0.0098414, 0.0006017, -0.0099511, -0.0005141, -0.0093272, 0.0105528
8: -0.0070695, -0.0001960, -0.0060873, -0.0001754, -0.0068941, 0.0058913
9: 0.9819222, 1.0031011, 0.9814653, 1.0020390, -0.0201167, 0.0216358

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135498, upper bound: 0.0134771
time: 2.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135498, upper bound: 0.0135830
time: 2.43 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0059585, 0.0010684, -0.0056893, 0.0006168, -0.0065753, 0.0067578
1: -0.0017161, 0.0130735, -0.0013733, 0.0129277, -0.0146438, 0.0144468
2: 0.0084131, 0.0230229, 0.0085391, 0.0222716, -0.0138585, 0.0144838
3: -0.0078928, -0.0025482, -0.0070381, -0.0025441, -0.0053488, 0.0044898
4: -0.0051848, 0.0038250, -0.0053017, 0.0029966, -0.0081814, 0.0091267
5: -0.0029168, 0.0093426, -0.0013302, 0.0094874, -0.0124041, 0.0106728
6: -0.0064998, 0.0008341, -0.0055349, 0.0009007, -0.0073181, 0.0062861
7: -0.0098685, 0.0009475, -0.0099511, -0.0005141, -0.0093544, 0.0108987
8: -0.0073397, -0.0001843, -0.0060873, -0.0001754, -0.0071643, 0.0059030
9: 0.9818552, 1.0038422, 0.9814653, 1.0020390, -0.0201838, 0.0223770

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135498, upper bound: 0.0133607
time: 2.15 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135498, upper bound: 0.0134427
time: 2.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0059139, 0.0008819, -0.0059875, 0.0011270, -0.0070408, 0.0068694
1: -0.0015332, 0.0130467, -0.0017412, 0.0132091, -0.0147423, 0.0147879
2: 0.0085650, 0.0228744, 0.0084041, 0.0232025, -0.0146375, 0.0144703
3: -0.0077371, -0.0025513, -0.0079351, -0.0025096, -0.0052276, 0.0053838
4: -0.0051312, 0.0035171, -0.0053540, 0.0037008, -0.0088320, 0.0088711
5: -0.0024359, 0.0092818, -0.0028073, 0.0094838, -0.0119197, 0.0120892
6: -0.0062402, 0.0008181, -0.0064117, 0.0009453, -0.0069780, 0.0070410
7: -0.0098414, 0.0006017, -0.0100399, 0.0008016, -0.0106430, 0.0106416
8: -0.0070695, -0.0001960, -0.0074170, -0.0001686, -0.0069010, 0.0072211
9: 0.9819222, 1.0031011, 0.9815348, 1.0036515, -0.0217293, 0.0215663

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135149, upper bound: 0.0132262
time: 2.10 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137317, upper bound: 0.0137575
time: 3.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0059585, 0.0010684, -0.0059875, 0.0011270, -0.0070855, 0.0070560
1: -0.0017161, 0.0130735, -0.0017412, 0.0132091, -0.0149252, 0.0148147
2: 0.0084131, 0.0230229, 0.0084041, 0.0232025, -0.0147895, 0.0146188
3: -0.0078928, -0.0025482, -0.0079351, -0.0025096, -0.0053833, 0.0053869
4: -0.0051848, 0.0038250, -0.0053540, 0.0037008, -0.0088856, 0.0091789
5: -0.0029168, 0.0093426, -0.0028073, 0.0094838, -0.0124005, 0.0121499
6: -0.0064998, 0.0008341, -0.0064117, 0.0009453, -0.0072066, 0.0070331
7: -0.0098685, 0.0009475, -0.0100399, 0.0008016, -0.0106701, 0.0109874
8: -0.0073397, -0.0001843, -0.0074170, -0.0001686, -0.0071711, 0.0072328
9: 0.9818552, 1.0038422, 0.9815348, 1.0036515, -0.0217963, 0.0223074

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135149, upper bound: 0.0131037
time: 2.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137317, upper bound: 0.0136182
time: 2.84 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.45 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134338, upper bound: 0.0133968
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134338, upper bound: 0.0134852
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134852, upper bound: 0.0133968
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134852, upper bound: 0.0134852
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0133774, upper bound: 0.0132328
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135829, upper bound: 0.0137576
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134272, upper bound: 0.0132328
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0136373, upper bound: 0.0137576
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135541, upper bound: 0.0133848
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135541, upper bound: 0.0134681
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0136221, upper bound: 0.0133847
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0136221, upper bound: 0.0134681
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134798, upper bound: 0.0132042
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0136881, upper bound: 0.0137317
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135461, upper bound: 0.0132042
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0137575, upper bound: 0.0137317
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134244, upper bound: 0.0135090
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134244, upper bound: 0.0136221
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134244, upper bound: 0.0133848
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134244, upper bound: 0.0134688
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140304
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0140978
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139202
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0138065, upper bound: 0.0139752
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134681, upper bound: 0.0135090
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134681, upper bound: 0.0136221
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134681, upper bound: 0.0133848
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134681, upper bound: 0.0134688
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134051, upper bound: 0.0133127
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0136182, upper bound: 0.0138702
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134051, upper bound: 0.0132042
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0136182, upper bound: 0.0137317
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135037, upper bound: 0.0133607
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135037, upper bound: 0.0134427
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135037, upper bound: 0.0133607
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135037, upper bound: 0.0134427
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134653, upper bound: 0.0131037
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0136769, upper bound: 0.0136182
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0134653, upper bound: 0.0131037
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0136769, upper bound: 0.0136182
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135622, upper bound: 0.0133607
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135622, upper bound: 0.0134427
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135622, upper bound: 0.0133607
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135622, upper bound: 0.0134427
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135317, upper bound: 0.0131037
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0136182
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135317, upper bound: 0.0131037
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0137469, upper bound: 0.0136182
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135022, upper bound: 0.0134771
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135022, upper bound: 0.0135830
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135022, upper bound: 0.0133607
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135022, upper bound: 0.0134427
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0139406
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0139904
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0138075
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0139167, upper bound: 0.0138580
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135498, upper bound: 0.0134771
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135498, upper bound: 0.0135830
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135498, upper bound: 0.0133607
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135498, upper bound: 0.0134427
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135149, upper bound: 0.0132262
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0137317, upper bound: 0.0137575
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0135149, upper bound: 0.0131037
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.45
Output dim: 9, lower bound: -0.0137317, upper bound: 0.0136182

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054652, 0.0004764, -0.0053555, 0.0005455, -0.0060108, 0.0058318
1: -0.0009938, 0.0125383, -0.0009866, 0.0122721, -0.0132659, 0.0135249
2: 0.0088165, 0.0213449, 0.0087295, 0.0208611, -0.0120446, 0.0126153
3: -0.0064154, -0.0025968, -0.0061500, -0.0025514, -0.0038641, 0.0035533
4: -0.0049981, 0.0025379, -0.0050989, 0.0026110, -0.0076091, 0.0076368
5: -0.0003342, 0.0092072, -0.0002302, 0.0092277, -0.0095620, 0.0094375
6: -0.0049825, 0.0007159, -0.0049021, 0.0008207, -0.0056325, 0.0054107
7: -0.0096546, -0.0012509, -0.0097836, -0.0012967, -0.0083579, 0.0085326
8: -0.0051166, -0.0002163, -0.0047367, -0.0002296, -0.0048869, 0.0045204
9: 0.9820836, 1.0008326, 0.9822048, 1.0008374, -0.0187538, 0.0186278

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126288, upper bound: 0.0129166
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131863, upper bound: 0.0131354
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053938, 0.0004689, -0.0053323, 0.0007189, -0.0061128, 0.0058011
1: -0.0009405, 0.0123889, -0.0014314, 0.0122523, -0.0131928, 0.0138204
2: 0.0088277, 0.0210433, 0.0083177, 0.0207894, -0.0119617, 0.0127256
3: -0.0062305, -0.0025973, -0.0060874, -0.0025114, -0.0037191, 0.0034901
4: -0.0049602, 0.0025145, -0.0050689, 0.0027566, -0.0077168, 0.0075834
5: -0.0001755, 0.0091577, -0.0003223, 0.0091862, -0.0093616, 0.0094801
6: -0.0048863, 0.0007034, -0.0048916, 0.0008155, -0.0055429, 0.0053731
7: -0.0096228, -0.0013708, -0.0097565, -0.0012052, -0.0084177, 0.0083857
8: -0.0048413, -0.0002250, -0.0046326, -0.0002139, -0.0046274, 0.0044076
9: 0.9822273, 1.0006964, 0.9822592, 1.0014162, -0.0191889, 0.0184373

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126288, upper bound: 0.0130228
time: 2.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131863, upper bound: 0.0132376
time: 2.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0055818, 0.0005002, -0.0053620, 0.0005437, -0.0061256, 0.0058622
1: -0.0011492, 0.0126953, -0.0009901, 0.0122866, -0.0134357, 0.0136854
2: 0.0087131, 0.0217622, 0.0087316, 0.0208896, -0.0121765, 0.0130307
3: -0.0067511, -0.0025868, -0.0061676, -0.0025510, -0.0042000, 0.0035808
4: -0.0050538, 0.0027898, -0.0051017, 0.0026120, -0.0076659, 0.0078915
5: -0.0008535, 0.0092472, -0.0002456, 0.0092278, -0.0100813, 0.0094928
6: -0.0052950, 0.0007520, -0.0049114, 0.0008233, -0.0059672, 0.0054626
7: -0.0097115, -0.0007990, -0.0097868, -0.0012874, -0.0084241, 0.0089878
8: -0.0056120, -0.0002088, -0.0047647, -0.0002303, -0.0053817, 0.0045560
9: 0.9819647, 1.0014023, 0.9821931, 1.0008472, -0.0188825, 0.0192091

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126589, upper bound: 0.0129166
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132376, upper bound: 0.0131354
time: 1.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0055144, 0.0004896, -0.0053390, 0.0007185, -0.0062329, 0.0058286
1: -0.0010930, 0.0125496, -0.0014350, 0.0122675, -0.0133604, 0.0139846
2: 0.0087244, 0.0214691, 0.0083188, 0.0208169, -0.0120925, 0.0131503
3: -0.0065769, -0.0025874, -0.0061051, -0.0025108, -0.0040661, 0.0035176
4: -0.0050146, 0.0027671, -0.0050705, 0.0027588, -0.0077735, 0.0078376
5: -0.0007008, 0.0091976, -0.0003383, 0.0091865, -0.0098872, 0.0095359
6: -0.0052021, 0.0007393, -0.0049013, 0.0008181, -0.0058844, 0.0054250
7: -0.0096783, -0.0009166, -0.0097609, -0.0011953, -0.0084830, 0.0088443
8: -0.0053500, -0.0002174, -0.0046611, -0.0002143, -0.0051357, 0.0044437
9: 0.9821079, 1.0012697, 0.9822491, 1.0014278, -0.0193198, 0.0190206

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126589, upper bound: 0.0130228
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132376, upper bound: 0.0132376
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130491, upper bound: 0.0133170
time: 2.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131599, upper bound: 0.0133170
time: 2.51 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0055016, 0.0004899, -0.0055845, 0.0005644, -0.0060660, 0.0060744
1: -0.0010846, 0.0125259, -0.0011896, 0.0123643, -0.0134489, 0.0137155
2: 0.0087277, 0.0214187, 0.0086324, 0.0214688, -0.0127411, 0.0127863
3: -0.0065505, -0.0025870, -0.0068958, -0.0025146, -0.0040359, 0.0043087
4: -0.0050164, 0.0027603, -0.0051185, 0.0032897, -0.0083061, 0.0078788
5: -0.0006789, 0.0091909, -0.0015500, 0.0091524, -0.0098313, 0.0107409
6: -0.0052060, 0.0007436, -0.0057473, 0.0008824, -0.0060126, 0.0064608
7: -0.0096731, -0.0009424, -0.0098450, -0.0001571, -0.0095160, 0.0089026
8: -0.0053314, -0.0002217, -0.0058639, -0.0002293, -0.0051021, 0.0056422
9: 0.9821731, 1.0012478, 0.9825439, 1.0022407, -0.0200676, 0.0187039

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128664, upper bound: 0.0126790
time: 2.12 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129719, upper bound: 0.0126790
time: 2.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

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

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131117, upper bound: 0.0133170
time: 2.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132092, upper bound: 0.0133170
time: 2.41 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055081, 0.0004954, -0.0053555, 0.0005455, -0.0060536, 0.0058508
1: -0.0011440, 0.0125560, -0.0009866, 0.0122721, -0.0134161, 0.0135426
2: 0.0086675, 0.0214989, 0.0087295, 0.0208611, -0.0121936, 0.0127694
3: -0.0065648, -0.0025924, -0.0061500, -0.0025514, -0.0040135, 0.0035576
4: -0.0050422, 0.0028330, -0.0050989, 0.0026110, -0.0076531, 0.0079319
5: -0.0007854, 0.0092617, -0.0002302, 0.0092277, -0.0100131, 0.0094919
6: -0.0052379, 0.0007443, -0.0049021, 0.0008207, -0.0059086, 0.0054651
7: -0.0096832, -0.0009570, -0.0097836, -0.0012967, -0.0083865, 0.0088266
8: -0.0053761, -0.0002057, -0.0047367, -0.0002296, -0.0051465, 0.0045310
9: 0.9819957, 1.0014795, 0.9822048, 1.0008374, -0.0188417, 0.0192747

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127201, upper bound: 0.0128982
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133131, upper bound: 0.0131246
time: 3.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054380, 0.0004861, -0.0053323, 0.0007189, -0.0061569, 0.0058184
1: -0.0010867, 0.0124205, -0.0014314, 0.0122523, -0.0133390, 0.0138519
2: 0.0086787, 0.0212084, 0.0083177, 0.0207894, -0.0121107, 0.0128907
3: -0.0063802, -0.0025929, -0.0060874, -0.0025114, -0.0038688, 0.0034944
4: -0.0050042, 0.0028094, -0.0050689, 0.0027566, -0.0077608, 0.0078782
5: -0.0006264, 0.0092147, -0.0003223, 0.0091862, -0.0098125, 0.0095370
6: -0.0051417, 0.0007322, -0.0048916, 0.0008155, -0.0058197, 0.0054276
7: -0.0096506, -0.0010770, -0.0097565, -0.0012052, -0.0084455, 0.0086795
8: -0.0050997, -0.0002132, -0.0046326, -0.0002139, -0.0048859, 0.0044193
9: 0.9821292, 1.0013417, 0.9822592, 1.0014162, -0.0192870, 0.0190825

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127201, upper bound: 0.0129994
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133131, upper bound: 0.0132208
time: 3.20 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056251, 0.0005268, -0.0053620, 0.0005437, -0.0061688, 0.0058888
1: -0.0013130, 0.0127040, -0.0009901, 0.0122866, -0.0135996, 0.0136941
2: 0.0085553, 0.0219109, 0.0087316, 0.0208896, -0.0123343, 0.0131794
3: -0.0069018, -0.0025822, -0.0061676, -0.0025510, -0.0043507, 0.0035854
4: -0.0051077, 0.0030899, -0.0051017, 0.0026120, -0.0077197, 0.0081916
5: -0.0013145, 0.0093160, -0.0002456, 0.0092278, -0.0105423, 0.0095616
6: -0.0055485, 0.0007797, -0.0049114, 0.0008233, -0.0062470, 0.0055147
7: -0.0097558, -0.0004877, -0.0097868, -0.0012874, -0.0084684, 0.0092991
8: -0.0058752, -0.0001975, -0.0047647, -0.0002303, -0.0056449, 0.0045673
9: 0.9818817, 1.0020707, 0.9821931, 1.0008472, -0.0189655, 0.0198776

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127642, upper bound: 0.0128982
time: 3.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0131246
time: 2.09 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0055579, 0.0005103, -0.0053390, 0.0007185, -0.0062764, 0.0058493
1: -0.0012539, 0.0125711, -0.0014350, 0.0122675, -0.0135214, 0.0140061
2: 0.0085665, 0.0216304, 0.0083188, 0.0208169, -0.0122504, 0.0133117
3: -0.0067256, -0.0025829, -0.0061051, -0.0025108, -0.0042148, 0.0035222
4: -0.0050689, 0.0030671, -0.0050705, 0.0027588, -0.0078277, 0.0081376
5: -0.0011561, 0.0092682, -0.0003383, 0.0091865, -0.0103425, 0.0096065
6: -0.0054517, 0.0007664, -0.0049013, 0.0008181, -0.0061597, 0.0054772
7: -0.0097219, -0.0006101, -0.0097609, -0.0011953, -0.0085265, 0.0091508
8: -0.0056079, -0.0002049, -0.0046611, -0.0002143, -0.0053936, 0.0044562
9: 0.9820170, 1.0019339, 0.9822491, 1.0014278, -0.0194108, 0.0196848

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127642, upper bound: 0.0129994
time: 3.05 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0132208
time: 2.37 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054305, 0.0004872, -0.0055776, 0.0005665, -0.0059970, 0.0060649
1: -0.0010830, 0.0123935, -0.0011855, 0.0123499, -0.0134329, 0.0135791
2: 0.0086813, 0.0211648, 0.0086291, 0.0214393, -0.0127581, 0.0125357
3: -0.0063695, -0.0025925, -0.0068773, -0.0025135, -0.0038560, 0.0042848
4: -0.0050042, 0.0028049, -0.0051090, 0.0032885, -0.0082927, 0.0079139
5: -0.0006183, 0.0092035, -0.0015325, 0.0091459, -0.0097642, 0.0107361
6: -0.0051518, 0.0007357, -0.0057366, 0.0008763, -0.0059595, 0.0064666
7: -0.0096434, -0.0010920, -0.0098376, -0.0001687, -0.0094747, 0.0087456
8: -0.0051044, -0.0002186, -0.0058347, -0.0002285, -0.0048759, 0.0056160
9: 0.9821993, 1.0013311, 0.9825593, 1.0022295, -0.0200301, 0.0187718

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128813, upper bound: 0.0126608
time: 2.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130403, upper bound: 0.0126608
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055229, 0.0004981, -0.0057894, 0.0006903, -0.0062132, 0.0062875
1: -0.0011549, 0.0125948, -0.0014144, 0.0128646, -0.0140194, 0.0140092
2: 0.0086665, 0.0215696, 0.0085745, 0.0224053, -0.0137388, 0.0129951
3: -0.0066012, -0.0025921, -0.0073870, -0.0025135, -0.0040877, 0.0047949
4: -0.0050484, 0.0028348, -0.0052152, 0.0033358, -0.0083842, 0.0080499
5: -0.0008152, 0.0092713, -0.0019593, 0.0093162, -0.0101314, 0.0112306
6: -0.0052585, 0.0007472, -0.0059693, 0.0009000, -0.0061042, 0.0066662
7: -0.0096872, -0.0009379, -0.0099326, 0.0001700, -0.0098572, 0.0089947
8: -0.0054307, -0.0002030, -0.0065710, -0.0001987, -0.0052320, 0.0063680
9: 0.9819598, 1.0015017, 0.9819455, 1.0026394, -0.0206797, 0.0195562

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131396, upper bound: 0.0133010
time: 2.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132808, upper bound: 0.0133010
time: 2.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0055454, 0.0005103, -0.0055845, 0.0005644, -0.0061098, 0.0060948
1: -0.0012467, 0.0125346, -0.0011896, 0.0123643, -0.0136109, 0.0137242
2: 0.0085695, 0.0215703, 0.0086324, 0.0214688, -0.0128994, 0.0129379
3: -0.0067034, -0.0025825, -0.0068958, -0.0025146, -0.0041888, 0.0043133
4: -0.0050699, 0.0030625, -0.0051185, 0.0032897, -0.0083596, 0.0081810
5: -0.0011388, 0.0092575, -0.0015500, 0.0091524, -0.0102911, 0.0108075
6: -0.0054603, 0.0007713, -0.0057473, 0.0008824, -0.0062946, 0.0065174
7: -0.0097165, -0.0006324, -0.0098450, -0.0001571, -0.0095594, 0.0092126
8: -0.0055969, -0.0002102, -0.0058639, -0.0002293, -0.0053676, 0.0056537
9: 0.9820941, 1.0019166, 0.9825439, 1.0022407, -0.0201465, 0.0193728

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129625, upper bound: 0.0126608
time: 2.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130910, upper bound: 0.0126608
time: 1.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056406, 0.0005324, -0.0057960, 0.0006964, -0.0063370, 0.0063284
1: -0.0013245, 0.0127423, -0.0014177, 0.0128788, -0.0142033, 0.0141600
2: 0.0085543, 0.0219806, 0.0085780, 0.0224353, -0.0138810, 0.0134025
3: -0.0069391, -0.0025820, -0.0074047, -0.0025142, -0.0044250, 0.0048227
4: -0.0051143, 0.0030917, -0.0052240, 0.0033367, -0.0084510, 0.0083158
5: -0.0013453, 0.0093266, -0.0019759, 0.0093225, -0.0106678, 0.0113025
6: -0.0055691, 0.0007824, -0.0059795, 0.0009054, -0.0064412, 0.0067160
7: -0.0097594, -0.0004664, -0.0099404, 0.0001812, -0.0099407, 0.0094739
8: -0.0059303, -0.0001947, -0.0065996, -0.0001996, -0.0057307, 0.0064049
9: 0.9818454, 1.0020938, 0.9819301, 1.0026491, -0.0208036, 0.0201637

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132218, upper bound: 0.0133010
time: 1.94 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0133010
time: 1.83 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054652, 0.0004764, -0.0053970, 0.0005621, -0.0060273, 0.0058734
1: -0.0009938, 0.0125383, -0.0011336, 0.0122875, -0.0132813, 0.0136719
2: 0.0088165, 0.0213449, 0.0085824, 0.0210072, -0.0121907, 0.0127625
3: -0.0064154, -0.0025968, -0.0062998, -0.0025466, -0.0038689, 0.0037030
4: -0.0049981, 0.0025379, -0.0051495, 0.0029021, -0.0079003, 0.0076873
5: -0.0003342, 0.0092072, -0.0006799, 0.0092838, -0.0096181, 0.0098871
6: -0.0049825, 0.0007159, -0.0051549, 0.0008514, -0.0056759, 0.0056882
7: -0.0096546, -0.0012509, -0.0098228, -0.0010018, -0.0086528, 0.0085719
8: -0.0051166, -0.0002163, -0.0049983, -0.0002205, -0.0048961, 0.0047820
9: 0.9820836, 1.0008326, 0.9821281, 1.0014832, -0.0193996, 0.0187045

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126155, upper bound: 0.0130154
time: 2.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131779, upper bound: 0.0132516
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053938, 0.0004689, -0.0053776, 0.0007387, -0.0061326, 0.0058464
1: -0.0009405, 0.0123889, -0.0015845, 0.0122965, -0.0132371, 0.0139735
2: 0.0088277, 0.0210433, 0.0081626, 0.0209606, -0.0121330, 0.0128807
3: -0.0062305, -0.0025973, -0.0062334, -0.0025063, -0.0037241, 0.0036361
4: -0.0049602, 0.0025145, -0.0051210, 0.0030484, -0.0080086, 0.0076355
5: -0.0001755, 0.0091577, -0.0007551, 0.0092542, -0.0094297, 0.0099128
6: -0.0048863, 0.0007034, -0.0051401, 0.0008464, -0.0055891, 0.0056462
7: -0.0096228, -0.0013708, -0.0097935, -0.0009206, -0.0087022, 0.0084226
8: -0.0048413, -0.0002250, -0.0048833, -0.0002009, -0.0046404, 0.0046583
9: 0.9822273, 1.0006964, 0.9821653, 1.0020704, -0.0198431, 0.0185311

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126155, upper bound: 0.0131467
time: 2.94 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131779, upper bound: 0.0133793
time: 1.43 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0055081, 0.0004954, -0.0053970, 0.0005621, -0.0060702, 0.0058924
1: -0.0011440, 0.0125560, -0.0011336, 0.0122875, -0.0134315, 0.0136896
2: 0.0086675, 0.0214989, 0.0085824, 0.0210072, -0.0123396, 0.0129165
3: -0.0065648, -0.0025924, -0.0062998, -0.0025466, -0.0040182, 0.0037074
4: -0.0050422, 0.0028330, -0.0051495, 0.0029021, -0.0079443, 0.0079825
5: -0.0007854, 0.0092617, -0.0006799, 0.0092838, -0.0100692, 0.0099416
6: -0.0052379, 0.0007443, -0.0051549, 0.0008514, -0.0058946, 0.0056755
7: -0.0096832, -0.0009570, -0.0098228, -0.0010018, -0.0086814, 0.0088658
8: -0.0053761, -0.0002057, -0.0049983, -0.0002205, -0.0051556, 0.0047926
9: 0.9819957, 1.0014795, 0.9821281, 1.0014832, -0.0194875, 0.0193514

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127201, upper bound: 0.0128982
time: 2.00 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133131, upper bound: 0.0131251
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0054380, 0.0004861, -0.0053776, 0.0007387, -0.0061767, 0.0058637
1: -0.0010867, 0.0124205, -0.0015845, 0.0122965, -0.0133833, 0.0140051
2: 0.0086787, 0.0212084, 0.0081626, 0.0209606, -0.0122819, 0.0130458
3: -0.0063802, -0.0025929, -0.0062334, -0.0025063, -0.0038738, 0.0036404
4: -0.0050042, 0.0028094, -0.0051210, 0.0030484, -0.0080526, 0.0079303
5: -0.0006264, 0.0092147, -0.0007551, 0.0092542, -0.0098806, 0.0099698
6: -0.0051417, 0.0007322, -0.0051401, 0.0008464, -0.0058097, 0.0056332
7: -0.0096506, -0.0010770, -0.0097935, -0.0009206, -0.0087300, 0.0087165
8: -0.0050997, -0.0002132, -0.0048833, -0.0002009, -0.0048989, 0.0046700
9: 0.9821292, 1.0013417, 0.9821653, 1.0020704, -0.0199412, 0.0191764

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 210

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127201, upper bound: 0.0129994
time: 2.08 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133131, upper bound: 0.0132213
time: 2.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055048, 0.0004810, -0.0058714, 0.0008140, -0.0063188, 0.0063524
1: -0.0010231, 0.0126230, -0.0015337, 0.0129864, -0.0140094, 0.0141567
2: 0.0088106, 0.0215174, 0.0085205, 0.0227349, -0.0139242, 0.0129970
3: -0.0065157, -0.0025965, -0.0076280, -0.0025551, -0.0039606, 0.0050316
4: -0.0050177, 0.0025498, -0.0051036, 0.0035584, -0.0085761, 0.0076534
5: -0.0004196, 0.0092337, -0.0024018, 0.0092754, -0.0096950, 0.0116355
6: -0.0050328, 0.0007224, -0.0062002, 0.0007863, -0.0057561, 0.0068785
7: -0.0096715, -0.0011866, -0.0097791, 0.0004561, -0.0101276, 0.0085925
8: -0.0052640, -0.0002102, -0.0069514, -0.0001921, -0.0050719, 0.0067411
9: 0.9819962, 1.0009048, 0.9819590, 1.0031875, -0.0211913, 0.0189458

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133033, upper bound: 0.0136240
time: 3.16 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133980, upper bound: 0.0136240
time: 2.55 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055048, 0.0004810, -0.0059632, 0.0010810, -0.0065858, 0.0064442
1: -0.0010231, 0.0126230, -0.0017205, 0.0130860, -0.0141091, 0.0143435
2: 0.0088106, 0.0215174, 0.0084129, 0.0230463, -0.0142357, 0.0131045
3: -0.0065157, -0.0025965, -0.0079041, -0.0025482, -0.0039674, 0.0053077
4: -0.0050177, 0.0025498, -0.0051849, 0.0038253, -0.0088431, 0.0077347
5: -0.0004196, 0.0092337, -0.0029275, 0.0093436, -0.0097632, 0.0121612
6: -0.0050328, 0.0007224, -0.0065055, 0.0008341, -0.0058068, 0.0071723
7: -0.0096715, -0.0011866, -0.0098716, 0.0009549, -0.0106264, 0.0086850
8: -0.0052640, -0.0002102, -0.0073575, -0.0001842, -0.0050798, 0.0071473
9: 0.9819962, 1.0009048, 0.9818428, 1.0038506, -0.0218543, 0.0190620

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133033, upper bound: 0.0136816
time: 2.47 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133980, upper bound: 0.0136816
time: 1.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0055479, 0.0005018, -0.0058714, 0.0008140, -0.0063620, 0.0063732
1: -0.0011752, 0.0126435, -0.0015337, 0.0129864, -0.0141616, 0.0141773
2: 0.0086616, 0.0216721, 0.0085205, 0.0227349, -0.0140733, 0.0131516
3: -0.0066647, -0.0025921, -0.0076280, -0.0025551, -0.0041096, 0.0050360
4: -0.0050620, 0.0028451, -0.0051036, 0.0035584, -0.0086204, 0.0079487
5: -0.0008715, 0.0092885, -0.0024018, 0.0092754, -0.0101469, 0.0116902
6: -0.0052894, 0.0007509, -0.0062002, 0.0007863, -0.0059778, 0.0068702
7: -0.0096999, -0.0008926, -0.0097791, 0.0004561, -0.0101561, 0.0088865
8: -0.0055225, -0.0001994, -0.0069514, -0.0001921, -0.0053304, 0.0067520
9: 0.9819055, 1.0015521, 0.9819590, 1.0031875, -0.0212820, 0.0195931

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0135040
time: 3.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135229, upper bound: 0.0135040
time: 2.13 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0055479, 0.0005018, -0.0059632, 0.0010810, -0.0066289, 0.0064650
1: -0.0011752, 0.0126435, -0.0017205, 0.0130860, -0.0142612, 0.0143640
2: 0.0086616, 0.0216721, 0.0084129, 0.0230463, -0.0143847, 0.0132592
3: -0.0066647, -0.0025921, -0.0079041, -0.0025482, -0.0041165, 0.0053121
4: -0.0050620, 0.0028451, -0.0051849, 0.0038253, -0.0088873, 0.0080300
5: -0.0008715, 0.0092885, -0.0029275, 0.0093436, -0.0102151, 0.0122160
6: -0.0052894, 0.0007509, -0.0065055, 0.0008341, -0.0060289, 0.0071656
7: -0.0096999, -0.0008926, -0.0098716, 0.0009549, -0.0106549, 0.0089790
8: -0.0055225, -0.0001994, -0.0073575, -0.0001842, -0.0053383, 0.0071581
9: 0.9819055, 1.0015521, 0.9818428, 1.0038506, -0.0219451, 0.0197093

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134081, upper bound: 0.0135498
time: 2.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135229, upper bound: 0.0135498
time: 2.40 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055818, 0.0005002, -0.0054035, 0.0005601, -0.0061419, 0.0059038
1: -0.0011492, 0.0126953, -0.0011373, 0.0123019, -0.0134511, 0.0138326
2: 0.0087131, 0.0217622, 0.0085837, 0.0210366, -0.0123235, 0.0131785
3: -0.0067511, -0.0025868, -0.0063174, -0.0025463, -0.0042048, 0.0037306
4: -0.0050538, 0.0027898, -0.0051558, 0.0029039, -0.0079578, 0.0079456
5: -0.0008535, 0.0092472, -0.0006959, 0.0092881, -0.0101415, 0.0099431
6: -0.0052950, 0.0007520, -0.0051645, 0.0008536, -0.0060097, 0.0057403
7: -0.0097115, -0.0007990, -0.0098306, -0.0009921, -0.0087194, 0.0090316
8: -0.0056120, -0.0002088, -0.0050264, -0.0002210, -0.0053909, 0.0048176
9: 0.9819647, 1.0014023, 0.9821123, 1.0014941, -0.0195293, 0.0192900

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126364, upper bound: 0.0130154
time: 2.28 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132208, upper bound: 0.0132516
time: 3.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055144, 0.0004896, -0.0053849, 0.0007378, -0.0062523, 0.0058745
1: -0.0010930, 0.0125496, -0.0015896, 0.0123127, -0.0134057, 0.0141392
2: 0.0087244, 0.0214691, 0.0081629, 0.0209928, -0.0122684, 0.0133062
3: -0.0065769, -0.0025874, -0.0062528, -0.0025060, -0.0040709, 0.0036654
4: -0.0050146, 0.0027671, -0.0051269, 0.0030514, -0.0080660, 0.0078940
5: -0.0007008, 0.0091976, -0.0007731, 0.0092586, -0.0099594, 0.0099707
6: -0.0052021, 0.0007393, -0.0051511, 0.0008478, -0.0059315, 0.0056994
7: -0.0096783, -0.0009166, -0.0098009, -0.0009087, -0.0087697, 0.0088843
8: -0.0053500, -0.0002174, -0.0049143, -0.0002012, -0.0051488, 0.0046969
9: 0.9821079, 1.0012697, 0.9821532, 1.0020840, -0.0199761, 0.0191165

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126364, upper bound: 0.0131467
time: 2.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132208, upper bound: 0.0133793
time: 1.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056251, 0.0005268, -0.0054035, 0.0005601, -0.0061852, 0.0059303
1: -0.0013130, 0.0127040, -0.0011373, 0.0123019, -0.0136149, 0.0138413
2: 0.0085553, 0.0219109, 0.0085837, 0.0210366, -0.0124812, 0.0133272
3: -0.0069018, -0.0025822, -0.0063174, -0.0025463, -0.0043554, 0.0037352
4: -0.0051077, 0.0030899, -0.0051558, 0.0029039, -0.0080116, 0.0082457
5: -0.0013145, 0.0093160, -0.0006959, 0.0092881, -0.0106026, 0.0100119
6: -0.0055485, 0.0007797, -0.0051645, 0.0008536, -0.0062357, 0.0057311
7: -0.0097558, -0.0004877, -0.0098306, -0.0009921, -0.0087638, 0.0093429
8: -0.0058752, -0.0001975, -0.0050264, -0.0002210, -0.0056542, 0.0048289
9: 0.9818817, 1.0020707, 0.9821123, 1.0014941, -0.0196123, 0.0199584

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127642, upper bound: 0.0128982
time: 1.96 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0131251
time: 1.57 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0055579, 0.0005103, -0.0053849, 0.0007378, -0.0062957, 0.0058952
1: -0.0012539, 0.0125711, -0.0015896, 0.0123127, -0.0135666, 0.0141607
2: 0.0085665, 0.0216304, 0.0081629, 0.0209928, -0.0124263, 0.0134676
3: -0.0067256, -0.0025829, -0.0062528, -0.0025060, -0.0042196, 0.0036700
4: -0.0050689, 0.0030671, -0.0051269, 0.0030514, -0.0081203, 0.0081940
5: -0.0011561, 0.0092682, -0.0007731, 0.0092586, -0.0104147, 0.0100413
6: -0.0054517, 0.0007664, -0.0051511, 0.0008478, -0.0061548, 0.0056896
7: -0.0097219, -0.0006101, -0.0098009, -0.0009087, -0.0088132, 0.0091908
8: -0.0056079, -0.0002049, -0.0049143, -0.0002012, -0.0054067, 0.0047094
9: 0.9820170, 1.0019339, 0.9821532, 1.0020840, -0.0200670, 0.0197807

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127642, upper bound: 0.0129994
time: 2.48 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133793, upper bound: 0.0132213
time: 2.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

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

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131008, upper bound: 0.0134412
time: 2.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131959, upper bound: 0.0134412
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056406, 0.0005324, -0.0058392, 0.0008043, -0.0064449, 0.0063716
1: -0.0013245, 0.0127423, -0.0015879, 0.0129146, -0.0142390, 0.0143302
2: 0.0085543, 0.0219806, 0.0084332, 0.0225888, -0.0140344, 0.0135474
3: -0.0069391, -0.0025820, -0.0075553, -0.0025107, -0.0044284, 0.0049733
4: -0.0051143, 0.0030917, -0.0052789, 0.0036420, -0.0087564, 0.0083707
5: -0.0013453, 0.0093266, -0.0024488, 0.0093872, -0.0107325, 0.0117754
6: -0.0055691, 0.0007824, -0.0062250, 0.0009252, -0.0064235, 0.0069264
7: -0.0097594, -0.0004664, -0.0099662, 0.0004972, -0.0102566, 0.0094997
8: -0.0059303, -0.0001947, -0.0068645, -0.0001882, -0.0057421, 0.0066698
9: 0.9818454, 1.0020938, 0.9818462, 1.0033220, -0.0214766, 0.0202476

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132218, upper bound: 0.0133010
time: 2.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133405, upper bound: 0.0133010
time: 2.51 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057852, 0.0006246, -0.0053555, 0.0005455, -0.0063307, 0.0059801
1: -0.0013248, 0.0128612, -0.0009866, 0.0122721, -0.0135969, 0.0138478
2: 0.0086726, 0.0223980, 0.0087295, 0.0208611, -0.0121885, 0.0136685
3: -0.0073697, -0.0025587, -0.0061500, -0.0025514, -0.0048184, 0.0035914
4: -0.0050372, 0.0032396, -0.0050989, 0.0026110, -0.0076481, 0.0083386
5: -0.0018214, 0.0091973, -0.0002302, 0.0092277, -0.0110491, 0.0094276
6: -0.0058951, 0.0007600, -0.0049021, 0.0008207, -0.0066257, 0.0055299
7: -0.0097485, 0.0000526, -0.0097836, -0.0012967, -0.0084518, 0.0098361
8: -0.0065265, -0.0002088, -0.0047367, -0.0002296, -0.0062968, 0.0045279
9: 0.9821140, 1.0024341, 0.9822048, 1.0008374, -0.0187234, 0.0202293

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126513, upper bound: 0.0128664
time: 2.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132625, upper bound: 0.0131117
time: 1.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057116, 0.0005488, -0.0053323, 0.0007189, -0.0064306, 0.0058810
1: -0.0012549, 0.0127174, -0.0014314, 0.0122523, -0.0135071, 0.0141488
2: 0.0086842, 0.0220849, 0.0083177, 0.0207894, -0.0121052, 0.0137672
3: -0.0071718, -0.0025593, -0.0060874, -0.0025114, -0.0046604, 0.0035280
4: -0.0049993, 0.0032156, -0.0050689, 0.0027566, -0.0077559, 0.0082844
5: -0.0016449, 0.0091489, -0.0003223, 0.0091862, -0.0108311, 0.0094712
6: -0.0057879, 0.0007483, -0.0048916, 0.0008155, -0.0065276, 0.0054906
7: -0.0097185, -0.0000926, -0.0097565, -0.0012052, -0.0085133, 0.0096639
8: -0.0062336, -0.0002176, -0.0046326, -0.0002139, -0.0060198, 0.0044149
9: 0.9822485, 1.0022737, 0.9822592, 1.0014162, -0.0191677, 0.0200145

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126513, upper bound: 0.0129719
time: 1.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132625, upper bound: 0.0132092
time: 2.12 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0058291, 0.0007388, -0.0053555, 0.0005455, -0.0063746, 0.0060943
1: -0.0014958, 0.0128908, -0.0009866, 0.0122721, -0.0137679, 0.0138774
2: 0.0085263, 0.0225485, 0.0087295, 0.0208611, -0.0123348, 0.0138189
3: -0.0075243, -0.0025554, -0.0061500, -0.0025514, -0.0049730, 0.0035946
4: -0.0050838, 0.0035465, -0.0050989, 0.0026110, -0.0076947, 0.0086454
5: -0.0023049, 0.0092477, -0.0002302, 0.0092277, -0.0115326, 0.0094779
6: -0.0061455, 0.0007797, -0.0049021, 0.0008207, -0.0068959, 0.0055651
7: -0.0097599, 0.0003780, -0.0097836, -0.0012967, -0.0084632, 0.0101615
8: -0.0067971, -0.0001985, -0.0047367, -0.0002296, -0.0065674, 0.0045383
9: 0.9820567, 1.0031055, 0.9822048, 1.0008374, -0.0187807, 0.0209007

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0128491
time: 1.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131008
time: 2.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057558, 0.0006272, -0.0053323, 0.0007189, -0.0064747, 0.0059595
1: -0.0014220, 0.0127496, -0.0014314, 0.0122523, -0.0136742, 0.0141810
2: 0.0085379, 0.0222403, 0.0083177, 0.0207894, -0.0122515, 0.0139226
3: -0.0073237, -0.0025561, -0.0060874, -0.0025114, -0.0048123, 0.0035313
4: -0.0050460, 0.0035219, -0.0050689, 0.0027566, -0.0078026, 0.0085908
5: -0.0021211, 0.0092006, -0.0003223, 0.0091862, -0.0113073, 0.0095229
6: -0.0060372, 0.0007676, -0.0048916, 0.0008155, -0.0067979, 0.0055262
7: -0.0097291, 0.0002247, -0.0097565, -0.0012052, -0.0085239, 0.0099812
8: -0.0064993, -0.0002062, -0.0046326, -0.0002139, -0.0062854, 0.0044264
9: 0.9821953, 1.0029429, 0.9822592, 1.0014162, -0.0192209, 0.0206838

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0129517
time: 2.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0133669, upper bound: 0.0131959
time: 2.07 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0057093, 0.0005542, -0.0055776, 0.0005665, -0.0062758, 0.0061318
1: -0.0012569, 0.0127029, -0.0011855, 0.0123499, -0.0136069, 0.0138885
2: 0.0086870, 0.0220747, 0.0086291, 0.0214393, -0.0127524, 0.0134456
3: -0.0071826, -0.0025591, -0.0068773, -0.0025135, -0.0046691, 0.0043181
4: -0.0049985, 0.0032112, -0.0051090, 0.0032885, -0.0082870, 0.0083202
5: -0.0016501, 0.0091410, -0.0015325, 0.0091459, -0.0107960, 0.0106736
6: -0.0058070, 0.0007511, -0.0057366, 0.0008763, -0.0064486, 0.0062635
7: -0.0097120, -0.0000924, -0.0098376, -0.0001687, -0.0095434, 0.0097452
8: -0.0062641, -0.0002217, -0.0058347, -0.0002285, -0.0060355, 0.0056130
9: 0.9823090, 1.0022751, 0.9825593, 1.0022295, -0.0199204, 0.0197158

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128805, upper bound: 0.0125904
time: 3.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0130192, upper bound: 0.0125904
time: 2.61 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0057987, 0.0006406, -0.0057894, 0.0006903, -0.0064890, 0.0064300
1: -0.0013349, 0.0128958, -0.0014144, 0.0128646, -0.0141994, 0.0143102
2: 0.0086718, 0.0224582, 0.0085745, 0.0224053, -0.0137335, 0.0138836
3: -0.0073998, -0.0025585, -0.0073870, -0.0025135, -0.0048863, 0.0048285
4: -0.0050436, 0.0032411, -0.0052152, 0.0033358, -0.0083794, 0.0084563
5: -0.0018479, 0.0092067, -0.0019593, 0.0093162, -0.0111641, 0.0111660
6: -0.0059138, 0.0007631, -0.0059693, 0.0009000, -0.0065875, 0.0064987
7: -0.0097526, 0.0000722, -0.0099326, 0.0001700, -0.0099227, 0.0100048
8: -0.0065737, -0.0002064, -0.0065710, -0.0001987, -0.0063750, 0.0063646
9: 0.9820838, 1.0024544, 0.9819455, 1.0026394, -0.0205556, 0.0205089

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 250

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0131369, upper bound: 0.0132092
time: 2.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0132625, upper bound: 0.0132092
time: 2.78 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.04 + 596.07 = 600.10 seconds
