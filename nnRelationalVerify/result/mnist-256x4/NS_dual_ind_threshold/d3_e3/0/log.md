## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 11.027958876


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610)
1: (-5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247)
2: (-6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207)
3: (-7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554)
4: (-7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166)
5: (-6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819)
6: (-6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685)
7: (-7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842)
8: (-7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869)
9: (-6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.19 + 5.77 = 7.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393524

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 215

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1375678, upper bound: 11.1376746
time: 3.49 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1379560, upper bound: 11.1379560
time: 3.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.02 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.02
Output dim: 7, lower bound: -11.1375678, upper bound: 11.1376746
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.02
Output dim: 7, lower bound: -11.1379560, upper bound: 11.1379560

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.6303740, 3.1593225, -5.7372198, 4.8913293, -8.5217037, 8.8965425
1: -2.7714946, 2.7723784, -4.5623875, 4.2496071, -7.0211020, 7.3347659
2: -3.7536540, 2.6865299, -5.9285626, 3.9897254, -7.7433786, 8.6150923
3: -4.0379457, 2.3275275, -6.6109314, 3.4848714, -7.5228152, 8.9384584
4: -4.0214934, 3.2869802, -6.4321103, 5.0563951, -9.0778885, 9.7190895
5: -3.5084109, 3.0033035, -5.5990210, 4.5031242, -8.0115356, 8.6023245
6: -3.3600686, 3.6286707, -5.2370024, 5.5732474, -8.9333153, 8.8656731
7: -4.1947308, 2.8535740, -6.4955425, 4.2086568, -8.4033871, 9.3491154
8: -4.1423469, 3.1006598, -6.4789906, 4.7292175, -8.8715649, 9.5796490
9: -3.3961940, 3.3829327, -5.2270908, 5.2996264, -8.6958199, 8.6100235

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1374329, upper bound: 11.1375092
time: 4.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1375678, upper bound: 11.1376746
time: 9.97 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.0455360, 4.3323078, -6.3461747, 5.3783674, -10.4239016, 10.6784821
1: -3.9848967, 3.7769749, -5.0694451, 4.6653833, -8.6502790, 8.8464193
2: -5.2240906, 3.5542545, -6.5475259, 4.3579969, -9.5820875, 10.1017799
3: -5.7972827, 3.1012523, -7.3373280, 3.8154747, -9.6127567, 10.4385777
4: -5.6666875, 4.4954319, -7.1121330, 5.5647254, -11.2314129, 11.6075649
5: -4.9111052, 3.9948406, -6.1805391, 4.9388285, -9.8499336, 10.1753788
6: -4.6290703, 4.9571333, -5.7646132, 6.1231718, -10.7522421, 10.7217455
7: -5.7724724, 3.7025704, -7.1376524, 4.6066875, -10.3791599, 10.8402233
8: -5.7194819, 4.1970921, -7.1466722, 5.1983638, -10.9178457, 11.3437643
9: -4.6295280, 4.6788139, -5.7514696, 5.8427496, -10.4722776, 10.4302826

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378788, upper bound: 11.1378314
time: 2.78 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1379560, upper bound: 11.1379560
time: 3.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 8.30 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 8.30
Output dim: 7, lower bound: -11.1374329, upper bound: 11.1375092
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 8.30
Output dim: 7, lower bound: -11.1375678, upper bound: 11.1376746
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 8.30
Output dim: 7, lower bound: -11.1378788, upper bound: 11.1378314
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 8.30
Output dim: 7, lower bound: -11.1379560, upper bound: 11.1379560

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.4223700, 2.9833481, -6.3247166, 5.3644209, -8.7867908, 9.3080645
1: -2.6000357, 2.6235795, -5.0482049, 4.6522527, -7.2522874, 7.6717834
2: -3.5319405, 2.5557799, -6.5158319, 4.3694410, -7.9013815, 9.0716114
3: -3.7815287, 2.2108853, -7.3160419, 3.8064609, -7.5879898, 9.5269270
4: -3.7834907, 3.1095929, -7.1193609, 5.5368929, -9.3203831, 10.2289543
5: -3.2949815, 2.8528790, -6.1742015, 4.9419675, -8.2369480, 9.0270786
6: -3.1706457, 3.4308615, -5.7495670, 6.0762568, -9.2469025, 9.1804276
7: -3.9517655, 2.7770216, -7.0736270, 4.6420512, -8.5938168, 9.8506489
8: -3.9078405, 2.9382780, -7.1383381, 5.1710205, -9.0788593, 10.0766163
9: -3.2103145, 3.1903248, -5.7362700, 5.8212538, -9.0315685, 8.9265938

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1355324, upper bound: 11.1356592
time: 3.66 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1352650, upper bound: 11.1353491
time: 2.90 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.6303740, 3.1593225, -5.5441713, 4.7361217, -8.3664951, 8.7034931
1: -2.7714946, 2.7723784, -4.4017377, 4.1180100, -6.8895044, 7.1741161
2: -3.7536540, 2.6865299, -5.7312355, 3.8736362, -7.6272898, 8.4177647
3: -4.0379457, 2.3275275, -6.3810863, 3.3810215, -7.4189658, 8.7086134
4: -4.0214934, 3.2869802, -6.2214909, 4.8978353, -8.9193277, 9.5084705
5: -3.5084109, 3.0033035, -5.4123859, 4.3628087, -7.8712196, 8.4156885
6: -3.3600686, 3.6286707, -5.0705476, 5.3972898, -8.7573586, 8.6992188
7: -4.1947308, 2.8535740, -6.2878618, 4.0790625, -8.2737932, 9.1414356
8: -4.1423469, 3.1006598, -6.2689295, 4.5828915, -8.7252388, 9.3695889
9: -3.3961940, 3.3829327, -5.0613055, 5.1278439, -8.5240364, 8.4442387

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1357811, upper bound: 11.1359009
time: 3.58 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1355492, upper bound: 11.1356550
time: 3.51 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.8278160, 4.1532068, -7.0147548, 5.9219279, -10.7497444, 11.1679611
1: -3.7999637, 3.6258264, -5.6208014, 5.1229057, -8.9228697, 9.2466278
2: -4.9993696, 3.4179881, -7.2191963, 4.7881098, -9.7874794, 10.6371841
3: -5.5364952, 2.9821241, -8.1339817, 4.1865873, -9.7230825, 11.1161060
4: -5.4262247, 4.3136230, -7.8907728, 6.1152840, -11.5415087, 12.2043953
5: -4.6953125, 3.8367918, -6.8312726, 5.4442101, -10.1395226, 10.6680641
6: -4.4362907, 4.7553072, -6.3483691, 6.6999211, -11.1362114, 11.1036758
7: -5.5283914, 3.5505602, -7.7999096, 5.0916481, -10.6200390, 11.3504696
8: -5.4815812, 4.0292091, -7.9049191, 5.7150240, -11.1966057, 11.9341278
9: -4.4402719, 4.4801502, -6.3366504, 6.4361939, -10.8764648, 10.8167992

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1360135, upper bound: 11.1360427
time: 3.28 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1352650, upper bound: 11.1359816
time: 4.13 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.0455360, 4.3323078, -6.1465526, 5.2165580, -10.2620935, 10.4788609
1: -3.9848967, 3.7769749, -4.9031734, 4.5294881, -8.5143852, 8.6801472
2: -5.2240906, 3.5542545, -6.3427272, 4.2374701, -9.4615593, 9.8969803
3: -5.7972827, 3.1012523, -7.0995979, 3.7060175, -9.5032997, 10.2008486
4: -5.6666875, 4.4954319, -6.8941636, 5.4004221, -11.0671072, 11.3895950
5: -4.9111052, 3.9948406, -5.9877944, 4.7935100, -9.7046146, 9.9826326
6: -4.6290703, 4.9571333, -5.5919099, 5.9414334, -10.5705032, 10.5490437
7: -5.7724724, 3.7025704, -6.9234333, 4.4716897, -10.2441616, 10.6260033
8: -5.7194819, 4.1970921, -6.9276366, 5.0458336, -10.7653160, 11.1247292
9: -4.6295280, 4.6788139, -5.5789709, 5.6638694, -10.2933979, 10.2577848

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378314, upper bound: 11.1378788
time: 3.91 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378314, upper bound: 11.1379560
time: 3.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 9.55 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 9.55
Output dim: 7, lower bound: -11.1355324, upper bound: 11.1356592
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 9.55
Output dim: 7, lower bound: -11.1352650, upper bound: 11.1353491
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 9.55
Output dim: 7, lower bound: -11.1357811, upper bound: 11.1359009
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 9.55
Output dim: 7, lower bound: -11.1355492, upper bound: 11.1356550
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 9.55
Output dim: 7, lower bound: -11.1360135, upper bound: 11.1360427
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 9.55
Output dim: 7, lower bound: -11.1352650, upper bound: 11.1359816
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 9.55
Output dim: 7, lower bound: -11.1378314, upper bound: 11.1378788
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 9.55
Output dim: 7, lower bound: -11.1378314, upper bound: 11.1379560

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.9481114, 1.7835484, -5.9383411, 5.0543685, -7.0024791, 7.7218895
1: -1.5435486, 1.5807998, -4.7286243, 4.3890228, -5.9325705, 6.3094234
2: -1.9682884, 1.6759942, -6.1211839, 4.1367326, -6.1050205, 7.7971783
3: -1.9708533, 1.4121602, -6.8528566, 3.5959864, -5.5668397, 8.2650166
4: -2.1136496, 1.9012936, -6.6902065, 5.2208905, -7.3345404, 8.5914993
5: -1.7952040, 1.8316526, -5.8011703, 4.6638861, -6.4590893, 7.6328230
6: -1.8662210, 2.0636110, -5.4149113, 5.7284861, -7.5947070, 7.4785213
7: -2.2750185, 2.5146799, -6.6682262, 4.3808517, -6.6558704, 9.1829052
8: -2.2413177, 1.8197199, -6.7153983, 4.8781486, -7.1194663, 8.5351162
9: -1.9229898, 1.8736517, -5.4061041, 5.4757409, -7.3987284, 7.2797537

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1352401
time: 3.10 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1356592
time: 3.02 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.1012230, 1.8818669, -5.6859875, 4.8484597, -6.9496827, 7.5678544
1: -1.6427524, 1.6882073, -4.5199184, 4.2170310, -5.8597827, 6.2081251
2: -2.1133909, 1.7331971, -5.8635302, 3.9756324, -6.0890231, 7.5967274
3: -2.1502800, 1.4797089, -6.5558739, 3.4555273, -5.6058073, 8.0355816
4: -2.2534671, 2.0287337, -6.4099107, 5.0170250, -7.2704921, 8.4386444
5: -1.9250152, 1.9093434, -5.5499649, 4.4738398, -6.3988552, 7.4593081
6: -1.9850335, 2.2054107, -5.1940374, 5.5051546, -7.4901881, 7.3994484
7: -2.4469190, 2.5303996, -6.4065804, 4.1900244, -6.6369424, 8.9369802
8: -2.4021993, 1.9259564, -6.4359279, 4.6857100, -7.0879092, 8.3618841
9: -2.0289431, 1.9925432, -5.1863670, 5.2496605, -7.2786036, 7.1789103

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350947, upper bound: 11.1350939
time: 2.25 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350947, upper bound: 11.1353491
time: 2.97 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.1339355, 1.9295669, -5.1826954, 4.4440012, -6.5779352, 7.1122622
1: -1.6763440, 1.7086787, -4.1022968, 3.8690314, -5.5453739, 5.8109741
2: -2.1586528, 1.7817266, -5.3606620, 3.6522830, -5.8109350, 7.1423879
3: -2.1856019, 1.5090646, -5.9509821, 3.1841176, -5.3697195, 7.4600468
4: -2.3057156, 2.0529051, -5.8176689, 4.6003799, -6.9060950, 7.8705740
5: -1.9829459, 1.9508190, -5.0593481, 4.1009130, -6.0838585, 7.0101666
6: -2.0266542, 2.2368402, -4.7552500, 5.0697155, -7.0963683, 6.9920893
7: -2.4965825, 2.5731692, -5.9035387, 3.8294179, -6.3259993, 8.4767075
8: -2.4529085, 1.9531921, -5.8717031, 4.3062739, -6.7591825, 7.8248949
9: -2.0727239, 2.0348778, -4.7502451, 4.8038697, -6.8765936, 6.7851229

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1355080, upper bound: 11.1355361
time: 2.71 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1355080, upper bound: 11.1359009
time: 3.02 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.2882481, 2.0300798, -4.8790579, 4.1914062, -6.4796524, 6.9091377
1: -1.7777171, 1.8185692, -3.8468482, 3.6589310, -5.4366484, 5.6654177
2: -2.3110275, 1.8399091, -5.0461960, 3.4585581, -5.7695856, 6.8861051
3: -2.3829169, 1.5785501, -5.5881805, 3.0144291, -5.3973460, 7.1667309
4: -2.4462066, 2.1862936, -5.4735622, 4.3514261, -6.7976327, 7.6598558
5: -2.1138322, 2.0334826, -4.7524800, 3.8777244, -5.9915566, 6.7859626
6: -2.1499176, 2.3810306, -4.4841824, 4.7959394, -6.9458570, 6.8652120
7: -2.6705880, 2.5882778, -5.5778384, 3.6009130, -6.2715001, 8.1661167
8: -2.6160305, 2.0675163, -5.5319967, 4.0732136, -6.6892443, 7.5995121
9: -2.1924689, 2.1579370, -4.4859114, 4.5261035, -6.7185717, 6.6438484

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1353949, upper bound: 11.1353949
time: 3.05 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1353949, upper bound: 11.1356550
time: 2.66 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.1106687, 2.7033679, -6.6134501, 5.5976534, -8.7083225, 9.3168182
1: -2.3637238, 2.3983910, -5.2896185, 4.8495340, -7.2132559, 7.6880093
2: -3.1868699, 2.3408685, -6.8081255, 4.5448923, -7.7317615, 9.1489944
3: -3.4101207, 2.0175810, -7.6557322, 3.9645009, -7.3746214, 9.6733122
4: -3.4256270, 2.8702152, -7.4450016, 5.7861176, -9.2117443, 10.3152142
5: -2.9345839, 2.6061769, -6.4438324, 5.1488895, -8.0834723, 9.0500088
6: -2.8837512, 3.1406932, -6.0000100, 6.3387227, -9.2224741, 9.1407013
7: -3.5994816, 2.6604283, -7.3799381, 4.8183093, -8.4177914, 10.0403662
8: -3.5447147, 2.6908290, -7.4608254, 5.4057465, -8.9504614, 10.1516542
9: -2.9202719, 2.8901606, -5.9901552, 6.0770407, -8.9973106, 8.8803158

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1353531, upper bound: 11.1353050
time: 3.31 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1353531, upper bound: 11.1360352
time: 2.72 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5095959, 3.0132291, -6.3717051, 5.3965998, -8.9061947, 9.3849344
1: -2.6713979, 2.6793702, -5.0889273, 4.6852140, -7.3566122, 7.7682967
2: -3.6029117, 2.5585771, -6.5593119, 4.3883848, -7.9912968, 9.1178894
3: -3.9062254, 2.2232537, -7.3707337, 3.8252220, -7.7314463, 9.5939875
4: -3.8738532, 3.2119787, -7.1752129, 5.5893626, -9.4632139, 10.3871889
5: -3.3092628, 2.8712878, -6.2027235, 4.9654889, -8.2747507, 9.0740108
6: -3.2272358, 3.5182319, -5.7867746, 6.1245165, -9.3517523, 9.3050060
7: -4.0486593, 2.7268999, -7.1288986, 4.6324329, -8.6810913, 9.8557968
8: -3.9793224, 2.9959116, -7.1876044, 5.2178936, -9.1972160, 10.1835155
9: -3.2600565, 3.2439332, -5.7773890, 5.8571234, -9.1171799, 9.0213223

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1353464, upper bound: 11.1352360
time: 4.28 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1353464, upper bound: 11.1359814
time: 3.50 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.7454338, 4.8978777, -6.1465526, 5.2165580, -10.9619923, 11.0444298
1: -4.5711040, 4.2618880, -4.9031734, 4.5294881, -9.1005917, 9.1650620
2: -5.9216757, 3.9974687, -6.3427272, 4.2374701, -10.1591454, 10.3401947
3: -6.6473250, 3.4786744, -7.0995979, 3.7060175, -10.3533421, 10.5782719
4: -6.4849114, 5.0728559, -6.8941636, 5.4004221, -11.8853312, 11.9670200
5: -5.6055107, 4.4986033, -5.9877944, 4.7935100, -10.3990211, 10.4863968
6: -5.2526865, 5.5694113, -5.5919099, 5.9414334, -11.1941204, 11.1613216
7: -6.4900904, 4.1909366, -6.9234333, 4.4716897, -10.9617805, 11.1143675
8: -6.4909239, 4.7331915, -6.9276366, 5.0458336, -11.5367565, 11.6608276
9: -5.2344575, 5.3073502, -5.5789709, 5.6638694, -10.8983269, 10.8863192

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359820, upper bound: 11.1360135
time: 3.82 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1358951, upper bound: 11.1359923
time: 2.92 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.8614578, 4.1831050, -6.1465526, 5.2165580, -10.0780163, 10.3296576
1: -3.8301578, 3.6507936, -4.9031734, 4.5294881, -8.3596458, 8.5539665
2: -5.0373816, 3.4429517, -6.3427272, 4.2374701, -9.2748508, 9.7856770
3: -5.5773373, 3.0022082, -7.0995979, 3.7060175, -9.2833548, 10.1018057
4: -5.4655628, 4.3436723, -6.8941636, 5.4004221, -10.8659830, 11.2378359
5: -4.7310486, 3.8629441, -5.9877944, 4.7935100, -9.5245581, 9.8507357
6: -4.4686842, 4.7887440, -5.5919099, 5.9414334, -10.4101162, 10.3806534
7: -5.5717645, 3.5785732, -6.9234333, 4.4716897, -10.0434542, 10.5020065
8: -5.5202322, 4.0571265, -6.9276366, 5.0458336, -10.5660648, 10.9847622
9: -4.4719934, 4.5138912, -5.5789709, 5.6638694, -10.1358624, 10.0928621

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359820, upper bound: 11.1362192
time: 3.61 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1358951, upper bound: 11.1362067
time: 3.10 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 8.79 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1352401
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1356592
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1350947, upper bound: 11.1350939
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1350947, upper bound: 11.1353491
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1355080, upper bound: 11.1355361
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1355080, upper bound: 11.1359009
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1353949, upper bound: 11.1353949
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1353949, upper bound: 11.1356550
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1353531, upper bound: 11.1353050
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1353531, upper bound: 11.1360352
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1353464, upper bound: 11.1352360
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1353464, upper bound: 11.1359814
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1359820, upper bound: 11.1360135
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1358951, upper bound: 11.1359923
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1359820, upper bound: 11.1362192
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.79
Output dim: 7, lower bound: -11.1358951, upper bound: 11.1362067

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.9481114, 1.7835484, -3.7827957, 3.2951903, -5.2433019, 5.5663443
1: -1.5435486, 1.5807998, -2.9017353, 2.8861072, -4.4296556, 4.4825354
2: -1.9682884, 1.6759942, -3.9047019, 2.8095207, -4.7778091, 5.5806961
3: -1.9708533, 1.4121602, -4.2250972, 2.4162335, -4.3870869, 5.6372566
4: -2.1136496, 1.9012936, -4.2324066, 3.4114237, -5.5250731, 6.1337004
5: -1.7952040, 1.8316526, -3.6810448, 3.1263404, -4.9215441, 5.5126972
6: -1.8662210, 2.0636110, -3.5086136, 3.7476497, -5.6138692, 5.5722246
7: -2.2750185, 2.5146799, -4.3354111, 2.9468405, -5.2218585, 6.8500905
8: -2.2413177, 1.8197199, -4.3249674, 3.2198422, -5.4611597, 6.1446872
9: -1.9229898, 1.8736517, -3.5369365, 3.5244958, -5.4474859, 5.4105883

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351823, upper bound: 11.1352056
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1352401
time: 3.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.9481114, 1.7835484, -5.3979197, 4.6199059, -6.5680175, 7.1814666
1: -1.5435486, 1.5807998, -4.2802105, 4.0224833, -5.5660305, 5.8610106
2: -1.9682884, 1.6759942, -5.5737529, 3.7878189, -5.7561073, 7.2497463
3: -1.9708533, 1.4121602, -6.2227554, 3.2937224, -5.2645760, 7.6349154
4: -2.1136496, 1.9012936, -6.0977616, 4.7883520, -6.9020014, 7.9990554
5: -1.7952040, 1.8316526, -5.2622852, 4.2563267, -6.0515308, 7.0939369
6: -1.8662210, 2.0636110, -4.9418936, 5.2532406, -7.1194601, 7.0055037
7: -2.2750185, 2.5146799, -6.1158113, 3.9670670, -6.2420845, 8.6304913
8: -2.2413177, 1.8197199, -6.1192255, 4.4632816, -6.7045994, 7.9389443
9: -1.9229898, 1.8736517, -4.9368329, 4.9943705, -6.9173594, 6.8104839

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351823, upper bound: 11.1356164
time: 3.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1356592
time: 4.36 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.1012230, 1.8818669, -3.5138817, 3.0617387, -5.1629615, 5.3957486
1: -1.6427524, 1.6882073, -2.6717944, 2.6917269, -4.3344793, 4.3600016
2: -2.1133909, 1.7331971, -3.6183810, 2.6337700, -4.7471609, 5.3515778
3: -2.1502800, 1.4797089, -3.8944695, 2.2587490, -4.4090285, 5.3741779
4: -2.2534671, 2.0287337, -3.9174998, 3.1869752, -5.4404416, 5.9462337
5: -1.9250152, 1.9093434, -3.3959155, 2.9274077, -4.8524227, 5.3052588
6: -1.9850335, 2.2054107, -3.2612433, 3.4957011, -5.4807348, 5.4666533
7: -2.4469190, 2.5303996, -4.0324888, 2.7605743, -5.2074924, 6.5628877
8: -2.4021993, 1.9259564, -4.0161891, 3.0084805, -5.4106798, 5.9421453
9: -2.0289431, 1.9925432, -3.2955339, 3.2711282, -5.3000708, 5.2880769

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350824, upper bound: 11.1350746
time: 3.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350947, upper bound: 11.1350939
time: 4.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.1012230, 1.8818669, -5.2082410, 4.4620523, -6.5632753, 7.0901079
1: -1.6427524, 1.6882073, -4.1216650, 3.8923192, -5.5350704, 5.8098722
2: -2.1133909, 1.7331971, -5.3784113, 3.6640770, -5.7774677, 7.1116085
3: -2.1502800, 1.4797089, -5.9985175, 3.1861296, -5.3364096, 7.4782267
4: -2.2534671, 2.0287337, -5.8846011, 4.6335812, -6.8870482, 7.9133334
5: -1.9250152, 1.9093434, -5.0687575, 4.1143880, -6.0394030, 6.9781008
6: -1.9850335, 2.2054107, -4.7733307, 5.0850210, -7.0700541, 6.9787407
7: -2.4469190, 2.5303996, -5.9147191, 3.8171613, -6.2640791, 8.4451180
8: -2.4021993, 1.9259564, -5.9052696, 4.3197432, -6.7219424, 7.8312259
9: -2.0289431, 1.9925432, -4.7715797, 4.8213925, -6.8503356, 6.7641230

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350824, upper bound: 11.1353279
time: 3.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350947, upper bound: 11.1353491
time: 3.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.1339355, 1.9295669, -3.1571269, 2.7626362, -4.8965712, 5.0866933
1: -1.6763440, 1.7086787, -2.4100082, 2.4331393, -4.1094823, 4.1186857
2: -2.1586528, 1.7817266, -3.2516975, 2.3963411, -4.5549936, 5.0334239
3: -2.1856019, 1.5090646, -3.4541128, 2.0633864, -4.2489882, 4.9631772
4: -2.3057156, 2.0529051, -3.4733453, 2.8901699, -5.1958847, 5.5262494
5: -1.9829459, 1.9508190, -3.0226030, 2.6640973, -4.6470432, 4.9734221
6: -2.0266542, 2.2368402, -2.9333212, 3.1861248, -5.2127786, 5.1701612
7: -2.4965825, 2.5731692, -3.6567106, 2.7434280, -5.2400093, 6.2298799
8: -2.4529085, 1.9531921, -3.6096737, 2.7339411, -5.1868496, 5.5628653
9: -2.0727239, 2.0348778, -2.9767938, 2.9509046, -5.0236282, 5.0116715

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1353344
time: 3.45 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1355314
time: 3.34 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.1339355, 1.9295669, -4.5146699, 3.8939621, -6.0278978, 6.4442363
1: -1.6763440, 1.7086787, -3.5319526, 3.4042315, -5.0805745, 5.2406311
2: -2.1586528, 1.7817266, -4.6761708, 3.2242422, -5.3828950, 6.4578962
3: -2.1856019, 1.5090646, -5.1492071, 2.8078556, -4.9934573, 6.6582718
4: -2.3057156, 2.0529051, -5.0656562, 4.0507965, -6.3565111, 7.1185613
5: -1.9829459, 1.9508190, -4.3788271, 3.6155701, -5.5985160, 6.3296461
6: -2.0266542, 2.2368402, -4.1540308, 4.4658070, -6.4924603, 6.3908710
7: -2.4965825, 2.5731692, -5.1891203, 3.3346128, -5.8311944, 7.7622895
8: -2.4529085, 1.9531921, -5.1312909, 3.7866611, -6.2395687, 7.0844827
9: -2.0727239, 2.0348778, -4.1661887, 4.1923218, -6.2650452, 6.2010665

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1357290
time: 3.53 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1359001
time: 3.46 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.2882481, 2.0300798, -2.8904681, 2.5370262, -4.8252735, 4.9205480
1: -1.7777171, 1.8185692, -2.2164125, 2.2407041, -4.0184212, 4.0349817
2: -2.3110275, 1.8399091, -2.9647920, 2.2256269, -4.5366545, 4.8047009
3: -2.3829169, 1.5785501, -3.1275761, 1.9115132, -4.2944298, 4.7061262
4: -2.4462066, 2.1862936, -3.1573172, 2.6722434, -5.1184492, 5.3436108
5: -2.1138322, 2.0334826, -2.7389550, 2.4701715, -4.5840034, 4.7724376
6: -2.1499176, 2.3810306, -2.6918461, 2.9379869, -5.0879040, 5.0728765
7: -2.6705880, 2.5882778, -3.3522999, 2.6825595, -5.3531470, 5.9405775
8: -2.6160305, 2.0675163, -3.3063493, 2.5259814, -5.1420116, 5.3738651
9: -2.1924689, 2.1579370, -2.7352579, 2.7065756, -4.8990445, 4.8931947

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1350947
time: 3.40 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1353949
time: 3.47 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.2882481, 2.0300798, -4.2981853, 3.7052093, -5.9934573, 6.3282652
1: -1.7777171, 1.8185692, -3.3431563, 3.2473710, -5.0250883, 5.1617255
2: -2.3110275, 1.8399091, -4.4457159, 3.0796194, -5.3906469, 6.2856250
3: -2.3829169, 1.5785501, -4.8790359, 2.6806717, -5.0635877, 6.4575863
4: -2.4462066, 2.1862936, -4.8108969, 3.8703141, -6.3165197, 6.9971895
5: -2.1138322, 2.0334826, -4.1487627, 3.4544172, -5.5682492, 6.1822453
6: -2.1499176, 2.3810306, -3.9526963, 4.2620564, -6.4119735, 6.3337269
7: -2.6705880, 2.5882778, -4.9438572, 3.1645565, -5.8351445, 7.5321350
8: -2.6160305, 2.0675163, -4.8809605, 3.6170545, -6.2330847, 6.9484768
9: -2.1924689, 2.1579370, -3.9706798, 3.9866600, -6.1791286, 6.1286159

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1353464
time: 3.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1356550
time: 3.46 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.1106687, 2.7033679, -3.7827957, 3.2951903, -6.4058585, 6.4861627
1: -2.3637238, 2.3983910, -2.9017353, 2.8861072, -5.2498307, 5.3001261
2: -3.1868699, 2.3408685, -3.9047019, 2.8095207, -5.9963903, 6.2455702
3: -3.4101207, 2.0175810, -4.2250972, 2.4162335, -5.8263540, 6.2426782
4: -3.4256270, 2.8702152, -4.2324066, 3.4114237, -6.8370504, 7.1026216
5: -2.9345839, 2.6061769, -3.6810448, 3.1263404, -6.0609236, 6.2872219
6: -2.8837512, 3.1406932, -3.5086136, 3.7476497, -6.6314006, 6.6493053
7: -3.5994816, 2.6604283, -4.3354111, 2.9468405, -6.5463219, 6.9958391
8: -3.5447147, 2.6908290, -4.3249674, 3.2198422, -6.7645569, 7.0157967
9: -2.9202719, 2.8901606, -3.5369365, 3.5244958, -6.4447680, 6.4270959

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1307044, upper bound: 11.1302095
time: 3.43 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301272, upper bound: 11.1298693
time: 4.16 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.1106687, 2.7033679, -5.3979197, 4.6199059, -7.7305746, 8.1012859
1: -2.3637238, 2.3983910, -4.2802105, 4.0224833, -6.3862047, 6.6786013
2: -3.1868699, 2.3408685, -5.5737529, 3.7878189, -6.9746885, 7.9146214
3: -3.4101207, 2.0175810, -6.2227554, 3.2937224, -6.7038426, 8.2403364
4: -3.4256270, 2.8702152, -6.0977616, 4.7883520, -8.2139788, 8.9679756
5: -2.9345839, 2.6061769, -5.2622852, 4.2563267, -7.1909084, 7.8684621
6: -2.8837512, 3.1406932, -4.9418936, 5.2532406, -8.1369905, 8.0825863
7: -3.5994816, 2.6604283, -6.1158113, 3.9670670, -7.5665483, 8.7762394
8: -3.5447147, 2.6908290, -6.1192255, 4.4632816, -8.0079966, 8.8100548
9: -2.9202719, 2.8901606, -4.9368329, 4.9943705, -7.9146423, 7.8269935

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1307044, upper bound: 11.1313913
time: 4.00 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301272, upper bound: 11.1309918
time: 2.85 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.5095959, 3.0132291, -3.5138817, 3.0617387, -6.5713334, 6.5271101
1: -2.6713979, 2.6793702, -2.6717944, 2.6917269, -5.3631248, 5.3511643
2: -3.6029117, 2.5585771, -3.6183810, 2.6337700, -6.2366815, 6.1769571
3: -3.9062254, 2.2232537, -3.8944695, 2.2587490, -6.1649737, 6.1177235
4: -3.8738532, 3.2119787, -3.9174998, 3.1869752, -7.0608277, 7.1294785
5: -3.3092628, 2.8712878, -3.3959155, 2.9274077, -6.2366695, 6.2672033
6: -3.2272358, 3.5182319, -3.2612433, 3.4957011, -6.7229357, 6.7794752
7: -4.0486593, 2.7268999, -4.0324888, 2.7605743, -6.8092308, 6.7593889
8: -3.9793224, 2.9959116, -4.0161891, 3.0084805, -6.9878030, 7.0121007
9: -3.2600565, 3.2439332, -3.2955339, 3.2711282, -6.5311847, 6.5394669

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1306928, upper bound: 11.1301602
time: 3.04 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301267, upper bound: 11.1298340
time: 2.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.5095959, 3.0132291, -5.2082410, 4.4620523, -7.9716482, 8.2214699
1: -2.6713979, 2.6793702, -4.1216650, 3.8923192, -6.5637169, 6.8010349
2: -3.6029117, 2.5585771, -5.3784113, 3.6640770, -7.2669883, 7.9369884
3: -3.9062254, 2.2232537, -5.9985175, 3.1861296, -7.0923538, 8.2217703
4: -3.8738532, 3.2119787, -5.8846011, 4.6335812, -8.5074348, 9.0965786
5: -3.3092628, 2.8712878, -5.0687575, 4.1143880, -7.4236503, 7.9400454
6: -3.2272358, 3.5182319, -4.7733307, 5.0850210, -8.3122549, 8.2915611
7: -4.0486593, 2.7268999, -5.9147191, 3.8171613, -7.8658209, 8.6416187
8: -3.9793224, 2.9959116, -5.9052696, 4.3197432, -8.2990656, 8.9011812
9: -3.2600565, 3.2439332, -4.7715797, 4.8213925, -8.0814495, 8.0155125

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1306928, upper bound: 11.1313562
time: 3.30 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301267, upper bound: 11.1309755
time: 3.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.3702435, 4.5953255, -4.2712283, 3.6816120, -9.0518551, 8.8665524
1: -4.2584782, 4.0036120, -3.3281317, 3.2267680, -7.4852462, 7.3317437
2: -5.5406184, 3.7697785, -4.4086599, 3.0750651, -8.6156826, 8.1784382
3: -6.1959505, 3.2757361, -4.8393097, 2.6687350, -8.8646851, 8.1150455
4: -6.0654278, 4.7638922, -4.7764378, 3.8421800, -9.9076080, 9.5403299
5: -5.2365303, 4.2325706, -4.1294413, 3.4384525, -8.6749830, 8.3620119
6: -4.9232883, 5.2296615, -3.9331429, 4.2283592, -9.1516476, 9.1628046
7: -6.0897589, 3.9380517, -4.9061098, 3.1619222, -9.2516813, 8.8441591
8: -6.0820246, 4.4450655, -4.8481131, 3.5975885, -9.6796131, 9.2931786
9: -4.9131632, 4.9700165, -3.9492452, 3.9595709, -8.8727331, 8.9192619

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359706, upper bound: 11.1359627
time: 2.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359706, upper bound: 11.1359627
time: 3.07 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.1921172, 4.4466643, -4.5964632, 3.9319887, -9.1241055, 9.0431271
1: -4.1084328, 3.8805048, -3.6104903, 3.4569182, -7.5653505, 7.4909940
2: -5.3591771, 3.6527016, -4.7455592, 3.2553487, -8.6145248, 8.3982611
3: -5.9818563, 3.1751950, -5.2418280, 2.8373833, -8.8192387, 8.4170227
4: -5.8644485, 4.6192193, -5.1363916, 4.1237707, -9.9882193, 9.7556105
5: -5.0524988, 4.1006241, -4.4285460, 3.6536789, -8.7061758, 8.5291691
6: -4.7624378, 5.0709376, -4.2134600, 4.5397835, -9.3022213, 9.2843971
7: -5.8991809, 3.7985160, -5.2686110, 3.3387711, -9.2379522, 9.0671253
8: -5.8830891, 4.3083396, -5.2005234, 3.8505926, -9.7336817, 9.5088615
9: -4.7569537, 4.8064957, -4.2241516, 4.2514286, -9.0083828, 9.0306463

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319498, upper bound: 11.1316869
time: 3.03 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1309756, upper bound: 11.1310188
time: 3.26 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.5146699, 3.8939621, -4.2712283, 3.6816120, -8.1962814, 8.1651907
1: -3.5319526, 3.4042315, -3.3281317, 3.2267680, -6.7587204, 6.7323623
2: -4.6761708, 3.2242422, -4.4086599, 3.0750651, -7.7512350, 7.6329021
3: -5.1492071, 2.8078556, -4.8393097, 2.6687350, -7.8179421, 7.6471634
4: -5.0656562, 4.0507965, -4.7764378, 3.8421800, -8.9078360, 8.8272324
5: -4.3788271, 3.6155701, -4.1294413, 3.4384525, -7.8172798, 7.7450113
6: -4.1540308, 4.4658070, -3.9331429, 4.2283592, -8.3823900, 8.3989496
7: -5.1891203, 3.3346128, -4.9061098, 3.1619222, -8.3510408, 8.2407207
8: -5.1312909, 3.7866611, -4.8481131, 3.5975885, -8.7288790, 8.6347742
9: -4.1661887, 4.1923218, -3.9492452, 3.9595709, -8.1257582, 8.1415663

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1361647, upper bound: 11.1361650
time: 2.94 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1361647, upper bound: 11.1361650
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.2981853, 3.7052093, -4.5964632, 3.9319887, -8.2301731, 8.3016720
1: -3.3431563, 3.2473710, -3.6104903, 3.4569182, -6.8000736, 6.8578610
2: -4.4457159, 3.0796194, -4.7455592, 3.2553487, -7.7010636, 7.8251781
3: -4.8790359, 2.6806717, -5.2418280, 2.8373833, -7.7164192, 7.9224997
4: -4.8108969, 3.8703141, -5.1363916, 4.1237707, -8.9346676, 9.0067043
5: -4.1487627, 3.4544172, -4.4285460, 3.6536789, -7.8024397, 7.8829608
6: -3.9526963, 4.2620564, -4.2134600, 4.5397835, -8.4924793, 8.4755154
7: -4.9438572, 3.1645565, -5.2686110, 3.3387711, -8.2826281, 8.4331665
8: -4.8809605, 3.6170545, -5.2005234, 3.8505926, -8.7315531, 8.8175774
9: -3.9706798, 3.9866600, -4.2241516, 4.2514286, -8.2221088, 8.2108116

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1322212, upper bound: 11.1319976
time: 3.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315763, upper bound: 11.1315803
time: 2.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 8.41 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1351823, upper bound: 11.1352056
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1352401
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1351823, upper bound: 11.1356164
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1356592
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1350824, upper bound: 11.1350746
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1350947, upper bound: 11.1350939
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1350824, upper bound: 11.1353279
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1350947, upper bound: 11.1353491
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1353344
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1355314
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1357290
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1359001
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1350947
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1353949
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1353464
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1356550
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1307044, upper bound: 11.1302095
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1301272, upper bound: 11.1298693
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1307044, upper bound: 11.1313913
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1301272, upper bound: 11.1309918
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1306928, upper bound: 11.1301602
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1301267, upper bound: 11.1298340
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1306928, upper bound: 11.1313562
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1301267, upper bound: 11.1309755
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1359706, upper bound: 11.1359627
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1359706, upper bound: 11.1359627
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1319498, upper bound: 11.1316869
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1309756, upper bound: 11.1310188
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1361647, upper bound: 11.1361650
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1361647, upper bound: 11.1361650
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1322212, upper bound: 11.1319976
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.41
Output dim: 7, lower bound: -11.1315763, upper bound: 11.1315803

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.0366821, 1.1294487, -3.3726606, 2.9440570, -3.9807391, 4.5021095
1: -0.8836873, 0.9393622, -2.5595658, 2.5824127, -3.4661000, 3.4989281
2: -1.0089200, 1.1475974, -3.4611201, 2.5483885, -3.5573084, 4.6087174
3: -0.9660726, 0.9333000, -3.7122908, 2.1809232, -3.1469958, 4.6455903
4: -1.1172843, 1.1083202, -3.7404990, 3.0561094, -4.1733937, 4.8488193
5: -0.9621598, 1.2682574, -3.2624285, 2.8332901, -3.7954500, 4.5306859
6: -1.0612345, 1.2332249, -3.1285300, 3.3568518, -4.4180861, 4.3617549
7: -1.0820520, 2.3275163, -3.8557882, 2.7028170, -3.7848687, 6.1833048
8: -1.2054007, 1.1982937, -3.8502388, 2.8989627, -4.1043634, 5.0485325
9: -1.1640354, 1.1141696, -3.1660120, 3.1398840, -4.3039193, 4.2801819

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351823, upper bound: 11.1352056
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351823, upper bound: 11.1352056
time: 3.53 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.5582193, 1.4868548, -3.7065353, 3.2283006, -4.7865200, 5.1933899
1: -1.2573355, 1.3035926, -2.8337045, 2.8292315, -4.0865669, 4.1372972
2: -1.5561496, 1.4517007, -3.8220358, 2.7594614, -4.3156109, 5.2737365
3: -1.5425440, 1.2013685, -4.1293440, 2.3710754, -3.9136195, 5.3307123
4: -1.6801794, 1.5882968, -4.1406517, 3.3461108, -5.0262899, 5.7289476
5: -1.4170804, 1.5896873, -3.6017392, 3.0714366, -4.4885168, 5.1914263
6: -1.5139066, 1.7081155, -3.4374158, 3.6744654, -5.1883721, 5.1455312
7: -1.7940495, 2.4350514, -4.2461228, 2.8902202, -4.6842699, 6.6811743
8: -1.7895389, 1.5539353, -4.2352052, 3.1600399, -4.9495788, 5.7891407
9: -1.5927870, 1.5432941, -3.4672897, 3.4518099, -5.0445967, 5.0105839

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1352401
time: 3.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1352109, upper bound: 11.1352401
time: 3.15 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.0366821, 1.1294487, -4.9357986, 4.2384329, -5.2751150, 6.0652475
1: -0.8836873, 0.9393622, -3.8831968, 3.6941929, -4.5778804, 4.8225589
2: -1.0089200, 1.1475974, -5.0895510, 3.4997926, -4.5087128, 6.2371483
3: -0.9660726, 0.9333000, -5.6583066, 3.0374768, -4.0035496, 6.5916066
4: -1.1172843, 1.1083202, -5.5651922, 4.3966761, -5.5139604, 6.6735125
5: -0.9621598, 1.2682574, -4.8088040, 3.9265554, -4.8887153, 6.0770617
6: -1.0612345, 1.2332249, -4.5278134, 4.8250666, -5.8863010, 5.7610373
7: -1.0820520, 2.3275163, -5.6001358, 3.6458933, -4.7279453, 7.9276524
8: -1.2054007, 1.1982937, -5.5980473, 4.1080832, -5.3134842, 6.7963409
9: -1.1640354, 1.1141696, -4.5324402, 4.5703454, -5.7343807, 5.6466098

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1355000, upper bound: 11.1356164
time: 4.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1355000, upper bound: 11.1356164
time: 3.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.5582193, 1.4868548, -5.3135562, 4.5511475, -6.1093669, 6.8004098
1: -1.2573355, 1.3035926, -4.2088633, 3.9634435, -5.2207789, 5.5124559
2: -1.5561496, 1.4517007, -5.4868264, 3.7357125, -5.2918620, 6.9385271
3: -1.5425440, 1.2013685, -6.1223207, 3.2470210, -4.7895651, 7.3236876
4: -1.6801794, 1.5882968, -6.0028696, 4.7189174, -6.3990965, 7.5911655
5: -1.4170804, 1.5896873, -5.1804781, 4.1954970, -5.6125774, 6.7701654
6: -1.5139066, 1.7081155, -4.8678179, 5.1768088, -6.6907153, 6.5759335
7: -1.7940495, 2.4350514, -6.0240173, 3.9073906, -5.7014399, 8.4590683
8: -1.7895389, 1.5539353, -6.0250063, 4.3997383, -6.1892772, 7.5789413
9: -1.5927870, 1.5432941, -4.8640189, 4.9186134, -6.5114002, 6.4073129

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1355324, upper bound: 11.1356592
time: 3.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1355324, upper bound: 11.1356592
time: 3.29 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.1622989, 1.2090102, -3.1159139, 2.7257710, -3.8880699, 4.3249240
1: -0.9653517, 1.0217161, -2.3729217, 2.3963590, -3.3617105, 3.3946378
2: -1.1276082, 1.1955432, -3.1869082, 2.3851414, -3.5127497, 4.3824515
3: -1.1146486, 0.9815911, -3.3970268, 2.0327532, -3.1474018, 4.3786178
4: -1.2488141, 1.2071750, -3.4394886, 2.8420310, -4.0908451, 4.6466637
5: -1.0637965, 1.3265548, -2.9888420, 2.6440153, -3.7078118, 4.3153968
6: -1.1627371, 1.3392450, -2.8913047, 3.1207695, -4.2835064, 4.2305498
7: -1.2176863, 2.3481715, -3.5652056, 2.6323614, -3.8500476, 5.9133768
8: -1.3313199, 1.2797718, -3.5572073, 2.6987391, -4.0300589, 4.8369789
9: -1.2560207, 1.2002938, -2.9350281, 2.9046814, -4.1607022, 4.1353216

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301449, upper bound: 11.1305624
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1295161, upper bound: 11.1294076
time: 8.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.7010288, 1.5812120, -3.4411111, 2.9978848, -4.6989136, 5.0223222
1: -1.3524939, 1.3974861, -2.6075225, 2.6368930, -3.9893866, 4.0050087
2: -1.6908520, 1.5037775, -3.5391874, 2.5861433, -4.2769952, 5.0429649
3: -1.7018156, 1.2636898, -3.8029032, 2.2161317, -3.9179473, 5.0665927
4: -1.8173356, 1.6979595, -3.8294849, 3.1242738, -4.9416094, 5.5274434
5: -1.5435944, 1.6561334, -3.3200762, 2.8746738, -4.4182682, 4.9762096
6: -1.6288460, 1.8341578, -3.1927657, 3.4259276, -5.0547738, 5.0269237
7: -1.9548051, 2.4544976, -3.9466109, 2.7278867, -4.6826916, 6.4011087
8: -1.9352900, 1.6447752, -3.9309459, 2.9510868, -4.8863769, 5.5757208
9: -1.6980934, 1.6444600, -3.2289934, 3.2023768, -4.9004703, 4.8734527

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1304043, upper bound: 11.1308313
time: 3.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1297654, upper bound: 11.1296722
time: 3.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.1622989, 1.2090102, -4.7603021, 4.0853271, -5.2476263, 5.9693122
1: -0.9653517, 1.0217161, -3.7291763, 3.5683601, -4.5337119, 4.7508926
2: -1.1276082, 1.1955432, -4.9043627, 3.3797317, -4.5073400, 6.0999060
3: -1.1146486, 0.9815911, -5.4410100, 2.9341478, -4.0487967, 6.4226012
4: -1.2488141, 1.2071750, -5.3595943, 4.2487574, -5.4975715, 6.5667696
5: -1.0637965, 1.3265548, -4.6207514, 3.7954447, -4.8592415, 5.9473062
6: -1.1627371, 1.3392450, -4.3648214, 4.6634569, -5.8261938, 5.7040663
7: -1.2176863, 2.3481715, -5.4064202, 3.5011234, -4.7188096, 7.7545919
8: -1.3313199, 1.2797718, -5.3934126, 3.9716401, -5.3029599, 6.6731844
9: -1.2560207, 1.2002938, -4.3730879, 4.4031563, -5.6591768, 5.5733814

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1302314, upper bound: 11.1307882
time: 3.27 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1296860, upper bound: 11.1298739
time: 2.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.7010288, 1.5812120, -5.1255932, 4.3936739, -6.0947027, 6.7068052
1: -1.3524939, 1.3974861, -4.0506134, 3.8337481, -5.1862411, 5.4480996
2: -1.6908520, 1.5037775, -5.2925653, 3.6122384, -5.3030906, 6.7963428
3: -1.7018156, 1.2636898, -5.8984337, 3.1398802, -4.8416958, 7.1621232
4: -1.8173356, 1.6979595, -5.7901092, 4.5650935, -6.3824291, 7.4880676
5: -1.5435944, 1.6561334, -4.9873009, 4.0551286, -5.5987229, 6.6434345
6: -1.6288460, 1.8341578, -4.6996336, 5.0093908, -6.6382370, 6.5337915
7: -1.9548051, 2.4544976, -5.8234792, 3.7577844, -5.7125893, 8.2779770
8: -1.9352900, 1.6447752, -5.8123865, 4.2564850, -6.1917748, 7.4571609
9: -1.6980934, 1.6444600, -4.6994939, 4.7460012, -6.4440947, 6.3439541

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1304986, upper bound: 11.1310392
time: 4.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1299388, upper bound: 11.1301328
time: 3.23 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.4759202, 2.2052619, -3.1571269, 2.7626362, -5.2385564, 5.3623881
1: -1.9209974, 1.9448106, -2.4100082, 2.4331393, -4.3541355, 4.3548183
2: -2.5026054, 2.0084023, -3.2516975, 2.3963411, -4.8989468, 5.2600994
3: -2.6021361, 1.6866988, -3.4541128, 2.0633864, -4.6655226, 5.1408114
4: -2.6887903, 2.3225532, -3.4733453, 2.8901699, -5.5789604, 5.7958980
5: -2.3388262, 2.1839335, -3.0226030, 2.6640973, -5.0029235, 5.2065363
6: -2.3287411, 2.5256596, -2.9333212, 3.1861248, -5.5148659, 5.4589806
7: -2.8677049, 2.5367522, -3.6567106, 2.7434280, -5.6111326, 6.1934624
8: -2.8559353, 2.2017016, -3.6096737, 2.7339411, -5.5898762, 5.8113756
9: -2.3546133, 2.3322062, -2.9767938, 2.9509046, -5.3055172, 5.3090000

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 12

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1353344
time: 3.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1353344
time: 3.63 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.9781940, 1.8079662, -3.1571269, 2.7626362, -4.7408304, 4.9650927
1: -1.5658681, 1.6022439, -2.4100082, 2.4331393, -3.9990072, 4.0122519
2: -2.0006208, 1.6958634, -3.2516975, 2.3963411, -4.3969622, 4.9475608
3: -2.0031581, 1.4287895, -3.4541128, 2.0633864, -4.0665436, 4.8829021
4: -2.1460299, 1.9267356, -3.4733453, 2.8901699, -5.0361996, 5.4000802
5: -1.8267264, 1.8514880, -3.0226030, 2.6640973, -4.4908237, 4.8740911
6: -1.8936270, 2.0926702, -2.9333212, 3.1861248, -5.0797515, 5.0259914
7: -2.3143888, 2.5432315, -3.6567106, 2.7434280, -5.0578160, 6.1999421
8: -2.2763715, 1.8420560, -3.6096737, 2.7339411, -5.0103111, 5.4517298
9: -1.9489374, 1.9005178, -2.9767938, 2.9509046, -4.8998423, 4.8773117

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 12

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1355314
time: 3.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351728, upper bound: 11.1355314
time: 3.16 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.4759202, 2.2052619, -4.5146699, 3.8939621, -6.3698826, 6.7199316
1: -1.9209974, 1.9448106, -3.5319526, 3.4042315, -5.3252282, 5.4767632
2: -2.5026054, 2.0084023, -4.6761708, 3.2242422, -5.7268476, 6.6845722
3: -2.6021361, 1.6866988, -5.1492071, 2.8078556, -5.4099917, 6.8359060
4: -2.6887903, 2.3225532, -5.0656562, 4.0507965, -6.7395859, 7.3882093
5: -2.3388262, 2.1839335, -4.3788271, 3.6155701, -5.9543962, 6.5627604
6: -2.3287411, 2.5256596, -4.1540308, 4.4658070, -6.7945480, 6.6796904
7: -2.8677049, 2.5367522, -5.1891203, 3.3346128, -6.2023172, 7.7258720
8: -2.8559353, 2.2017016, -5.1312909, 3.7866611, -6.6425962, 7.3329926
9: -2.3546133, 2.3322062, -4.1661887, 4.1923218, -6.5469351, 6.4983950

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1354801, upper bound: 11.1357290
time: 2.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1354801, upper bound: 11.1357290
time: 2.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.9781940, 1.8079662, -4.5146699, 3.8939621, -5.8721561, 6.3226357
1: -1.5658681, 1.6022439, -3.5319526, 3.4042315, -4.9700994, 5.1341953
2: -2.0006208, 1.6958634, -4.6761708, 3.2242422, -5.2248631, 6.3720331
3: -2.0031581, 1.4287895, -5.1492071, 2.8078556, -4.8110132, 6.5779967
4: -2.1460299, 1.9267356, -5.0656562, 4.0507965, -6.1968260, 6.9923916
5: -1.8267264, 1.8514880, -4.3788271, 3.6155701, -5.4422965, 6.2303152
6: -1.8936270, 2.0926702, -4.1540308, 4.4658070, -6.3594322, 6.2466998
7: -2.3143888, 2.5432315, -5.1891203, 3.3346128, -5.6490011, 7.7323518
8: -2.2763715, 1.8420560, -5.1312909, 3.7866611, -6.0630322, 6.9733462
9: -1.9489374, 1.9005178, -4.1661887, 4.1923218, -6.1412592, 6.0667067

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1354801, upper bound: 11.1359001
time: 3.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1354801, upper bound: 11.1359001
time: 2.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.6521246, 2.3263724, -2.8904681, 2.5370262, -5.1891489, 5.2168398
1: -2.0446608, 2.0809975, -2.2164125, 2.2407041, -4.2853651, 4.2974095
2: -2.7097640, 2.0814281, -2.9647920, 2.2256269, -4.9353905, 5.0462198
3: -2.8495383, 1.7706244, -3.1275761, 1.9115132, -4.7610502, 4.8982005
4: -2.8642821, 2.4896882, -3.1573172, 2.6722434, -5.5365248, 5.6470051
5: -2.4940369, 2.2863274, -2.7389550, 2.4701715, -4.9642076, 5.0252824
6: -2.4832761, 2.7030678, -2.6918461, 2.9379869, -5.4212627, 5.3949137
7: -3.0734477, 2.5547447, -3.3522999, 2.6825595, -5.7560067, 5.9070444
8: -3.0578136, 2.3457003, -3.3063493, 2.5259814, -5.5837951, 5.6520491
9: -2.5247838, 2.4823110, -2.7352579, 2.7065756, -5.2313595, 5.2175689

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1349898, upper bound: 11.1350824
time: 3.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1350947
time: 7.13 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.1326923, 1.9079105, -2.8904681, 2.5370262, -4.6697178, 4.7983780
1: -1.6663493, 1.7108555, -2.2164125, 2.2407041, -3.9070535, 3.9272676
2: -2.1478918, 1.7539401, -2.9647920, 2.2256269, -4.3735189, 4.7187319
3: -2.1895509, 1.4972043, -3.1275761, 1.9115132, -4.1010637, 4.6247797
4: -2.2870698, 2.0558407, -3.1573172, 2.6722434, -4.9593124, 5.2131577
5: -1.9580733, 1.9308828, -2.7389550, 2.4701715, -4.4282436, 4.6698380
6: -2.0141110, 2.2360489, -2.6918461, 2.9379869, -4.9520979, 4.9278946
7: -2.4880722, 2.5578649, -3.3522999, 2.6825595, -5.1706314, 5.9101648
8: -2.4395995, 1.9505744, -3.3063493, 2.5259814, -4.9655800, 5.2569237
9: -2.0562296, 2.0220921, -2.7352579, 2.7065756, -4.7628050, 4.7573500

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1349898, upper bound: 11.1353724
time: 10.29 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1350057, upper bound: 11.1353949
time: 2.37 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.6521246, 2.3263724, -4.2981853, 3.7052093, -6.3573337, 6.6245565
1: -2.0446608, 2.0809975, -3.3431563, 3.2473710, -5.2920318, 5.4241533
2: -2.7097640, 2.0814281, -4.4457159, 3.0796194, -5.7893829, 6.5271425
3: -2.8495383, 1.7706244, -4.8790359, 2.6806717, -5.5302100, 6.6496601
4: -2.8642821, 2.4896882, -4.8108969, 3.8703141, -6.7345953, 7.3005838
5: -2.4940369, 2.2863274, -4.1487627, 3.4544172, -5.9484539, 6.4350901
6: -2.4832761, 2.7030678, -3.9526963, 4.2620564, -6.7453313, 6.6557636
7: -3.0734477, 2.5547447, -4.9438572, 3.1645565, -6.2380042, 7.4986019
8: -3.0578136, 2.3457003, -4.8809605, 3.6170545, -6.6748681, 7.2266607
9: -2.5247838, 2.4823110, -3.9706798, 3.9866600, -6.5114441, 6.4529901

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301124, upper bound: 11.1306928
time: 2.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1297063, upper bound: 11.1301266
time: 2.52 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.1326923, 1.9079105, -4.2981853, 3.7052093, -5.8379016, 6.2060947
1: -1.6663493, 1.7108555, -3.3431563, 3.2473710, -4.9137201, 5.0540118
2: -2.1478918, 1.7539401, -4.4457159, 3.0796194, -5.2275114, 6.1996555
3: -2.1895509, 1.4972043, -4.8790359, 2.6806717, -4.8702226, 6.3762398
4: -2.2870698, 2.0558407, -4.8108969, 3.8703141, -6.1573820, 6.8667374
5: -1.9580733, 1.9308828, -4.1487627, 3.4544172, -5.4124904, 6.0796456
6: -2.0141110, 2.2360489, -3.9526963, 4.2620564, -6.2761674, 6.1887455
7: -2.4880722, 2.5578649, -4.9438572, 3.1645565, -5.6526289, 7.5017223
8: -2.4395995, 1.9505744, -4.8809605, 3.6170545, -6.0566540, 6.8315349
9: -2.0562296, 2.0220921, -3.9706798, 3.9866600, -6.0428896, 5.9927716

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301124, upper bound: 11.1314251
time: 3.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1297063, upper bound: 11.1307708
time: 2.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.0700850, 1.8522511, -3.7827957, 3.2951903, -5.3652754, 5.6350465
1: -1.6128281, 1.6475128, -2.9017353, 2.8861072, -4.4989352, 4.5492482
2: -2.0605006, 1.6930301, -3.9047019, 2.8095207, -4.8700213, 5.5977321
3: -2.1187716, 1.4356531, -4.2250972, 2.4162335, -4.5350051, 5.6607504
4: -2.2105248, 2.0083089, -4.2324066, 3.4114237, -5.6219482, 6.2407146
5: -1.8896530, 1.8870926, -3.6810448, 3.1263404, -5.0159931, 5.5681372
6: -1.9458218, 2.1700959, -3.5086136, 3.7476497, -5.6934714, 5.6787095
7: -2.3862846, 2.4634528, -4.3354111, 2.9468405, -5.3331251, 6.7988639
8: -2.3513331, 1.8971951, -4.3249674, 3.2198422, -5.5711756, 6.2221622
9: -1.9916382, 1.9473342, -3.5369365, 3.5244958, -5.5161343, 5.4842706

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1307044, upper bound: 11.1302095
time: 4.09 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1307044, upper bound: 11.1302095
time: 3.17 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.6114042, 1.5128739, -3.0226240, 2.6527181, -4.2641220, 4.5354977
1: -1.2857106, 1.3230542, -2.3081326, 2.3331451, -3.6188555, 3.6311870
2: -1.5815821, 1.4615809, -3.0986924, 2.3149395, -3.8965216, 4.5602732
3: -1.6023346, 1.1864302, -3.2969232, 1.9700270, -3.5723615, 4.4833536
4: -1.7114117, 1.6385217, -3.3432546, 2.7864854, -4.4978971, 4.9817762
5: -1.4540634, 1.6271017, -2.8947253, 2.5874727, -4.0415354, 4.5218267
6: -1.5434947, 1.7480379, -2.8080602, 3.0468917, -4.5903864, 4.5560970
7: -1.8312027, 2.4597728, -3.4776754, 2.6163151, -4.4475179, 5.9374485
8: -1.8199761, 1.5858837, -3.4561696, 2.6312642, -4.4512405, 5.0420527
9: -1.6365480, 1.5633770, -2.8626356, 2.8272305, -4.4637785, 4.4260120

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301272, upper bound: 11.1298693
time: 3.09 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301272, upper bound: 11.1298693
time: 3.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.0700850, 1.8522511, -5.3979197, 4.6199059, -6.6899910, 7.2501702
1: -1.6128281, 1.6475128, -4.2802105, 4.0224833, -5.6353111, 5.9277229
2: -2.0605006, 1.6930301, -5.5737529, 3.7878189, -5.8483195, 7.2667828
3: -2.1187716, 1.4356531, -6.2227554, 3.2937224, -5.4124937, 7.6584072
4: -2.2105248, 2.0083089, -6.0977616, 4.7883520, -6.9988761, 8.1060677
5: -1.8896530, 1.8870926, -5.2622852, 4.2563267, -6.1459780, 7.1493759
6: -1.9458218, 2.1700959, -4.9418936, 5.2532406, -7.1990614, 7.1119890
7: -2.3862846, 2.4634528, -6.1158113, 3.9670670, -6.3533506, 8.5792637
8: -2.3513331, 1.8971951, -6.1192255, 4.4632816, -6.8146148, 8.0164204
9: -1.9916382, 1.9473342, -4.9368329, 4.9943705, -6.9860086, 6.8841667

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1316047, upper bound: 11.1313913
time: 4.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1316047, upper bound: 11.1313913
time: 4.96 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.6114042, 1.5128739, -4.5745220, 3.9338031, -5.5452070, 6.0873957
1: -1.2857106, 1.3230542, -3.5709195, 3.4371743, -4.7228851, 4.8939738
2: -1.5815821, 1.4615809, -4.7199326, 3.2489336, -4.8305159, 6.1815133
3: -1.6023346, 1.1864302, -5.2284555, 2.8159585, -4.4182930, 6.4148855
4: -1.7114117, 1.6385217, -5.1562271, 4.1088471, -5.8202591, 6.7947488
5: -1.4540634, 1.6271017, -4.4332037, 3.6677465, -5.1218090, 6.0603056
6: -1.5434947, 1.7480379, -4.1961656, 4.5033216, -6.0468163, 5.9442024
7: -1.8312027, 2.4597728, -5.2191105, 3.3495448, -5.1807475, 7.6788836
8: -1.8199761, 1.5858837, -5.1879139, 3.8313441, -5.6513205, 6.7737975
9: -1.6365480, 1.5633770, -4.2188644, 4.2364783, -5.8730264, 5.7822409

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1310252, upper bound: 11.1309918
time: 5.24 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1310252, upper bound: 11.1309918
time: 2.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.4337096, 2.1277008, -3.5138817, 3.0617387, -5.4954481, 5.6415820
1: -1.8655181, 1.8994603, -2.6717944, 2.6917269, -4.5572453, 4.5712547
2: -2.4388204, 1.8728986, -3.6183810, 2.6337700, -5.0725904, 5.4912796
3: -2.5798173, 1.6143669, -3.8944695, 2.2587490, -4.8385658, 5.5088363
4: -2.5871410, 2.3145995, -3.9174998, 3.1869752, -5.7741165, 6.2320995
5: -2.2303979, 2.1095214, -3.3959155, 2.9274077, -5.1578054, 5.5054369
6: -2.2591159, 2.5077868, -3.2612433, 3.4957011, -5.7548170, 5.7690301
7: -2.7985439, 2.5124106, -4.0324888, 2.7605743, -5.5591183, 6.5448995
8: -2.7579620, 2.1711023, -4.0161891, 3.0084805, -5.7664423, 6.1872911
9: -2.3034930, 2.2565482, -3.2955339, 3.2711282, -5.5746212, 5.5520821

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301267, upper bound: 11.1298340
time: 3.43 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301267, upper bound: 11.1298340
time: 3.88 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.9503245, 1.7576987, -2.7700634, 2.4402616, -4.3905859, 4.5277619
1: -1.5236439, 1.5504494, -2.1255851, 2.1522102, -3.6758542, 3.6760345
2: -1.9224323, 1.6152831, -2.8286629, 2.1542816, -4.0767140, 4.4439459
3: -1.9729252, 1.3493168, -2.9859624, 1.8277439, -3.8006692, 4.3352785
4: -2.0549929, 1.9139297, -3.0444262, 2.5816033, -4.6365962, 4.9583559
5: -1.7646469, 1.8154390, -2.6265635, 2.4051752, -4.1698222, 4.4420023
6: -1.8266568, 2.0617313, -2.5803752, 2.8121195, -4.6387763, 4.6421061
7: -2.2305408, 2.5078595, -3.1907668, 2.5579088, -4.7884493, 5.6986265
8: -2.1998677, 1.8151746, -3.1696167, 2.4358850, -4.6357527, 4.9847908
9: -1.8964636, 1.8362263, -2.6348908, 2.5960917, -4.4925556, 4.4711156

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1299331, upper bound: 11.1295872
time: 4.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301267, upper bound: 11.1298340
time: 15.30 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.4337096, 2.1277008, -5.2082410, 4.4620523, -6.8957620, 7.3359413
1: -1.8655181, 1.8994603, -4.1216650, 3.8923192, -5.7578363, 6.0211253
2: -2.4388204, 1.8728986, -5.3784113, 3.6640770, -6.1028972, 7.2513099
3: -2.5798173, 1.6143669, -5.9985175, 3.1861296, -5.7659469, 7.6128845
4: -2.5871410, 2.3145995, -5.8846011, 4.6335812, -7.2207222, 8.1991997
5: -2.2303979, 2.1095214, -5.0687575, 4.1143880, -6.3447857, 7.1782789
6: -2.2591159, 2.5077868, -4.7733307, 5.0850210, -7.3441362, 7.2811174
7: -2.7985439, 2.5124106, -5.9147191, 3.8171613, -6.6157055, 8.4271297
8: -2.7579620, 2.1711023, -5.9052696, 4.3197432, -7.0777030, 8.0763712
9: -2.3034930, 2.2565482, -4.7715797, 4.8213925, -7.1248856, 7.0281277

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1310188, upper bound: 11.1309756
time: 3.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1310188, upper bound: 11.1309755
time: 17.98 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.9503245, 1.7576987, -4.4102726, 3.7907891, -5.7411137, 6.1679711
1: -1.5236439, 1.5504494, -3.4269993, 3.3185661, -4.8422098, 4.9774485
2: -1.9224323, 1.6152831, -4.5463576, 3.1358671, -5.0582995, 6.1616406
3: -1.9729252, 1.3493168, -5.0239162, 2.7183449, -4.6912699, 6.3732319
4: -2.0549929, 1.9139297, -4.9630480, 3.9715970, -6.0265899, 6.8769779
5: -1.7646469, 1.8154390, -4.2567472, 3.5445232, -5.3091702, 6.0721865
6: -1.8266568, 2.0617313, -4.0422916, 4.3505669, -6.1772237, 6.1040230
7: -2.2305408, 2.5078595, -5.0347905, 3.2138023, -5.4443431, 7.5426502
8: -2.1998677, 1.8151746, -4.9970236, 3.7026587, -5.9025264, 6.8121977
9: -1.8964636, 1.8362263, -4.0689211, 4.0793591, -5.9758224, 5.9051466

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1310159, upper bound: 11.1309604
time: 3.34 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1310159, upper bound: 11.1309753
time: 3.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.8747616, 3.3514919, -4.2712283, 3.6816120, -7.5563736, 7.6227193
1: -2.9832888, 2.9466987, -3.3281317, 3.2267680, -6.2100568, 6.2748294
2: -3.9848781, 2.8379266, -4.4086599, 3.0750651, -7.0599432, 7.2465858
3: -4.3526115, 2.4407072, -4.8393097, 2.6687350, -7.0213466, 7.2800169
4: -4.3472972, 3.5092354, -4.7764378, 3.8421800, -8.1894770, 8.2856722
5: -3.7256112, 3.1623192, -4.1294413, 3.4384525, -7.1640635, 7.2917604
6: -3.5783389, 3.8379726, -3.9331429, 4.2283592, -7.8066978, 7.7711153
7: -4.4410048, 2.9062433, -4.9061098, 3.1619222, -7.6029272, 7.8123531
8: -4.4102068, 3.2816570, -4.8481131, 3.5975885, -8.0077953, 8.1297703
9: -3.6032693, 3.5899425, -3.9492452, 3.9595709, -7.5628386, 7.5391874

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1356592, upper bound: 11.1355324
time: 2.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1356592, upper bound: 11.1360135
time: 3.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.4817257, 3.8408513, -4.2712283, 3.6816120, -8.1633358, 8.1120796
1: -3.5062194, 3.3774667, -3.3281317, 3.2267680, -6.7329874, 6.7055984
2: -4.6227469, 3.1893656, -4.4086599, 3.0750651, -7.6978121, 7.5980225
3: -5.1075735, 2.7668097, -4.8393097, 2.6687350, -7.7763085, 7.6061182
4: -5.0369434, 4.0300832, -4.7764378, 3.8421800, -8.8791237, 8.8065205
5: -4.3113694, 3.5787861, -4.1294413, 3.4384525, -7.7498217, 7.7082272
6: -4.1136346, 4.4184623, -3.9331429, 4.2283592, -8.3419933, 8.3516054
7: -5.1229801, 3.2661667, -4.9061098, 3.1619222, -8.2849007, 8.1722755
8: -5.0814462, 3.7569907, -4.8481131, 3.5975885, -8.6790352, 8.6051035
9: -4.1261520, 4.1449056, -3.9492452, 3.9595709, -8.0857210, 8.0941505

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1356592, upper bound: 11.1355324
time: 3.03 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1356592, upper bound: 11.1360135
time: 3.20 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9136858, 3.3567977, -4.5964632, 3.9319887, -7.8456740, 7.9532609
1: -2.9918089, 2.9533749, -3.6104903, 3.4569182, -6.4487271, 6.5638633
2: -4.0150146, 2.7921395, -4.7455592, 3.2553487, -7.2703614, 7.5376987
3: -4.4199438, 2.4208252, -5.2418280, 2.8373833, -7.2573266, 7.6626530
4: -4.3703680, 3.5556159, -5.1363916, 4.1237707, -8.4941387, 8.6920061
5: -3.7406230, 3.1886971, -4.4285460, 3.6536789, -7.3943005, 7.6172404
6: -3.5738804, 3.8846557, -4.2134600, 4.5397835, -8.1136637, 8.0981159
7: -4.4690466, 2.8218193, -5.2686110, 3.3387711, -7.8078175, 8.0904293
8: -4.4216461, 3.3121078, -5.2005234, 3.8505926, -8.2722387, 8.5126305
9: -3.6216877, 3.6108351, -4.2241516, 4.2514286, -7.8731165, 7.8349867

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1310392, upper bound: 11.1304986
time: 4.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1310392, upper bound: 11.1316592
time: 3.29 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.4084721, 2.9298792, -3.8070891, 3.2567265, -6.6651983, 6.7369657
1: -2.5643389, 2.5861664, -2.9189150, 2.8805318, -5.4448709, 5.5050812
2: -3.4790292, 2.4481630, -3.9089816, 2.7257905, -6.2048197, 6.3571444
3: -3.8028722, 2.1184878, -4.2757893, 2.3708816, -6.1737537, 6.3942771
4: -3.7648377, 3.1512618, -4.2124457, 3.4678023, -7.2326403, 7.3637075
5: -3.2247710, 2.8339119, -3.6134417, 3.0895782, -6.3143492, 6.4473515
6: -3.1180856, 3.4133804, -3.4806223, 3.8009605, -6.9190454, 6.8940015
7: -3.9101517, 2.5908709, -4.3762054, 2.7660365, -6.6761880, 6.9670763
8: -3.8468966, 2.9187708, -4.2963667, 3.2321587, -7.0790544, 7.2151375
9: -3.1847816, 3.1377277, -3.5212889, 3.5133979, -6.6981773, 6.6590166

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301328, upper bound: 11.1299388
time: 2.94 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1301328, upper bound: 11.1310082
time: 3.11 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.1394293, 2.7286744, -4.2712283, 3.6816120, -6.8210411, 6.9999027
1: -2.3855271, 2.4198594, -3.3281317, 3.2267680, -5.6122952, 5.7479911
2: -3.2197902, 2.3616676, -4.4086599, 3.0750651, -6.2948551, 6.7703276
3: -3.4455047, 2.0345397, -4.8393097, 2.6687350, -6.1142397, 6.8738489
4: -3.4599383, 2.8958287, -4.7764378, 3.8421800, -7.3021183, 7.6722660
5: -2.9655170, 2.6283078, -4.1294413, 3.4384525, -6.4039693, 6.7577486
6: -2.9111078, 3.1691720, -3.9331429, 4.2283592, -7.1394668, 7.1023149
7: -3.6372988, 2.6873622, -4.9061098, 3.1619222, -6.7992196, 7.5934715
8: -3.5780981, 2.7142856, -4.8481131, 3.5975885, -7.1756868, 7.5623989
9: -2.9477859, 2.9182186, -3.9492452, 3.9595709, -6.9073534, 6.8674641

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359009, upper bound: 11.1357811
time: 3.18 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359009, upper bound: 11.1362192
time: 3.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5399783, 3.0403802, -4.2712283, 3.6816120, -7.2215900, 7.3116083
1: -2.6985719, 2.7019701, -3.3281317, 3.2267680, -5.9253397, 6.0301018
2: -3.6372669, 2.5806541, -4.4086599, 3.0750651, -6.7123318, 6.9893131
3: -3.9433038, 2.2413454, -4.8393097, 2.6687350, -6.6120377, 7.0806541
4: -3.9097598, 3.2388346, -4.7764378, 3.8421800, -7.7519388, 8.0152712
5: -3.3418965, 2.8948724, -4.1294413, 3.4384525, -6.7803473, 7.0243139
6: -3.2559361, 3.5484056, -3.9331429, 4.2283592, -7.4842954, 7.4815483
7: -4.0877209, 2.7517262, -4.9061098, 3.1619222, -7.2496433, 7.6578350
8: -4.0142646, 3.0209842, -4.8481131, 3.5975885, -7.6118526, 7.8690972
9: -3.2888074, 3.2739127, -3.9492452, 3.9595709, -7.2483759, 7.2231569

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 215

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359009, upper bound: 11.1357811
time: 3.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359009, upper bound: 11.1362192
time: 3.52 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 7.96 + 598.26 = 606.22 seconds
