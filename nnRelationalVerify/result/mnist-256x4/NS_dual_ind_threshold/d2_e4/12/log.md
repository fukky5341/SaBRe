## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 16.772132178899998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.0813789, 11.2827454, -15.0813789, 11.2827454, -26.3641243, 26.3641205)
1: (-11.9968710, 9.8736877, -11.9968710, 9.8736877, -21.8705597, 21.8705597)
2: (-20.3788204, 6.2278857, -20.3788204, 6.2278857, -26.6067066, 26.6067066)
3: (-17.6044922, 8.0591736, -17.6044922, 8.0591736, -25.6636658, 25.6636639)
4: (-17.5211048, 10.9149046, -17.5211048, 10.9149046, -28.4360085, 28.4360085)
5: (-13.4232674, 11.1254854, -13.4232674, 11.1254854, -24.5487480, 24.5487499)
6: (-14.3546829, 12.1642437, -14.3546829, 12.1642437, -26.5189266, 26.5189247)
7: (-15.6885843, 11.7738943, -15.6885843, 11.7738943, -27.4624786, 27.4624786)
8: (-18.0344696, 10.8528175, -18.0344696, 10.8528175, -28.8872871, 28.8872871)
9: (-12.8600473, 14.7708082, -12.8600473, 14.7708082, -27.6308556, 27.6308556)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.00 + 6.09 = 8.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -16.7889211, upper bound: 16.7889210

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 97

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7886564, upper bound: 16.7884349
time: 3.47 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7884231, upper bound: 16.7884231
time: 2.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.48 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.48
Output dim: 2, lower bound: -16.7886564, upper bound: 16.7884349
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.48
Output dim: 2, lower bound: -16.7884231, upper bound: 16.7884231

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -13.1813297, 9.8878269, -14.5231848, 10.8715200, -24.0528488, 24.4110088
1: -10.4467020, 8.6383743, -11.5422430, 9.5096140, -19.9563160, 20.1806164
2: -17.8418465, 5.4266028, -19.6355267, 5.9911404, -23.8329868, 25.0621262
3: -15.3386250, 7.0623465, -16.9398727, 7.7664104, -23.1050358, 24.0022144
4: -15.3066406, 9.5693970, -16.8722019, 10.5172749, -25.8239155, 26.4415970
5: -11.7473125, 9.7261295, -12.9308081, 10.7148294, -22.4621391, 22.6569366
6: -12.5676880, 10.6401472, -13.8305483, 11.7158251, -24.2835121, 24.4706955
7: -13.6896210, 10.3107548, -15.1026649, 11.3440180, -25.0336380, 25.4134197
8: -15.7598648, 9.5340424, -17.3674870, 10.4633541, -26.2232189, 26.9015293
9: -11.2322197, 12.9281416, -12.3819199, 14.2294397, -25.4616566, 25.3100586

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879223
time: 4.14 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880186, upper bound: 16.7878156
time: 6.41 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13.5456219, 10.1642570, -14.1408577, 10.5926952, -24.1383171, 24.3051147
1: -10.7396545, 8.8749199, -11.2289143, 9.2587910, -19.9984436, 20.1038342
2: -18.3422832, 5.5906639, -19.1293602, 5.8180499, -24.1603336, 24.7200241
3: -15.7648544, 7.2581749, -16.4822273, 7.5615859, -23.3264389, 23.7404022
4: -15.7280169, 9.8288593, -16.4284554, 10.2467251, -25.9747429, 26.2573147
5: -12.0697851, 9.9976034, -12.5947800, 10.4329176, -22.5027008, 22.5923824
6: -12.9204655, 10.9304743, -13.4727783, 11.4083433, -24.3288078, 24.4032516
7: -14.0692215, 10.5971327, -14.7003736, 11.0465984, -25.1158180, 25.2975063
8: -16.2042503, 9.8035870, -16.9085236, 10.1969805, -26.4012280, 26.7121105
9: -11.5464306, 13.2845898, -12.0518293, 13.8622561, -25.4086857, 25.3364182

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878140, upper bound: 16.7879126
time: 3.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878023, upper bound: 16.7878018
time: 2.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.71 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.71
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879223
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.71
Output dim: 2, lower bound: -16.7880186, upper bound: 16.7878156
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.71
Output dim: 2, lower bound: -16.7878140, upper bound: 16.7879126
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.71
Output dim: 2, lower bound: -16.7878023, upper bound: 16.7878018

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -12.9533577, 9.7210979, -13.0021191, 9.7540455, -22.7074032, 22.7232151
1: -10.2637882, 8.4924469, -10.3056040, 8.5268288, -18.7906151, 18.7980499
2: -17.5397243, 5.3227978, -17.6089287, 5.3059487, -22.8456726, 22.9317245
3: -15.0693455, 6.9418211, -15.1352768, 6.9702177, -22.0395584, 22.0770969
4: -15.0416155, 9.4103909, -15.0999517, 9.4441795, -24.4857941, 24.5103416
5: -11.5469456, 9.5586643, -11.5893860, 9.5928917, -21.1398373, 21.1480503
6: -12.3536606, 10.4569092, -12.3969183, 10.4899788, -22.8436394, 22.8538284
7: -13.4465284, 10.1345997, -13.4850283, 10.1615591, -23.6080856, 23.6196251
8: -15.4856834, 9.3768511, -15.5380650, 9.4085217, -24.8942032, 24.9149151
9: -11.0370569, 12.7092447, -11.0745487, 12.7586126, -23.7956696, 23.7837944

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880185, upper bound: 16.7878153
time: 2.62 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880185, upper bound: 16.7878158
time: 4.62 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -12.2945976, 9.2346992, -13.9764996, 10.4827271, -22.7773247, 23.2111950
1: -9.7333279, 8.0689240, -11.0962582, 9.1552963, -18.8886242, 19.1651802
2: -16.6497993, 5.0359879, -18.9607658, 5.7084284, -22.3582268, 23.9967537
3: -14.2813063, 6.5998201, -16.2876148, 7.4766431, -21.7579498, 22.8874321
4: -14.2714863, 8.9457893, -16.2373085, 10.1388769, -24.4103622, 25.1830978
5: -10.9656143, 9.0711918, -12.4540586, 10.3147392, -21.2803535, 21.5252495
6: -11.7289448, 9.9266768, -13.3310509, 11.2670555, -22.9960003, 23.2577286
7: -12.7508774, 9.6254730, -14.5113211, 10.9180384, -23.6689148, 24.1367874
8: -14.6906843, 8.9169827, -16.7139740, 10.1086035, -24.7992878, 25.6309547
9: -10.4738407, 12.0659389, -11.9044523, 13.7190037, -24.1928444, 23.9703903

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874797, upper bound: 16.7873601
time: 3.98 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875761, upper bound: 16.7873879
time: 4.47 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13.3112240, 9.9917231, -12.6421423, 9.4886961, -22.7999191, 22.6338654
1: -10.5526667, 8.7257738, -10.0148163, 8.2939672, -18.8466301, 18.7405891
2: -18.0332260, 5.4839253, -17.1226807, 5.1543851, -23.1876106, 22.6066055
3: -15.4904242, 7.1338325, -14.7011881, 6.7810884, -22.2715111, 21.8350201
4: -15.4571466, 9.6665668, -14.6789970, 9.1896610, -24.6468086, 24.3455639
5: -11.8642988, 9.8249750, -11.2714500, 9.3269196, -21.1912174, 21.0964241
6: -12.7014627, 10.7426825, -12.0562067, 10.2013292, -22.9027901, 22.7988873
7: -13.8200569, 10.4153528, -13.1056566, 9.8843613, -23.7044125, 23.5210094
8: -15.9231224, 9.6428823, -15.1063776, 9.1568537, -25.0799751, 24.7492599
9: -11.3463793, 13.0603161, -10.7667294, 12.4058867, -23.7522602, 23.8270454

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878023, upper bound: 16.7878021
time: 2.61 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878023, upper bound: 16.7878019
time: 2.53 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.6525908, 9.5050402, -13.5218973, 10.1473560, -22.7999458, 23.0269375
1: -10.0232086, 8.3026962, -10.7291155, 8.8613777, -18.8845863, 19.0318108
2: -17.1482506, 5.1964030, -18.3462009, 5.5195904, -22.6678410, 23.5426044
3: -14.7053633, 6.7876945, -15.7421169, 7.2384768, -21.9438400, 22.5298119
4: -14.6897678, 9.2034969, -15.7057934, 9.8175192, -24.5072861, 24.9092903
5: -11.2833729, 9.3384247, -12.0524340, 9.9791632, -21.2625351, 21.3908558
6: -12.0787325, 10.2136717, -12.8994217, 10.9032688, -22.9820023, 23.1130924
7: -13.1237249, 9.9065590, -14.0334330, 10.5693083, -23.6930313, 23.9399910
8: -15.1303730, 9.1836262, -16.1670647, 9.7901812, -24.9205551, 25.3506908
9: -10.7835684, 12.4186878, -11.5166540, 13.2726870, -24.0562553, 23.9353390

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872547, upper bound: 16.7873439
time: 3.39 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873762, upper bound: 16.7873759
time: 3.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 9.27 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7880185, upper bound: 16.7878153
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7880185, upper bound: 16.7878158
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7874797, upper bound: 16.7873601
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7875761, upper bound: 16.7873879
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7878023, upper bound: 16.7878021
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7878023, upper bound: 16.7878019
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7872547, upper bound: 16.7873439
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 9.27
Output dim: 2, lower bound: -16.7873762, upper bound: 16.7873759

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.7300854, 8.8148861, -13.0021191, 9.7540455, -21.4841270, 21.8170052
1: -9.2753878, 7.7086868, -10.3056040, 8.5268288, -17.8022118, 18.0142899
2: -15.8883123, 4.7938633, -17.6089287, 5.3059487, -21.1942616, 22.4027901
3: -13.6119719, 6.3095388, -15.1352768, 6.9702177, -20.5821896, 21.4448147
4: -13.6088190, 8.5450840, -15.0999517, 9.4441795, -23.0529976, 23.6450348
5: -10.4625282, 8.6544476, -11.5893860, 9.5928917, -20.0554161, 20.2438335
6: -11.1897135, 9.4724636, -12.3969183, 10.4899788, -21.6796913, 21.8693810
7: -12.1446791, 9.1887398, -13.4850283, 10.1615591, -22.3062363, 22.6737633
8: -14.0139036, 8.5245228, -15.5380650, 9.4085217, -23.4224243, 24.0625877
9: -9.9907265, 11.5104055, -11.0745487, 12.7586126, -22.7493343, 22.5849533

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879222
time: 3.06 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879224
time: 3.94 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.6236229, 9.4836922, -13.0021191, 9.7540455, -22.3776665, 22.4858093
1: -10.0006857, 8.2841740, -10.3056040, 8.5268288, -18.5275116, 18.5897789
2: -17.1298695, 5.1659675, -17.6089287, 5.3059487, -22.4358177, 22.7748947
3: -14.6703176, 6.7739773, -15.1352768, 6.9702177, -21.6405354, 21.9092522
4: -14.6523581, 9.1825809, -15.0999517, 9.4441795, -24.0965385, 24.2825317
5: -11.2555904, 9.3172874, -11.5893860, 9.5928917, -20.8484783, 20.9066734
6: -12.0461922, 10.1855412, -12.3969183, 10.4899788, -22.5361710, 22.5824585
7: -13.0864277, 9.8823919, -13.4850283, 10.1615591, -23.2479820, 23.3674164
8: -15.0923710, 9.1683226, -15.5380650, 9.4085217, -24.5008926, 24.7063866
9: -10.7525148, 12.3903303, -11.0745487, 12.7586126, -23.5111256, 23.4648781

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879222
time: 3.44 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879224
time: 3.94 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.9692307, 8.2473040, -13.6176090, 10.2169762, -21.1862068, 21.8649120
1: -8.6555157, 7.2189274, -10.8060093, 8.9246492, -17.5801659, 18.0249367
2: -14.8558550, 4.4678316, -18.4767685, 5.5537229, -20.4095745, 22.9445992
3: -12.6996489, 5.9104276, -15.8583841, 7.2903461, -19.9899940, 21.7688103
4: -12.7146463, 8.0040607, -15.8162460, 9.8843403, -22.5989838, 23.8203049
5: -9.7849255, 8.0907040, -12.1358738, 10.0494328, -19.8343582, 20.2265778
6: -10.4614553, 8.8611507, -12.9900160, 10.9780769, -21.4395332, 21.8511658
7: -11.3365431, 8.6002254, -14.1286716, 10.6402740, -21.9768162, 22.7288971
8: -13.0971718, 7.9881897, -16.2832642, 9.8580532, -22.9552231, 24.2714500
9: -9.3376408, 10.7598066, -11.5966415, 13.3663387, -22.7039795, 22.3564453

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874379, upper bound: 16.7873354
time: 3.39 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874361, upper bound: 16.7872951
time: 3.00 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.6193314, 8.7346478, -13.3745604, 10.0366688, -21.6559982, 22.1092072
1: -9.1805401, 7.6353459, -10.6081219, 8.7669621, -17.9475021, 18.2434673
2: -15.7515182, 4.7518239, -18.1460953, 5.4528470, -21.2043648, 22.8979187
3: -13.4608078, 6.2423420, -15.5640841, 7.1609120, -20.6217194, 21.8064270
4: -13.4730129, 8.4687901, -15.5310545, 9.7121572, -23.1851692, 23.9998436
5: -10.3637915, 8.5711966, -11.9208508, 9.8690662, -20.2328568, 20.4920464
6: -11.0889511, 9.3819504, -12.7590675, 10.7834349, -21.8723869, 22.1410179
7: -12.0302067, 9.1070509, -13.8740454, 10.4533167, -22.4835243, 22.9810925
8: -13.8859749, 8.4551115, -15.9910259, 9.6863613, -23.5723362, 24.4461365
9: -9.8912859, 11.4002342, -11.3880186, 13.1269970, -23.0182838, 22.7882538

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875046, upper bound: 16.7873586
time: 6.59 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875024, upper bound: 16.7873190
time: 4.26 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.0772381, 9.0770397, -12.6421423, 9.4886961, -21.5659332, 21.7191811
1: -9.5562763, 7.9356146, -10.0148163, 8.2939672, -17.8502388, 17.9504299
2: -16.3716736, 4.9510994, -17.1226807, 5.1543851, -21.5260563, 22.0737801
3: -14.0229406, 6.4921365, -14.7011881, 6.7810884, -20.8040276, 21.1933250
4: -14.0138779, 8.7949123, -14.6789970, 9.1896610, -23.2035389, 23.4739094
5: -10.7702284, 8.9136171, -11.2714500, 9.3269196, -20.0971489, 20.1850662
6: -11.5296602, 9.7507792, -12.0562067, 10.2013292, -21.7309875, 21.8069839
7: -12.5054417, 9.4604807, -13.1056566, 9.8843613, -22.3898029, 22.5661373
8: -14.4423056, 8.7842598, -15.1063776, 9.1568537, -23.5991554, 23.8906364
9: -10.2909136, 11.8526239, -10.7667294, 12.4058867, -22.6968002, 22.6193523

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873647, upper bound: 16.7873621
time: 4.05 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873831, upper bound: 16.7874517
time: 5.53 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.9314556, 9.7166882, -12.6421423, 9.4886961, -22.4201508, 22.3588295
1: -10.2499132, 8.4861364, -10.0148163, 8.2939672, -18.5438805, 18.5009518
2: -17.5610466, 5.3049002, -17.1226807, 5.1543851, -22.7154274, 22.4275818
3: -15.0350924, 6.9357486, -14.7011881, 6.7810884, -21.8161774, 21.6369362
4: -15.0119295, 9.4046679, -14.6789970, 9.1896610, -24.2015877, 24.0836620
5: -11.5284538, 9.5472984, -11.2714500, 9.3269196, -20.8553734, 20.8187485
6: -12.3478422, 10.4324865, -12.0562067, 10.2013292, -22.5491714, 22.4886894
7: -13.4062319, 10.1241331, -13.1056566, 9.8843613, -23.2905922, 23.2297897
8: -15.4716148, 9.3997459, -15.1063776, 9.1568537, -24.6284676, 24.5061226
9: -11.0192289, 12.6937790, -10.7667294, 12.4058867, -23.4251118, 23.4605064

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873647, upper bound: 16.7873622
time: 3.97 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873831, upper bound: 16.7874517
time: 16.59 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.3141346, 8.5082016, -13.1675196, 9.8838224, -21.1979561, 21.6757202
1: -8.9351692, 7.4447608, -10.4414864, 8.6337681, -17.5689373, 17.8862476
2: -15.3368931, 4.6240921, -17.8669891, 5.3671589, -20.7040520, 22.4910812
3: -13.1090126, 6.0919633, -15.3198166, 7.0541940, -20.1632061, 21.4117794
4: -13.1180639, 8.2526989, -15.2902012, 9.5658846, -22.6839466, 23.5429001
5: -10.0908203, 8.3485203, -11.7372227, 9.7172070, -19.8080273, 20.0857410
6: -10.7998762, 9.1381531, -12.5611305, 10.6183109, -21.4181862, 21.6992836
7: -11.6957102, 8.8706179, -13.6549559, 10.2943926, -21.9901028, 22.5255718
8: -13.5245447, 8.2464895, -15.7416401, 9.5421772, -23.0667229, 23.9881248
9: -9.6358404, 11.1007061, -11.2127447, 12.9236460, -22.5594845, 22.3134499

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872084, upper bound: 16.7873185
time: 2.78 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872062, upper bound: 16.7872804
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.9651995, 8.9963102, -12.9379320, 9.7132053, -21.6784058, 21.9342384
1: -9.4607172, 7.8615689, -10.2541943, 8.4847193, -17.9454308, 18.1157627
2: -16.2339439, 4.9091148, -17.5540390, 5.2720060, -21.5059509, 22.4631538
3: -13.8708124, 6.4241095, -15.0420256, 6.9316230, -20.8024349, 21.4661350
4: -13.8774099, 8.7178154, -15.0209074, 9.4031391, -23.2805481, 23.7387199
5: -10.6704626, 8.8297653, -11.5338860, 9.5468340, -20.2172966, 20.3636513
6: -11.4275217, 9.6594839, -12.3425770, 10.4345150, -21.8620338, 22.0020599
7: -12.3907166, 9.3784876, -13.4147758, 10.1176109, -22.5083256, 22.7932625
8: -14.3122740, 8.7139969, -15.4653387, 9.3795109, -23.6917839, 24.1793365
9: -10.1909952, 11.7415266, -11.0156298, 12.6974344, -22.8884277, 22.7571564

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873092, upper bound: 16.7873445
time: 3.55 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873072, upper bound: 16.7873072
time: 5.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 10.51 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879222
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879224
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879222
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7880319, upper bound: 16.7879224
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7874379, upper bound: 16.7873354
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7874361, upper bound: 16.7872951
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7875046, upper bound: 16.7873586
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7875024, upper bound: 16.7873190
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7873647, upper bound: 16.7873621
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7873831, upper bound: 16.7874517
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7873647, upper bound: 16.7873622
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7873831, upper bound: 16.7874517
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7872084, upper bound: 16.7873185
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7872062, upper bound: 16.7872804
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7873092, upper bound: 16.7873445
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.51
Output dim: 2, lower bound: -16.7873072, upper bound: 16.7873072

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.7300854, 8.8148861, -11.7300854, 8.8148861, -20.5449715, 20.5449715
1: -9.2753878, 7.7086868, -9.2753878, 7.7086868, -16.9840736, 16.9840736
2: -15.8883123, 4.7938633, -15.8883123, 4.7938633, -20.6821747, 20.6821747
3: -13.6119719, 6.3095388, -13.6119719, 6.3095388, -19.9215069, 19.9215088
4: -13.6088190, 8.5450840, -13.6088190, 8.5450840, -22.1539040, 22.1539001
5: -10.4625282, 8.6544476, -10.4625282, 8.6544476, -19.1169758, 19.1169758
6: -11.1897135, 9.4724636, -11.1897135, 9.4724636, -20.6621780, 20.6621780
7: -12.1446791, 9.1887398, -12.1446791, 9.1887398, -21.3334160, 21.3334160
8: -14.0139036, 8.5245228, -14.0139036, 8.5245228, -22.5384254, 22.5384254
9: -9.9907265, 11.5104055, -9.9907265, 11.5104055, -21.5011272, 21.5011292

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877262, upper bound: 16.7875486
time: 4.17 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877668, upper bound: 16.7875610
time: 3.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.7300854, 8.8148861, -12.0772381, 9.0770397, -20.8071251, 20.8921242
1: -9.2753878, 7.7086868, -9.5562763, 7.9356146, -17.2110023, 17.2649612
2: -15.8883123, 4.7938633, -16.3716736, 4.9510994, -20.8394127, 21.1655369
3: -13.6119719, 6.3095388, -14.0229406, 6.4921365, -20.1041088, 20.3324757
4: -13.6088190, 8.5450840, -14.0138779, 8.7949123, -22.4037285, 22.5589619
5: -10.4625282, 8.6544476, -10.7702284, 8.9136171, -19.3761444, 19.4246750
6: -11.1897135, 9.4724636, -11.5296602, 9.7507792, -20.9404926, 21.0021248
7: -12.1446791, 9.1887398, -12.5054417, 9.4604807, -21.6051598, 21.6941795
8: -14.0139036, 8.5245228, -14.4423056, 8.7842598, -22.7981644, 22.9668274
9: -9.9907265, 11.5104055, -10.2909136, 11.8526239, -21.8433475, 21.8013191

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 131

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877262, upper bound: 16.7875486
time: 3.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877668, upper bound: 16.7875611
time: 2.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.6236229, 9.4836922, -11.7300854, 8.8148861, -21.4385090, 21.2137756
1: -10.0006857, 8.2841740, -9.2753878, 7.7086868, -17.7093716, 17.5595627
2: -17.1298695, 5.1659675, -15.8883123, 4.7938633, -21.9237328, 21.0542793
3: -14.6703176, 6.7739773, -13.6119719, 6.3095388, -20.9798546, 20.3859482
4: -14.6523581, 9.1825809, -13.6088190, 8.5450840, -23.1974411, 22.7914009
5: -11.2555904, 9.3172874, -10.4625282, 8.6544476, -19.9100342, 19.7798157
6: -12.0461922, 10.1855412, -11.1897135, 9.4724636, -21.5186558, 21.3752556
7: -13.0864277, 9.8823919, -12.1446791, 9.1887398, -22.2751656, 22.0270691
8: -15.0923710, 9.1683226, -14.0139036, 8.5245228, -23.6168938, 23.1822262
9: -10.7525148, 12.3903303, -9.9907265, 11.5104055, -22.2629204, 22.3810539

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880061, upper bound: 16.7878725
time: 2.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7879871, upper bound: 16.7878686
time: 2.98 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.6236229, 9.4836922, -12.0772381, 9.0770397, -21.7006626, 21.5609303
1: -10.0006857, 8.2841740, -9.5562763, 7.9356146, -17.9363003, 17.8404503
2: -17.1298695, 5.1659675, -16.3716736, 4.9510994, -22.0809689, 21.5376396
3: -14.6703176, 6.7739773, -14.0229406, 6.4921365, -21.1624527, 20.7969170
4: -14.6523581, 9.1825809, -14.0138779, 8.7949123, -23.4472694, 23.1964588
5: -11.2555904, 9.3172874, -10.7702284, 8.9136171, -20.1692009, 20.0875168
6: -12.0461922, 10.1855412, -11.5296602, 9.7507792, -21.7969704, 21.7152023
7: -13.0864277, 9.8823919, -12.5054417, 9.4604807, -22.5469093, 22.3878326
8: -15.0923710, 9.1683226, -14.4423056, 8.7842598, -23.8766308, 23.6106262
9: -10.7525148, 12.3903303, -10.2909136, 11.8526239, -22.6051350, 22.6812439

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7880061, upper bound: 16.7878725
time: 4.35 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7879871, upper bound: 16.7878686
time: 5.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.7728958, 8.1008568, -13.1007071, 9.8321924, -20.6050873, 21.2015629
1: -8.4962120, 7.0923719, -10.3871069, 8.5905876, -17.0867996, 17.4794769
2: -14.5853004, 4.3850946, -17.7661915, 5.3368998, -19.9221992, 22.1512871
3: -12.4666166, 5.8080540, -15.2429962, 7.0201731, -19.4867897, 21.0510502
4: -12.4854631, 7.8645277, -15.2123642, 9.5172434, -22.0027046, 23.0768890
5: -9.6109648, 7.9455953, -11.6790161, 9.6669130, -19.2778740, 19.6246090
6: -10.2736378, 8.7047119, -12.4970722, 10.5654202, -20.8390579, 21.2017841
7: -11.1300154, 8.4484940, -13.5857668, 10.2414341, -21.3714485, 22.0342598
8: -12.8596783, 7.8476038, -15.6591644, 9.4884109, -22.3480873, 23.5067673
9: -9.1706409, 10.5664320, -11.1552439, 12.8576317, -22.0282726, 21.7216759

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873720, upper bound: 16.7872962
time: 3.21 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874379, upper bound: 16.7873355
time: 2.37 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.7795362, 8.1059132, -13.9321938, 10.4447088, -21.2242451, 22.0381012
1: -8.5016136, 7.0967159, -11.0579166, 9.1264963, -17.6281090, 18.1546326
2: -14.5947905, 4.3882842, -18.8924999, 5.6765509, -20.2713394, 23.2807827
3: -12.4745111, 5.8116398, -16.2407360, 7.4511747, -19.9256821, 22.0523739
4: -12.4931431, 7.8693037, -16.1868782, 10.1049557, -22.5980949, 24.0561829
5: -9.6168442, 7.9505663, -12.4117479, 10.2810020, -19.8978462, 20.3623123
6: -10.2800570, 8.7099953, -13.2825680, 11.2313461, -21.5114021, 21.9925632
7: -11.1370182, 8.4537954, -14.4591770, 10.8762960, -22.0133114, 22.9129696
8: -12.8678265, 7.8526506, -16.6577892, 10.0699673, -22.9377937, 24.5104370
9: -9.1763287, 10.5730190, -11.8650322, 13.6724329, -22.8487587, 22.4380512

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873694, upper bound: 16.7872644
time: 4.02 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874361, upper bound: 16.7872954
time: 2.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.4264307, 8.5908823, -12.8626165, 9.6552162, -21.0816460, 21.4534988
1: -9.0242281, 7.5110855, -10.1928692, 8.4360409, -17.4602680, 17.7039528
2: -15.4858179, 4.6706715, -17.4416523, 5.2383194, -20.7241364, 22.1123238
3: -13.2321758, 6.1418557, -14.9549103, 6.8932676, -20.1254425, 21.0967655
4: -13.2480612, 8.3318434, -14.9330101, 9.3484211, -22.5964813, 23.2648544
5: -10.1930923, 8.4286919, -11.4680233, 9.4902401, -19.6833324, 19.8967152
6: -10.9046164, 9.2284145, -12.2703695, 10.3747969, -21.2794132, 21.4987831
7: -11.8273964, 8.9581127, -13.3364639, 10.0579786, -21.8853741, 22.2945747
8: -13.6529131, 8.3170338, -15.3728294, 9.3199558, -22.9728642, 23.6898632
9: -9.7272263, 11.2104340, -10.9508333, 12.6231174, -22.3503437, 22.1612663

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874299, upper bound: 16.7873145
time: 5.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875046, upper bound: 16.7873584
time: 4.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -11.4309959, 8.5944376, -13.6777611, 10.2594280, -21.6904240, 22.2721977
1: -9.0279350, 7.5140834, -10.8543673, 8.9611530, -17.9890862, 18.3684502
2: -15.4925470, 4.6733146, -18.5512409, 5.5680876, -21.0606346, 23.2245560
3: -13.2375154, 6.1443744, -15.9276724, 7.3171492, -20.5546627, 22.0720444
4: -13.2532635, 8.3351707, -15.8884010, 9.9255209, -23.1787834, 24.2235718
5: -10.1971197, 8.4321318, -12.1897240, 10.0921650, -20.2892818, 20.6218529
6: -10.9090738, 9.2320290, -13.0457249, 11.0262718, -21.9353447, 22.2777538
7: -11.8322611, 8.9619064, -14.1935024, 10.6826019, -22.5148621, 23.1554070
8: -13.6585484, 8.3207359, -16.3524971, 9.8924084, -23.5509567, 24.6732292
9: -9.7311745, 11.2149563, -11.6463470, 13.4246712, -23.1558456, 22.8613014

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874280, upper bound: 16.7872847
time: 3.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875024, upper bound: 16.7873189
time: 7.34 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -11.7364120, 8.8230667, -11.3512030, 8.5279341, -20.2643471, 20.1742687
1: -9.2791481, 7.7171774, -8.9664574, 7.4658022, -16.7449493, 16.6836357
2: -15.9104481, 4.8051844, -15.3767538, 4.6010761, -20.5115242, 20.1819382
3: -13.6163902, 6.3149266, -13.1602802, 6.1102676, -19.7266541, 19.4752064
4: -13.6135235, 8.5527344, -13.1631212, 8.2729073, -21.8864288, 21.7158546
5: -10.4664221, 8.6614799, -10.1222782, 8.3722153, -18.8386383, 18.7837582
6: -11.2038746, 9.4768066, -10.8232288, 9.1635580, -20.3674316, 20.3000355
7: -12.1415949, 9.1966524, -11.7291241, 8.8866968, -21.0282917, 20.9257774
8: -14.0333939, 8.5454750, -13.5555916, 8.2526894, -22.2860832, 22.1010647
9: -9.9985218, 11.5169067, -9.6598463, 11.1351452, -21.1336670, 21.1767540

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875089, upper bound: 16.7874432
time: 2.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874690, upper bound: 16.7874394
time: 3.07 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -11.4353180, 8.5986242, -11.9012699, 8.9399357, -20.3752537, 20.4998932
1: -9.0335627, 7.5229340, -9.4102793, 7.8184853, -16.8520470, 16.9332123
2: -15.5011692, 4.6773448, -16.1374626, 4.8377652, -20.3389339, 20.8148060
3: -13.2537470, 6.1554594, -13.8032389, 6.3908300, -19.6445770, 19.9586945
4: -13.2600336, 8.3390093, -13.8042822, 8.6658983, -21.9259319, 22.1432915
5: -10.1986618, 8.4381456, -10.6111097, 8.7787476, -18.9774094, 19.0492554
6: -10.9164839, 9.2355595, -11.3537550, 9.6033878, -20.5198689, 20.5893135
7: -11.8240671, 8.9643250, -12.3138781, 9.3146648, -21.1387329, 21.2782021
8: -13.6717072, 8.3327770, -14.2224598, 8.6491699, -22.3208752, 22.5552349
9: -9.7398586, 11.2203045, -10.1279974, 11.6764069, -21.4162655, 21.3483009

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875173, upper bound: 16.7874838
time: 3.24 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874808, upper bound: 16.7874805
time: 3.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.5771599, 9.4526930, -11.3512030, 8.5279341, -21.1050949, 20.8038960
1: -9.9622469, 8.2585707, -8.9664574, 7.4658022, -17.4280472, 17.2250271
2: -17.0817184, 5.1519356, -15.3767538, 4.6010761, -21.6827946, 20.5286865
3: -14.6126966, 6.7517610, -13.1602802, 6.1102676, -20.7229652, 19.9120388
4: -14.5963421, 9.1527472, -13.1631212, 8.2729073, -22.8692455, 22.3158684
5: -11.2126646, 9.2854099, -10.1222782, 8.3722153, -19.5848789, 19.4076862
6: -12.0091696, 10.1475372, -10.8232288, 9.1635580, -21.1727276, 20.9707661
7: -13.0279999, 9.8488522, -11.7291241, 8.8866968, -21.9146957, 21.5779762
8: -15.0462379, 9.1516190, -13.5555916, 8.2526894, -23.2989235, 22.7072105
9: -10.7151814, 12.3450975, -9.6598463, 11.1351452, -21.8503265, 22.0049400

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873411, upper bound: 16.7873166
time: 3.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872992, upper bound: 16.7873140
time: 3.18 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.3473158, 9.2816467, -11.9012699, 8.9399357, -21.2872505, 21.1829166
1: -9.7744207, 8.1096783, -9.4102793, 7.8184853, -17.5929070, 17.5199547
2: -16.7680836, 5.0567889, -16.1374626, 4.8377652, -21.6058483, 21.1942520
3: -14.3344746, 6.6289682, -13.8032389, 6.3908300, -20.7253036, 20.4322052
4: -14.3264761, 8.9897194, -13.8042822, 8.6658983, -22.9923744, 22.7940025
5: -11.0087271, 9.1147423, -10.6111097, 8.7787476, -19.7874718, 19.7258511
6: -11.7902889, 9.9636488, -11.3537550, 9.6033878, -21.3936768, 21.3174019
7: -12.7875566, 9.6720734, -12.3138781, 9.3146648, -22.1022224, 21.9859505
8: -14.7700958, 8.9886684, -14.2224598, 8.6491699, -23.4192657, 23.2111282
9: -10.5177193, 12.1187468, -10.1279974, 11.6764069, -22.1941261, 22.2467442

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873549, upper bound: 16.7873941
time: 3.14 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873137, upper bound: 16.7873922
time: 2.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -11.1225109, 8.3648701, -12.6587057, 9.5046682, -20.6271782, 21.0235748
1: -8.7792988, 7.3210416, -10.0286961, 8.3049498, -17.0842476, 17.3497372
2: -15.0725536, 4.5428467, -17.1667862, 5.1539249, -20.2264786, 21.7096329
3: -12.8806286, 5.9915690, -14.7142086, 6.7883162, -19.6689453, 20.7057781
4: -12.8938513, 8.1161709, -14.6957979, 9.2043381, -22.0981865, 22.8119698
5: -9.9208193, 8.2064972, -11.2870445, 9.3407488, -19.2615681, 19.4935417
6: -10.6162243, 8.9850760, -12.0754890, 10.2121267, -20.8283501, 21.0605659
7: -11.4941292, 8.7223911, -13.1207838, 9.9013472, -21.3954735, 21.8431740
8: -13.2923098, 8.1086540, -15.1276236, 9.1781454, -22.4704552, 23.2362747
9: -9.4722328, 10.9116278, -10.7781382, 12.4232216, -21.8954506, 21.6897659

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7871342, upper bound: 16.7872801
time: 2.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872084, upper bound: 16.7873188
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -11.1264038, 8.3679228, -13.4572611, 10.0967808, -21.2231846, 21.8251839
1: -8.7825127, 7.3236446, -10.6766520, 8.8194141, -17.6019249, 18.0002956
2: -15.0782433, 4.5451641, -18.2538662, 5.4782987, -20.5565414, 22.7990303
3: -12.8853626, 5.9938092, -15.6669369, 7.2034421, -20.0888042, 21.6607437
4: -12.8983459, 8.1190481, -15.6315527, 9.7698107, -22.6681557, 23.7506008
5: -9.9242783, 8.2094555, -11.9941902, 9.9303799, -19.8546562, 20.2036457
6: -10.6200619, 8.9882202, -12.8353567, 10.8503590, -21.4704208, 21.8235779
7: -11.4982481, 8.7256107, -13.9602518, 10.5134621, -22.0117073, 22.6858616
8: -13.2972059, 8.1118736, -16.0876942, 9.7393360, -23.0365410, 24.1995678
9: -9.4756842, 10.9155388, -11.4596529, 13.2081261, -22.6838093, 22.3751907

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 158

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7871323, upper bound: 16.7872461
time: 4.89 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872062, upper bound: 16.7872806
time: 4.54 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -11.7761202, 8.8550129, -12.4350405, 9.3383560, -21.1144753, 21.2900543
1: -9.3071032, 7.7395105, -9.8461103, 8.1597137, -17.4668159, 17.5856209
2: -15.9731255, 4.8290172, -16.8616982, 5.0609989, -21.0341244, 21.6907120
3: -13.6458464, 6.3251257, -14.4433851, 6.6688390, -20.3146858, 20.7685089
4: -13.6563873, 8.5832777, -14.4333916, 9.0457182, -22.7021027, 23.0166702
5: -10.5029898, 8.6896009, -11.0888433, 9.1747284, -19.6777153, 19.7784443
6: -11.2465773, 9.5085068, -11.8624907, 10.0330544, -21.2796288, 21.3709984
7: -12.1919470, 9.2324085, -12.8868408, 9.7289677, -21.9209137, 22.1192455
8: -14.0835657, 8.5779266, -14.8584623, 9.0195150, -23.1030807, 23.4363861
9: -10.0293760, 11.5552025, -10.5859413, 12.2029276, -22.2323036, 22.1411438

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872319, upper bound: 16.7873001
time: 3.02 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873092, upper bound: 16.7873447
time: 3.81 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -11.7786684, 8.8570871, -13.2273159, 9.9260406, -21.7047081, 22.0844040
1: -9.3092222, 7.7412415, -10.4891262, 8.6703415, -17.9795647, 18.2303677
2: -15.9770699, 4.8308764, -17.9408073, 5.3829851, -21.3600521, 22.7716827
3: -13.6489143, 6.3266730, -15.3889990, 7.0806355, -20.7295494, 21.7156715
4: -13.6592922, 8.5851889, -15.3617363, 9.6069269, -23.2662163, 23.9469242
5: -10.5052347, 8.6915903, -11.7906017, 9.7597799, -20.2650127, 20.4821892
6: -11.2491179, 9.5105715, -12.6166515, 10.6663933, -21.9155121, 22.1272202
7: -12.1946716, 9.2346563, -13.7189474, 10.3363466, -22.5310154, 22.9536018
8: -14.0867996, 8.5802460, -15.8115444, 9.5768433, -23.6636429, 24.3917904
9: -10.0316906, 11.5577822, -11.2623110, 12.9812593, -23.0129509, 22.8200932

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 74
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 158

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872295, upper bound: 16.7872705
time: 4.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873072, upper bound: 16.7873072
time: 2.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 9.47 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7877262, upper bound: 16.7875486
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7877668, upper bound: 16.7875610
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7877262, upper bound: 16.7875486
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7877668, upper bound: 16.7875611
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7880061, upper bound: 16.7878725
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7879871, upper bound: 16.7878686
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7880061, upper bound: 16.7878725
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7879871, upper bound: 16.7878686
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7873720, upper bound: 16.7872962
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7874379, upper bound: 16.7873355
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7873694, upper bound: 16.7872644
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7874361, upper bound: 16.7872954
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7874299, upper bound: 16.7873145
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7875046, upper bound: 16.7873584
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7874280, upper bound: 16.7872847
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7875024, upper bound: 16.7873189
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7875089, upper bound: 16.7874432
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7874690, upper bound: 16.7874394
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7875173, upper bound: 16.7874838
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7874808, upper bound: 16.7874805
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7873411, upper bound: 16.7873166
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7872992, upper bound: 16.7873140
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7873549, upper bound: 16.7873941
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7873137, upper bound: 16.7873922
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7871342, upper bound: 16.7872801
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7872084, upper bound: 16.7873188
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7871323, upper bound: 16.7872461
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7872062, upper bound: 16.7872806
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7872319, upper bound: 16.7873001
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7873092, upper bound: 16.7873447
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7872295, upper bound: 16.7872705
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 9.47
Output dim: 2, lower bound: -16.7873072, upper bound: 16.7873072

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.4580345, 7.8647637, -11.3874474, 8.5596752, -19.0177097, 19.2522106
1: -8.2408047, 6.8924475, -8.9968023, 7.4888926, -15.7296972, 15.8892488
2: -14.1641560, 4.2472825, -15.4246321, 4.6471949, -18.8113499, 19.6719151
3: -12.0933285, 5.6476440, -13.2030077, 6.1312447, -18.2245731, 18.8506508
4: -12.1130133, 7.6404610, -13.2063446, 8.3016748, -20.4146881, 20.8468056
5: -9.3266249, 7.7134547, -10.1573067, 8.4009733, -17.7275982, 17.8707600
6: -9.9710875, 8.4492817, -10.8620844, 9.1970606, -19.1681480, 19.3113670
7: -10.7845163, 8.2035303, -11.7792206, 8.9238110, -19.7083282, 19.9827499
8: -12.4828377, 7.6332083, -13.6022797, 8.2843637, -20.7672005, 21.2354889
9: -8.9002008, 10.2557926, -9.6968555, 11.1728792, -20.0730801, 19.9526482

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7879029, upper bound: 16.7879214
time: 2.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878983, upper bound: 16.7878935
time: 2.48 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.0099659, 8.2796249, -11.0809050, 8.3308744, -19.3408394, 19.3605289
1: -8.6866112, 7.2473240, -8.7464790, 7.2909966, -15.9776077, 15.9938011
2: -14.9295053, 4.4842863, -15.0073776, 4.5167718, -19.4462757, 19.4916649
3: -12.7404222, 5.9299912, -12.8332376, 5.9688630, -18.7092857, 18.7632275
4: -12.7582769, 8.0353889, -12.8460550, 8.0837803, -20.8420563, 20.8814430
5: -9.8183794, 8.1220245, -9.8844290, 8.1736507, -17.9920311, 18.0064545
6: -10.5051556, 8.8918018, -10.5690279, 8.9513550, -19.4565105, 19.4608307
7: -11.3723555, 8.6332588, -11.4554462, 8.6868391, -20.0591946, 20.0887032
8: -13.1548500, 8.0312605, -13.2331963, 8.0677481, -21.2225952, 21.2644577
9: -9.3707142, 10.8000717, -9.4338303, 10.8703814, -20.2410946, 20.2339020

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878994, upper bound: 16.7879260
time: 7.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7878970, upper bound: 16.7878969
time: 3.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.4580345, 7.8647637, -11.7364120, 8.8230667, -19.2811012, 19.6011753
1: -8.2408047, 6.8924475, -9.2791481, 7.7171774, -15.9579821, 16.1715965
2: -14.1641560, 4.2472825, -15.9104481, 4.8051844, -18.9693413, 20.1577301
3: -12.0933285, 5.6476440, -13.6163902, 6.3149266, -18.4082546, 19.2640343
4: -12.1130133, 7.6404610, -13.6135235, 8.5527344, -20.6657486, 21.2539845
5: -9.3266249, 7.7134547, -10.4664221, 8.6614799, -17.9881058, 18.1798763
6: -9.9710875, 8.4492817, -11.2038746, 9.4768066, -19.4478951, 19.6531563
7: -10.7845163, 8.2035303, -12.1415949, 9.1966524, -19.9811687, 20.3451252
8: -12.4828377, 7.6332083, -14.0333939, 8.5454750, -21.0283127, 21.6666031
9: -8.9002008, 10.2557926, -9.9985218, 11.5169067, -20.4171066, 20.2543144

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876839, upper bound: 16.7875244
time: 7.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876800, upper bound: 16.7874835
time: 3.93 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.0099659, 8.2796249, -11.4353180, 8.5986242, -19.6085892, 19.7149429
1: -8.6866112, 7.2473240, -9.0335627, 7.5229340, -16.2095451, 16.2808857
2: -14.9295053, 4.4842863, -15.5011692, 4.6773448, -19.6068497, 19.9854546
3: -12.7404222, 5.9299912, -13.2537470, 6.1554594, -18.8958817, 19.1837349
4: -12.7582769, 8.0353889, -13.2600336, 8.3390093, -21.0972843, 21.2954216
5: -9.8183794, 8.1220245, -10.1986618, 8.4381456, -18.2565250, 18.3206863
6: -10.5051556, 8.8918018, -10.9164839, 9.2355595, -19.7407150, 19.8082848
7: -11.3723555, 8.6332588, -11.8240671, 8.9643250, -20.3366814, 20.4573250
8: -13.1548500, 8.0312605, -13.6717072, 8.3327770, -21.4876270, 21.7029686
9: -9.3707142, 10.8000717, -9.7398586, 11.2203045, -20.5910149, 20.5399303

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877031, upper bound: 16.7875312
time: 4.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877021, upper bound: 16.7874929
time: 4.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.1141214, 9.1036758, -11.5311251, 8.6663847, -20.7805061, 20.6348000
1: -9.5868788, 7.9554234, -9.1138172, 7.5802870, -17.1671658, 17.0692368
2: -16.4284439, 4.9523883, -15.6142569, 4.7097230, -21.1381664, 20.5666447
3: -14.0637913, 6.5075006, -13.3752728, 6.2054424, -20.2692318, 19.8827744
4: -14.0566139, 8.8204956, -13.3761930, 8.4036903, -22.4603043, 22.1966896
5: -10.8044205, 8.9400196, -10.2864285, 8.5070457, -19.3114662, 19.2264481
6: -11.5594177, 9.7788868, -10.9995213, 9.3136873, -20.8731003, 20.7784042
7: -12.5512466, 9.4891949, -11.9357471, 9.0350084, -21.5862541, 21.4249420
8: -14.4774323, 8.8032875, -13.7734289, 8.3816366, -22.8590698, 22.5767174
9: -10.3171139, 11.8889866, -9.8206034, 11.3145723, -21.6316872, 21.7095871

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877245, upper bound: 16.7877716
time: 2.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7877297, upper bound: 16.7877889
time: 4.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.9285793, 9.7081604, -11.5373068, 8.6711483, -21.5997257, 21.2454662
1: -10.2485542, 8.4797688, -9.1188889, 7.5843558, -17.8329067, 17.5986576
2: -17.5380058, 5.2822409, -15.6231565, 4.7128630, -22.2508698, 20.9053974
3: -15.0365458, 6.9310603, -13.3827076, 6.2088552, -21.2454014, 20.3137684
4: -15.0120115, 9.3974380, -13.3833895, 8.4081688, -23.4201794, 22.7808266
5: -11.5263062, 9.5418100, -10.2918968, 8.5117283, -20.0380325, 19.8337059
6: -12.3348122, 10.4298611, -11.0055227, 9.3186493, -21.6534595, 21.4353809
7: -13.4075022, 10.1132126, -11.9422493, 9.0399933, -22.4474945, 22.0554619
8: -15.4563360, 9.3755808, -13.7810469, 8.3864708, -23.8428059, 23.1566277
9: -11.0124989, 12.6895866, -9.8260050, 11.3207178, -22.3332138, 22.5155907

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876933, upper bound: 16.7877678
time: 3.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7876999, upper bound: 16.7877859
time: 2.92 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.1141214, 9.1036758, -11.8851261, 8.9335709, -21.0476913, 20.9888020
1: -9.5868788, 7.9554234, -9.4002428, 7.8116379, -17.3985157, 17.3556671
2: -16.4284439, 4.9523883, -16.1070480, 4.8698907, -21.2983322, 21.0594368
3: -14.0637913, 6.5075006, -13.7943401, 6.3915691, -20.4553566, 20.3018417
4: -14.0566139, 8.8204956, -13.7892694, 8.6583042, -22.7149181, 22.6097641
5: -10.8044205, 8.9400196, -10.6000938, 8.7712364, -19.5756569, 19.5401115
6: -11.5594177, 9.7788868, -11.3458786, 9.5973721, -21.1567898, 21.1247616
7: -12.5512466, 9.4891949, -12.3035736, 9.3121376, -21.8633823, 21.7927685
8: -14.4774323, 8.8032875, -14.2099657, 8.6461916, -23.1236229, 23.0132523
9: -10.3171139, 11.8889866, -10.1266689, 11.6634064, -21.9805202, 22.0156555

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875302, upper bound: 16.7873262
time: 3.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875477, upper bound: 16.7874045
time: 4.11 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.9285793, 9.7081604, -11.8902483, 8.9375439, -21.8661213, 21.5984077
1: -10.2485542, 8.4797688, -9.4044466, 7.8150244, -18.0635777, 17.8842125
2: -17.5380058, 5.2822409, -16.1144257, 4.8727179, -22.4107246, 21.3966675
3: -15.0365458, 6.9310603, -13.8004971, 6.3944359, -21.4309807, 20.7315578
4: -15.0120115, 9.3974380, -13.7952061, 8.6620388, -23.6740494, 23.1926441
5: -11.5263062, 9.5418100, -10.6046352, 8.7751026, -20.3014069, 20.1464462
6: -12.3348122, 10.4298611, -11.3508883, 9.6014948, -21.9363022, 21.7807465
7: -13.4075022, 10.1132126, -12.3089924, 9.3162937, -22.7237930, 22.4222050
8: -15.4563360, 9.3755808, -14.2163391, 8.6502810, -24.1066093, 23.5919189
9: -11.0124989, 12.6895866, -10.1311731, 11.6685181, -22.6810131, 22.8207588

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 131

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874946, upper bound: 16.7873228
time: 5.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875134, upper bound: 16.7874013
time: 3.45 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.5126085, 7.9061828, -12.6225653, 9.4769955, -19.9896049, 20.5287476
1: -8.2838192, 6.9255395, -9.9994946, 8.2817259, -16.5655441, 16.9250336
2: -14.2200003, 4.2940454, -17.1113129, 5.1413164, -19.3613167, 21.4053574
3: -12.1556072, 5.6735883, -14.6731062, 6.7708197, -18.9264259, 20.3466949
4: -12.1791906, 7.6784353, -14.6537075, 9.1780367, -21.3572273, 22.3321419
5: -9.3781452, 7.7523136, -11.2555008, 9.3137035, -18.6918488, 19.0078144
6: -10.0245523, 8.4971542, -12.0408106, 10.1838617, -20.2084141, 20.5379639
7: -10.8569288, 8.2497139, -13.0842085, 9.8734856, -20.7304153, 21.3339195
8: -12.5482616, 7.6634669, -15.0831575, 9.1496496, -21.6979103, 22.7466240
9: -8.9506474, 10.3096666, -10.7478695, 12.3877249, -21.3383675, 21.0575352

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873720, upper bound: 16.7872963
time: 3.51 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873720, upper bound: 16.7872961
time: 3.14 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.5612192, 7.9431901, -13.0763969, 9.8141527, -20.3753719, 21.0195866
1: -8.3244791, 6.9556928, -10.3674059, 8.5748959, -16.8993740, 17.3230972
2: -14.2947521, 4.2991371, -17.7329330, 5.3270030, -19.6217537, 22.0320702
3: -12.2140579, 5.6975565, -15.2140341, 7.0074940, -19.2215500, 20.9115906
4: -12.2374010, 7.7141867, -15.1839542, 9.5000134, -21.7374115, 22.8981400
5: -9.4226961, 7.7892342, -11.6574955, 9.6489611, -19.0716572, 19.4467297
6: -10.0709314, 8.5357189, -12.4738960, 10.5460320, -20.6169624, 21.0096149
7: -10.9070978, 8.2856407, -13.5602512, 10.2227545, -21.1298485, 21.8458920
8: -12.6038380, 7.6981106, -15.6299000, 9.4712143, -22.0750523, 23.3280087
9: -8.9904861, 10.3577366, -11.1345482, 12.8337164, -21.8242016, 21.4922848

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874379, upper bound: 16.7873355
time: 45.82 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874379, upper bound: 16.7873355
time: 2.30 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.5247221, 7.9153275, -13.4601545, 10.0949450, -20.6196671, 21.3754826
1: -8.2936506, 6.9334002, -10.6762400, 8.8216991, -17.1153488, 17.6096382
2: -14.2370577, 4.2994709, -18.2472610, 5.4821577, -19.7192154, 22.5467319
3: -12.1699934, 5.6800103, -15.6772785, 7.2050085, -19.3750000, 21.3572884
4: -12.1932869, 7.6870966, -15.6355076, 9.7703867, -21.9636688, 23.3226013
5: -9.3888874, 7.7613134, -11.9946060, 9.9322443, -19.3211288, 19.7559204
6: -10.0361910, 8.5067968, -12.8334770, 10.8545008, -20.8906898, 21.3402710
7: -10.8697023, 8.2592182, -13.9636688, 10.5134668, -21.3831654, 22.2228870
8: -12.5630217, 7.6723886, -16.0895271, 9.7358856, -22.2989044, 23.7619133
9: -8.9609861, 10.3216391, -11.4628649, 13.2087936, -22.1697769, 21.7845039

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873694, upper bound: 16.7872644
time: 3.33 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873694, upper bound: 16.7872644
time: 3.11 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.5667210, 7.9473972, -13.9078455, 10.4266834, -20.9934025, 21.8552437
1: -8.3289461, 6.9593091, -11.0382318, 9.1107798, -17.4397259, 17.9975395
2: -14.3026552, 4.3019061, -18.8592491, 5.6665921, -19.9692459, 23.1611519
3: -12.2206116, 5.7005677, -16.2116699, 7.4384947, -19.6591053, 21.9122353
4: -12.2437449, 7.7181511, -16.1584167, 10.0877066, -22.3314514, 23.8765678
5: -9.4275656, 7.7933607, -12.3902359, 10.2630157, -19.6905823, 20.1835938
6: -10.0762711, 8.5400963, -13.2594223, 11.2119045, -21.2881756, 21.7995167
7: -10.9129057, 8.2900887, -14.4336233, 10.8576078, -21.7705135, 22.7237129
8: -12.6106091, 7.7023458, -16.6284866, 10.0527725, -22.6633816, 24.3308334
9: -8.9952154, 10.3631983, -11.8442926, 13.6485186, -22.6437302, 22.2074909

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874361, upper bound: 16.7872954
time: 11.79 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874361, upper bound: 16.7872952
time: 3.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -11.1403160, 8.3778954, -12.3853045, 9.3005352, -20.4408512, 20.7631989
1: -8.7910652, 7.3277993, -9.8058462, 8.1278105, -16.9188766, 17.1336441
2: -15.0875702, 4.5697646, -16.7878952, 5.0429063, -20.1304741, 21.3576584
3: -12.8902378, 5.9942288, -14.3861008, 6.6443667, -19.5346050, 20.3803291
4: -12.9122019, 8.1280069, -14.3752022, 9.0097704, -21.9219723, 22.5032082
5: -9.9379616, 8.2166090, -11.0451612, 9.1375465, -19.0755062, 19.2617702
6: -10.6314182, 9.0008106, -11.8147192, 9.9938669, -20.6252842, 20.8155289
7: -11.5286636, 8.7404184, -12.8356609, 9.6907158, -21.2193794, 21.5760803
8: -13.3106470, 8.1153965, -14.7975664, 8.9816341, -22.2922821, 22.9129639
9: -9.4860840, 10.9288454, -10.5440636, 12.1539526, -21.6400356, 21.4729080

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874299, upper bound: 16.7873143
time: 3.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874299, upper bound: 16.7873144
time: 7.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.2007828, 8.4232521, -12.8387308, 9.6374750, -20.8382568, 21.2619820
1: -8.8412018, 7.3658204, -10.1735067, 8.4206142, -17.2618160, 17.5393276
2: -15.1768036, 4.5790548, -17.4089546, 5.2285514, -20.4053516, 21.9880104
3: -12.9636354, 6.0244179, -14.9264507, 6.8808217, -19.8444519, 20.9508667
4: -12.9843254, 8.1718655, -14.9051018, 9.3314819, -22.3158035, 23.0769653
5: -9.9930134, 8.2622108, -11.4468718, 9.4726000, -19.4656105, 19.7090816
6: -10.6891212, 9.0487242, -12.2475863, 10.3557348, -21.0448570, 21.2963104
7: -11.5903168, 8.7848511, -13.3114100, 10.0396032, -21.6299210, 22.0962601
8: -13.3810005, 8.1575193, -15.3440571, 9.3030367, -22.6840363, 23.5015755
9: -9.5356865, 10.9884052, -10.9304790, 12.5996494, -22.1353340, 21.9188843

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875046, upper bound: 16.7873585
time: 3.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875046, upper bound: 16.7873586
time: 3.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.1515131, 8.3863354, -13.2068882, 9.9097109, -21.0612183, 21.5932236
1: -8.8001652, 7.3350668, -10.4727840, 8.6571465, -17.4573116, 17.8078461
2: -15.1033382, 4.5749092, -17.9064598, 5.3750515, -20.4783897, 22.4813690
3: -12.9034986, 6.0001769, -15.3668299, 7.0713396, -19.9748383, 21.3670044
4: -12.9251871, 8.1360140, -15.3383770, 9.5915937, -22.5167809, 23.4743900
5: -9.9478531, 8.2249203, -11.7728500, 9.7443132, -19.6921654, 19.9977703
6: -10.6421585, 9.0097132, -12.5965052, 10.6507301, -21.2928848, 21.6062183
7: -11.5404682, 8.7492390, -13.6990595, 10.3202457, -21.8607140, 22.4482994
8: -13.3242741, 8.1237164, -15.7854271, 9.5587502, -22.8830242, 23.9091434
9: -9.4956446, 10.9399109, -11.2453260, 12.9614687, -22.4571114, 22.1852341

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874280, upper bound: 16.7872847
time: 4.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874280, upper bound: 16.7872847
time: 4.01 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.2048473, 8.4264164, -13.6533470, 10.2413082, -21.4461555, 22.0797634
1: -8.8445110, 7.3685055, -10.8345814, 8.9453945, -17.7899055, 18.2030869
2: -15.1828165, 4.5814066, -18.5178375, 5.5581417, -20.7409554, 23.0992432
3: -12.9684181, 6.0266867, -15.8985939, 7.3044162, -20.2728310, 21.9252796
4: -12.9889555, 8.1748228, -15.8598690, 9.9082184, -22.8971748, 24.0346909
5: -9.9965897, 8.2652769, -12.1681156, 10.0741320, -20.0707207, 20.4333897
6: -10.6930857, 9.0519390, -13.0224485, 11.0067987, -21.6998825, 22.0743866
7: -11.5946217, 8.7882385, -14.1678734, 10.6638393, -22.2584591, 22.9561081
8: -13.3860521, 8.1608257, -16.3231010, 9.8751392, -23.2611885, 24.4839249
9: -9.5391998, 10.9924536, -11.6255608, 13.4006557, -22.9398518, 22.6180134

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875024, upper bound: 16.7873190
time: 4.18 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875024, upper bound: 16.7873191
time: 5.19 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -11.2548046, 8.4634752, -11.1530876, 8.3800859, -19.6348915, 19.6165619
1: -8.8879681, 7.4064674, -8.8056602, 7.3380613, -16.2260284, 16.2121239
2: -15.2473278, 4.6016741, -15.1038408, 4.5175829, -19.7649117, 19.7055130
3: -13.0432291, 6.0629239, -12.9249430, 6.0068831, -19.0501099, 18.9878654
4: -13.0504265, 8.2102985, -12.9317560, 8.1320581, -21.1824837, 21.1420536
5: -10.0398655, 8.3046513, -9.9466543, 8.2257195, -18.2655849, 18.2513046
6: -10.7432327, 9.0923166, -10.6335936, 9.0056629, -19.7488899, 19.7259102
7: -11.6355705, 8.8248425, -11.5206976, 8.7336445, -20.3692150, 20.3455353
8: -13.4509773, 8.1996632, -13.3157921, 8.1108580, -21.5618362, 21.5154552
9: -9.5869761, 11.0425901, -9.4912596, 10.9399300, -20.5269051, 20.5338478

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875012, upper bound: 16.7874079
time: 4.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875089, upper bound: 16.7874431
time: 3.02 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.0649748, 9.0654192, -11.1590910, 8.3846760, -20.4496498, 20.2245083
1: -9.5464020, 7.9273276, -8.8105278, 7.3420043, -16.8884029, 16.7378559
2: -16.3510132, 4.9319668, -15.1124678, 4.5206461, -20.8716583, 20.0444336
3: -14.0101156, 6.4841919, -12.9320297, 6.0101404, -20.0202560, 19.4162216
4: -14.0013142, 8.7844296, -12.9386539, 8.1363850, -22.1376991, 21.7230816
5: -10.7584743, 8.9035025, -9.9519472, 8.2302189, -18.9886932, 18.8554459
6: -11.5152397, 9.7397165, -10.6394348, 9.0104370, -20.5256729, 20.3791504
7: -12.4884377, 9.4458132, -11.5270252, 8.7384777, -21.2269115, 20.9728355
8: -14.4250965, 8.7697401, -13.3232393, 8.1155243, -22.5406170, 22.0929794
9: -10.2788324, 11.8395395, -9.4964027, 10.9458952, -21.2247276, 21.3359413

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874592, upper bound: 16.7874032
time: 2.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874690, upper bound: 16.7874393
time: 9.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.9583740, 8.2420921, -11.7079268, 8.7959137, -19.7542858, 19.9500198
1: -8.6457233, 7.2151537, -9.2537155, 7.6938987, -16.3396168, 16.4688683
2: -14.8436241, 4.4755526, -15.8712959, 4.7561288, -19.5997524, 20.3468475
3: -12.6854734, 5.9057112, -13.5742760, 6.2901411, -18.9756126, 19.4799881
4: -12.7020340, 7.9994555, -13.5789557, 8.5286884, -21.2307224, 21.5784092
5: -9.7757511, 8.0847502, -10.4400539, 8.6359062, -18.4116573, 18.5248013
6: -10.4597893, 8.8547268, -11.1690025, 9.4494448, -19.9092293, 20.0237293
7: -11.3226023, 8.5956612, -12.1105824, 9.1654224, -20.4880238, 20.7062435
8: -13.0942917, 7.9901133, -13.9889545, 8.5107327, -21.6050186, 21.9790688
9: -9.3326511, 10.7500429, -9.9634809, 11.4862604, -20.8189125, 20.7135239

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875076, upper bound: 16.7874516
time: 6.44 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7875173, upper bound: 16.7874836
time: 3.13 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.7599373, 8.8380527, -11.7141390, 8.8006983, -20.5606327, 20.5521908
1: -9.2975931, 7.7305083, -9.2587423, 7.6979742, -16.9955635, 16.9892483
2: -15.9365444, 4.8023758, -15.8802986, 4.7595472, -20.6960907, 20.6826725
3: -13.6427345, 6.3227072, -13.5815411, 6.2935171, -19.9362488, 19.9042435
4: -13.6430197, 8.5677853, -13.5860567, 8.5331821, -22.1762009, 22.1538391
5: -10.4871254, 8.6772118, -10.4455261, 8.6405621, -19.1276855, 19.1227379
6: -11.2238836, 9.4950781, -11.1750498, 9.4543839, -20.6782665, 20.6701241
7: -12.1662960, 9.2102013, -12.1171923, 9.1704950, -21.3367863, 21.3273888
8: -14.0585804, 8.5542707, -13.9966612, 8.5156565, -22.5742340, 22.5509319
9: -10.0166435, 11.5389204, -9.9688330, 11.4924297, -21.5090714, 21.5077515

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874708, upper bound: 16.7874485
time: 6.27 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7874808, upper bound: 16.7874805
time: 4.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.0866814, 9.0864038, -11.1530876, 8.3800859, -20.4667625, 20.2394905
1: -9.5638590, 7.9421449, -8.8056602, 7.3380613, -16.9019184, 16.7478046
2: -16.4057140, 4.9448462, -15.1038408, 4.5175829, -20.9232979, 20.0486851
3: -14.0292158, 6.4950914, -12.9249430, 6.0068831, -20.0360966, 19.4200325
4: -14.0230532, 8.8039875, -12.9317560, 8.1320581, -22.1551094, 21.7357445
5: -10.7782707, 8.9219675, -9.9466543, 8.2257195, -19.0039864, 18.8686218
6: -11.5402069, 9.7558575, -10.6335936, 9.0056629, -20.5458679, 20.3894501
7: -12.5124722, 9.4702587, -11.5206976, 8.7336445, -21.2461166, 20.9909534
8: -14.4534655, 8.7992039, -13.3157921, 8.1108580, -22.5643234, 22.1149960
9: -10.2960510, 11.8621054, -9.4912596, 10.9399300, -21.2359810, 21.3533630

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873150, upper bound: 16.7872811
time: 2.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7873411, upper bound: 16.7873168
time: 2.79 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.8778515, 9.6741276, -11.1590910, 8.3846760, -21.2625256, 20.8332157
1: -10.2063046, 8.4517536, -8.8105278, 7.3420043, -17.5483074, 17.2622814
2: -17.4835854, 5.2682118, -15.1124678, 4.5206461, -22.0042305, 20.3806763
3: -14.9734802, 6.9067755, -12.9320297, 6.0101404, -20.9836197, 19.8388062
4: -14.9506426, 9.3648615, -12.9386539, 8.1363850, -23.0870285, 22.3035145
5: -11.4799089, 9.5067072, -9.9519472, 8.2302189, -19.7101288, 19.4586525
6: -12.2946930, 10.3885365, -10.6394348, 9.0104370, -21.3051281, 21.0279694
7: -13.3447495, 10.0771141, -11.5270252, 8.7384777, -22.0832233, 21.6041393
8: -15.4064474, 9.3567133, -13.3232393, 8.1155243, -23.5219727, 22.6799526
9: -10.9716606, 12.6404524, -9.4964027, 10.9458952, -21.9175549, 22.1368561

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 74
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 158

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872752, upper bound: 16.7872769
time: 5.14 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -16.7872992, upper bound: 16.7873142
time: 2.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.8585796, 8.9166012, -11.7079268, 8.7959137, -20.6544914, 20.6245270
1: -9.3773813, 7.7944517, -9.2537155, 7.6938987, -17.0712757, 17.0481644
2: -16.0942631, 4.8504367, -15.8712959, 4.7561288, -20.8503914, 20.7217331
3: -13.7530222, 6.3731942, -13.5742760, 6.2901411, -20.0431633, 19.9474697
4: -13.7551651, 8.6421967, -13.5789557, 8.5286884, -22.2838497, 22.2211533
5: -10.5758677, 8.7525463, -10.4400539, 8.6359062, -19.2117729, 19.1926003
6: -11.3230267, 9.5734587, -11.1690025, 9.4494448, -20.7724686, 20.7424622
7: -12.2738028, 9.2948055, -12.1105824, 9.1654224, -21.4392242, 21.4053879
8: -14.1795607, 8.6375160, -13.9889545, 8.5107327, -22.6902924, 22.6264706
9: -10.1001139, 11.6374044, -9.9634809, 11.4862604, -21.5863743, 21.6008854

Time for backsubstitution: 2.17 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 8.09 + 593.31 = 601.40 seconds
