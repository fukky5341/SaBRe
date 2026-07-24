## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 14.544726514199999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500)
1: (-8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229)
2: (-10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675)
3: (-11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024)
4: (-10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131)
5: (-9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238)
6: (-8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851)
7: (-10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482)
8: (-11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781)
9: (-8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.03 + 8.64 = 10.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -14.5592858, upper bound: 14.5592858

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5582913, upper bound: 14.5585405
time: 4.53 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5581795, upper bound: 14.5581795
time: 3.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.43 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.43
Output dim: 7, lower bound: -14.5582913, upper bound: 14.5585405
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.43
Output dim: 7, lower bound: -14.5581795, upper bound: 14.5581795

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -9.2332058, 7.5500565, -9.5691223, 7.8232427, -17.0564461, 17.1191788
1: -7.6223903, 6.7887859, -7.9051175, 7.0272355, -14.6496258, 14.6939030
2: -9.5624714, 6.0221243, -9.9190397, 6.2403669, -15.8028383, 15.9411640
3: -11.1595211, 5.3600512, -11.5723495, 5.5530062, -16.7125244, 16.9323997
4: -10.1810751, 8.1270733, -10.5500622, 8.4095707, -18.5906448, 18.6771355
5: -8.7630548, 6.8487358, -9.0884361, 7.0976272, -15.8606815, 15.9371719
6: -8.2509880, 8.8466702, -8.5532036, 9.1604691, -17.4114571, 17.3998737
7: -10.1813774, 6.3747544, -10.5468979, 6.6263194, -16.8076973, 16.9216518
8: -10.4311209, 7.4013281, -10.8207664, 7.6601825, -18.0913029, 18.2220955
9: -8.4925251, 8.5917301, -8.7972698, 8.9020777, -17.3946037, 17.3889999

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5572736, upper bound: 14.5571389
time: 4.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5569158, upper bound: 14.5570582
time: 4.17 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.4336948, 8.5151262, -9.5239668, 7.7865181, -18.2202129, 18.0390911
1: -8.6247463, 7.6292572, -7.8682852, 6.9958572, -15.6206036, 15.4975424
2: -10.8023262, 6.7426581, -9.8733940, 6.2130933, -17.0154190, 16.6160507
3: -12.6426868, 6.0132675, -11.5166206, 5.5267096, -18.1693954, 17.5298882
4: -11.4782667, 9.1388054, -10.5016251, 8.3740215, -19.8522873, 19.6404266
5: -9.9137287, 7.7186980, -9.0447302, 7.0647287, -16.9784584, 16.7634258
6: -9.2961464, 9.9447584, -8.5137606, 9.1205149, -18.4166603, 18.4585190
7: -11.4429817, 7.1891727, -10.5030088, 6.5939770, -18.0369568, 17.6921806
8: -11.7977543, 8.3138504, -10.7687931, 7.6273317, -19.4250870, 19.0826435
9: -9.5566216, 9.6469698, -8.7576714, 8.8622665, -18.4188862, 18.4046402

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5572509, upper bound: 14.5569950
time: 4.78 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5568552, upper bound: 14.5568552
time: 4.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 11.24 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 11.24
Output dim: 7, lower bound: -14.5572736, upper bound: 14.5571389
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 11.24
Output dim: 7, lower bound: -14.5569158, upper bound: 14.5570582
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 11.24
Output dim: 7, lower bound: -14.5572509, upper bound: 14.5569950
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 11.24
Output dim: 7, lower bound: -14.5568552, upper bound: 14.5568552

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -9.1166639, 7.4551277, -9.1604519, 7.4904327, -16.6070938, 16.6155796
1: -7.5243444, 6.7057462, -7.5614710, 6.7362747, -14.2606192, 14.2672176
2: -9.4384470, 5.9466739, -9.4845362, 5.9758625, -15.4143095, 15.4312096
3: -11.0177517, 5.2923470, -11.0756798, 5.3156691, -16.3334198, 16.3680267
4: -10.0537767, 8.0281696, -10.1041260, 8.0632133, -18.1169891, 18.1322937
5: -8.6487751, 6.7618694, -8.6878738, 6.7928553, -15.4416304, 15.4497433
6: -8.1463242, 8.7383909, -8.1865139, 8.7807436, -16.9270668, 16.9249039
7: -10.0556936, 6.2834730, -10.1065006, 6.3072805, -16.3629723, 16.3899727
8: -10.2967930, 7.3110042, -10.3501587, 7.3433800, -17.6401730, 17.6611633
9: -8.3860235, 8.4825869, -8.4239426, 8.5200052, -16.9060287, 16.9065285

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5553731, upper bound: 14.5551759
time: 3.99 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5572736, upper bound: 14.5571389
time: 5.57 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.9616213, 7.3291378, -10.9010344, 8.8909578, -17.8525772, 18.2301693
1: -7.3938541, 6.5949306, -9.0124092, 7.9510579, -15.3449116, 15.6073380
2: -9.2745552, 5.8477669, -11.2775574, 7.0405650, -16.3151207, 17.1253223
3: -10.8297386, 5.2011395, -13.2486343, 6.2676687, -17.0974064, 18.4497738
4: -9.8854065, 7.8988309, -11.9751549, 9.5459118, -19.4313145, 19.8739853
5: -8.4963436, 6.6465578, -10.3510866, 8.0608482, -16.5571899, 16.9976444
6: -8.0076714, 8.5960598, -9.7079449, 10.3846169, -18.3922882, 18.3040047
7: -9.8917122, 6.1602230, -11.9279099, 7.4803400, -17.3720512, 18.0881310
8: -10.1178570, 7.1920061, -12.3177242, 8.6984711, -18.8163280, 19.5097294
9: -8.2448177, 8.3380194, -9.9885588, 10.0631571, -18.3079758, 18.3265743

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5552231, upper bound: 14.5551354
time: 13.28 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5569158, upper bound: 14.5570582
time: 3.89 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.3159332, 8.4192390, -9.1147213, 7.4532914, -17.7692242, 17.5339584
1: -8.5260143, 7.5453691, -7.5241594, 6.7043562, -15.2303696, 15.0695257
2: -10.6772251, 6.6663322, -9.4383221, 5.9483256, -16.6255512, 16.1046543
3: -12.5000353, 5.9450707, -11.0190563, 5.2890882, -17.7891235, 16.9641266
4: -11.3499813, 9.0391121, -10.0549412, 8.0271311, -19.3771133, 19.0940533
5: -9.7985229, 7.6305070, -8.6436243, 6.7596054, -16.5581284, 16.2741318
6: -9.1907473, 9.8355236, -8.1464481, 8.7403612, -17.9311085, 17.9819698
7: -11.3162460, 7.0975161, -10.0616827, 6.2743864, -17.5906334, 17.1591988
8: -11.6626034, 8.2225733, -10.2975492, 7.3101511, -18.9727554, 18.5201168
9: -9.4489384, 9.5373354, -8.3838434, 8.4796343, -17.9285679, 17.9211731

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5550557, upper bound: 14.5544968
time: 3.37 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5572509, upper bound: 14.5569950
time: 5.12 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.1621847, 8.2942142, -10.8550949, 8.8537388, -19.0159225, 19.1493073
1: -8.3970776, 7.4355021, -8.9749088, 7.9191046, -16.3161812, 16.4104118
2: -10.5147285, 6.5680060, -11.2310762, 7.0128350, -17.5275593, 17.7990818
3: -12.3145580, 5.8549771, -13.1919537, 6.2413340, -18.5558929, 19.0469303
4: -11.1833134, 8.9109325, -11.9256630, 9.5096226, -20.6929359, 20.8365936
5: -9.6477575, 7.5155268, -10.3069668, 8.0274200, -17.6751766, 17.8224926
6: -9.0537910, 9.6947660, -9.6678448, 10.3438015, -19.3975925, 19.3626060
7: -11.1543674, 6.9757490, -11.8827686, 7.4480648, -18.6024323, 18.8585167
8: -11.4856844, 8.1045904, -12.2648849, 8.6650610, -20.1507454, 20.3694763
9: -9.3085346, 9.3947392, -9.9483557, 10.0226374, -19.3311729, 19.3430901

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5548260, upper bound: 14.5544032
time: 6.18 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5568552, upper bound: 14.5568552
time: 4.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 12.49 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 12.49
Output dim: 7, lower bound: -14.5553731, upper bound: 14.5551759
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.49
Output dim: 7, lower bound: -14.5572736, upper bound: 14.5571389
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.49
Output dim: 7, lower bound: -14.5552231, upper bound: 14.5551354
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.49
Output dim: 7, lower bound: -14.5569158, upper bound: 14.5570582
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 12.49
Output dim: 7, lower bound: -14.5550557, upper bound: 14.5544968
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 12.49
Output dim: 7, lower bound: -14.5572509, upper bound: 14.5569950
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 12.49
Output dim: 7, lower bound: -14.5548260, upper bound: 14.5544032
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 12.49
Output dim: 7, lower bound: -14.5568552, upper bound: 14.5568552

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.3733463, 7.6622195, -8.8075247, 7.2076697, -16.5810165, 16.4697437
1: -7.7361426, 6.8884258, -7.2654805, 6.4902811, -14.2264233, 14.1539040
2: -9.6967163, 6.0653362, -9.1130896, 5.7467895, -15.4435043, 15.1784258
3: -11.3692942, 5.4350014, -10.6465092, 5.1187558, -16.4880505, 16.0815086
4: -10.3155956, 8.2353363, -9.7151070, 7.7621365, -18.0777283, 17.9504414
5: -8.9041576, 6.9469900, -8.3498955, 6.5346966, -15.4388542, 15.2968836
6: -8.3562250, 8.9653254, -7.8712349, 8.4517889, -16.8080139, 16.8365555
7: -10.3034430, 6.4422359, -9.7231970, 6.0511694, -16.3546104, 16.1654320
8: -10.5925503, 7.5024781, -9.9474697, 7.0715251, -17.6640739, 17.4499474
9: -8.6179724, 8.7157326, -8.1058254, 8.1970997, -16.8150692, 16.8215561

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 171

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5534219, upper bound: 14.5530550
time: 4.67 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5533528, upper bound: 14.5530281
time: 5.38 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.9247465, 7.3010063, -9.1604519, 7.4904327, -16.4151764, 16.4614582
1: -7.3631897, 6.5717564, -7.5614710, 6.7362747, -14.0994635, 14.1332273
2: -9.2360897, 5.8193226, -9.4845362, 5.9758625, -15.2119522, 15.3038588
3: -10.7865372, 5.1842108, -11.0756798, 5.3156691, -16.1022053, 16.2598915
4: -9.8420238, 7.8639188, -10.1041260, 8.0632133, -17.9052353, 17.9680443
5: -8.4650526, 6.6205029, -8.6878738, 6.7928553, -15.2579079, 15.3083754
6: -7.9736018, 8.5614128, -8.1865139, 8.7807436, -16.7543449, 16.7479267
7: -9.8517513, 6.1378365, -10.1065006, 6.3072805, -16.1590309, 16.2443352
8: -10.0762234, 7.1636000, -10.3501587, 7.3433800, -17.4196033, 17.5137577
9: -8.2114105, 8.3066216, -8.4239426, 8.5200052, -16.7314148, 16.7305641

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555416, upper bound: 14.5556077
time: 32.24 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555416, upper bound: 14.5571389
time: 9.39 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.2175512, 7.5359540, -10.5385323, 8.5995092, -17.8170605, 18.0744858
1: -7.6048055, 6.7773647, -8.7095013, 7.6990080, -15.3038139, 15.4868660
2: -9.5324726, 5.9667845, -10.8959627, 6.8042097, -16.3366814, 16.8627472
3: -11.1802216, 5.3439379, -12.8094177, 6.0649419, -17.2451630, 18.1533508
4: -10.1471901, 8.1064358, -11.5778122, 9.2373943, -19.3845844, 19.6842480
5: -8.7512970, 6.8318591, -10.0038233, 7.7935333, -16.5448303, 16.8356819
6: -8.2175732, 8.8220482, -9.3852139, 10.0468025, -18.2643757, 18.2072620
7: -10.1383696, 6.3212938, -11.5367680, 7.2166324, -17.3549995, 17.8580627
8: -10.4130526, 7.3831682, -11.9050274, 8.4187183, -18.8317680, 19.2881966
9: -8.4771118, 8.5710726, -9.6612720, 9.7315445, -18.2086563, 18.2323418

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5532776, upper bound: 14.5530128
time: 4.85 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5531839, upper bound: 14.5529659
time: 4.53 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.7729807, 7.1779251, -10.9010344, 8.8909578, -17.6639366, 18.0789604
1: -7.2352448, 6.4634161, -9.0124092, 7.9510579, -15.1863022, 15.4758253
2: -9.0758018, 5.7226467, -11.2775574, 7.0405650, -16.1163654, 17.0002022
3: -10.6026773, 5.0947647, -13.2486343, 6.2676687, -16.8703442, 18.3433990
4: -9.6771584, 7.7374735, -11.9751549, 9.5459118, -19.2230701, 19.7126274
5: -8.3157339, 6.5075984, -10.3510866, 8.0608482, -16.3765793, 16.8586845
6: -7.8377419, 8.4220581, -9.7079449, 10.3846169, -18.2223587, 18.1300030
7: -9.6909704, 6.0170403, -11.9279099, 7.4803400, -17.1713104, 17.9449501
8: -9.9009447, 7.0473170, -12.3177242, 8.6984711, -18.5994148, 19.3650417
9: -8.0732613, 8.1649704, -9.9885588, 10.0631571, -18.1364174, 18.1535263

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5544639, upper bound: 14.5549915
time: 4.48 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5544639, upper bound: 14.5570582
time: 5.75 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.4431400, 8.5213022, -8.7613897, 7.1702504, -17.6133900, 17.2826900
1: -8.6326790, 7.6381369, -7.2276773, 6.4580975, -15.0907764, 14.8658123
2: -10.7998314, 6.7106829, -9.0664110, 5.7191515, -16.5189819, 15.7770929
3: -12.6903038, 6.0165257, -10.5893879, 5.0919199, -17.7822227, 16.6059113
4: -11.4720640, 9.1374464, -9.6654701, 7.7257423, -19.1978073, 18.8029175
5: -9.9291363, 7.7207313, -8.3051548, 6.5012817, -16.4304180, 16.0258865
6: -9.2903156, 9.9439144, -7.8307543, 8.4109020, -17.7012177, 17.7746696
7: -11.4290886, 7.1699195, -9.6779613, 6.0180817, -17.4471703, 16.8478813
8: -11.8114433, 8.3164024, -9.8942471, 7.0379696, -18.8494110, 18.2106495
9: -9.5663109, 9.6570034, -8.0653963, 8.1564274, -17.7227325, 17.7223969

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 171

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5532384, upper bound: 14.5525572
time: 4.88 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5531089, upper bound: 14.5524866
time: 23.01 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.1216431, 8.2631321, -9.1147213, 7.4532914, -17.5749321, 17.3778534
1: -8.3637791, 7.4097567, -7.5241594, 6.7043562, -15.0681343, 14.9339085
2: -10.4722195, 6.5374403, -9.4383221, 5.9483256, -16.4205456, 15.9757614
3: -12.2673101, 5.8354411, -11.0190563, 5.2890882, -17.5563984, 16.8544979
4: -11.1361694, 8.8733768, -10.0549412, 8.0271311, -19.1632996, 18.9283180
5: -9.6132135, 7.4870715, -8.6436243, 6.7596054, -16.3728180, 16.1306953
6: -9.0167150, 9.6571779, -8.1464481, 8.7403612, -17.7570763, 17.8036232
7: -11.1110334, 6.9496779, -10.0616827, 6.2743864, -17.3854198, 17.0113602
8: -11.4398928, 8.0737019, -10.2975492, 7.3101511, -18.7500439, 18.3712502
9: -9.2718754, 9.3597107, -8.3838434, 8.4796343, -17.7515049, 17.7435532

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555361, upper bound: 14.5555208
time: 4.84 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555361, upper bound: 14.5569950
time: 6.20 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.2800217, 8.3890781, -10.4923725, 8.5620661, -18.8420868, 18.8814507
1: -8.4960232, 7.5222764, -8.6717739, 7.6668973, -16.1629200, 16.1940460
2: -10.6283522, 6.6079683, -10.8491735, 6.7762890, -17.4046383, 17.4571419
3: -12.4926853, 5.9221725, -12.7524014, 6.0384421, -18.5311260, 18.6745739
4: -11.2961988, 9.0026169, -11.5281582, 9.2009716, -20.4971676, 20.5307732
5: -9.7700434, 7.5998249, -9.9593945, 7.7599239, -17.5299683, 17.5592194
6: -9.1462193, 9.7942324, -9.3448944, 10.0057869, -19.1520042, 19.1391258
7: -11.2572899, 7.0449672, -11.4913406, 7.1840906, -18.4413795, 18.5363045
8: -11.6245165, 8.1919432, -11.8518353, 8.3851242, -20.0096397, 20.0437775
9: -9.4189701, 9.5067406, -9.6208763, 9.6908188, -19.1097870, 19.1276169

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5530488, upper bound: 14.5524630
time: 4.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5528666, upper bound: 14.5523741
time: 6.71 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.9690399, 8.1390772, -10.8550949, 8.8537388, -18.8227787, 18.9941673
1: -8.2358475, 7.3007317, -8.9749088, 7.9191046, -16.1549492, 16.2756405
2: -10.3109531, 6.4399042, -11.2310762, 7.0128350, -17.3237858, 17.6709785
3: -12.0832729, 5.7459292, -13.1919537, 6.2413340, -18.3246078, 18.9378796
4: -10.9707642, 8.7462597, -11.9256630, 9.5096226, -20.4803848, 20.6719208
5: -9.4636335, 7.3730083, -10.3069668, 8.0274200, -17.4910526, 17.6799755
6: -8.8808880, 9.5175037, -9.6678448, 10.3438015, -19.2246895, 19.1853447
7: -10.9504719, 6.8286619, -11.8827686, 7.4480648, -18.3985329, 18.7114296
8: -11.2642879, 7.9567003, -12.2648849, 8.6650610, -19.9293480, 20.2215843
9: -9.1325245, 9.2180643, -9.9483557, 10.0226374, -19.1551628, 19.1664200

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5544032, upper bound: 14.5548260
time: 9.84 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5544032, upper bound: 14.5568552
time: 4.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.79 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5534219, upper bound: 14.5530550
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5533528, upper bound: 14.5530281
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5555416, upper bound: 14.5556077
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5555416, upper bound: 14.5571389
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5532776, upper bound: 14.5530128
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5531839, upper bound: 14.5529659
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5544639, upper bound: 14.5549915
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5544639, upper bound: 14.5570582
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5532384, upper bound: 14.5525572
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5531089, upper bound: 14.5524866
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5555361, upper bound: 14.5555208
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5555361, upper bound: 14.5569950
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5530488, upper bound: 14.5524630
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5528666, upper bound: 14.5523741
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5544032, upper bound: 14.5548260
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -14.5544032, upper bound: 14.5568552

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.1366367, 7.4729986, -8.0846615, 6.6263790, -15.7630148, 15.5576572
1: -7.5372524, 6.7194786, -6.6517773, 5.9742956, -13.5115471, 13.3712559
2: -9.4452686, 5.9047632, -8.3462772, 5.2610559, -14.7063246, 14.2510395
3: -11.0855827, 5.2975445, -9.7617769, 4.7100906, -15.7956724, 15.0593214
4: -10.0617695, 8.0354729, -8.9285564, 7.1544704, -17.2162399, 16.9640293
5: -8.6775208, 6.7680693, -7.6509981, 5.9973626, -14.6748810, 14.4190674
6: -8.1374531, 8.7497454, -7.2163639, 7.7900195, -15.9274673, 15.9661093
7: -10.0559454, 6.2523384, -8.9615650, 5.4928594, -15.5488052, 15.2139025
8: -10.3153019, 7.3197789, -9.1033611, 6.5214467, -16.8367481, 16.4231377
9: -8.4022322, 8.4931669, -7.4423771, 7.5308604, -15.9330921, 15.9355440

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5534219, upper bound: 14.5530550
time: 4.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5534219, upper bound: 14.5530550
time: 4.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.9157772, 7.2960997, -9.0920057, 7.4196682, -16.3354454, 16.3881035
1: -7.3509016, 6.5606098, -7.4808388, 6.6613450, -14.0122461, 14.0414476
2: -9.2072029, 5.7459726, -9.3519630, 5.8049517, -15.0121546, 15.0979347
3: -10.8228931, 5.1661978, -11.0548897, 5.2422037, -16.0650978, 16.2210884
4: -9.8266068, 7.8508215, -10.0276031, 7.9821196, -17.8087273, 17.8784218
5: -8.4644394, 6.5966911, -8.5974207, 6.6854506, -15.1498890, 15.1941109
6: -7.9302573, 8.5537233, -8.0650930, 8.7330685, -16.6633263, 16.6188126
7: -9.8277168, 6.0596886, -10.0354156, 6.0562830, -15.8839989, 16.0951042
8: -10.0517378, 7.1511397, -10.2018633, 7.2754512, -17.3271885, 17.3530025
9: -8.1969252, 8.2818756, -8.3055763, 8.4035168, -16.6004410, 16.5874519

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 171

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5533528, upper bound: 14.5530281
time: 4.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5533528, upper bound: 14.5530281
time: 6.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.9247465, 7.3010063, -9.4564638, 7.7294493, -16.6541958, 16.7574673
1: -7.3631897, 6.5717564, -7.8078623, 6.9488430, -14.3120317, 14.3796186
2: -9.2360897, 5.8193226, -9.7857199, 6.1210260, -15.3571157, 15.6050415
3: -10.7865372, 5.1842108, -11.4753075, 5.4809413, -16.2674789, 16.6595192
4: -9.8420238, 7.8639188, -10.4119873, 8.3055820, -18.1476059, 18.2759056
5: -8.4650526, 6.6205029, -8.9805202, 7.0076075, -15.4726582, 15.6010199
6: -7.9736018, 8.5614128, -8.4330273, 9.0447464, -17.0183487, 16.9944401
7: -9.8517513, 6.1378365, -10.3971634, 6.4963717, -16.3481236, 16.5349960
8: -10.0762234, 7.1636000, -10.6927881, 7.5661612, -17.6423836, 17.8563881
9: -8.2114105, 8.3066216, -8.6930714, 8.7900772, -17.0014877, 16.9996929

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5537358, upper bound: 14.5538840
time: 5.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5536159, upper bound: 14.5536639
time: 4.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.9247465, 7.3010063, -8.9691067, 7.3371000, -16.2618465, 16.2701130
1: -7.3631897, 6.5717564, -7.4006968, 6.6028385, -13.9660273, 13.9724531
2: -9.2360897, 5.8193226, -9.2830086, 5.8489933, -15.0850830, 15.1023283
3: -10.7865372, 5.1842108, -10.8452711, 5.2079096, -15.9944468, 16.0294819
4: -9.8420238, 7.8639188, -9.8928223, 7.8996048, -17.7416286, 17.7567406
5: -8.4650526, 6.6205029, -8.5047970, 6.6520095, -15.1170616, 15.1252995
6: -7.9736018, 8.5614128, -8.0142832, 8.6042051, -16.5778065, 16.5756950
7: -9.8517513, 6.1378365, -9.9027071, 6.1623707, -16.0141220, 16.0405426
8: -10.0762234, 7.1636000, -10.1303272, 7.1965842, -17.2728081, 17.2939262
9: -8.2114105, 8.3066216, -8.2500143, 8.3444929, -16.5559044, 16.5566368

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 1

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5537358, upper bound: 14.5556575
time: 4.50 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5536159, upper bound: 14.5553950
time: 4.11 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.9942303, 7.3581438, -9.7465916, 7.9590373, -16.9532661, 17.1047363
1: -7.4168224, 6.6177859, -8.0424461, 7.1312551, -14.5480766, 14.6602306
2: -9.2949142, 5.8145680, -10.0544872, 6.2625031, -15.5574150, 15.8690548
3: -10.9125519, 5.2136736, -11.8573265, 5.6079273, -16.5204773, 17.0709991
4: -9.9077721, 7.9181700, -10.7288065, 8.5658913, -18.4736633, 18.6469727
5: -8.5375919, 6.6627045, -9.2411327, 7.1851430, -15.7227345, 15.9038343
6: -8.0102863, 8.6192455, -8.6585388, 9.3249340, -17.3352165, 17.2777824
7: -9.9049034, 6.1403022, -10.7063332, 6.5877552, -16.4926586, 16.8466358
8: -10.1506729, 7.2109337, -10.9773378, 7.8067861, -17.9574585, 18.1882668
9: -8.2735815, 8.3597145, -8.9333801, 8.9969635, -17.2705460, 17.2930927

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5524191, upper bound: 14.5524103
time: 15.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5523403, upper bound: 14.5521280
time: 3.31 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.7782526, 7.1840496, -10.7128334, 8.7210188, -17.4992676, 17.8968811
1: -7.2338667, 6.4619427, -8.8375111, 7.7857599, -15.0196257, 15.2994518
2: -9.0617514, 5.6583862, -11.0110779, 6.7758560, -15.8376074, 16.6694641
3: -10.6528091, 5.0859776, -13.1026821, 6.1125860, -16.7653923, 18.1886578
4: -9.6763296, 7.7368584, -11.7837124, 9.3544712, -19.0307961, 19.5205708
5: -8.3276463, 6.4951124, -10.1506405, 7.8364677, -16.1641083, 16.6457500
6: -7.8090658, 8.4264059, -9.4620647, 10.2279940, -18.0370522, 17.8884697
7: -9.6799421, 5.9537606, -11.7312260, 7.1132803, -16.7932224, 17.6849861
8: -9.8923807, 7.0462008, -12.0207472, 8.5270004, -18.4193783, 19.0669479
9: -8.0708923, 8.1542606, -9.7549496, 9.8309803, -17.9018707, 17.9092102

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 171

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 185

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5523503, upper bound: 14.5523928
time: 3.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5522519, upper bound: 14.5520712
time: 6.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.7729807, 7.1779251, -11.1465178, 9.0870838, -17.8600636, 18.3244438
1: -7.2352448, 6.4634161, -9.2156467, 8.1208544, -15.3560991, 15.6790628
2: -9.0758018, 5.7226467, -11.5219421, 7.1585474, -16.2343483, 17.2445889
3: -10.6026773, 5.0947647, -13.5789490, 6.4051352, -17.0078125, 18.6737137
4: -9.6771584, 7.7374735, -12.2227154, 9.7396374, -19.4167957, 19.9601898
5: -8.3157339, 6.5075984, -10.5932922, 8.2346563, -16.5503902, 17.1008911
6: -7.8377419, 8.4220581, -9.9085941, 10.6037884, -18.4415283, 18.3306503
7: -9.6909704, 6.0170403, -12.1537647, 7.6317949, -17.3227654, 18.1708050
8: -9.9009447, 7.0473170, -12.6040154, 8.8818302, -18.7827759, 19.6513309
9: -8.0732613, 8.1649704, -10.2090759, 10.2818747, -18.3551350, 18.3740463

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5525241, upper bound: 14.5531955
time: 5.14 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5524142, upper bound: 14.5529650
time: 8.94 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.7729807, 7.1779251, -10.7123566, 8.7396002, -17.5125771, 17.8902817
1: -7.2352448, 6.4634161, -8.8541069, 7.8197622, -15.0550070, 15.3175220
2: -9.0758018, 5.7226467, -11.0786781, 6.9138432, -15.9896431, 16.8013248
3: -10.6026773, 5.0947647, -13.0225649, 6.1609392, -16.7636166, 18.1173286
4: -9.6771584, 7.7374735, -11.7676973, 9.3847752, -19.0619335, 19.5051708
5: -8.3157339, 6.5075984, -10.1705036, 7.9208970, -16.2366314, 16.6781025
6: -7.8377419, 8.4220581, -9.5381241, 10.2103672, -18.0481071, 17.9601822
7: -9.6909704, 6.0170403, -11.7279243, 7.3364458, -17.0274162, 17.7449646
8: -9.9009447, 7.0473170, -12.1008368, 8.5531778, -18.4541225, 19.1481495
9: -8.0732613, 8.1649704, -9.8164759, 9.8900976, -17.9633579, 17.9814453

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5525241, upper bound: 14.5555526
time: 4.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5524142, upper bound: 14.5552867
time: 32.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -10.1784630, 8.3086472, -8.0417385, 6.5921021, -16.7705631, 16.3503838
1: -8.4142494, 7.4511232, -6.6169577, 5.9445219, -14.3587704, 14.0680799
2: -10.5214481, 6.5337586, -8.3035583, 5.2368298, -15.7582779, 14.8373165
3: -12.3746948, 5.8655024, -9.7089252, 4.6854053, -17.0601006, 15.5744276
4: -11.1911144, 8.9166851, -8.8823719, 7.1211390, -18.3122520, 17.7990532
5: -9.6770105, 7.5193319, -7.6100702, 5.9665985, -15.6436090, 15.1294022
6: -9.0506439, 9.7054548, -7.1799741, 7.7521210, -16.8027649, 16.8854294
7: -11.1550198, 6.9617639, -8.9199209, 5.4636679, -16.6186867, 15.8816853
8: -11.5064449, 8.1148033, -9.0548553, 6.4907427, -17.9971886, 17.1696587
9: -9.3264437, 9.4123774, -7.4047441, 7.4940219, -16.8204651, 16.8171215

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5532384, upper bound: 14.5525572
time: 6.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5532384, upper bound: 14.5525572
time: 6.14 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.9355478, 8.1133432, -9.0504169, 7.3866096, -17.3221569, 17.1637592
1: -8.2104130, 7.2756367, -7.4471326, 6.6325827, -14.8429909, 14.7227678
2: -10.2593946, 6.3594589, -9.3106556, 5.7817221, -16.0411167, 15.6701117
3: -12.0910473, 5.7181010, -11.0036707, 5.2183971, -17.3094425, 16.7217712
4: -10.9337339, 8.7114372, -9.9829903, 7.9496593, -18.8833923, 18.6944237
5: -9.4438705, 7.3285270, -8.5577297, 6.6558151, -16.0996838, 15.8862534
6: -8.8196182, 9.4882812, -8.0299330, 8.6964321, -17.5160503, 17.5182152
7: -10.9050989, 6.7444296, -9.9949903, 6.0284209, -16.9335175, 16.7394161
8: -11.2160130, 7.9271984, -10.1548767, 7.2457438, -18.4617558, 18.0820751
9: -9.1007824, 9.1764183, -8.2692394, 8.3678246, -17.4686069, 17.4456577

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5531089, upper bound: 14.5524866
time: 5.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5531089, upper bound: 14.5524866
time: 8.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.1216431, 8.2631321, -9.4162645, 7.6968017, -17.8184452, 17.6793976
1: -8.3637791, 7.4097567, -7.7750406, 6.9207506, -15.2845297, 15.1847973
2: -10.4722195, 6.5374403, -9.7450514, 6.0969143, -16.5691338, 16.2824917
3: -12.2673101, 5.8354411, -11.4257698, 5.4572773, -17.7245827, 17.2612114
4: -11.1361694, 8.8733768, -10.3686905, 8.2742767, -19.4104462, 19.2420654
5: -9.6132135, 7.4870715, -8.9414101, 6.9783583, -16.5915718, 16.4284821
6: -9.0167150, 9.6571779, -8.3979292, 9.0093622, -18.0260754, 18.0551033
7: -11.1110334, 6.9496779, -10.3582277, 6.4672709, -17.5783024, 17.3079052
8: -11.4398928, 8.0737019, -10.6461077, 7.5371447, -18.9770374, 18.7198067
9: -9.2718754, 9.3597107, -8.6580420, 8.7548637, -18.0267334, 18.0177517

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5528658, upper bound: 14.5538102
time: 6.51 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5536117, upper bound: 14.5536179
time: 4.86 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.1216431, 8.2631321, -8.9248343, 7.3011365, -17.4227753, 17.1879654
1: -8.3637791, 7.4097567, -7.3645973, 6.5719643, -14.9357433, 14.7743511
2: -10.4722195, 6.5374403, -9.2383289, 5.8223796, -16.2945995, 15.7757692
3: -12.2673101, 5.8354411, -10.7904482, 5.1821103, -17.4494190, 16.6258888
4: -11.1361694, 8.8733768, -9.8452520, 7.8647337, -19.0009041, 18.7186260
5: -9.6132135, 7.4870715, -8.4619293, 6.6198192, -16.2330322, 15.9490013
6: -9.0167150, 9.6571779, -7.9754648, 8.5651646, -17.5818787, 17.6326427
7: -11.1110334, 6.9496779, -9.8594246, 6.1304703, -17.2415047, 16.8091011
8: -11.4398928, 8.0737019, -10.0793781, 7.1644745, -18.6043663, 18.1530800
9: -9.2718754, 9.3597107, -8.2111998, 8.3054333, -17.5773048, 17.5709114

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5537199, upper bound: 14.5555377
time: 5.13 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5536117, upper bound: 14.5552975
time: 6.21 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -10.0311565, 8.1893978, -9.7012386, 7.9224567, -17.9536133, 17.8906326
1: -8.2902479, 7.3458071, -8.0052404, 7.0995536, -15.3898010, 15.3510475
2: -10.3662767, 6.4406662, -10.0085754, 6.2354970, -16.6017685, 16.4492416
3: -12.1962767, 5.7794538, -11.8014088, 5.5818777, -17.7781506, 17.5808620
4: -11.0323200, 8.7946777, -10.6798592, 8.5299740, -19.5622940, 19.4745369
5: -9.5329542, 7.4099846, -9.1976700, 7.1523008, -16.6852493, 16.6076508
6: -8.9197874, 9.5703812, -8.6188841, 9.2846775, -18.2044640, 18.1892643
7: -10.9992628, 6.8473487, -10.6616840, 6.5561113, -17.5553741, 17.5090313
8: -11.3368788, 8.0019608, -10.9250889, 7.7738600, -19.1107388, 18.9270496
9: -9.1928425, 9.2757444, -8.8937445, 8.9571400, -18.1499805, 18.1694889

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 184

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 232

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5529389, upper bound: 14.5524067
time: 5.46 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5529389, upper bound: 14.5524067
time: 4.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.7897320, 7.9950485, -10.6701508, 8.6870213, -18.4767532, 18.6651974
1: -8.0867367, 7.1709266, -8.8028240, 7.7561240, -15.8428612, 15.9737473
2: -10.1053562, 6.2672043, -10.9686489, 6.7519274, -16.8572845, 17.2358532
3: -11.9129667, 5.6327782, -13.0502739, 6.0884223, -18.0013866, 18.6830521
4: -10.7756252, 8.5902519, -11.7379093, 9.3207121, -20.0963364, 20.3281612
5: -9.3007355, 7.2203703, -10.1103115, 7.8059220, -17.1066551, 17.3306808
6: -8.6895008, 9.3541069, -9.4253407, 10.1903372, -18.8798370, 18.7794476
7: -10.7500544, 6.6312742, -11.6897335, 7.0850334, -17.8350868, 18.3210068
8: -11.0473022, 7.8154421, -11.9725685, 8.4962244, -19.5435257, 19.7880096
9: -8.9683475, 9.0406590, -9.7177935, 9.7941704, -18.7625179, 18.7584515

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 232

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5528154, upper bound: 14.5523371
time: 4.05 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5528154, upper bound: 14.5523370
time: 6.25 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.9690399, 8.1390772, -11.1041708, 9.0526953, -19.0217361, 19.2432480
1: -8.2358475, 7.3007317, -9.1810980, 8.0913353, -16.3271809, 16.4818306
2: -10.3109531, 6.4399042, -11.4789677, 7.1329165, -17.4438629, 17.9188709
3: -12.0832729, 5.7459292, -13.5267143, 6.3807707, -18.4640427, 19.2726402
4: -10.9707642, 8.7462597, -12.1770859, 9.7063065, -20.6770706, 20.9233456
5: -9.4636335, 7.3730083, -10.5524368, 8.2036648, -17.6672974, 17.9254456
6: -8.8808880, 9.5175037, -9.8715944, 10.5662518, -19.4471397, 19.3890991
7: -10.9504719, 6.8286619, -12.1124048, 7.6018000, -18.5522690, 18.9410667
8: -11.2642879, 7.9567003, -12.5550146, 8.8510380, -20.1153259, 20.5117111
9: -9.1325245, 9.2180643, -10.1720934, 10.2447014, -19.3772259, 19.3901558

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5524631, upper bound: 14.5530488
time: 5.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5523741, upper bound: 14.5528666
time: 5.56 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.9690399, 8.1390772, -10.6678219, 8.7034988, -18.6725388, 18.8068962
1: -8.2358475, 7.3007317, -8.8177299, 7.7888117, -16.0246544, 16.1184616
2: -10.3109531, 6.4399042, -11.0336533, 6.8869896, -17.1979408, 17.4735565
3: -12.0832729, 5.7459292, -12.9675751, 6.1353536, -18.2186260, 18.7134991
4: -10.9707642, 8.7462597, -11.7197695, 9.3496370, -20.3203983, 20.4660301
5: -9.4636335, 7.3730083, -10.1276665, 7.8885016, -17.3521347, 17.5006752
6: -8.8808880, 9.5175037, -9.4992294, 10.1708298, -19.0517178, 19.0167313
7: -10.9504719, 6.8286619, -11.6842546, 7.3051324, -18.2556038, 18.5129147
8: -11.2642879, 7.9567003, -12.0495710, 8.5208130, -19.7851009, 20.0062695
9: -9.1325245, 9.2180643, -9.7775211, 9.8508205, -18.9833431, 18.9955864

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5524631, upper bound: 14.5553921
time: 4.96 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5523741, upper bound: 14.5551605
time: 6.25 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 13.24 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5534219, upper bound: 14.5530550
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5534219, upper bound: 14.5530550
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5533528, upper bound: 14.5530281
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5533528, upper bound: 14.5530281
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5537358, upper bound: 14.5538840
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5536159, upper bound: 14.5536639
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5537358, upper bound: 14.5556575
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5536159, upper bound: 14.5553950
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5524191, upper bound: 14.5524103
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5523403, upper bound: 14.5521280
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5523503, upper bound: 14.5523928
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5522519, upper bound: 14.5520712
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5525241, upper bound: 14.5531955
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5524142, upper bound: 14.5529650
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5525241, upper bound: 14.5555526
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5524142, upper bound: 14.5552867
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5532384, upper bound: 14.5525572
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5532384, upper bound: 14.5525572
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5531089, upper bound: 14.5524866
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5531089, upper bound: 14.5524866
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5528658, upper bound: 14.5538102
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5536117, upper bound: 14.5536179
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5537199, upper bound: 14.5555377
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5536117, upper bound: 14.5552975
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5529389, upper bound: 14.5524067
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5529389, upper bound: 14.5524067
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5528154, upper bound: 14.5523371
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5528154, upper bound: 14.5523370
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5524631, upper bound: 14.5530488
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5523741, upper bound: 14.5528666
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5524631, upper bound: 14.5553921
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.24
Output dim: 7, lower bound: -14.5523741, upper bound: 14.5551605

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.8871155, 7.2709875, -8.0846615, 6.6263790, -15.5134935, 15.3556480
1: -7.3249979, 6.5407753, -6.6517773, 5.9742956, -13.2992935, 13.1925526
2: -9.1792297, 5.7433615, -8.3462772, 5.2610559, -14.4402857, 14.0896387
3: -10.7806463, 5.1512470, -9.7617769, 4.7100906, -15.4907370, 14.9130239
4: -9.7907028, 7.8259010, -8.9285564, 7.1544704, -16.9451733, 16.7544556
5: -8.4314556, 6.5824938, -7.6509981, 5.9973626, -14.4288177, 14.2334909
6: -7.9125023, 8.5177460, -7.2163639, 7.7900195, -15.7025185, 15.7341089
7: -9.7848797, 6.0564342, -8.9615650, 5.4928594, -15.2777386, 15.0179996
8: -10.0261269, 7.1264062, -9.1033611, 6.5214467, -16.5475731, 16.2297668
9: -8.1743851, 8.2575932, -7.4423771, 7.5308604, -15.7052460, 15.6999693

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5524969, upper bound: 14.5519354
time: 4.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5521869, upper bound: 14.5518105
time: 3.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -10.4818726, 8.5472460, -8.0846615, 6.6263790, -17.1082516, 16.6319084
1: -8.6571484, 7.6522255, -6.6517773, 5.9742956, -14.6314430, 14.3040028
2: -10.8188934, 6.7129145, -8.3462772, 5.2610559, -16.0799484, 15.0591917
3: -12.7829552, 6.0189600, -9.7617769, 4.7100906, -17.4930420, 15.7807369
4: -11.5134716, 9.1873064, -8.9285564, 7.1544704, -18.6679420, 18.1158638
5: -9.9498825, 7.7310033, -7.6509981, 5.9973626, -15.9472427, 15.3820019
6: -9.3097677, 9.9890442, -7.2163639, 7.7900195, -17.0997849, 17.2054081
7: -11.4582596, 7.1142550, -8.9615650, 5.4928594, -16.9511185, 16.0758171
8: -11.8311253, 8.3686247, -9.1033611, 6.5214467, -18.3525677, 17.4719810
9: -9.6019754, 9.6689777, -7.4423771, 7.5308604, -17.1328354, 17.1113548

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5524969, upper bound: 14.5519354
time: 5.12 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5521869, upper bound: 14.5518105
time: 4.25 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.6788492, 7.1024876, -9.0920057, 7.4196682, -16.0985184, 16.1944923
1: -7.1475849, 6.3901024, -7.4808388, 6.6613450, -13.8089294, 13.8709412
2: -8.9536219, 5.5921183, -9.3519630, 5.8049517, -14.7585735, 14.9440804
3: -10.5277252, 5.0284863, -11.0548897, 5.2422037, -15.7699270, 16.0833740
4: -9.5657587, 7.6506519, -10.0276031, 7.9821196, -17.5478783, 17.6782532
5: -8.2277737, 6.4213185, -8.5974207, 6.6854506, -14.9132242, 15.0187397
6: -7.7194748, 8.3311195, -8.0650930, 8.7330685, -16.4525414, 16.3962078
7: -9.5673971, 5.8775330, -10.0354156, 6.0562830, -15.6236801, 15.9129486
8: -9.7764530, 6.9676189, -10.2018633, 7.2754512, -17.0519047, 17.1694832
9: -7.9779367, 8.0599356, -8.3055763, 8.4035168, -16.3814526, 16.3655109

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5523857, upper bound: 14.5518903
time: 4.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5521291, upper bound: 14.5517890
time: 4.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.1949368, 8.3181992, -9.0920057, 7.4196682, -17.6146049, 17.4102020
1: -8.4182529, 7.4342122, -7.4808388, 6.6613450, -15.0795975, 14.9150505
2: -10.4983559, 6.4998326, -9.3519630, 5.8049517, -16.3033066, 15.8517952
3: -12.4504004, 5.8351889, -11.0548897, 5.2422037, -17.6926041, 16.8900776
4: -11.2082386, 8.9284077, -10.0276031, 7.9821196, -19.1903572, 18.9560089
5: -9.6833324, 7.4930744, -8.5974207, 6.6854506, -16.3687801, 16.0904961
6: -9.0251970, 9.7228413, -8.0650930, 8.7330685, -17.7582645, 17.7879333
7: -11.1495323, 6.8574696, -10.0354156, 6.0562830, -17.2058144, 16.8928852
8: -11.4738369, 8.1286106, -10.2018633, 7.2754512, -18.7492886, 18.3304749
9: -9.3284168, 9.3877220, -8.3055763, 8.4035168, -17.7319317, 17.6932983

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5523857, upper bound: 14.5518903
time: 4.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5521291, upper bound: 14.5517890
time: 10.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.2063236, 6.7229242, -9.2285175, 7.5472937, -15.7536144, 15.9514418
1: -6.7560234, 6.0591741, -7.6155806, 6.7857666, -13.5417900, 13.6747532
2: -8.4742241, 5.3362823, -9.5427504, 5.9651384, -14.4393597, 14.8790321
3: -9.9139738, 4.7756658, -11.2021675, 5.3480062, -15.2619772, 15.9778318
4: -9.0617380, 7.2590232, -10.1671085, 8.1126995, -17.1744385, 17.4261322
5: -7.7717185, 6.0838022, -8.7621765, 6.8349133, -14.6066303, 14.8459787
6: -7.3224907, 7.9060097, -8.2213202, 8.8368988, -16.1593876, 16.1273270
7: -9.0969791, 5.5751305, -10.1578779, 6.3113694, -15.4083462, 15.7330084
8: -9.2377949, 6.6168747, -10.4244289, 7.3900471, -16.6278419, 17.0413017
9: -7.5505476, 7.6419239, -8.4845991, 8.5746336, -16.1251793, 16.1265221

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 128

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5531396, upper bound: 14.5530726
time: 5.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5522688, upper bound: 14.5522309
time: 14.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.1976595, 7.5033822, -9.0094090, 7.3714390, -16.5690975, 16.5127907
1: -7.5715609, 6.7351556, -7.4305191, 6.6267872, -14.1983480, 14.1656742
2: -9.4632683, 5.8700480, -9.3055601, 5.8062162, -15.2694836, 15.1756067
3: -11.1877174, 5.2987428, -10.9417419, 5.2168694, -16.4045868, 16.2404842
4: -10.1424770, 8.0734892, -9.9331894, 7.9280276, -18.0705051, 18.0066757
5: -8.7022028, 6.7596946, -8.5510159, 6.6634140, -15.3656158, 15.3107109
6: -8.1576385, 8.8341856, -8.0145416, 8.6413918, -16.7990265, 16.8487282
7: -10.1533718, 6.1270809, -9.9294167, 6.1201601, -16.2735310, 16.0564976
8: -10.3171635, 7.3592863, -10.1615219, 7.2216129, -17.5387764, 17.5208092
9: -8.3991995, 8.5004711, -8.2798786, 8.3651772, -16.7643776, 16.7803459

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 128

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5530537, upper bound: 14.5528911
time: 18.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5527965, upper bound: 14.5527975
time: 8.19 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.2063236, 6.7229242, -8.7294216, 7.1451921, -15.3515129, 15.4523458
1: -6.7560234, 6.0591741, -7.1988816, 6.4321709, -13.1881943, 13.2580528
2: -8.4742241, 5.3362823, -9.0277195, 5.6861019, -14.1603260, 14.3639994
3: -9.9139738, 4.7756658, -10.5580988, 5.0695419, -14.9835138, 15.3337650
4: -9.0617380, 7.2590232, -9.6349754, 7.6967573, -16.7584953, 16.8939991
5: -7.7717185, 6.0838022, -8.2750435, 6.4714131, -14.2431297, 14.3588457
6: -7.3224907, 7.9060097, -7.7928886, 8.3852453, -15.7077351, 15.6988974
7: -9.0969791, 5.5751305, -9.6520271, 5.9677911, -15.0647669, 15.2271576
8: -9.2377949, 6.6168747, -9.8491020, 7.0123158, -16.2501106, 16.4659767
9: -7.5505476, 7.6419239, -8.0306120, 8.1199379, -15.6704845, 15.6725359

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 171

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5549298, upper bound: 14.5548003
time: 4.43 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5546298, upper bound: 14.5547174
time: 5.54 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 10.67 + 593.75 = 604.42 seconds
