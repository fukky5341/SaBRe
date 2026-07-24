## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 154.56034074419998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-87.9151230, 68.5816803, -87.9151230, 68.5816803, -156.4967957, 156.4967957)
1: (-70.3884354, 62.2707710, -70.3884354, 62.2707710, -132.6591797, 132.6591644)
2: (-94.4465866, 64.4209595, -94.4465866, 64.4209595, -158.8675385, 158.8675537)
3: (-99.7921600, 55.1154213, -99.7921600, 55.1154213, -154.9075623, 154.9075775)
4: (-103.3198853, 65.3530502, -103.3198853, 65.3530502, -168.6729431, 168.6729431)
5: (-81.0754700, 65.9836578, -81.0754700, 65.9836578, -147.0591278, 147.0591278)
6: (-83.0478287, 77.3565063, -83.0478287, 77.3565063, -160.4043274, 160.4043274)
7: (-88.2885132, 75.0491028, -88.2885132, 75.0491028, -163.3376160, 163.3376160)
8: (-104.6999664, 72.3314667, -104.6999664, 72.3314667, -177.0314331, 177.0314331)
9: (-84.2536926, 75.4037476, -84.2536926, 75.4037476, -159.6574249, 159.6574249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.85 + 13.26 = 14.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -154.7150558, upper bound: 154.7150558

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7094079, upper bound: 154.7100752
time: 10.72 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7136831, upper bound: 154.7136831
time: 10.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 21.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 21.05
Output dim: 4, lower bound: -154.7094079, upper bound: 154.7100752
NS_A2, status: Status.UNKNOWN, split count: 1, time: 21.05
Output dim: 4, lower bound: -154.7136831, upper bound: 154.7136831

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -78.1110840, 60.8416214, -84.2285233, 65.6670074, -143.7780914, 145.0701447
1: -62.1475143, 55.3199234, -67.2769165, 59.6561127, -121.8036270, 122.5968399
2: -83.7355194, 57.4565086, -90.4149170, 61.8032837, -145.5387573, 147.8713989
3: -88.4660645, 49.0245056, -95.5290833, 52.8229332, -141.2890015, 144.5535889
4: -92.5861588, 57.3921318, -99.3052063, 62.3420029, -154.9281464, 156.6973267
5: -71.8089142, 58.6708832, -77.5844879, 63.2328873, -135.0417938, 136.2553711
6: -73.9255219, 68.6816177, -79.6215744, 74.0920715, -148.0175781, 148.3031616
7: -78.4739304, 66.8039246, -84.5962753, 71.9510803, -150.4250031, 151.4001923
8: -92.7796097, 64.0383682, -100.2095566, 69.2089310, -161.9885254, 164.2479248
9: -75.2036514, 66.6150513, -80.8588943, 72.0851746, -147.2888184, 147.4739380

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6469897, upper bound: 154.6407124
time: 11.86 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7079469, upper bound: 154.7086554
time: 10.08 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -82.4793930, 64.2899170, -85.7685165, 66.8867722, -149.3661652, 150.0584259
1: -65.8094025, 58.4177246, -68.5805511, 60.7502022, -126.5595627, 126.9982758
2: -88.5034561, 60.5652313, -92.1007690, 62.8974571, -151.4009094, 152.6660004
3: -93.5097885, 51.7365265, -97.3124008, 53.7803459, -147.2901306, 149.0489197
4: -97.4018021, 60.9177780, -100.9806824, 63.6041298, -161.0059357, 161.8984528
5: -75.9326935, 61.9275970, -79.0450439, 64.3821335, -140.3148193, 140.9726410
6: -77.9988632, 72.5447617, -81.0534210, 75.4568710, -153.4556885, 153.5981293
7: -82.8456421, 70.4815216, -86.1400604, 73.2447205, -156.0903320, 156.6215668
8: -98.0842896, 67.7380829, -102.0872650, 70.5174026, -168.6016693, 169.8253326
9: -79.2458572, 70.5169907, -82.2753906, 73.4750595, -152.7209167, 152.7923279

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6498743, upper bound: 154.6433765
time: 12.09 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7124971, upper bound: 154.7124971
time: 8.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 21.78 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.78
Output dim: 4, lower bound: -154.6469897, upper bound: 154.6407124
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.78
Output dim: 4, lower bound: -154.7079469, upper bound: 154.7086554
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.78
Output dim: 4, lower bound: -154.6498743, upper bound: 154.6433765
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.78
Output dim: 4, lower bound: -154.7124971, upper bound: 154.7124971

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -70.4228745, 54.6894150, -66.4248047, 51.2497292, -121.6725922, 121.1142044
1: -55.6461411, 49.8676529, -51.9296989, 47.0211754, -102.6673050, 101.7973251
2: -75.2288055, 52.0178146, -70.5255814, 49.3287354, -124.5575256, 122.5433807
3: -79.5048599, 44.3138008, -74.5316162, 42.0020638, -121.5069199, 118.8454132
4: -84.4666595, 50.8490295, -81.3808289, 46.3640289, -130.8306732, 132.2298279
5: -64.4108276, 52.8829575, -60.1173477, 49.7720261, -114.1828537, 113.0003052
6: -66.8773804, 61.8791733, -63.4553528, 58.2953033, -125.1726761, 125.3345108
7: -70.8017960, 60.3550034, -66.8436661, 57.0650177, -127.8668137, 127.1986694
8: -83.4577637, 57.5659256, -78.4479218, 54.0891304, -137.5468903, 136.0138397
9: -68.2179413, 59.5853157, -64.9767456, 55.3823090, -123.6002426, 124.5620422

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6359814, upper bound: 154.6294650
time: 12.07 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6370444, upper bound: 154.6303505
time: 12.55 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -78.1110840, 60.8416214, -81.7621460, 63.7012024, -141.8122864, 142.6037598
1: -62.1475143, 55.3199234, -65.1951294, 57.9080200, -120.0555344, 120.5150528
2: -83.7355194, 57.4565086, -87.6928787, 60.0546951, -143.7901917, 145.1493683
3: -88.4660645, 49.0245056, -92.6549377, 51.3104820, -139.7765503, 141.6794281
4: -92.5861588, 57.3921318, -96.6807480, 60.2719307, -152.8580933, 154.0728760
5: -71.8089142, 58.6708832, -75.2192383, 61.3835411, -133.1924438, 133.8901215
6: -73.9255219, 68.6816177, -77.3582916, 71.9089050, -145.8343964, 146.0399017
7: -78.4739304, 66.8039246, -82.1314621, 69.8784637, -148.3523865, 148.9353485
8: -92.7796097, 64.0383682, -97.2264862, 67.1470490, -159.9266510, 161.2648315
9: -75.2036514, 66.6150513, -78.6092148, 69.8383255, -145.0419769, 145.2242432

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6406068, upper bound: 154.6474466
time: 13.92 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6406068, upper bound: 154.7086554
time: 19.18 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -74.7671814, 58.1178017, -67.8828430, 52.4065094, -127.1736908, 126.0006409
1: -59.2831345, 52.9471054, -53.1625824, 48.0608902, -107.3440170, 106.1096878
2: -79.9682083, 55.1087112, -72.1232681, 50.3659096, -130.3340912, 127.2319794
3: -84.5132294, 47.0102005, -76.2197495, 42.9050865, -127.4183121, 123.2299500
4: -89.2509308, 54.3566170, -82.9525833, 47.5647964, -136.8157349, 137.3091888
5: -68.5117035, 56.1230927, -61.4994240, 50.8658485, -119.3775482, 117.6225128
6: -70.9274292, 65.7184906, -64.8103485, 59.5855904, -130.5130157, 130.5288391
7: -75.1480789, 64.0087357, -68.3062515, 58.2937508, -133.4418335, 132.3149872
8: -88.7273941, 61.2432671, -80.2249451, 55.3250847, -144.0524750, 141.4682159
9: -72.2363815, 63.4594116, -66.3222809, 56.6946487, -128.9310150, 129.7816925

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6386053, upper bound: 154.6320285
time: 12.77 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6392308, upper bound: 154.6324711
time: 12.20 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -82.4793930, 64.2899170, -83.2981796, 64.9178848, -147.3972778, 147.5880890
1: -65.8094025, 58.4177246, -66.4947433, 59.0004311, -124.8098297, 124.9124451
2: -88.5034561, 60.5652313, -89.3745728, 61.1455612, -149.6490173, 149.9398041
3: -93.5097885, 51.7365265, -94.4350815, 52.2653465, -145.7751312, 146.1716003
4: -97.4018021, 60.9177780, -98.3510666, 61.5330467, -158.9348450, 159.2688446
5: -75.9326935, 61.9275970, -76.6759415, 62.5312157, -138.4639130, 138.6035461
6: -77.9988632, 72.5447617, -78.7867508, 73.2695007, -151.2683563, 151.3314819
7: -82.8456421, 70.4815216, -83.6712494, 71.1681671, -154.0137482, 154.1527405
8: -98.0842896, 67.7380829, -99.0993576, 68.4526215, -166.5368805, 166.8374023
9: -79.2458572, 70.5169907, -80.0219803, 71.2244644, -150.4703217, 150.5389709

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 166

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6433765, upper bound: 154.6498743
time: 9.43 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6433765, upper bound: 154.7124971
time: 13.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 23.85 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.85
Output dim: 4, lower bound: -154.6359814, upper bound: 154.6294650
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.85
Output dim: 4, lower bound: -154.6370444, upper bound: 154.6303505
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.85
Output dim: 4, lower bound: -154.6406068, upper bound: 154.6474466
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.85
Output dim: 4, lower bound: -154.6406068, upper bound: 154.7086554
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 23.85
Output dim: 4, lower bound: -154.6386053, upper bound: 154.6320285
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.85
Output dim: 4, lower bound: -154.6392308, upper bound: 154.6324711
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 23.85
Output dim: 4, lower bound: -154.6433765, upper bound: 154.6498743
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.85
Output dim: 4, lower bound: -154.6433765, upper bound: 154.7124971

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -61.8878632, 47.9617233, -63.7555161, 49.1437988, -111.0316391, 111.7172165
1: -48.6519966, 43.8407211, -49.7377586, 45.1378365, -93.7898331, 93.5784760
2: -65.9121628, 45.9265213, -67.6131592, 47.4302673, -113.3424301, 113.5396805
3: -69.7016907, 39.0275993, -71.4529114, 40.3523598, -110.0540314, 110.4805145
4: -75.0527725, 44.0806198, -78.5099487, 44.2093582, -119.2621231, 122.5905685
5: -56.3591232, 46.5311012, -57.5908890, 47.7866173, -104.1457062, 104.1219864
6: -58.9704742, 54.3771553, -60.9961395, 55.9561920, -114.9266586, 115.3732910
7: -62.3103333, 53.2145920, -64.1977234, 54.8354492, -117.1457825, 117.4123001
8: -73.2653885, 50.5017357, -75.2610016, 51.8871460, -125.1525192, 125.7627411
9: -60.3227921, 52.1073494, -62.5294113, 53.0360832, -113.3588715, 114.6367645

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6296025, upper bound: 154.6218482
time: 18.19 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6338123, upper bound: 154.6271081
time: 12.18 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -65.2767639, 50.6123085, -64.4043198, 49.6517944, -114.9285431, 115.0166321
1: -51.4015121, 46.2345848, -50.2621231, 45.5943069, -96.9958191, 96.4967041
2: -69.5957947, 48.3620720, -68.3184891, 47.8954086, -117.4912033, 116.6805573
3: -73.5629578, 41.1247902, -72.1948013, 40.7543793, -114.3173370, 113.3195877
4: -78.9233551, 46.6684647, -79.2230759, 44.7175179, -123.6408691, 125.8915405
5: -59.5259628, 49.0552788, -58.1995621, 48.2683144, -107.7942734, 107.2548370
6: -62.1407204, 57.3572083, -61.5985489, 56.5232315, -118.6639557, 118.9557571
7: -65.6997833, 56.0698242, -64.8404083, 55.3792419, -121.0790253, 120.9102325
8: -77.3024292, 53.2947426, -76.0335922, 52.4218140, -129.7242432, 129.3283234
9: -63.5127029, 55.0309677, -63.1301231, 53.5982018, -117.1109009, 118.1610794

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6306999, upper bound: 154.6225141
time: 11.56 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6348608, upper bound: 154.6279742
time: 12.32 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -61.0134735, 46.9926720, -81.7621460, 63.7012024, -124.7146759, 128.7548218
1: -47.4360580, 43.2015343, -65.1951294, 57.9080200, -105.3440781, 108.3966599
2: -64.6352997, 45.4873924, -87.6928787, 60.0546951, -124.6899796, 133.1802521
3: -68.2957535, 38.6569023, -92.6549377, 51.3104820, -119.6062317, 131.3118286
4: -75.4870224, 41.9992981, -96.6807480, 60.2719307, -135.7589569, 138.6800385
5: -55.0194321, 45.7228470, -75.2192383, 61.3835411, -116.4029694, 120.9420853
6: -58.4250526, 53.5161018, -77.3582916, 71.9089050, -130.3339386, 130.8743896
7: -61.4343147, 52.4935989, -82.1314621, 69.8784637, -131.3127747, 134.6250458
8: -71.8887329, 49.5572662, -97.2264862, 67.1470490, -139.0357819, 146.7837524
9: -59.9705887, 50.5815125, -78.6092148, 69.8383255, -129.8089142, 129.1907043

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6209757, upper bound: 154.6369888
time: 9.04 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6204433, upper bound: 154.6374724
time: 10.02 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -75.6670303, 58.8925705, -81.7621460, 63.7012024, -139.3682251, 140.6547241
1: -60.0846748, 53.5883102, -65.1951294, 57.9080200, -117.9926910, 118.7834396
2: -81.0385742, 55.7261200, -87.6928787, 60.0546951, -141.0932617, 143.4190063
3: -85.6200790, 47.5263901, -92.6549377, 51.3104820, -136.9305573, 140.1813354
4: -89.9919891, 55.3355865, -96.6807480, 60.2719307, -150.2639160, 152.0163269
5: -69.4646378, 56.8362579, -75.2192383, 61.3835411, -130.8481750, 132.0554962
6: -71.6836777, 66.5193253, -77.3582916, 71.9089050, -143.5925751, 143.8776245
7: -76.0339203, 64.7512207, -82.1314621, 69.8784637, -145.9123840, 146.8826599
8: -89.8267441, 61.9960556, -97.2264862, 67.1470490, -156.9737854, 159.2225342
9: -72.9766541, 64.3889694, -78.6092148, 69.8383255, -142.8149719, 142.9981689

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6209757, upper bound: 154.7035861
time: 16.80 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6204433, upper bound: 154.7057054
time: 10.54 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -66.2269440, 51.3867760, -65.1983643, 50.2845688, -116.5115128, 116.5851364
1: -52.2849274, 46.9123268, -50.9579468, 46.1652527, -98.4501724, 97.8702698
2: -70.6469193, 49.0150452, -69.1893921, 48.4560661, -119.1029816, 118.2044373
3: -74.7003708, 41.7214622, -73.1233597, 41.2456818, -115.9460449, 114.8448181
4: -79.8349533, 47.5812569, -80.0638657, 45.3923683, -125.2273254, 127.6451187
5: -60.4599419, 49.7802315, -58.9578247, 48.8679504, -109.3278961, 108.7380524
6: -63.0175972, 58.2145996, -62.3370895, 57.2325592, -120.2501526, 120.5516739
7: -66.6597519, 56.8710670, -65.6441650, 56.0521049, -122.7118454, 122.5152283
8: -78.5156326, 54.1742859, -77.0179520, 53.1040993, -131.6197357, 131.1921997
9: -64.3496780, 55.9776611, -63.8597565, 54.3313942, -118.6810608, 119.8374100

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6317973, upper bound: 154.6242144
time: 12.33 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6365295, upper bound: 154.6296728
time: 11.30 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -69.4783096, 53.9245148, -65.8203278, 50.7714272, -120.2497253, 119.7448273
1: -54.9107895, 49.2087402, -51.4591942, 46.6033592, -101.5141373, 100.6679382
2: -74.1783752, 51.3524361, -69.8662872, 48.9011536, -123.0795288, 121.2187195
3: -78.3993301, 43.7311592, -73.8348236, 41.6303520, -120.0296783, 117.5659790
4: -83.5614548, 50.0532494, -80.7492981, 45.8810387, -129.4424896, 130.8025360
5: -63.4862900, 52.1956863, -59.5400810, 49.3299255, -112.8162155, 111.7357635
6: -66.0569382, 61.0731277, -62.9135628, 57.7754593, -123.8323898, 123.9866943
7: -69.9044342, 59.6058273, -66.2609863, 56.5729675, -126.4774017, 125.8668137
8: -82.3925400, 56.8532257, -77.7576218, 53.6175385, -136.0100708, 134.6108398
9: -67.4082565, 58.7691689, -64.4363403, 54.8691711, -122.2774277, 123.2054901

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6326227, upper bound: 154.6245952
time: 12.17 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6370874, upper bound: 154.6300531
time: 14.03 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -64.8345795, 50.0017662, -83.2981796, 64.9178848, -129.7524719, 133.2999420
1: -50.6013718, 45.8979492, -66.4947433, 59.0004311, -109.6018066, 112.3926849
2: -68.7912064, 48.2071762, -89.3745728, 61.1455612, -129.9367676, 137.5817413
3: -72.6958542, 41.0185890, -94.4350815, 52.2653465, -124.9611893, 135.4536591
4: -79.6702118, 45.0614700, -98.3510666, 61.5330467, -141.2032623, 143.4125366
5: -58.6162605, 48.5807838, -76.6759415, 62.5312157, -121.1474686, 125.2567291
6: -61.9852524, 56.8900986, -78.7867508, 73.2695007, -135.2547455, 135.6768494
7: -65.2527542, 55.7268562, -83.6712494, 71.1681671, -136.4208832, 139.3981018
8: -76.5235519, 52.7604256, -99.0993576, 68.4526215, -144.9761200, 151.8597870
9: -63.5092888, 53.9645233, -80.0219803, 71.2244644, -134.7337494, 133.9865112

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6386053
time: 8.66 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6225817, upper bound: 154.6392308
time: 10.46 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -80.0182724, 62.3286095, -83.2981796, 64.9178848, -144.9361572, 145.6267853
1: -63.7334671, 56.6745224, -66.4947433, 59.0004311, -122.7339020, 123.1692657
2: -85.7877960, 58.8206749, -89.3745728, 61.1455612, -146.9333496, 148.1952057
3: -90.6423111, 50.2274284, -94.4350815, 52.2653465, -142.9076538, 144.6625061
4: -94.7824707, 58.8527756, -98.3510666, 61.5330467, -156.3155212, 157.2038422
5: -73.5729752, 60.0822754, -76.6759415, 62.5312157, -136.1041870, 136.7582092
6: -75.7399292, 70.3668365, -78.7867508, 73.2695007, -149.0094299, 149.1535950
7: -80.3863068, 68.4134293, -83.6712494, 71.1681671, -151.5543976, 152.0846863
8: -95.1086044, 65.6820602, -99.0993576, 68.4526215, -163.5612030, 164.7814026
9: -77.0006256, 68.2752304, -80.0219803, 71.2244644, -148.2250519, 148.2972107

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6236331
time: 9.39 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6225817, upper bound: 154.7090299
time: 11.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 22.00 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6296025, upper bound: 154.6218482
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6338123, upper bound: 154.6271081
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6306999, upper bound: 154.6225141
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6348608, upper bound: 154.6279742
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6209757, upper bound: 154.6369888
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6204433, upper bound: 154.6374724
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6209757, upper bound: 154.7035861
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6204433, upper bound: 154.7057054
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6317973, upper bound: 154.6242144
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6365295, upper bound: 154.6296728
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6326227, upper bound: 154.6245952
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6370874, upper bound: 154.6300531
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6386053
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6225817, upper bound: 154.6392308
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6230221, upper bound: 154.6236331
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.00
Output dim: 4, lower bound: -154.6225817, upper bound: 154.7090299

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -56.5128059, 43.7232399, -51.1164665, 39.2168045, -95.7296066, 94.8397064
1: -44.2237358, 40.0636177, -39.4164734, 36.3016014, -80.5253372, 79.4800720
2: -60.0498238, 42.1138649, -53.8930664, 38.4809341, -98.5307541, 96.0069199
3: -63.4787140, 35.7133713, -56.8229675, 32.5830612, -96.0617676, 92.5363388
4: -69.2525711, 39.7273178, -64.8519058, 34.0493622, -103.3019333, 104.5792236
5: -51.2370491, 42.5046539, -45.5617714, 38.3314247, -89.5684738, 88.0664215
6: -54.0410309, 49.6479225, -49.4109993, 44.8604546, -98.9014893, 99.0589218
7: -56.9263306, 48.7037277, -51.5375214, 44.2155571, -101.1418915, 100.2412262
8: -66.8801117, 46.1288071, -60.3102036, 41.7087517, -108.5888672, 106.4390106
9: -55.4032288, 47.3140564, -50.9770813, 41.8577919, -97.2610092, 98.2911377

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6288152, upper bound: 154.6208644
time: 12.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6276745, upper bound: 154.6202176
time: 13.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -59.9422150, 46.4251213, -58.6226006, 45.1034775, -105.0456924, 105.0477142
1: -47.0392265, 42.4739571, -45.5316544, 41.5459671, -88.5851898, 88.0056152
2: -63.7861710, 44.5527000, -62.0249977, 43.8005180, -107.5866776, 106.5776978
3: -67.4506226, 37.8250237, -65.5169296, 37.2013893, -104.6520081, 103.3419266
4: -72.9666290, 42.4911003, -73.0097427, 40.0424080, -113.0090332, 115.5008392
5: -54.5004044, 45.0741196, -52.6993942, 43.9399834, -98.4403839, 97.7735138
6: -57.1906433, 52.6638298, -56.2980156, 51.4492264, -108.6398697, 108.9618454
7: -60.3701859, 51.5873947, -59.0825386, 50.5297394, -110.8999252, 110.6699219
8: -70.9524460, 48.9082985, -69.1574402, 47.7163010, -118.6687469, 118.0657349
9: -58.5494385, 50.3649902, -57.8496094, 48.4782333, -107.0276718, 108.2145996

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6326062, upper bound: 154.6257520
time: 11.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6318290, upper bound: 154.6252501
time: 11.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -59.8856010, 46.3599052, -51.7575150, 39.7175064, -99.6030807, 98.1174164
1: -46.9433670, 42.4408264, -39.9283638, 36.7478943, -83.6912613, 82.3691864
2: -63.7098961, 44.5436935, -54.5870285, 38.9411087, -102.6510010, 99.1307220
3: -67.3161011, 37.7886391, -57.5527649, 32.9765778, -100.2926636, 95.3414001
4: -73.1116104, 42.2904282, -65.5632706, 34.5481339, -107.6597443, 107.8536987
5: -54.3803940, 45.0180702, -46.1616936, 38.8066444, -93.1870422, 91.1797485
6: -57.1986694, 52.6066742, -50.0078125, 45.4180374, -102.6167068, 102.6144867
7: -60.3007965, 51.5491943, -52.1744232, 44.7533417, -105.0541382, 103.7236099
8: -70.9085083, 48.9001465, -61.0705299, 42.2317505, -113.1402588, 109.9706726
9: -58.5856133, 50.2091293, -51.5717697, 42.4070206, -100.9926300, 101.7808990

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6300249, upper bound: 154.6217242
time: 13.30 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6290180, upper bound: 154.6211283
time: 13.45 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -63.3090096, 49.0568962, -59.2474174, 45.5930710, -108.9020844, 108.3043137
1: -49.7706757, 44.8515434, -46.0316086, 41.9843254, -91.7549973, 90.8831482
2: -67.4429245, 46.9729958, -62.7021408, 44.2505569, -111.6934814, 109.6751328
3: -71.2863617, 39.9083710, -66.2316284, 37.5866699, -108.8730087, 106.1399994
4: -76.8128586, 45.0622673, -73.7026596, 40.5293846, -117.3422394, 118.7649231
5: -57.6465683, 47.5812645, -53.2856636, 44.4045525, -102.0511017, 100.8669281
6: -60.3410301, 55.6227074, -56.8814926, 51.9929962, -112.3340149, 112.5041962
7: -63.7367630, 54.4242592, -59.7040062, 51.0536041, -114.7903671, 114.1282654
8: -74.9641418, 51.6805382, -69.9009094, 48.2288780, -123.1930237, 121.5814438
9: -61.7183304, 53.2690315, -58.4288902, 49.0172157, -110.7355499, 111.6979218

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6337689, upper bound: 154.6265985
time: 13.46 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6329516, upper bound: 154.6261106
time: 11.03 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -58.5494003, 45.0480499, -73.0024872, 56.7920341, -115.3414307, 118.0505295
1: -45.4391823, 41.4753990, -58.0022850, 51.7124443, -97.1516190, 99.4776840
2: -61.9553642, 43.7310715, -78.1278229, 53.8006477, -115.7560120, 121.8588867
3: -65.4566650, 37.1347427, -82.5811539, 45.8843536, -111.3410034, 119.7158966
4: -72.8346634, 40.0183754, -87.0247192, 53.3223572, -126.1570206, 127.0430679
5: -52.6853104, 43.8911743, -66.9576797, 54.8823776, -107.5676727, 110.8488541
6: -56.1520309, 51.3619919, -69.2439957, 64.2107620, -120.3627625, 120.6059647
7: -58.9924507, 50.4327011, -73.4284286, 62.5522614, -121.5447083, 123.8611298
8: -68.9544525, 47.5448227, -86.7415466, 59.8907623, -128.8451843, 134.2863770
9: -57.7121696, 48.4354019, -70.5244217, 62.1467896, -119.8589554, 118.9598236

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 119

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6279901, upper bound: 154.6347553
time: 12.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6279235, upper bound: 154.6347553
time: 12.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -59.1368980, 45.5081940, -76.4984589, 59.5279541, -118.6648560, 122.0066528
1: -45.9105606, 41.8871346, -60.8425293, 54.1873093, -100.0978622, 102.7296600
2: -62.5924225, 44.1525002, -81.9290848, 56.3119545, -118.9043732, 126.0815887
3: -66.1300201, 37.4983330, -86.5659866, 48.0477905, -114.1778107, 124.0643082
4: -73.4788284, 40.4821854, -91.0030975, 56.0036774, -129.4824982, 131.4852905
5: -53.2386551, 44.3276291, -70.2222977, 57.4787903, -110.7174301, 114.5499115
6: -56.6985054, 51.8735809, -72.5108719, 67.2830353, -123.9815063, 124.3844528
7: -59.5756035, 50.9252281, -76.9144974, 65.4919510, -125.0675507, 127.8397141
8: -69.6510925, 48.0258179, -90.9142990, 62.7774506, -132.4285278, 138.9401245
9: -58.2551422, 48.9424133, -73.8018036, 65.1672287, -123.4223709, 122.7442169

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6285091, upper bound: 154.6354519
time: 12.56 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6284189, upper bound: 154.6354519
time: 12.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -72.6428452, 56.4995461, -73.0024872, 56.7920341, -129.4348755, 129.5020294
1: -57.5921707, 51.4508095, -58.0022850, 51.7124443, -109.3046112, 109.4530792
2: -77.7278519, 53.5720367, -78.1278229, 53.8006477, -131.5284882, 131.6998596
3: -82.1326828, 45.6511154, -82.5811539, 45.8843536, -128.0170288, 128.2322693
4: -86.7100601, 52.8956909, -87.0247192, 53.3223572, -140.0323944, 139.9203949
5: -66.6006622, 54.5933685, -66.9576797, 54.8823776, -121.4830399, 121.5510483
6: -68.8923492, 63.8626518, -69.2439957, 64.2107620, -133.1031036, 133.1066437
7: -73.0346985, 62.2294312, -73.4284286, 62.5522614, -135.5869598, 135.6578674
8: -86.2003479, 59.4814453, -86.7415466, 59.8907623, -146.0911102, 146.2229614
9: -70.2059021, 61.7147293, -70.5244217, 62.1467896, -132.3526917, 132.2391357

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6991621, upper bound: 154.7000002
time: 10.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7018055, upper bound: 154.7023688
time: 12.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -73.4405975, 57.1298676, -76.4984589, 59.5279541, -132.9685516, 133.6283264
1: -58.2476158, 52.0154076, -60.8425293, 54.1873093, -112.4349213, 112.8579407
2: -78.6017685, 54.1408081, -81.9290848, 56.3119545, -134.9137268, 136.0698853
3: -83.0502167, 46.1451263, -86.5659866, 48.0477905, -131.0980072, 132.7111206
4: -87.5780258, 53.5395508, -91.0030975, 56.0036774, -143.5816956, 144.5426483
5: -67.3562241, 55.1850433, -70.2222977, 57.4787903, -124.8350143, 125.4073410
6: -69.6317596, 64.5619736, -72.5108719, 67.2830353, -136.9147797, 137.0728455
7: -73.8269882, 62.8950806, -76.9144974, 65.4919510, -139.3189392, 139.8095703
8: -87.1563339, 60.1474876, -90.9142990, 62.7774506, -149.9337769, 151.0617676
9: -70.9393463, 62.4191208, -73.8018036, 65.1672287, -136.1065674, 136.2209167

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7003586, upper bound: 154.7017025
time: 10.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.7038294, upper bound: 154.7045408
time: 11.46 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -60.7780266, 47.0882912, -52.3294678, 40.1760635, -100.9540863, 99.4177551
1: -47.7756386, 43.0767899, -40.4173431, 37.1487350, -84.9243774, 83.4941101
2: -64.6962051, 45.1559677, -55.2095337, 39.3349991, -104.0311966, 100.3655014
3: -68.3858109, 38.3493729, -58.2240829, 33.3295517, -101.7153549, 96.5734482
4: -73.9573669, 43.1570053, -66.1569061, 35.0405464, -108.9979019, 109.3139038
5: -55.2590942, 45.6997185, -46.7130775, 39.2397461, -94.4988403, 92.4127960
6: -58.0231476, 53.4110832, -50.5328865, 45.9290466, -103.9521790, 103.9439545
7: -61.2018509, 52.3039627, -52.7517662, 45.2397919, -106.4416428, 105.0557251
8: -72.0512848, 49.7265854, -61.7756462, 42.7115555, -114.7628403, 111.5022278
9: -59.3698120, 51.1015930, -52.0874786, 42.9281693, -102.2979813, 103.1890640

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6311372, upper bound: 154.6232719
time: 12.18 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -154.6299129, upper bound: 154.6222959
time: 12.87 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 14.11 + 600.85 = 614.96 seconds
