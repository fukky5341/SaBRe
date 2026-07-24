## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 197.2433907684


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486)
1: (-87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961)
2: (-114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986)
3: (-122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255)
4: (-112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691)
5: (-100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425)
6: (-96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658)
7: (-105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948)
8: (-125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392)
9: (-96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 8.24 = 9.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -197.4408316, upper bound: 197.4408316

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4093851, upper bound: 197.4098506
time: 7.73 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4388501, upper bound: 197.4388501
time: 4.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.17 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.17
Output dim: 4, lower bound: -197.4093851, upper bound: 197.4098506
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.17
Output dim: 4, lower bound: -197.4388501, upper bound: 197.4388501

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -79.9444504, 63.3750114, -93.6110077, 74.2906799, -154.2351227, 156.9860229
1: -66.2200775, 56.2719460, -77.8618393, 65.9795837, -132.1996613, 134.1337891
2: -87.7009277, 57.5595970, -102.8611526, 67.3055725, -155.0064850, 160.4207458
3: -93.7524338, 49.1665268, -109.6692657, 57.5932503, -151.3456879, 158.8357544
4: -85.9368134, 66.0450745, -100.7016296, 77.4250488, -163.3618622, 166.7467041
5: -76.8184204, 59.8392181, -89.9155273, 70.0633163, -146.8817291, 149.7547302
6: -74.1077042, 70.4479294, -86.6628723, 82.6478958, -156.7556000, 157.1108093
7: -80.7441788, 67.8362045, -94.4707870, 79.3106003, -160.0547485, 162.3069611
8: -96.0069962, 65.3933868, -112.6965866, 76.9817963, -172.9888000, 178.0899506
9: -73.5597076, 71.9806671, -86.1536560, 84.4908371, -158.0505066, 158.1343231

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 249

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4034076, upper bound: 197.4040576
time: 7.87 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4037714, upper bound: 197.4043222
time: 6.62 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -99.1355057, 78.7032318, -103.5517502, 82.2159271, -181.3513794, 182.2549591
1: -82.5673141, 69.9038849, -86.3098831, 73.0283813, -155.5956573, 156.2137604
2: -108.9954758, 71.2377625, -113.8802032, 74.3851089, -183.3805847, 185.1179352
3: -116.1185074, 61.0004578, -121.2779465, 63.7236328, -179.8421326, 182.2783813
4: -106.6865845, 82.0365067, -111.4516983, 85.6835556, -192.3701477, 193.4881592
5: -95.2138977, 74.1955032, -99.4504166, 77.4904404, -172.7043152, 173.6459198
6: -91.7468567, 87.5859756, -95.8004303, 91.5136414, -183.2604980, 183.3863983
7: -100.0214844, 83.9518967, -104.4575272, 87.6625214, -187.6839905, 188.4094238
8: -119.4582748, 81.6738663, -124.8128204, 85.3636856, -204.8219452, 206.4866791
9: -91.2429123, 89.5477371, -95.2969131, 93.5584030, -184.8013000, 184.8446350

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4098506, upper bound: 197.4093851
time: 6.89 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4098506, upper bound: 197.4388501
time: 6.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.25 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.25
Output dim: 4, lower bound: -197.4034076, upper bound: 197.4040576
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.25
Output dim: 4, lower bound: -197.4037714, upper bound: 197.4043222
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.25
Output dim: 4, lower bound: -197.4098506, upper bound: 197.4093851
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.25
Output dim: 4, lower bound: -197.4098506, upper bound: 197.4388501

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -73.0316467, 57.9180489, -75.6849442, 60.1346016, -133.1662445, 133.6029663
1: -60.3441200, 51.3801651, -62.6311302, 53.2998581, -113.6439743, 114.0112839
2: -80.0673218, 52.6203270, -83.0660782, 54.5087700, -134.5760345, 135.6864014
3: -85.6491470, 44.8958778, -88.6635284, 46.5233040, -132.1724548, 133.5594025
4: -78.5099258, 60.3406448, -81.4453278, 62.6406708, -141.1506042, 141.7859497
5: -70.2130051, 54.6945190, -72.7830734, 56.7273750, -126.9403839, 127.4775925
6: -67.7568207, 64.3030548, -70.1905365, 66.7155838, -134.4724121, 134.4935760
7: -73.7721176, 62.0215416, -76.3979111, 64.2283859, -138.0005035, 138.4194183
8: -87.6442566, 59.6669273, -91.0238953, 62.1415443, -149.7857971, 150.6908112
9: -67.1719589, 65.6571884, -69.6005783, 68.1161194, -135.2880859, 135.2577667

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3990703, upper bound: 197.3998343
time: 9.03 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4012846, upper bound: 197.4017169
time: 6.64 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -75.3317871, 59.7350540, -82.0703354, 65.1801453, -140.5119324, 141.8053894
1: -62.3005295, 53.0110550, -68.0527954, 57.8200951, -120.1206207, 121.0638504
2: -82.6068497, 54.2722511, -90.1104736, 59.0871429, -141.6939697, 144.3827209
3: -88.3486023, 46.3182869, -96.1490173, 50.4647293, -138.8133240, 142.4672852
4: -80.9839172, 62.2458687, -88.3116684, 67.9164124, -148.9003296, 150.5575256
5: -72.4125366, 56.4119797, -78.8885956, 61.4874153, -133.8999481, 135.3005676
6: -69.8801880, 66.3476715, -76.0839462, 72.3854599, -142.2656250, 142.4316101
7: -76.1012115, 63.9697990, -82.8512955, 69.6323013, -145.7334900, 146.8210907
8: -90.4331894, 61.5680618, -98.7549820, 67.4131393, -157.8463135, 160.3230438
9: -69.3129883, 67.7797928, -75.5298309, 73.9788437, -143.2918091, 143.3096313

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3995486, upper bound: 197.4002069
time: 8.19 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4017135, upper bound: 197.4020573
time: 7.34 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -99.1355057, 78.7032318, -79.9444504, 63.3750114, -162.5104828, 158.6476593
1: -82.5673141, 69.9038849, -66.2200775, 56.2719460, -138.8392639, 136.1239624
2: -108.9954758, 71.2377625, -87.7009277, 57.5595970, -166.5550690, 158.9386597
3: -116.1185074, 61.0004578, -93.7524338, 49.1665268, -165.2850037, 154.7528839
4: -106.6865845, 82.0365067, -85.9368134, 66.0450745, -172.7316589, 167.9733124
5: -95.2138977, 74.1955032, -76.8184204, 59.8392181, -155.0531158, 151.0139160
6: -91.7468567, 87.5859756, -74.1077042, 70.4479294, -162.1947937, 161.6936798
7: -100.0214844, 83.9518967, -80.7441788, 67.8362045, -167.8576660, 164.6960754
8: -119.4582748, 81.6738663, -96.0069962, 65.3933868, -184.8516235, 177.6808624
9: -91.2429123, 89.5477371, -73.5597076, 71.9806671, -163.2235718, 163.1074371

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4040576, upper bound: 197.4034076
time: 6.83 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4043222, upper bound: 197.4037714
time: 8.22 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -99.1355057, 78.7032318, -99.1355057, 78.7032318, -177.8386688, 177.8386688
1: -82.5673141, 69.9038849, -82.5673141, 69.9038849, -152.4711609, 152.4711609
2: -108.9954758, 71.2377625, -108.9954758, 71.2377625, -180.2332306, 180.2332306
3: -116.1185074, 61.0004578, -116.1185074, 61.0004578, -177.1189117, 177.1189117
4: -106.6865845, 82.0365067, -106.6865845, 82.0365067, -188.7230530, 188.7230530
5: -95.2138977, 74.1955032, -95.2138977, 74.1955032, -169.4093933, 169.4093933
6: -91.7468567, 87.5859756, -91.7468567, 87.5859756, -179.3328247, 179.3328247
7: -100.0214844, 83.9518967, -100.0214844, 83.9518967, -183.9733887, 183.9733887
8: -119.4582748, 81.6738663, -119.4582748, 81.6738663, -201.1321259, 201.1321259
9: -91.2429123, 89.5477371, -91.2429123, 89.5477371, -180.7906494, 180.7906494

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4040576, upper bound: 197.4349020
time: 6.46 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4043222, upper bound: 197.4353032
time: 8.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 15.48 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.48
Output dim: 4, lower bound: -197.3990703, upper bound: 197.3998343
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.48
Output dim: 4, lower bound: -197.4012846, upper bound: 197.4017169
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.48
Output dim: 4, lower bound: -197.3995486, upper bound: 197.4002069
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.48
Output dim: 4, lower bound: -197.4017135, upper bound: 197.4020573
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.48
Output dim: 4, lower bound: -197.4040576, upper bound: 197.4034076
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.48
Output dim: 4, lower bound: -197.4043222, upper bound: 197.4037714
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.48
Output dim: 4, lower bound: -197.4040576, upper bound: 197.4349020
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.48
Output dim: 4, lower bound: -197.4043222, upper bound: 197.4353032

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -57.9427338, 46.0502853, -70.4626999, 56.0282097, -113.9709473, 116.5129852
1: -47.8214684, 40.8501358, -58.2938728, 49.6530418, -97.4745102, 99.1440125
2: -63.5096283, 41.9536591, -77.3342590, 50.8208275, -114.3304596, 119.2879181
3: -68.0391769, 35.6473465, -82.5662689, 43.3153915, -111.3545609, 118.2135925
4: -62.3432198, 48.0855103, -75.8566132, 58.3983650, -120.7415848, 123.9421234
5: -55.8076591, 43.5713387, -67.7981415, 52.8739662, -108.6816254, 111.3694687
6: -54.0077972, 51.0143890, -65.4401321, 62.1155243, -116.1233063, 116.4545212
7: -58.6748962, 49.5177422, -71.1705475, 59.9031448, -118.5780334, 120.6882858
8: -69.5280075, 47.2781334, -84.7654190, 57.8496094, -127.3776169, 132.0435486
9: -53.4291344, 52.0453949, -64.8390274, 63.3995895, -116.8287201, 116.8844223

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3932489, upper bound: 197.3936768
time: 6.16 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3932489, upper bound: 197.3998343
time: 5.45 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -65.8794403, 52.2954102, -72.9649277, 57.9946861, -123.8741226, 125.2603378
1: -54.3890953, 46.3860855, -60.3662300, 51.4006844, -105.7897797, 106.7523193
2: -72.2206726, 47.5700035, -80.0800705, 52.5894127, -124.8100891, 127.6500702
3: -77.3069305, 40.5152245, -85.4911652, 44.8560715, -122.1630020, 126.0063934
4: -70.8522568, 54.5207787, -78.5336304, 60.4258194, -131.2780762, 133.0544128
5: -63.3780975, 49.4197197, -70.1838226, 54.7208481, -118.0989304, 119.6035461
6: -61.2465439, 57.9995079, -67.7139282, 64.3172913, -125.5638351, 125.7134399
7: -66.6315079, 56.0905304, -73.6806335, 61.9724655, -128.6039581, 129.7711639
8: -79.0618134, 53.7890968, -87.7609787, 59.9042397, -138.9660492, 141.5500793
9: -60.6512718, 59.2058563, -67.1188202, 65.6608353, -126.3121033, 126.3246765

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3955396, upper bound: 197.3955912
time: 6.24 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3955396, upper bound: 197.4017169
time: 6.66 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -60.2420197, 47.8670578, -76.8394470, 61.0653992, -121.3074112, 124.7065048
1: -49.7753067, 42.4780426, -63.7083130, 54.1665840, -103.9418945, 106.1863556
2: -66.0479279, 43.6061211, -84.3665390, 55.3923912, -121.4403076, 127.9726410
3: -70.7364349, 37.0704155, -90.0398178, 47.2520943, -117.9885254, 127.1102295
4: -64.8158646, 49.9908714, -82.7122116, 63.6658783, -128.4817047, 132.7030792
5: -58.0063400, 45.2878685, -73.8942566, 57.6271324, -115.6334686, 119.1821289
6: -56.1291389, 53.0588417, -71.3235855, 67.7776871, -123.9068146, 124.3824310
7: -61.0022278, 51.4662247, -77.6131973, 65.2984772, -126.3007050, 129.0794067
8: -72.3168259, 49.1797638, -92.4851456, 63.1140747, -135.4308777, 141.6648865
9: -55.5688286, 54.1698914, -70.7597351, 69.2548294, -124.8236542, 124.9296265

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935548, upper bound: 197.3939135
time: 5.76 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935548, upper bound: 197.4002070
time: 5.56 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -68.1773148, 54.1091690, -79.3142090, 63.0113335, -131.1886292, 133.4233704
1: -56.3429832, 48.0154800, -65.7577209, 55.8949928, -112.2379684, 113.7732010
2: -74.7554169, 49.2199211, -87.0832596, 57.1420746, -131.8974915, 136.3031769
3: -80.0022659, 41.9368668, -92.9332886, 48.7755280, -128.7778015, 134.8701477
4: -73.3227768, 56.4227982, -85.3604202, 65.6712646, -138.9940491, 141.7832031
5: -65.5744629, 51.1348572, -76.2549057, 59.4534836, -125.0279465, 127.3897629
6: -63.3671684, 60.0410423, -73.5738220, 69.9542389, -133.3214111, 133.6148529
7: -68.9566498, 58.0363235, -80.0968933, 67.3466797, -136.3033295, 138.1331940
8: -81.8438416, 55.6883812, -95.4462814, 65.1454468, -146.9892883, 151.1346588
9: -62.7911453, 61.3247604, -73.0153503, 71.4899216, -134.2810516, 134.3400879

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3556151, upper bound: 197.3552974
time: 6.83 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3998998, upper bound: 197.4002612
time: 6.94 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -80.9900284, 64.3731308, -73.0316467, 57.9180489, -138.9080658, 137.4047546
1: -67.1509247, 57.0728683, -60.3441200, 51.3801651, -118.5310745, 117.4169922
2: -88.9628677, 58.2864723, -80.0673218, 52.6203270, -141.5831909, 138.3537445
3: -94.8495255, 49.7938385, -85.6491470, 44.8958778, -139.7454071, 135.4429932
4: -87.1935120, 67.0708923, -78.5099258, 60.3406448, -147.5341492, 145.5808105
5: -77.8680878, 60.6965294, -70.2130051, 54.6945190, -132.5626068, 130.9095306
6: -75.0712967, 71.4620056, -67.7568207, 64.3030548, -139.3743439, 139.2188263
7: -81.7263260, 68.6844559, -73.7721176, 62.0215416, -143.7478485, 142.4565735
8: -97.5274811, 66.6552429, -87.6442566, 59.6669273, -157.1943817, 154.2994995
9: -74.4901352, 72.9769440, -67.1719589, 65.6571884, -140.1473236, 140.1488953

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 168

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3998342, upper bound: 197.3990703
time: 8.40 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4017169, upper bound: 197.4012846
time: 5.98 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -87.6675110, 69.6542511, -75.3317871, 59.7350540, -147.4025574, 144.9860382
1: -72.8256454, 61.8019753, -62.3005295, 53.0110550, -125.8367004, 124.1025009
2: -96.3304672, 63.0746346, -82.6068497, 54.2722511, -150.6027222, 145.6814575
3: -102.6804047, 53.9181709, -88.3486023, 46.3182869, -148.9986725, 142.2667542
4: -94.3752365, 72.5914841, -80.9839172, 62.2458687, -156.6211090, 153.5753937
5: -84.2571259, 65.6748962, -72.4125366, 56.4119797, -140.6690979, 138.0874329
6: -81.2340698, 77.3911514, -69.8801880, 66.3476715, -147.5817413, 147.2713165
7: -88.4754868, 74.3349457, -76.1012115, 63.9697990, -152.4452820, 150.4361572
8: -105.6106262, 72.1767273, -90.4331894, 61.5680618, -167.1786652, 162.6099091
9: -80.6891785, 79.1061096, -69.3129883, 67.7797928, -148.4689636, 148.4190826

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4002069, upper bound: 197.3995486
time: 6.86 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4020573, upper bound: 197.4017135
time: 6.82 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -80.9900284, 64.3731308, -92.0331955, 73.0945358, -154.0845490, 156.4063110
1: -67.1509247, 57.0728683, -76.5306625, 64.8809509, -132.0318756, 133.6035309
2: -88.9628677, 58.2864723, -101.1530609, 66.1669846, -155.1298523, 159.4395294
3: -94.8495255, 49.7938385, -107.7886047, 56.6104317, -151.4599609, 157.5824432
4: -87.1935120, 67.0708923, -99.0544281, 76.1752396, -163.3687439, 166.1253204
5: -77.8680878, 60.6965294, -88.4246063, 68.9097595, -146.7778473, 149.1211395
6: -75.0712967, 71.4620056, -85.2191925, 81.2735596, -156.3448486, 156.6811829
7: -81.7263260, 68.6844559, -92.8564529, 77.9755554, -159.7018738, 161.5409088
8: -97.5274811, 66.6552429, -110.8734207, 75.7917480, -173.3192291, 177.5286560
9: -74.4901352, 72.9769440, -84.6815414, 83.0557251, -157.5458527, 157.6584778

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4317643, upper bound: 197.4316913
time: 5.80 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4326845, upper bound: 197.4326838
time: 6.72 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -87.6675110, 69.6542511, -94.5711670, 75.1020737, -162.7695923, 164.2254028
1: -72.8256454, 61.8019753, -78.6913071, 66.6797943, -139.5054321, 140.4932556
2: -96.3304672, 63.0746346, -103.9555054, 67.9882889, -164.3187561, 167.0301361
3: -102.6804047, 53.9181709, -110.7691803, 58.1814957, -160.8618774, 164.6873322
4: -94.3752365, 72.5914841, -101.7866516, 78.2773743, -172.6526031, 174.3781281
5: -84.2571259, 65.6748962, -90.8530884, 70.8041534, -155.0612793, 156.5279846
6: -81.2340698, 77.3911514, -87.5619812, 83.5290527, -164.7631226, 164.9531250
7: -88.4754868, 74.3349457, -95.4252090, 80.1242294, -168.5997162, 169.7601624
8: -105.6106262, 72.1767273, -113.9477844, 77.8950272, -183.5056305, 186.1245117
9: -80.6891785, 79.1061096, -87.0421753, 85.3920898, -166.0812683, 166.1482849

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 249

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4323521, upper bound: 197.4322804
time: 6.57 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4332869, upper bound: 197.4332890
time: 5.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 12.83 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3932489, upper bound: 197.3936768
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3932489, upper bound: 197.3998343
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3955396, upper bound: 197.3955912
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3955396, upper bound: 197.4017169
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3935548, upper bound: 197.3939135
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3935548, upper bound: 197.4002070
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3556151, upper bound: 197.3552974
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3998998, upper bound: 197.4002612
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.3998342, upper bound: 197.3990703
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.4017169, upper bound: 197.4012846
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.4002069, upper bound: 197.3995486
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.4020573, upper bound: 197.4017135
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.4317643, upper bound: 197.4316913
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.4326845, upper bound: 197.4326838
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.4323521, upper bound: 197.4322804
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 12.83
Output dim: 4, lower bound: -197.4332869, upper bound: 197.4332890

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -57.9427338, 46.0502853, -56.8708534, 45.1799011, -103.1226349, 102.9211426
1: -47.8214684, 40.8501358, -46.7256660, 40.0073929, -87.8288574, 87.5758057
2: -63.5096283, 41.9536591, -62.2637253, 41.1164856, -104.6261063, 104.2173843
3: -68.0391769, 35.6473465, -66.7398453, 34.9410629, -102.9802170, 102.3871689
4: -62.3432198, 48.0855103, -61.1706085, 47.0911636, -109.4343872, 109.2561111
5: -55.8076591, 43.5713387, -54.7701302, 42.7085228, -98.5161819, 98.3414688
6: -54.0077972, 51.0143890, -52.9590340, 49.9874458, -103.9952393, 103.9734192
7: -58.6748962, 49.5177422, -57.5324364, 48.4921188, -107.1670151, 107.0501633
8: -69.5280075, 47.2781334, -68.1527634, 46.3337097, -115.8617172, 115.4308853
9: -53.4291344, 52.0453949, -52.3251114, 50.9505348, -104.3796692, 104.3704987

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3435790, upper bound: 197.3448669
time: 7.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3914039, upper bound: 197.3917330
time: 7.26 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -57.9427338, 46.0502853, -75.8119965, 60.2978096, -118.2405396, 121.8622818
1: -47.8214684, 40.8501358, -62.8509064, 53.4563637, -101.2778244, 103.7010422
2: -63.5096283, 41.9536591, -83.2740860, 54.6312675, -118.1408997, 125.2277451
3: -68.0391769, 35.6473465, -88.8035431, 46.6117477, -114.6509094, 124.4508667
4: -62.3432198, 48.0855103, -81.6525497, 62.8625793, -125.2057953, 129.7380524
5: -55.8076591, 43.5713387, -72.9256134, 56.8753815, -112.6830292, 116.4969406
6: -54.0077972, 51.0143890, -70.3589935, 66.8992462, -120.9070435, 121.3733826
7: -58.6748962, 49.5177422, -76.5403214, 64.3945618, -123.0694504, 126.0580597
8: -69.5280075, 47.2781334, -91.3225021, 62.3986130, -131.9265747, 138.6006317
9: -53.4291344, 52.0453949, -69.7656937, 68.2994537, -121.7285843, 121.8110886

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3435790, upper bound: 197.3509764
time: 7.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3914039, upper bound: 197.3917330
time: 7.19 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -65.8794403, 52.2954102, -59.3823814, 47.1555328, -113.0349731, 111.6777725
1: -54.3890953, 46.3860855, -48.8047943, 41.7608223, -96.1499176, 95.1908722
2: -72.2206726, 47.5700035, -65.0233459, 42.8941574, -115.1148224, 112.5933533
3: -77.3069305, 40.5152245, -69.6765671, 36.4887199, -113.7956390, 110.1917877
4: -70.8522568, 54.5207787, -63.8604088, 49.1275330, -119.9797897, 118.3811646
5: -63.3780975, 49.4197197, -57.1678123, 44.5630150, -107.9411163, 106.5875320
6: -61.2465439, 57.9995079, -55.2435722, 52.1976051, -113.4441528, 113.2430801
7: -66.6315079, 56.0905304, -60.0515900, 50.5713768, -117.2028809, 116.1421051
8: -79.0618134, 53.7890968, -71.1646042, 48.4001923, -127.4620056, 124.9536896
9: -60.6512718, 59.2058563, -54.6130562, 53.2264557, -113.8777313, 113.8189087

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 16

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
time: 7.48 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
time: 6.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -65.8794403, 52.2954102, -78.2111206, 62.1873703, -128.0668030, 130.5065308
1: -54.3890953, 46.3860855, -64.8361664, 55.1317062, -109.5207901, 111.2222519
2: -72.2206726, 47.5700035, -85.9116669, 56.3260231, -128.5466919, 133.4816742
3: -77.3069305, 40.5152245, -91.6079941, 48.0906143, -125.3975372, 132.1232147
4: -70.8522568, 54.5207787, -84.2190933, 64.8059158, -135.6581573, 138.7398529
5: -63.3780975, 49.4197197, -75.2123718, 58.6466026, -122.0246811, 124.6320953
6: -61.2465439, 57.9995079, -72.5405655, 69.0117416, -130.2582855, 130.5400696
7: -66.6315079, 56.0905304, -78.9501953, 66.3804626, -133.0119629, 135.0407257
8: -79.0618134, 53.7890968, -94.1931534, 64.3665237, -143.4283142, 147.9822235
9: -60.6512718, 59.2058563, -71.9540558, 70.4683914, -131.1196442, 131.1599121

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 16

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
time: 6.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
time: 6.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -60.2420197, 47.8670578, -62.8464165, 49.9019470, -110.1439514, 110.7134705
1: -49.7753067, 42.4780426, -51.7920532, 44.2308540, -94.0061569, 94.2700958
2: -66.0479279, 43.6061211, -68.8544846, 45.4061241, -111.4540558, 112.4606018
3: -70.7364349, 37.0704155, -73.7485275, 38.6310387, -109.3674545, 110.8189392
4: -64.8158646, 49.9908714, -67.5910187, 52.0301514, -116.8460007, 117.5818787
5: -58.0063400, 45.2878685, -60.4884338, 47.1658325, -105.1721725, 105.7763062
6: -56.1291389, 53.0588417, -58.4762383, 55.2907486, -111.4198914, 111.5350800
7: -61.0022278, 51.4662247, -63.5737572, 53.5554466, -114.5576782, 115.0399551
8: -72.3168259, 49.1797638, -75.3903580, 51.2594261, -123.5762482, 124.5701218
9: -55.5688286, 54.1698914, -57.8748322, 56.4527016, -112.0215302, 112.0447083

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 140

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3458486
time: 7.16 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3919603
time: 5.77 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -60.2420197, 47.8670578, -82.5053101, 65.5929260, -125.8349457, 130.3723755
1: -49.7753067, 42.4780426, -68.5392532, 58.1979141, -107.9732132, 111.0172958
2: -66.0479279, 43.6061211, -90.6620712, 59.4303551, -125.4782867, 134.2681885
3: -70.7364349, 37.0704155, -96.6505051, 50.7477188, -121.4841537, 133.7209167
4: -64.8158646, 49.9908714, -88.8516541, 68.3969727, -133.2128296, 138.8425140
5: -58.0063400, 45.2878685, -79.3276138, 61.8655396, -119.8718796, 124.6154785
6: -56.1291389, 53.0588417, -76.5364380, 72.8436508, -128.9727631, 129.5952759
7: -61.0022278, 51.4662247, -83.3060150, 70.0573196, -131.0595398, 134.7722015
8: -72.3168259, 49.1797638, -99.4258194, 67.9357758, -140.2525787, 148.6055756
9: -55.5688286, 54.1698914, -75.9818802, 74.4452362, -130.0140381, 130.1517639

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 140

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3517804
time: 6.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3984237
time: 5.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -62.0469666, 49.2285309, -67.0202026, 53.1995049, -115.2464752, 116.2487259
1: -51.0891342, 43.6434135, -55.0931129, 47.0479889, -98.1371231, 98.7365265
2: -67.9903717, 44.8529129, -73.4976349, 48.3682632, -116.3586349, 118.3505478
3: -72.8682632, 38.1401863, -78.6756744, 41.1268463, -113.9951019, 116.8158569
4: -66.6970749, 51.3261032, -72.0598450, 55.3605576, -122.0576324, 123.3859482
5: -59.6983490, 46.5770988, -64.4900131, 50.2861137, -109.9844666, 111.0671082
6: -57.7032280, 54.5625877, -62.2053375, 58.9094963, -116.6127167, 116.7679291
7: -62.8005638, 52.8869438, -67.7314758, 57.0015945, -119.8021469, 120.6184158
8: -74.3370590, 50.4757309, -80.3169785, 54.5408249, -128.8778839, 130.7927094
9: -57.1556816, 55.7152481, -61.6664352, 60.1348381, -117.2905197, 117.3816757

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2743508, upper bound: 197.2751606
time: 8.98 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 123

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3429292, upper bound: 197.3423829
time: 6.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3552385, upper bound: 197.3549356
time: 8.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -68.1773148, 54.1091690, -76.5980301, 60.8542404, -129.0315552, 130.7071991
1: -56.3429832, 48.0154800, -63.4378738, 53.9610176, -110.3039932, 111.4533539
2: -74.7554169, 49.2199211, -84.0879135, 55.2139778, -129.9693909, 133.3078308
3: -80.0022659, 41.9368668, -89.7706451, 47.0953178, -127.0975800, 131.7075195
4: -73.3227768, 56.4227982, -82.4272842, 63.4185333, -136.7413025, 138.8500671
5: -65.5744629, 51.1348572, -73.6554718, 57.4330177, -123.0074768, 124.7903290
6: -63.3671684, 60.0410423, -71.0655365, 67.5311279, -130.8982849, 131.1065674
7: -68.9566498, 58.0363235, -77.3692703, 65.0638504, -134.0205078, 135.4055939
8: -81.8438416, 55.6883812, -92.1338272, 62.8546524, -144.6984863, 147.8222046
9: -62.7911453, 61.3247604, -70.5179138, 69.0138245, -131.8049622, 131.8426819

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 166

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3517679, upper bound: 197.3527371
time: 8.41 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3517679, upper bound: 197.4002612
time: 7.04 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -75.8261261, 60.3121338, -57.9427338, 46.0502853, -121.8764114, 118.2548676
1: -62.8633270, 53.4675026, -47.8214684, 40.8501358, -103.7134552, 101.2889709
2: -83.2939453, 54.6399155, -63.5096283, 41.9536591, -125.2476044, 118.1495438
3: -88.8187943, 46.6220093, -68.0391769, 35.6473465, -124.4661255, 114.6611710
4: -81.6678467, 62.8759117, -62.3432198, 48.0855103, -129.7533417, 125.2191238
5: -72.9381943, 56.8863258, -55.8076591, 43.5713387, -116.5095215, 112.6939850
6: -70.3727188, 66.9132614, -54.0077972, 51.0143890, -121.3871078, 120.9210587
7: -76.5561447, 64.4066925, -58.6748962, 49.5177422, -126.0738831, 123.0815811
8: -91.3393250, 62.4127159, -69.5280075, 47.2781334, -138.6174622, 131.9407043
9: -69.7811584, 68.3138657, -53.4291344, 52.0453949, -121.8265533, 121.7429962

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3524509, upper bound: 197.3522171
time: 6.62 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3980564, upper bound: 197.3974015
time: 6.36 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -78.2179031, 62.1927681, -65.8794403, 52.2954102, -130.5133057, 128.0722046
1: -64.8415833, 55.1365471, -54.3890953, 46.3860855, -111.2276688, 109.5256424
2: -85.9191360, 56.3308945, -72.2206726, 47.5700035, -133.4891357, 128.5515747
3: -91.6161194, 48.0945930, -77.3069305, 40.5152245, -132.1313477, 125.4015198
4: -84.2264862, 64.8117371, -70.8522568, 54.5207787, -138.7472687, 135.6639862
5: -75.2188950, 58.6516495, -63.3780975, 49.4197197, -124.6386108, 122.0297318
6: -72.5468826, 69.0175552, -61.2465439, 57.9995079, -130.5463867, 130.2640991
7: -78.9570618, 66.3858948, -66.6315079, 56.0905304, -135.0475769, 133.0173950
8: -94.2015152, 64.3725281, -79.0618134, 53.7890968, -147.9906158, 143.4343414
9: -71.9602432, 70.4743423, -60.6512718, 59.2058563, -131.1661072, 131.1255951

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3534659, upper bound: 197.3534748
time: 8.43 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3999677, upper bound: 197.3995213
time: 9.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -82.5053101, 65.5929260, -60.2420197, 47.8670578, -130.3723755, 125.8349457
1: -68.5392532, 58.1979141, -49.7753067, 42.4780426, -111.0172958, 107.9732132
2: -90.6620712, 59.4303551, -66.0479279, 43.6061211, -134.2681885, 125.4782867
3: -96.6505051, 50.7477188, -70.7364349, 37.0704155, -133.7209167, 121.4841537
4: -88.8516541, 68.3969727, -64.8158646, 49.9908714, -138.8425140, 133.2128296
5: -79.3276138, 61.8655396, -58.0063400, 45.2878685, -124.6154785, 119.8718796
6: -76.5364380, 72.8436508, -56.1291389, 53.0588417, -129.5952759, 128.9727631
7: -83.3060150, 70.0573196, -61.0022278, 51.4662247, -134.7722015, 131.0595398
8: -99.4258194, 67.9357758, -72.3168259, 49.1797638, -148.6055756, 140.2525787
9: -75.9818802, 74.4452362, -55.5688286, 54.1698914, -130.1517639, 130.0140381

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3540110, upper bound: 197.3540892
time: 7.96 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3984237, upper bound: 197.3978360
time: 7.58 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -84.8675461, 67.4506226, -68.1773148, 54.1091690, -138.9767151, 135.6278992
1: -70.4933014, 59.8462105, -56.3429832, 48.0154800, -118.5087814, 116.1891937
2: -93.2551422, 61.0984306, -74.7554169, 49.2199211, -142.4750519, 135.8538361
3: -99.4132309, 52.2015228, -80.0022659, 41.9368668, -141.3500977, 132.2037964
4: -91.3773499, 70.3094864, -73.3227768, 56.4227982, -147.8001251, 143.6322632
5: -81.5809326, 63.6083488, -65.5744629, 51.1348572, -132.7157593, 129.1828156
6: -78.6837616, 74.9215012, -63.3671684, 60.0410423, -138.7248077, 138.2886658
7: -85.6768112, 72.0121231, -68.9566498, 58.0363235, -143.7131348, 140.9687653
8: -102.2498474, 69.8722305, -81.8438416, 55.6883812, -157.9382324, 151.7160645
9: -78.1340103, 76.5771179, -62.7911453, 61.3247604, -139.4587555, 139.3682556

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3552974, upper bound: 197.3556151
time: 8.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4002612, upper bound: 197.3998998
time: 7.22 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -75.8261261, 60.3121338, -77.2992630, 61.5009766, -137.3271027, 137.6113739
1: -62.8633270, 53.4675026, -64.3019257, 54.5988350, -117.4621429, 117.7694244
2: -83.2939453, 54.6399155, -84.9845963, 55.7626038, -139.0565186, 139.6244965
3: -88.8187943, 46.6220093, -90.5878448, 47.5759583, -136.3947449, 137.2098236
4: -81.6678467, 62.8759117, -83.2826004, 64.2062607, -145.8740845, 146.1585083
5: -72.9381943, 56.8863258, -74.3595276, 58.0471001, -130.9852753, 131.2458496
6: -70.3727188, 66.9132614, -71.7897568, 68.2973099, -138.6700287, 138.7030182
7: -76.5561447, 64.4066925, -78.1059036, 65.7645874, -142.3207092, 142.5126038
8: -91.3393250, 62.4127159, -93.2075729, 63.7042542, -155.0435791, 155.6202850
9: -69.7811584, 68.3138657, -71.2577591, 69.7733765, -139.5545349, 139.5716248

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3889231, upper bound: 197.3892909
time: 7.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4309055, upper bound: 197.4308090
time: 6.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -78.2179031, 62.1927681, -84.8210297, 67.4192200, -145.6371155, 147.0137939
1: -64.8415833, 55.1365471, -70.5221481, 59.8427505, -124.6843338, 125.6586914
2: -85.9191360, 56.3308945, -93.2328949, 61.0774345, -146.9965668, 149.5637817
3: -91.6161194, 48.0945930, -99.3765945, 52.1892471, -143.8053436, 147.4711761
4: -84.2264862, 64.8117371, -91.3349609, 70.2952423, -154.5217133, 156.1466675
5: -75.2188950, 58.6516495, -81.5312271, 63.5877037, -138.8065948, 140.1828766
6: -72.5468826, 69.0175552, -78.6498032, 74.9121475, -147.4590149, 147.6673431
7: -78.9570618, 66.3858948, -85.6495056, 71.9921265, -150.9491882, 152.0354004
8: -94.2015152, 64.3725281, -102.2190781, 69.8545380, -164.0560455, 166.5916138
9: -71.9602432, 70.4743423, -78.0998611, 76.5440521, -148.5042877, 148.5742035

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 168

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3893478, upper bound: 197.3900271
time: 7.36 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4318249, upper bound: 197.4318312
time: 5.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -82.5053101, 65.5929260, -79.8546524, 63.5200233, -146.0253296, 145.4475708
1: -68.5392532, 58.1979141, -66.4755096, 56.4093628, -124.9486084, 124.6734161
2: -90.6620712, 59.4303551, -87.8040314, 57.5958214, -148.2578888, 147.2343903
3: -96.6505051, 50.7477188, -93.5866852, 49.1573524, -145.8078461, 144.3344116
4: -88.8516541, 68.3969727, -86.0329437, 66.3216858, -155.1733398, 154.4299164
5: -79.3276138, 61.8655396, -76.8029633, 59.9532547, -139.2808380, 138.6685028
6: -76.5364380, 72.8436508, -74.1473465, 70.5671692, -147.1035919, 146.9909668
7: -83.3060150, 70.0573196, -80.6907272, 67.9256592, -151.2316589, 150.7480469
8: -99.4258194, 67.9357758, -96.3019333, 65.8203430, -165.2461548, 164.2377014
9: -75.9818802, 74.4452362, -73.6327667, 72.1244125, -148.1062927, 148.0780029

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3899987, upper bound: 197.3905494
time: 6.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4315746, upper bound: 197.4315091
time: 5.97 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -84.8675461, 67.4506226, -87.3436508, 69.4137726, -154.2813110, 154.7942352
1: -70.4933014, 59.8462105, -72.6687927, 61.6308365, -132.1241455, 132.5149994
2: -93.2551422, 61.0984306, -96.0182571, 62.8872414, -156.1423798, 157.1166534
3: -99.4132309, 52.2015228, -102.3376160, 53.7498741, -153.1631012, 154.5391235
4: -91.3773499, 70.3094864, -94.0491791, 72.3854446, -163.7627716, 164.3586731
5: -81.5809326, 63.6083488, -83.9450302, 65.4696503, -147.0505829, 147.5533752
6: -78.6837616, 74.9215012, -80.9792633, 77.1534271, -155.8371887, 155.9007568
7: -85.6768112, 72.0121231, -88.2022552, 74.1273880, -159.8041992, 160.2143250
8: -102.2498474, 69.8722305, -105.2730026, 71.9457550, -174.1955872, 175.1452332
9: -78.1340103, 76.5771179, -80.4464798, 78.8647079, -156.9986877, 157.0235901

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3903241, upper bound: 197.3911440
time: 8.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4325254, upper bound: 197.4325262
time: 7.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 16.76 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3435790, upper bound: 197.3448669
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3914039, upper bound: 197.3917330
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3435790, upper bound: 197.3509764
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3914039, upper bound: 197.3917330
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3445014, upper bound: 197.3455256
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3935833, upper bound: 197.3936362
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3458486
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3919603
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3446313, upper bound: 197.3517804
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3916919, upper bound: 197.3984237
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3429292, upper bound: 197.3423829
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3552385, upper bound: 197.3549356
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3517679, upper bound: 197.3527371
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3517679, upper bound: 197.4002612
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3524509, upper bound: 197.3522171
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3980564, upper bound: 197.3974015
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3534659, upper bound: 197.3534748
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3999677, upper bound: 197.3995213
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3540110, upper bound: 197.3540892
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3984237, upper bound: 197.3978360
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3552974, upper bound: 197.3556151
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.4002612, upper bound: 197.3998998
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3889231, upper bound: 197.3892909
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.4309055, upper bound: 197.4308090
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3893478, upper bound: 197.3900271
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.4318249, upper bound: 197.4318312
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3899987, upper bound: 197.3905494
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.4315746, upper bound: 197.4315091
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.3903241, upper bound: 197.3911440
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.76
Output dim: 4, lower bound: -197.4325254, upper bound: 197.4325262

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -47.4679146, 37.6418762, -51.1771278, 40.6312370, -88.0991516, 88.8190002
1: -38.8449783, 33.3431625, -41.8857231, 35.9514542, -74.7964325, 75.2288666
2: -51.9211960, 34.4692459, -55.9766922, 37.0592918, -88.9804840, 90.4459229
3: -55.8201180, 29.1703510, -60.0769081, 31.4306316, -87.2507477, 89.2472458
4: -51.0849304, 39.2998428, -55.0345345, 42.3612747, -93.4462051, 94.3343811
5: -45.7621841, 35.7236023, -49.3151779, 38.4544220, -84.2166061, 85.0387802
6: -44.3253365, 41.6139870, -47.7023163, 44.9141312, -89.2394714, 89.3162994
7: -48.1248817, 40.6121941, -51.7996445, 43.6828079, -91.8076935, 92.4118271
8: -56.5187950, 38.2916603, -61.1576614, 41.5104675, -98.0292587, 99.4493256
9: -43.7176476, 42.3769341, -47.0778770, 45.7437401, -89.4613800, 89.4547958

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3306206, upper bound: 197.3315094
time: 6.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3431080, upper bound: 197.3442736
time: 7.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -55.3833160, 44.0194168, -56.8708534, 45.1799011, -100.5632172, 100.8902740
1: -45.6325645, 39.0242767, -46.7256660, 40.0073929, -85.6399536, 85.7499390
2: -60.6905785, 40.1326218, -62.2637253, 41.1164856, -101.8070526, 102.3963470
3: -65.0504074, 34.0663147, -66.7398453, 34.9410629, -99.9914627, 100.8061600
4: -59.5827065, 45.9672623, -61.1706085, 47.0911636, -106.6738739, 107.1378632
5: -53.3589630, 41.6650772, -54.7701302, 42.7085228, -96.0674667, 96.4352112
6: -51.6435699, 48.7329788, -52.9590340, 49.9874458, -101.6310120, 101.6920090
7: -56.1016235, 47.3591499, -57.5324364, 48.4921188, -104.5937424, 104.8915710
8: -66.4026184, 45.1274033, -68.1527634, 46.3337097, -112.7363205, 113.2801590
9: -51.0727882, 49.7116470, -52.3251114, 50.9505348, -102.0233231, 102.0367508

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3439151, upper bound: 197.3430297
time: 6.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3439151, upper bound: 197.3917330
time: 7.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -47.4679146, 37.6418762, -69.5849533, 55.3434868, -102.8114014, 107.2268295
1: -38.8449783, 33.3431625, -57.5094337, 49.0119171, -87.8568954, 90.8525848
2: -51.9211960, 34.4692459, -76.4041748, 50.2001266, -102.1213226, 110.8734131
3: -55.8201180, 29.1703510, -81.5516968, 42.7532845, -98.5733948, 110.7220459
4: -51.0849304, 39.2998428, -74.9236908, 57.6881180, -108.7730484, 114.2235336
5: -45.7621841, 35.7236023, -66.9579086, 52.2439461, -98.0061188, 102.6815109
6: -44.3253365, 41.6139870, -64.6052551, 61.3337822, -105.6591034, 106.2192383
7: -48.1248817, 40.6121941, -70.2841644, 59.1597023, -107.2845840, 110.8963623
8: -56.5187950, 38.2916603, -83.7072678, 57.1163826, -113.6351776, 121.9989243
9: -43.7176476, 42.3769341, -64.0404968, 62.6071358, -106.3247757, 106.4174347

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 16

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3319312, upper bound: 197.3332771
time: 8.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3330548, upper bound: 197.3345685
time: 7.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -55.3833160, 44.0194168, -75.8119965, 60.2978096, -115.6811218, 119.8314133
1: -45.6325645, 39.0242767, -62.8509064, 53.4563637, -99.0889282, 101.8751755
2: -60.6905785, 40.1326218, -83.2740860, 54.6312675, -115.3218460, 123.4067078
3: -65.0504074, 34.0663147, -88.8035431, 46.6117477, -111.6621552, 122.8698578
4: -59.5827065, 45.9672623, -81.6525497, 62.8625793, -122.4452744, 127.6198120
5: -53.3589630, 41.6650772, -72.9256134, 56.8753815, -110.2343140, 114.5906906
6: -51.6435699, 48.7329788, -70.3589935, 66.8992462, -118.5428085, 119.0919724
7: -56.1016235, 47.3591499, -76.5403214, 64.3945618, -120.4961853, 123.8994675
8: -66.4026184, 45.1274033, -91.3225021, 62.3986130, -128.8012238, 136.4499054
9: -51.0727882, 49.7116470, -69.7656937, 68.2994537, -119.3722382, 119.4773407

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 168

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3522171, upper bound: 197.3524509
time: 6.06 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3522171, upper bound: 197.3980564
time: 7.40 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -54.7215843, 43.3499985, -53.5525131, 42.4985390, -97.2201080, 96.9024963
1: -44.7856941, 38.3751106, -43.8243408, 37.6009789, -82.3866577, 82.1994476
2: -59.8806915, 39.5832481, -58.5819130, 38.7380905, -98.6187820, 98.1651611
3: -64.3329544, 33.5935249, -62.8623543, 32.8848343, -97.2177887, 96.4558792
4: -58.8240967, 45.1662483, -57.5734825, 44.2801704, -103.1042633, 102.7397308
5: -52.7031136, 41.0701408, -51.5768814, 40.2142334, -92.9173431, 92.6470184
6: -50.9307022, 47.9796181, -49.8584633, 46.9951553, -97.9258575, 97.8380814
7: -55.3959122, 46.6520004, -54.1852798, 45.6539421, -101.0498505, 100.8372803
8: -65.2225189, 44.1653137, -64.0118866, 43.4517593, -108.6742783, 108.1772003
9: -50.3221130, 48.8782005, -49.2435493, 47.8847427, -98.2068558, 98.1217346

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 123

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3309015, upper bound: 197.3317689
time: 7.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3440366, upper bound: 197.3450530
time: 6.35 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -63.2079582, 50.1743317, -59.3823814, 47.1555328, -110.3634949, 109.5567017
1: -52.1058426, 44.4840164, -48.8047943, 41.7608223, -93.8666687, 93.2887955
2: -69.2766647, 45.6721153, -65.0233459, 42.8941574, -112.1708221, 110.6954651
3: -74.1985397, 38.8619728, -69.6765671, 36.4887199, -110.6872482, 108.5385361
4: -67.9682007, 52.3077011, -63.8604088, 49.1275330, -117.0957336, 116.1680984
5: -60.8209152, 47.4334183, -57.1678123, 44.5630150, -105.3839264, 104.6012268
6: -58.7809258, 55.6166687, -55.2435722, 52.1976051, -110.9785309, 110.8602371
7: -63.9507828, 53.8463707, -60.0515900, 50.5713768, -114.5221558, 113.8979492
8: -75.8026123, 51.5363121, -71.1646042, 48.4001923, -124.2027969, 122.7009125
9: -58.1971397, 56.7696991, -54.6130562, 53.2264557, -111.4235992, 111.3827515

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3452886, upper bound: 197.3441037
time: 8.06 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3452886, upper bound: 197.3936362
time: 7.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -54.7215843, 43.3499985, -71.9626389, 57.2119904, -111.9335632, 115.3126297
1: -44.7856941, 38.3751106, -59.4743156, 50.6704521, -95.4561462, 97.8494263
2: -59.8806915, 39.5832481, -79.0130463, 51.8808098, -111.7615051, 118.5962906
3: -64.3329544, 33.5935249, -84.3316269, 44.2165146, -108.5494690, 117.9251480
4: -58.8240967, 45.1662483, -77.4662857, 59.6112022, -118.4353027, 122.6325302
5: -52.7031136, 41.0701408, -69.2237854, 53.9985428, -106.7016525, 110.2939224
6: -50.9307022, 47.9796181, -66.7659988, 63.4245262, -114.3552170, 114.7456131
7: -55.3959122, 46.6520004, -72.6700211, 61.1262703, -116.5221863, 119.3220215
8: -65.2225189, 44.1653137, -86.5514526, 59.0627899, -124.2853088, 130.7167664
9: -50.3221130, 48.8782005, -66.2064590, 64.7532654, -115.0753784, 115.0846558

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 166

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3326032, upper bound: 197.3338226
time: 8.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3337295, upper bound: 197.3349887
time: 8.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -63.2079582, 50.1743317, -78.2111206, 62.1873703, -125.3953247, 128.3854523
1: -52.1058426, 44.4840164, -64.8361664, 55.1317062, -107.2375412, 109.3201828
2: -69.2766647, 45.6721153, -85.9116669, 56.3260231, -125.6026917, 131.5837860
3: -74.1985397, 38.8619728, -91.6079941, 48.0906143, -122.2891464, 130.4699249
4: -67.9682007, 52.3077011, -84.2190933, 64.8059158, -132.7740936, 136.5267944
5: -60.8209152, 47.4334183, -75.2123718, 58.6466026, -119.4674988, 122.6457825
6: -58.7809258, 55.6166687, -72.5405655, 69.0117416, -127.7926636, 128.1572266
7: -63.9507828, 53.8463707, -78.9501953, 66.3804626, -130.3312378, 132.7965546
8: -75.8026123, 51.5363121, -94.1931534, 64.3665237, -140.1690979, 145.7294312
9: -58.1971397, 56.7696991, -71.9540558, 70.4683914, -128.6655273, 128.7237549

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 168

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3534748, upper bound: 197.3534659
time: 6.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3534748, upper bound: 197.3999677
time: 7.66 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 9.13 + 600.11 = 609.24 seconds
