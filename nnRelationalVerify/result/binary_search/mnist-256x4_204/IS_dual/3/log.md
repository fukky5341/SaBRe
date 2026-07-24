## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 385.180259218
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-251.0046997, 198.6933441, -251.0046997, 198.6933441, -449.6980591, 449.6980591)
1: (-210.6947174, 176.4580078, -210.6947174, 176.4580078, -387.1527100, 387.1527100)
2: (-276.5284119, 179.2005463, -276.5284119, 179.2005463, -455.7289429, 455.7289429)
3: (-294.5504456, 154.9829407, -294.5504456, 154.9829407, -449.5333862, 449.5333862)
4: (-269.3182068, 206.1575470, -269.3182068, 206.1575470, -475.4757690, 475.4757690)
5: (-241.0453339, 187.2388306, -241.0453339, 187.2388306, -428.2841797, 428.2841797)
6: (-230.5612183, 222.3140869, -230.5612183, 222.3140869, -452.8753052, 452.8753052)
7: (-251.9901886, 210.9694214, -251.9901886, 210.9694214, -462.9595947, 462.9595947)
8: (-303.3696594, 206.9290314, -303.3696594, 206.9290314, -510.2987061, 510.2987061)
9: (-228.5547943, 225.5001526, -228.5547943, 225.5001526, -454.0549316, 454.0549316)

## BASE Result
execution time: IAR + LP analysis = 1.14 + 10.94 = 12.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -385.1965521, upper bound: 385.1965521


# Binary Search by BASE starts (time budget: 2687.92 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=387.1527099609375
rel_dist={1: [-385.1965121761082, 385.19651217610806]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=387.1527099609375
rel_dist={1: [-385.1964844738563, 385.19648447385634]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=387.1527099609375
rel_dist={1: [-385.1964274816625, 385.19642747694013]}

## Binary Search Result
Binary search time: 44.69 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2643.24 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1888955, upper bound: 385.1871984
time: 11.10 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
time: 8.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.26
Output dim: 1, lower bound: -385.1888955, upper bound: 385.1871984
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.26
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -251.0046997, 198.6933441, -442.2391357, 443.8168335
1: -204.4127197, 171.1914368, -210.6947174, 176.4580078, -380.8707275, 381.8861694
2: -268.3244019, 173.8122711, -276.5284119, 179.2005463, -447.5249329, 450.3406677
3: -285.7283020, 150.3652191, -294.5504456, 154.9829407, -440.7112427, 444.9156494
4: -261.2940063, 200.0122070, -269.3182068, 206.1575470, -467.4515381, 469.3304138
5: -233.8536224, 181.6165619, -241.0453339, 187.2388306, -421.0924683, 422.6618652
6: -223.7100830, 215.6901703, -230.5612183, 222.3140869, -446.0241394, 446.2513428
7: -244.5041962, 204.6670990, -251.9901886, 210.9694214, -455.4736023, 456.6572876
8: -294.3796692, 200.7644348, -303.3696594, 206.9290314, -501.3087158, 504.1340637
9: -221.7438049, 218.7707672, -228.5547943, 225.5001526, -447.2439575, 447.3255615

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1664008, upper bound: 385.1670925
time: 9.04 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1484161, upper bound: 385.1454211
time: 7.97 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -244.8681488, 193.9198761, -248.6522064, 196.8404236, -441.7085571, 442.5720825
1: -205.4636383, 172.0779266, -208.7127838, 174.7948303, -380.2584839, 380.7906799
2: -269.8109131, 174.6556702, -273.9429626, 177.5025482, -447.3134766, 448.5985718
3: -287.2557373, 151.1343994, -291.7710266, 153.5300140, -440.7857666, 442.9054260
4: -262.6797791, 200.9863586, -266.7790527, 204.2151489, -466.8949280, 467.7654114
5: -235.1186066, 182.4603882, -238.7845764, 185.4650421, -420.5836182, 421.2449646
6: -224.9225464, 216.8661041, -228.3950653, 220.2264252, -445.1489868, 445.2611084
7: -245.7872772, 205.7355347, -249.6228790, 208.9833832, -454.7706604, 455.3583984
8: -295.9555969, 201.7678680, -300.5292664, 204.9837189, -500.9393311, 502.2971191
9: -222.8706055, 219.9002075, -226.3978424, 223.3750305, -446.2456055, 446.2980347

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 118

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1497712, upper bound: 385.1537338
time: 8.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1372244, upper bound: 385.1372244
time: 7.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.62 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 17.62
Output dim: 1, lower bound: -385.1664008, upper bound: 385.1670925
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 17.62
Output dim: 1, lower bound: -385.1484161, upper bound: 385.1454211
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 17.62
Output dim: 1, lower bound: -385.1497712, upper bound: 385.1537338
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 17.62
Output dim: 1, lower bound: -385.1372244, upper bound: 385.1372244
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=387.1527099609375
rel_dist={1: [-385.1965121761082, 385.19651217610806]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1900728, upper bound: 385.1880729
time: 8.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850250, upper bound: 385.1850250
time: 9.10 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.09 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.09
Output dim: 1, lower bound: -385.1900728, upper bound: 385.1880729
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.09
Output dim: 1, lower bound: -385.1850250, upper bound: 385.1850250

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -251.0046997, 198.6933441, -442.2391357, 443.8168335
1: -204.4127197, 171.1914368, -210.6947174, 176.4580078, -380.8707275, 381.8861694
2: -268.3244019, 173.8122711, -276.5284119, 179.2005463, -447.5249329, 450.3406677
3: -285.7283020, 150.3652191, -294.5504456, 154.9829407, -440.7112427, 444.9156494
4: -261.2940063, 200.0122070, -269.3182068, 206.1575470, -467.4515381, 469.3304138
5: -233.8536224, 181.6165619, -241.0453339, 187.2388306, -421.0924683, 422.6618652
6: -223.7100830, 215.6901703, -230.5612183, 222.3140869, -446.0241394, 446.2513428
7: -244.5041962, 204.6670990, -251.9901886, 210.9694214, -455.4736023, 456.6572876
8: -294.3796692, 200.7644348, -303.3696594, 206.9290314, -501.3087158, 504.1340637
9: -221.7438049, 218.7707672, -228.5547943, 225.5001526, -447.2439575, 447.3255615

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1709852, upper bound: 385.1726133
time: 9.23 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1517100, upper bound: 385.1479502
time: 7.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -244.8681488, 193.9198761, -250.7811737, 198.5173187, -443.3854675, 444.7010498
1: -205.4636383, 172.0779266, -210.5063629, 176.3000031, -381.7636108, 382.5842590
2: -269.8109131, 174.6556702, -276.2827759, 179.0392456, -448.8501587, 450.9384155
3: -287.2557373, 151.1343994, -294.2863464, 154.8448486, -442.1005859, 445.4207458
4: -262.6797791, 200.9863586, -269.0769958, 205.9729614, -468.6527405, 470.0633545
5: -235.1186066, 182.4603882, -240.8304901, 187.0703430, -422.1889648, 423.2908936
6: -224.9225464, 216.8661041, -230.3553772, 222.1157684, -447.0383301, 447.2214661
7: -245.7872772, 205.7355347, -251.7651978, 210.7807312, -456.5679932, 457.5007324
8: -295.9555969, 201.7678680, -303.0998535, 206.7442169, -502.6997986, 504.8677368
9: -222.8706055, 219.9002075, -228.3498993, 225.2982635, -448.1688538, 448.2500916

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 118

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1532494, upper bound: 385.1586522
time: 7.53 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1372379, upper bound: 385.1372379
time: 7.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.95 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 15.95
Output dim: 1, lower bound: -385.1709852, upper bound: 385.1726133
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 15.95
Output dim: 1, lower bound: -385.1517100, upper bound: 385.1479502
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 15.95
Output dim: 1, lower bound: -385.1532494, upper bound: 385.1586522
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 15.95
Output dim: 1, lower bound: -385.1372379, upper bound: 385.1372379
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=387.1527099609375
rel_dist={1: [-385.1965333370153, 385.19653334436805]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1906028, upper bound: 385.1885583
time: 9.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850475, upper bound: 385.1850475
time: 8.13 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.55
Output dim: 1, lower bound: -385.1906028, upper bound: 385.1885583
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.55
Output dim: 1, lower bound: -385.1850475, upper bound: 385.1850475

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -251.0046997, 198.6933441, -442.2391357, 443.8168335
1: -204.4127197, 171.1914368, -210.6947174, 176.4580078, -380.8707275, 381.8861694
2: -268.3244019, 173.8122711, -276.5284119, 179.2005463, -447.5249329, 450.3406677
3: -285.7283020, 150.3652191, -294.5504456, 154.9829407, -440.7112427, 444.9156494
4: -261.2940063, 200.0122070, -269.3182068, 206.1575470, -467.4515381, 469.3304138
5: -233.8536224, 181.6165619, -241.0453339, 187.2388306, -421.0924683, 422.6618652
6: -223.7100830, 215.6901703, -230.5612183, 222.3140869, -446.0241394, 446.2513428
7: -244.5041962, 204.6670990, -251.9901886, 210.9694214, -455.4736023, 456.6572876
8: -294.3796692, 200.7644348, -303.3696594, 206.9290314, -501.3087158, 504.1340637
9: -221.7438049, 218.7707672, -228.5547943, 225.5001526, -447.2439575, 447.3255615

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1878441, upper bound: 385.1850013
time: 7.74 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1904694, upper bound: 385.1885497
time: 9.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -244.8681488, 193.9198761, -251.0046997, 198.6933441, -443.5614929, 444.9245605
1: -205.4636383, 172.0779266, -210.6947174, 176.4580078, -381.9216309, 382.7726135
2: -269.8109131, 174.6556702, -276.5284119, 179.2005463, -449.0114746, 451.1840515
3: -287.2557373, 151.1343994, -294.5504456, 154.9829407, -442.2386780, 445.6848145
4: -262.6797791, 200.9863586, -269.3182068, 206.1575470, -468.8373413, 470.3045654
5: -235.1186066, 182.4603882, -241.0453339, 187.2388306, -422.3574219, 423.5057373
6: -224.9225464, 216.8661041, -230.5612183, 222.3140869, -447.2366333, 447.4272766
7: -245.7872772, 205.7355347, -251.9901886, 210.9694214, -456.7566833, 457.7257080
8: -295.9555969, 201.7678680, -303.3696594, 206.9290314, -502.8846436, 505.1375122
9: -222.8706055, 219.9002075, -228.5547943, 225.5001526, -448.3707581, 448.4549866

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 118

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1553006, upper bound: 385.1615348
time: 7.44 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1372447, upper bound: 385.1372447
time: 7.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.42 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.42
Output dim: 1, lower bound: -385.1878441, upper bound: 385.1850013
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.42
Output dim: 1, lower bound: -385.1904694, upper bound: 385.1885497
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 16.42
Output dim: 1, lower bound: -385.1553006, upper bound: 385.1615348
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 16.42
Output dim: 1, lower bound: -385.1372447, upper bound: 385.1372447

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -244.2093658, 193.3718872, -436.9176636, 437.0214844
1: -204.4127197, 171.1914368, -205.0571442, 171.6760864, -376.0888062, 376.2485962
2: -268.3244019, 173.8122711, -268.9541321, 174.3889313, -442.7132874, 442.7663879
3: -285.7283020, 150.3652191, -286.5935364, 150.6956940, -436.4239502, 436.9587402
4: -261.2940063, 200.0122070, -262.1099548, 200.6207428, -461.9147339, 462.1221619
5: -233.8536224, 181.6165619, -234.6154785, 182.2501373, -416.1037598, 416.2320251
6: -223.7100830, 215.6901703, -224.4102631, 216.2925262, -440.0025330, 440.1004028
7: -244.5041962, 204.6670990, -245.1510010, 205.2819977, -449.7861328, 449.8181152
8: -294.3796692, 200.7644348, -295.1891174, 201.4533691, -495.8330383, 495.9535522
9: -221.7438049, 218.7707672, -222.4554138, 219.4731140, -441.2169189, 441.2261658

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 117

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1701291, upper bound: 385.1711888
time: 9.13 seconds

## Relational analysis of IS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1851054, upper bound: 385.1819767
time: 9.24 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845361, upper bound: 385.1812538
time: 9.14 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -247.0763092, 195.6066284, -439.1523743, 439.8884277
1: -204.4127197, 171.1914368, -207.4265747, 173.6944275, -378.1071472, 378.6179810
2: -268.3244019, 173.8122711, -272.1664734, 176.4153595, -444.7396851, 445.9787292
3: -285.7283020, 150.3652191, -289.9555359, 152.5218048, -438.2500916, 440.3207397
4: -261.2940063, 200.0122070, -265.1312866, 202.9466095, -464.2406006, 465.1434937
5: -233.8536224, 181.6165619, -237.3134460, 184.3449707, -418.1986084, 418.9299316
6: -223.7100830, 215.6901703, -226.9823761, 218.8345184, -442.5445557, 442.6725159
7: -244.5041962, 204.6670990, -248.0450592, 207.6798553, -452.1840515, 452.7121582
8: -294.3796692, 200.7644348, -298.6455994, 203.7384491, -498.1181030, 499.4100342
9: -221.7438049, 218.7707672, -225.0018921, 221.9935608, -443.7373657, 443.7726440

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1729825, upper bound: 385.1749043
time: 8.38 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1531168, upper bound: 385.1486953
time: 10.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.28 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 20.28
Output dim: 1, lower bound: -385.1851054, upper bound: 385.1819767
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 20.28
Output dim: 1, lower bound: -385.1845361, upper bound: 385.1812538
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 20.28
Output dim: 1, lower bound: -385.1729825, upper bound: 385.1749043
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 20.28
Output dim: 1, lower bound: -385.1531168, upper bound: 385.1486953

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -236.9843597, 187.6765747, -431.2223511, 429.7965088
1: -204.4127197, 171.1914368, -198.9614563, 166.6070251, -371.0197449, 370.1528931
2: -268.3244019, 173.8122711, -261.0065613, 169.2618408, -437.5861816, 434.8187866
3: -285.7283020, 150.3652191, -278.1372375, 146.3132629, -432.0415649, 428.5024414
4: -261.2940063, 200.0122070, -254.3480377, 194.6784821, -455.9724731, 454.3602295
5: -233.8536224, 181.6165619, -227.6886139, 176.8377686, -410.6914062, 409.3051147
6: -223.7100830, 215.6901703, -217.7956696, 209.9308167, -433.6408997, 433.4858398
7: -244.5041962, 204.6670990, -237.8521423, 199.2113800, -443.7155762, 442.5192261
8: -294.3796692, 200.7644348, -286.5350342, 195.6402893, -490.0198975, 487.2994690
9: -221.7438049, 218.7707672, -215.8903046, 212.9733887, -434.7171936, 434.6610718

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 117

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1661979, upper bound: 385.1671961
time: 9.88 seconds

## Relational analysis of IS_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1815409, upper bound: 385.1795681
time: 8.33 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1851054, upper bound: 385.1819767
time: 9.12 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -239.2343597, 189.4757538, -433.0215149, 432.0465088
1: -204.4127197, 171.1914368, -200.8483429, 168.1969910, -372.6096802, 372.0397339
2: -268.3244019, 173.8122711, -263.4966125, 170.8766327, -439.2009888, 437.3088684
3: -285.7283020, 150.3652191, -280.7887268, 147.7045135, -433.4328003, 431.1539307
4: -261.2940063, 200.0122070, -256.7720642, 196.5230560, -457.8170776, 456.7842712
5: -233.8536224, 181.6165619, -229.8620148, 178.5121765, -412.3657837, 411.4785156
6: -223.7100830, 215.6901703, -219.8798828, 211.9316406, -435.6416931, 435.5700073
7: -244.5041962, 204.6670990, -240.1170502, 201.0999603, -445.6040955, 444.7841187
8: -294.3796692, 200.7644348, -289.2776489, 197.5100861, -491.8897400, 490.0420532
9: -221.7438049, 218.7707672, -217.9517517, 214.9941559, -436.7379456, 436.7225037

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 218

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1655947, upper bound: 385.1665549
time: 9.36 seconds

## Relational analysis of IS_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1809664, upper bound: 385.1789013
time: 9.40 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845361, upper bound: 385.1812538
time: 9.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 36.03 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 36.03
Output dim: 1, lower bound: -385.1815409, upper bound: 385.1795681
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 36.03
Output dim: 1, lower bound: -385.1851054, upper bound: 385.1819767
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 36.03
Output dim: 1, lower bound: -385.1809664, upper bound: 385.1789013
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 36.03
Output dim: 1, lower bound: -385.1845361, upper bound: 385.1812538

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -244.7019806, 193.7428741, -236.9843597, 187.6765747, -432.3785400, 430.7272339
1: -205.4266663, 171.9966583, -198.9614563, 166.6070251, -372.0336914, 370.9581299
2: -269.5385437, 174.6662292, -261.0065613, 169.2618408, -438.8003540, 435.6727905
3: -287.0465698, 151.0273895, -278.1372375, 146.3132629, -433.3598328, 429.1646118
4: -262.6747437, 200.9801331, -254.3480377, 194.6784821, -457.3532104, 455.3281250
5: -235.0068817, 182.5113678, -227.6886139, 176.8377686, -411.8446045, 410.1999512
6: -224.8174286, 216.7113190, -217.7956696, 209.9308167, -434.7482300, 434.5069885
7: -245.6354675, 205.6291351, -237.8521423, 199.2113800, -444.8468628, 443.4812317
8: -295.7739868, 201.7994995, -286.5350342, 195.6402893, -491.4142151, 488.3344727
9: -222.8391113, 219.8530579, -215.8903046, 212.9733887, -435.8125000, 435.7433472

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1779601, upper bound: 385.1759978
time: 9.20 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1754266, upper bound: 385.1742913
time: 9.07 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -242.4687347, 191.9583130, -236.9843597, 187.6765747, -430.1452942, 428.9426880
1: -203.5082855, 170.4314270, -198.9614563, 166.6070251, -370.1152344, 369.3928833
2: -267.1310120, 173.0465698, -261.0065613, 169.2618408, -436.3928528, 434.0531311
3: -284.4601746, 149.6999512, -278.1372375, 146.3132629, -430.7734375, 427.8371582
4: -260.1350708, 199.1241608, -254.3480377, 194.6784821, -454.8135376, 453.4721985
5: -232.8220062, 180.8194580, -227.6886139, 176.8377686, -409.6597900, 408.5080566
6: -222.7167358, 214.7347870, -217.7956696, 209.9308167, -432.6475525, 432.5304565
7: -243.4166718, 203.7606506, -237.8521423, 199.2113800, -442.6280518, 441.6127014
8: -293.0816956, 199.8827820, -286.5350342, 195.6402893, -488.7219849, 486.4178162
9: -220.7583771, 217.8075562, -215.8903046, 212.9733887, -433.7317505, 433.6978760

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1661979, upper bound: 385.1671961
time: 9.28 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1394741, upper bound: 385.1464845
time: 8.31 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1779601, upper bound: 385.1782988
time: 9.15 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1810000, upper bound: 385.1779691
time: 8.32 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -244.7019806, 193.7428741, -239.2343597, 189.4757538, -434.1777344, 432.9772034
1: -205.4266663, 171.9966583, -200.8483429, 168.1969910, -373.6236267, 372.8450012
2: -269.5385437, 174.6662292, -263.4966125, 170.8766327, -440.4151611, 438.1628113
3: -287.0465698, 151.0273895, -280.7887268, 147.7045135, -434.7510681, 431.8161011
4: -262.6747437, 200.9801331, -256.7720642, 196.5230560, -459.1977539, 457.7521667
5: -235.0068817, 182.5113678, -229.8620148, 178.5121765, -413.5190430, 412.3733521
6: -224.8174286, 216.7113190, -219.8798828, 211.9316406, -436.7490845, 436.5911865
7: -245.6354675, 205.6291351, -240.1170502, 201.0999603, -446.7354126, 445.7461548
8: -295.7739868, 201.7994995, -289.2776489, 197.5100861, -493.2840576, 491.0770569
9: -222.8391113, 219.8530579, -217.9517517, 214.9941559, -437.8331909, 437.8048096

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1773638, upper bound: 385.1751721
time: 8.25 seconds

## Relational analysis of IS_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1748842, upper bound: 385.1734711
time: 7.84 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -242.4687347, 191.9583130, -239.2343597, 189.4757538, -431.9444580, 431.1926575
1: -203.5082855, 170.4314270, -200.8483429, 168.1969910, -371.7052002, 371.2797546
2: -267.1310120, 173.0465698, -263.4966125, 170.8766327, -438.0075989, 436.5431824
3: -284.4601746, 149.6999512, -280.7887268, 147.7045135, -432.1646729, 430.4886475
4: -260.1350708, 199.1241608, -256.7720642, 196.5230560, -456.6581116, 455.8962402
5: -232.8220062, 180.8194580, -229.8620148, 178.5121765, -411.3341675, 410.6814575
6: -222.7167358, 214.7347870, -219.8798828, 211.9316406, -434.6483459, 434.6146851
7: -243.4166718, 203.7606506, -240.1170502, 201.0999603, -444.5165710, 443.8775940
8: -293.0816956, 199.8827820, -289.2776489, 197.5100861, -490.5917969, 489.1604004
9: -220.7583771, 217.8075562, -217.9517517, 214.9941559, -435.7524719, 435.7592773

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1655947, upper bound: 385.1665549
time: 8.65 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1389265, upper bound: 385.1456467
time: 8.39 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1808976, upper bound: 385.1773734
time: 9.11 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1803670, upper bound: 385.1770352
time: 8.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 49.24 seconds
IS_A1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 49.24
Output dim: 1, lower bound: -385.1779601, upper bound: 385.1759978
IS_A1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 49.24
Output dim: 1, lower bound: -385.1754266, upper bound: 385.1742913
IS_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 49.24
Output dim: 1, lower bound: -385.1779601, upper bound: 385.1782988
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 49.24
Output dim: 1, lower bound: -385.1810000, upper bound: 385.1779691
IS_A1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 49.24
Output dim: 1, lower bound: -385.1773638, upper bound: 385.1751721
IS_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 49.24
Output dim: 1, lower bound: -385.1748842, upper bound: 385.1734711
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 49.24
Output dim: 1, lower bound: -385.1808976, upper bound: 385.1773734
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 49.24
Output dim: 1, lower bound: -385.1803670, upper bound: 385.1770352

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -236.4857788, 187.2598419, -236.9843597, 187.6765747, -424.1623535, 424.2442017
1: -198.5041046, 166.2397766, -198.9614563, 166.6070251, -365.1111450, 365.2012329
2: -260.6116028, 168.7163391, -261.0065613, 169.2618408, -429.8734131, 429.7228394
3: -277.5300293, 146.0063477, -278.1372375, 146.3132629, -423.8432617, 424.1435852
4: -253.7036285, 194.2079010, -254.3480377, 194.6784821, -448.3821106, 448.5558777
5: -227.1259308, 176.3223267, -227.6886139, 176.8377686, -403.9636841, 404.0108643
6: -217.2797394, 209.4933472, -217.7956696, 209.9308167, -427.2105713, 427.2890015
7: -237.3958893, 198.7750092, -237.8521423, 199.2113800, -436.6072693, 436.6271362
8: -285.8770752, 194.9155273, -286.5350342, 195.6402893, -481.5173035, 481.4505615
9: -215.3512115, 212.4472656, -215.8903046, 212.9733887, -428.3245850, 428.3375854

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1620866, upper bound: 385.1630372
time: 9.62 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1346201, upper bound: 385.1421833
time: 9.37 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1763425, upper bound: 385.1718718
time: 8.63 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1803393, upper bound: 385.1776337
time: 9.76 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -232.8372040, 184.3764343, -239.2343597, 189.4757538, -422.3129578, 423.6107788
1: -195.4661865, 163.6889648, -200.8483429, 168.1969910, -363.6631165, 364.5372925
2: -256.5744324, 166.1516571, -263.4966125, 170.8766327, -427.4510498, 429.6482544
3: -273.2261047, 143.7992554, -280.7887268, 147.7045135, -420.9306030, 424.5879822
4: -249.8061066, 191.2268372, -256.7720642, 196.5230560, -446.3290710, 447.9989014
5: -223.6269989, 173.6613922, -229.8620148, 178.5121765, -402.1391602, 403.5233459
6: -213.9082489, 206.2582703, -219.8798828, 211.9316406, -425.8398743, 426.1381531
7: -233.7688599, 195.7424011, -240.1170502, 201.0999603, -434.8687439, 435.8594055
8: -281.4695740, 191.9439545, -289.2776489, 197.5100861, -478.9796448, 481.2216187
9: -212.0572662, 209.1838989, -217.9517517, 214.9941559, -427.0513306, 427.1356506

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1614552, upper bound: 385.1623299
time: 9.82 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1312227, upper bound: 385.1392499
time: 9.33 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1754913, upper bound: 385.1702751
time: 9.67 seconds

## Relational analysis of IS_A1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1800540, upper bound: 385.1769522
time: 9.75 seconds

## BFS IS instance: IS_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -236.4857788, 187.2598419, -239.2343597, 189.4757538, -425.9615479, 426.4942017
1: -198.5041046, 166.2397766, -200.8483429, 168.1969910, -366.7010803, 367.0881348
2: -260.6116028, 168.7163391, -263.4966125, 170.8766327, -431.4882202, 432.2128601
3: -277.5300293, 146.0063477, -280.7887268, 147.7045135, -425.2345581, 426.7950745
4: -253.7036285, 194.2079010, -256.7720642, 196.5230560, -450.2266846, 450.9799194
5: -227.1259308, 176.3223267, -229.8620148, 178.5121765, -405.6381226, 406.1842651
6: -217.2797394, 209.4933472, -219.8798828, 211.9316406, -429.2113647, 429.3732300
7: -237.3958893, 198.7750092, -240.1170502, 201.0999603, -438.4958496, 438.8920593
8: -285.8770752, 194.9155273, -289.2776489, 197.5100861, -483.3871460, 484.1931763
9: -215.3512115, 212.4472656, -217.9517517, 214.9941559, -430.3453674, 430.3990173

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1613935, upper bound: 385.1622123
time: 9.32 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1340085, upper bound: 385.1411960
time: 8.86 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1751336, upper bound: 385.1700332
time: 10.09 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1797130, upper bound: 385.1767349
time: 8.51 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 51.39 seconds
IS_A1_B1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 51.39
Output dim: 1, lower bound: -385.1763425, upper bound: 385.1718718
IS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 51.39
Output dim: 1, lower bound: -385.1803393, upper bound: 385.1776337
IS_A1_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 51.39
Output dim: 1, lower bound: -385.1754913, upper bound: 385.1702751
IS_A1_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 51.39
Output dim: 1, lower bound: -385.1800540, upper bound: 385.1769522
IS_A1_B1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 51.39
Output dim: 1, lower bound: -385.1751336, upper bound: 385.1700332
IS_A1_B1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 51.39
Output dim: 1, lower bound: -385.1797130, upper bound: 385.1767349

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -236.4857788, 187.2598419, -233.2178345, 184.6820221, -421.1677856, 420.4776611
1: -198.5041046, 166.2397766, -195.8060303, 163.9724274, -362.4765320, 362.0458069
2: -260.6116028, 168.7163391, -256.8397217, 166.5584717, -427.1700439, 425.5560303
3: -277.5300293, 146.0063477, -273.7318115, 144.0270081, -421.5570374, 419.7381592
4: -253.7036285, 194.2079010, -250.2745972, 191.5880280, -445.2916565, 444.4824219
5: -227.1259308, 176.3223267, -224.0620270, 174.0357666, -401.1616821, 400.3843079
6: -217.2797394, 209.4933472, -214.2854462, 206.5954742, -423.8751526, 423.7788086
7: -237.3958893, 198.7750092, -234.0839233, 196.0755310, -433.4714050, 432.8588867
8: -285.8770752, 194.9155273, -281.9509583, 192.4903564, -478.3674011, 476.8664856
9: -215.3512115, 212.4472656, -212.4891663, 209.6014862, -424.9526978, 424.9364319

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 218

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1616029, upper bound: 385.1625931
time: 9.85 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1340090, upper bound: 385.1412862
time: 9.36 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1426025, upper bound: 385.1273698
time: 9.53 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1803300, upper bound: 385.1775812
time: 7.52 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 51.08 seconds
IS_A1_B1_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 51.08
Output dim: 1, lower bound: -385.1426025, upper bound: 385.1273698
IS_A1_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 51.08
Output dim: 1, lower bound: -385.1803300, upper bound: 385.1775812

## BFS IS instance: IS_A1_B1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -236.4857788, 187.2598419, -232.3417969, 183.9898071, -420.4755859, 419.6016235
1: -198.5041046, 166.2397766, -195.0619202, 163.3596039, -361.8636780, 361.3016968
2: -260.6116028, 168.7163391, -255.8756866, 165.9290009, -426.5405884, 424.5919495
3: -277.5300293, 146.0063477, -272.7090149, 143.4984741, -421.0284424, 418.7153625
4: -253.7036285, 194.2079010, -249.3244476, 190.8659821, -444.5696106, 443.5322571
5: -227.1259308, 176.3223267, -223.2258301, 173.3785858, -400.5044556, 399.5480652
6: -217.2797394, 209.4933472, -213.4839172, 205.8170929, -423.0968018, 422.9772644
7: -237.3958893, 198.7750092, -233.1978455, 195.3398743, -432.7357788, 431.9728394
8: -285.8770752, 194.9155273, -280.8934326, 191.7663269, -477.6434021, 475.8089600
9: -215.3512115, 212.4472656, -211.6933289, 208.8169098, -424.1681213, 424.1405945

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 117

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1616001, upper bound: 385.1624811
time: 9.45 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1265031, upper bound: 385.1411973
time: 10.79 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1475121, upper bound: 385.1518559
time: 9.64 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=387.1527099609375
rel_dist={1: [-385.1965462217105, 385.19654622393296]}

## Binary search (step 3) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1903546, upper bound: 385.1883223
time: 8.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1850363, upper bound: 385.1850363
time: 8.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.50 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.50
Output dim: 1, lower bound: -385.1903546, upper bound: 385.1883223
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.50
Output dim: 1, lower bound: -385.1850363, upper bound: 385.1850363

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -251.0046997, 198.6933441, -442.2391357, 443.8168335
1: -204.4127197, 171.1914368, -210.6947174, 176.4580078, -380.8707275, 381.8861694
2: -268.3244019, 173.8122711, -276.5284119, 179.2005463, -447.5249329, 450.3406677
3: -285.7283020, 150.3652191, -294.5504456, 154.9829407, -440.7112427, 444.9156494
4: -261.2940063, 200.0122070, -269.3182068, 206.1575470, -467.4515381, 469.3304138
5: -233.8536224, 181.6165619, -241.0453339, 187.2388306, -421.0924683, 422.6618652
6: -223.7100830, 215.6901703, -230.5612183, 222.3140869, -446.0241394, 446.2513428
7: -244.5041962, 204.6670990, -251.9901886, 210.9694214, -455.4736023, 456.6572876
8: -294.3796692, 200.7644348, -303.3696594, 206.9290314, -501.3087158, 504.1340637
9: -221.7438049, 218.7707672, -228.5547943, 225.5001526, -447.2439575, 447.3255615

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 103
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 103
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1721484, upper bound: 385.1741135
time: 9.40 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1526517, upper bound: 385.1486409
time: 8.84 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -244.8681488, 193.9198761, -251.0046997, 198.6933441, -443.5614929, 444.9245605
1: -205.4636383, 172.0779266, -210.6947174, 176.4580078, -381.9216309, 382.7726135
2: -269.8109131, 174.6556702, -276.5284119, 179.2005463, -449.0114746, 451.1840515
3: -287.2557373, 151.1343994, -294.5504456, 154.9829407, -442.2386780, 445.6848145
4: -262.6797791, 200.9863586, -269.3182068, 206.1575470, -468.8373413, 470.3045654
5: -235.1186066, 182.4603882, -241.0453339, 187.2388306, -422.3574219, 423.5057373
6: -224.9225464, 216.8661041, -230.5612183, 222.3140869, -447.2366333, 447.4272766
7: -245.7872772, 205.7355347, -251.9901886, 210.9694214, -456.7566833, 457.7257080
8: -295.9555969, 201.7678680, -303.3696594, 206.9290314, -502.8846436, 505.1375122
9: -222.8706055, 219.9002075, -228.5547943, 225.5001526, -448.3707581, 448.4549866

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 244
type: B, layer: 1, pos: 244
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 118

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1542849, upper bound: 385.1601475
time: 8.26 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1372415, upper bound: 385.1372415
time: 6.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.99 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 15.99
Output dim: 1, lower bound: -385.1721484, upper bound: 385.1741135
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 15.99
Output dim: 1, lower bound: -385.1526517, upper bound: 385.1486409
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 15.99
Output dim: 1, lower bound: -385.1542849, upper bound: 385.1601475
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 15.99
Output dim: 1, lower bound: -385.1372415, upper bound: 385.1372415
Binary search (step 3): status=Status.VERIFIED, k_low=10, k_high=10, k_mid=10, eps_mid=0.0390625, abs_max=387.1527099609375
rel_dist={1: [-385.1965400148607, 385.19654000300284]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0390625
execution time: 805.39 seconds
