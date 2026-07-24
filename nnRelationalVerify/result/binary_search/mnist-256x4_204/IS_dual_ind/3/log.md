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
execution time: IAR + LP analysis = 1.18 + 11.01 = 12.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -385.1965521, upper bound: 385.1965521


# Binary Search by BASE starts (time budget: 2687.81 seconds, max iter: 100)

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
Binary search time: 44.97 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2642.84 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1888955, upper bound: 385.1871984
time: 11.12 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
time: 8.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.59
Output dim: 1, lower bound: -385.1888955, upper bound: 385.1871984
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.59
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

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
time: 8.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
time: 8.56 seconds

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

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1810879, upper bound: 385.1807313
time: 8.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1813893, upper bound: 385.1813893
time: 8.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.69 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.69
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.69
Output dim: 1, lower bound: -385.1849891, upper bound: 385.1849891
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.69
Output dim: 1, lower bound: -385.1810879, upper bound: 385.1807313
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.69
Output dim: 1, lower bound: -385.1813893, upper bound: 385.1813893

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -243.5457764, 192.8121338, -436.3579102, 436.3579102
1: -204.4127197, 171.1914368, -204.4127197, 171.1914368, -375.6041565, 375.6041565
2: -268.3244019, 173.8122711, -268.3244019, 173.8122711, -442.1365967, 442.1365967
3: -285.7283020, 150.3652191, -285.7283020, 150.3652191, -436.0935059, 436.0935059
4: -261.2940063, 200.0122070, -261.2940063, 200.0122070, -461.3062134, 461.3062134
5: -233.8536224, 181.6165619, -233.8536224, 181.6165619, -415.4701843, 415.4701843
6: -223.7100830, 215.6901703, -223.7100830, 215.6901703, -439.4002075, 439.4002075
7: -244.5041962, 204.6670990, -244.5041962, 204.6670990, -449.1712646, 449.1712646
8: -294.3796692, 200.7644348, -294.3796692, 200.7644348, -495.1441040, 495.1441040
9: -221.7438049, 218.7707672, -221.7438049, 218.7707672, -440.5145874, 440.5145874

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1847806, upper bound: 385.1833382
time: 10.87 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1847744, upper bound: 385.1833706
time: 9.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -244.8681488, 193.9198761, -437.4656372, 437.6802979
1: -204.4127197, 171.1914368, -205.4636383, 172.0779266, -376.4906311, 376.6550598
2: -268.3244019, 173.8122711, -269.8109131, 174.6556702, -442.9800110, 443.6231689
3: -285.7283020, 150.3652191, -287.2557373, 151.1343994, -436.8626709, 437.6209717
4: -261.2940063, 200.0122070, -262.6797791, 200.9863586, -462.2803650, 462.6919861
5: -233.8536224, 181.6165619, -235.1186066, 182.4603882, -416.3140259, 416.7350769
6: -223.7100830, 215.6901703, -224.9225464, 216.8661041, -440.5761108, 440.6127014
7: -244.5041962, 204.6670990, -245.7872772, 205.7355347, -450.2397156, 450.4543457
8: -294.3796692, 200.7644348, -295.9555969, 201.7678680, -496.1475220, 496.7200012
9: -221.7438049, 218.7707672, -222.8706055, 219.9002075, -441.6440125, 441.6413574

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1847806, upper bound: 385.1833382
time: 9.46 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1847744, upper bound: 385.1833706
time: 9.49 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -244.8681488, 193.9198761, -238.8824005, 189.1493073, -434.0174561, 432.8022766
1: -205.4636383, 172.0779266, -200.5513763, 167.9540405, -373.4176636, 372.6292725
2: -269.8109131, 174.6556702, -263.2330322, 170.5083923, -440.3193054, 437.8887024
3: -287.2557373, 151.1343994, -280.3717651, 147.5429077, -434.7986145, 431.5061646
4: -262.6797791, 200.9863586, -256.2998962, 196.2016449, -458.8814087, 457.2862549
5: -235.1186066, 182.4603882, -229.4580078, 178.2023468, -413.3209229, 411.9183350
6: -224.9225464, 216.8661041, -219.4587860, 211.6257324, -436.5482788, 436.3248291
7: -245.7872772, 205.7355347, -239.8339996, 200.8481903, -446.6353760, 445.5695190
8: -295.9555969, 201.7678680, -288.7486572, 196.9311066, -492.8866577, 490.5165405
9: -222.8706055, 219.9002075, -217.5705872, 214.6241455, -437.4947510, 437.4707947

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1806213
time: 8.39 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1807313
time: 6.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -243.7186890, 193.0135193, -242.7910461, 192.2370605, -435.9556885, 435.8045654
1: -204.5043030, 171.2726135, -203.8125000, 170.6885986, -375.1928711, 375.0851135
2: -268.5508728, 173.8335266, -267.5561523, 173.2598267, -441.8106689, 441.3896484
3: -285.9150696, 150.4305267, -284.9802856, 149.9122620, -435.8273315, 435.4108276
4: -261.4442139, 200.0436859, -260.4773560, 199.3984680, -460.8426819, 460.5210571
5: -234.0196991, 181.6049194, -233.2022400, 181.0597839, -415.0794678, 414.8071289
6: -223.8714142, 215.8549805, -223.0684509, 215.0919495, -438.9632874, 438.9234314
7: -244.6354065, 204.7772827, -243.7280121, 204.0994873, -448.7348938, 448.5052185
8: -294.5720520, 200.8229065, -293.4702454, 200.1164551, -494.6885071, 494.2931519
9: -221.8298187, 218.8706360, -221.1016083, 218.1244812, -439.9542542, 439.9722290

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1807313, upper bound: 385.1810879
time: 8.20 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1807313, upper bound: 385.1813893
time: 9.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.89 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.89
Output dim: 1, lower bound: -385.1847806, upper bound: 385.1833382
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.89
Output dim: 1, lower bound: -385.1847744, upper bound: 385.1833706
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.89
Output dim: 1, lower bound: -385.1847806, upper bound: 385.1833382
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.89
Output dim: 1, lower bound: -385.1847744, upper bound: 385.1833706
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.89
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1806213
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.89
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1807313
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.89
Output dim: 1, lower bound: -385.1807313, upper bound: 385.1810879
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.89
Output dim: 1, lower bound: -385.1807313, upper bound: 385.1813893

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -243.5457764, 192.8121338, -426.7368774, 428.7841797
1: -196.3794403, 164.4563293, -204.4127197, 171.1914368, -367.5708008, 368.8690491
2: -257.7795410, 166.9246521, -268.3244019, 173.8122711, -431.5917969, 435.2489929
3: -274.5066223, 144.4709167, -285.7283020, 150.3652191, -424.8718262, 430.1992188
4: -250.9762573, 192.1235657, -261.2940063, 200.0122070, -450.9884644, 453.4175720
5: -224.6685638, 174.4661713, -233.8536224, 181.6165619, -406.2850647, 408.3197937
6: -214.9111786, 207.2229767, -223.7100830, 215.6901703, -430.6012573, 430.9330444
7: -234.8669281, 196.6575928, -244.5041962, 204.6670990, -439.5340271, 441.1617126
8: -282.7800598, 192.8340759, -294.3796692, 200.7644348, -483.5444641, 487.2137451
9: -213.0522919, 210.1563110, -221.7438049, 218.7707672, -431.8230591, 431.9001160

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1900419, upper bound: 385.1900419
time: 7.54 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1900419, upper bound: 385.1900712
time: 7.43 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -237.5785980, 188.1260834, -242.3500977, 191.8693695, -429.4479065, 430.4761963
1: -199.4214020, 167.0109253, -203.4141235, 170.3534698, -369.7748718, 370.4250488
2: -261.8223267, 169.4930878, -267.0136719, 172.9566498, -434.7789001, 436.5067444
3: -278.8170166, 146.6810913, -284.3335876, 149.6327667, -428.4497681, 431.0146790
4: -254.8795471, 195.1088715, -260.0089111, 199.0314789, -453.9109802, 455.1177673
5: -228.1727142, 177.1310883, -232.7107086, 180.7257385, -408.8983765, 409.8417664
6: -218.2875519, 210.4626465, -222.6168060, 214.6379852, -432.9255371, 433.0794678
7: -238.4990692, 199.6945343, -243.3046722, 203.6697388, -442.1688232, 442.9992065
8: -287.1939697, 195.8099365, -292.9399109, 199.7818909, -486.9758301, 488.7498474
9: -216.3509674, 213.4246216, -220.6605377, 217.6997681, -434.0507202, 434.0851440

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1900712, upper bound: 385.1900419
time: 6.05 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1900712, upper bound: 385.1900713
time: 7.02 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -244.8681488, 193.9198761, -427.8446655, 430.1065674
1: -196.3794403, 164.4563293, -205.4636383, 172.0779266, -368.4572754, 369.9199829
2: -257.7795410, 166.9246521, -269.8109131, 174.6556702, -432.4351807, 436.7355652
3: -274.5066223, 144.4709167, -287.2557373, 151.1343994, -425.6410217, 431.7266235
4: -250.9762573, 192.1235657, -262.6797791, 200.9863586, -451.9626160, 454.8033447
5: -224.6685638, 174.4661713, -235.1186066, 182.4603882, -407.1289673, 409.5847168
6: -214.9111786, 207.2229767, -224.9225464, 216.8661041, -431.7771912, 432.1455078
7: -234.8669281, 196.6575928, -245.7872772, 205.7355347, -440.6024780, 442.4448242
8: -282.7800598, 192.8340759, -295.9555969, 201.7678680, -484.5479126, 488.7896729
9: -213.0522919, 210.1563110, -222.8706055, 219.9002075, -432.9525146, 433.0269165

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1844970, upper bound: 385.1827406
time: 9.93 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1844970, upper bound: 385.1833202
time: 9.75 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -237.5785980, 188.1260834, -243.7186890, 193.0135193, -430.5921021, 431.8447571
1: -199.4214020, 167.0109253, -204.5043030, 171.2726135, -370.6940002, 371.5152283
2: -261.8223267, 169.4930878, -268.5508728, 173.8335266, -435.6557922, 438.0439453
3: -278.8170166, 146.6810913, -285.9150696, 150.4305267, -429.2475586, 432.5961609
4: -254.8795471, 195.1088715, -261.4442139, 200.0436859, -454.9231873, 456.5531006
5: -228.1727142, 177.1310883, -234.0196991, 181.6049194, -409.7776184, 411.1507874
6: -218.2875519, 210.4626465, -223.8714142, 215.8549805, -434.1425171, 434.3340454
7: -238.4990692, 199.6945343, -244.6354065, 204.7772827, -443.2763062, 444.3299561
8: -287.1939697, 195.8099365, -294.5720520, 200.8229065, -488.0168762, 490.3819885
9: -216.3509674, 213.4246216, -221.8298187, 218.8706360, -435.2215881, 435.2544250

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 131

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845257, upper bound: 385.1827429
time: 11.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1845257, upper bound: 385.1833706
time: 9.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -238.8824005, 189.1493073, -424.1998596, 425.0723572
1: -197.2579956, 165.2008362, -200.5513763, 167.9540405, -365.2120361, 365.7521973
2: -259.0479431, 167.6212616, -263.2330322, 170.5083923, -429.5563354, 430.8542480
3: -275.7994385, 145.1157227, -280.3717651, 147.5429077, -423.3422852, 425.4874878
4: -252.1463470, 192.9314423, -256.2998962, 196.2016449, -448.3479919, 449.2313232
5: -225.7425232, 175.1535187, -229.4580078, 178.2023468, -403.9447937, 404.6114807
6: -215.9430695, 208.2210846, -219.4587860, 211.6257324, -427.5687866, 427.6798706
7: -235.9456177, 197.5567017, -239.8339996, 200.8481903, -436.7937317, 437.3906860
8: -284.1164551, 193.6714935, -288.7486572, 196.9311066, -481.0474854, 482.4201660
9: -213.9953918, 211.1070557, -217.5705872, 214.6241455, -428.6195068, 428.6776428

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806263, upper bound: 385.1806213
time: 8.10 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806263, upper bound: 385.1806213
time: 8.69 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -238.8824005, 189.1493073, -428.4382324, 428.4173889
1: -200.7943878, 168.1681061, -200.5513763, 167.9540405, -368.7484131, 368.7194519
2: -263.7323914, 170.6133423, -263.2330322, 170.5083923, -434.2407532, 433.8463440
3: -280.7873535, 147.6848602, -280.3717651, 147.5429077, -428.3302307, 428.0566406
4: -256.6813049, 196.3964844, -256.2998962, 196.2016449, -452.8828735, 452.6963806
5: -229.8008270, 178.2671509, -229.4580078, 178.2023468, -408.0031738, 407.7250977
6: -219.8521881, 211.9787750, -219.4587860, 211.6257324, -431.4779053, 431.4375610
7: -240.1766357, 201.0883026, -239.8339996, 200.8481903, -441.0247803, 440.9223022
8: -289.2331848, 197.1266174, -288.7486572, 196.9311066, -486.1642456, 485.8752747
9: -217.8293915, 214.8981171, -217.5705872, 214.6241455, -432.4535522, 432.4686890

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806263, upper bound: 385.1807313
time: 8.15 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806263, upper bound: 385.1807313
time: 8.16 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -242.7910461, 192.2370605, -427.2875671, 428.9810181
1: -197.2579956, 165.2008362, -203.8125000, 170.6885986, -367.9465942, 369.0133362
2: -259.0479431, 167.6212616, -267.5561523, 173.2598267, -432.3077393, 435.1773376
3: -275.7994385, 145.1157227, -284.9802856, 149.9122620, -425.7117004, 430.0959778
4: -252.1463470, 192.9314423, -260.4773560, 199.3984680, -451.5447998, 453.4088135
5: -225.7425232, 175.1535187, -233.2022400, 181.0597839, -406.8022461, 408.3557129
6: -215.9430695, 208.2210846, -223.0684509, 215.0919495, -431.0349426, 431.2895508
7: -235.9456177, 197.5567017, -243.7280121, 204.0994873, -440.0451050, 441.2847290
8: -284.1164551, 193.6714935, -293.4702454, 200.1164551, -484.2329102, 487.1417236
9: -213.9953918, 211.1070557, -221.1016083, 218.1244812, -432.1197815, 432.2086792

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1810879
time: 8.54 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1810879
time: 8.96 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -242.7910461, 192.2370605, -431.5260010, 432.3260498
1: -200.7943878, 168.1681061, -203.8125000, 170.6885986, -371.4829712, 371.9805908
2: -263.7323914, 170.6133423, -267.5561523, 173.2598267, -436.9921570, 438.1694336
3: -280.7873535, 147.6848602, -284.9802856, 149.9122620, -430.6996155, 432.6651306
4: -256.6813049, 196.3964844, -260.4773560, 199.3984680, -456.0797729, 456.8738403
5: -229.8008270, 178.2671509, -233.2022400, 181.0597839, -410.8605957, 411.4693604
6: -219.8521881, 211.9787750, -223.0684509, 215.0919495, -434.9440918, 435.0472107
7: -240.1766357, 201.0883026, -243.7280121, 204.0994873, -444.2761230, 444.8163147
8: -289.2331848, 197.1266174, -293.4702454, 200.1164551, -489.3496399, 490.5968628
9: -217.8293915, 214.8981171, -221.1016083, 218.1244812, -435.9538269, 435.9997253

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1813624
time: 8.70 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1813624
time: 9.22 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.25 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1900419, upper bound: 385.1900419
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1900419, upper bound: 385.1900712
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1900712, upper bound: 385.1900419
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1900712, upper bound: 385.1900713
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1844970, upper bound: 385.1827406
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1844970, upper bound: 385.1833202
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1845257, upper bound: 385.1827429
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1845257, upper bound: 385.1833706
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1806263, upper bound: 385.1806213
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1806263, upper bound: 385.1806213
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1806263, upper bound: 385.1807313
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1806263, upper bound: 385.1807313
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1810879
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1810879
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1813624
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.25
Output dim: 1, lower bound: -385.1806213, upper bound: 385.1813624

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -233.9247894, 185.2384186, -419.1631775, 419.1631775
1: -196.3794403, 164.4563293, -196.3794403, 164.4563293, -360.8357544, 360.8357544
2: -257.7795410, 166.9246521, -257.7795410, 166.9246521, -424.7041626, 424.7041626
3: -274.5066223, 144.4709167, -274.5066223, 144.4709167, -418.9775085, 418.9775085
4: -250.9762573, 192.1235657, -250.9762573, 192.1235657, -443.0997925, 443.0997925
5: -224.6685638, 174.4661713, -224.6685638, 174.4661713, -399.1347046, 399.1347046
6: -214.9111786, 207.2229767, -214.9111786, 207.2229767, -422.1341248, 422.1341248
7: -234.8669281, 196.6575928, -234.8669281, 196.6575928, -431.5245056, 431.5245056
8: -282.7800598, 192.8340759, -282.7800598, 192.8340759, -475.6141357, 475.6141357
9: -213.0522919, 210.1563110, -213.0522919, 210.1563110, -423.2086182, 423.2086182

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1878840, upper bound: 385.1881327
time: 9.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1890187, upper bound: 385.1889195
time: 8.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -237.5785980, 188.1260834, -422.0508118, 422.8170166
1: -196.3794403, 164.4563293, -199.4214020, 167.0109253, -363.3903198, 363.8777466
2: -257.7795410, 166.9246521, -261.8223267, 169.4930878, -427.2726135, 428.7469177
3: -274.5066223, 144.4709167, -278.8170166, 146.6810913, -421.1877136, 423.2879333
4: -250.9762573, 192.1235657, -254.8795471, 195.1088715, -446.0851135, 447.0030518
5: -224.6685638, 174.4661713, -228.1727142, 177.1310883, -401.7996521, 402.6388550
6: -214.9111786, 207.2229767, -218.2875519, 210.4626465, -425.3738098, 425.5105286
7: -234.8669281, 196.6575928, -238.4990692, 199.6945343, -434.5614624, 435.1566467
8: -282.7800598, 192.8340759, -287.1939697, 195.8099365, -478.5899963, 480.0280457
9: -213.0522919, 210.1563110, -216.3509674, 213.4246216, -426.4768982, 426.5072632

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1878840, upper bound: 385.1881327
time: 7.79 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1890187, upper bound: 385.1889267
time: 8.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -237.5785980, 188.1260834, -233.9247894, 185.2384186, -422.8170166, 422.0508118
1: -199.4214020, 167.0109253, -196.3794403, 164.4563293, -363.8777466, 363.3903198
2: -261.8223267, 169.4930878, -257.7795410, 166.9246521, -428.7469177, 427.2726135
3: -278.8170166, 146.6810913, -274.5066223, 144.4709167, -423.2879333, 421.1877136
4: -254.8795471, 195.1088715, -250.9762573, 192.1235657, -447.0030518, 446.0851135
5: -228.1727142, 177.1310883, -224.6685638, 174.4661713, -402.6388550, 401.7996521
6: -218.2875519, 210.4626465, -214.9111786, 207.2229767, -425.5105286, 425.3738098
7: -238.4990692, 199.6945343, -234.8669281, 196.6575928, -435.1566467, 434.5614624
8: -287.1939697, 195.8099365, -282.7800598, 192.8340759, -480.0280457, 478.5899963
9: -216.3509674, 213.4246216, -213.0522919, 210.1563110, -426.5072632, 426.4768982

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1872868, upper bound: 385.1877910
time: 8.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1888443, upper bound: 385.1888091
time: 10.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -237.5785980, 188.1260834, -237.5785980, 188.1260834, -425.7046509, 425.7046509
1: -199.4214020, 167.0109253, -199.4214020, 167.0109253, -366.4323120, 366.4323120
2: -261.8223267, 169.4930878, -261.8223267, 169.4930878, -431.3153687, 431.3153687
3: -278.8170166, 146.6810913, -278.8170166, 146.6810913, -425.4981079, 425.4981079
4: -254.8795471, 195.1088715, -254.8795471, 195.1088715, -449.9883728, 449.9883728
5: -228.1727142, 177.1310883, -228.1727142, 177.1310883, -405.3038025, 405.3038025
6: -218.2875519, 210.4626465, -218.2875519, 210.4626465, -428.7501831, 428.7501831
7: -238.4990692, 199.6945343, -238.4990692, 199.6945343, -438.1936035, 438.1936035
8: -287.1939697, 195.8099365, -287.1939697, 195.8099365, -483.0039062, 483.0039062
9: -216.3509674, 213.4246216, -216.3509674, 213.4246216, -429.7755127, 429.7755127

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1872868, upper bound: 385.1877934
time: 9.27 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1888443, upper bound: 385.1888560
time: 9.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -235.0505981, 186.1899567, -420.1147156, 420.2890015
1: -196.3794403, 164.4563293, -197.2579956, 165.2008362, -361.5802307, 361.7143250
2: -257.7795410, 166.9246521, -259.0479431, 167.6212616, -425.4007263, 425.9725647
3: -274.5066223, 144.4709167, -275.7994385, 145.1157227, -419.6223450, 420.2703247
4: -250.9762573, 192.1235657, -252.1463470, 192.9314423, -443.9077148, 444.2698975
5: -224.6685638, 174.4661713, -225.7425232, 175.1535187, -399.8220825, 400.2086182
6: -214.9111786, 207.2229767, -215.9430695, 208.2210846, -423.1322327, 423.1660461
7: -234.8669281, 196.6575928, -235.9456177, 197.5567017, -432.4236450, 432.6031799
8: -282.7800598, 192.8340759, -284.1164551, 193.6714935, -476.4515381, 476.9505310
9: -213.0522919, 210.1563110, -213.9953918, 211.1070557, -424.1593628, 424.1516724

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1817360, upper bound: 385.1806246
time: 11.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1837364, upper bound: 385.1822363
time: 9.25 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -239.2889404, 189.5350037, -423.4597473, 424.5273438
1: -196.3794403, 164.4563293, -200.7943878, 168.1681061, -364.5474548, 365.2507324
2: -257.7795410, 166.9246521, -263.7323914, 170.6133423, -428.3928223, 430.6569824
3: -274.5066223, 144.4709167, -280.7873535, 147.6848602, -422.1914673, 425.2582397
4: -250.9762573, 192.1235657, -256.6813049, 196.3964844, -447.3727417, 448.8048096
5: -224.6685638, 174.4661713, -229.8008270, 178.2671509, -402.9356995, 404.2669983
6: -214.9111786, 207.2229767, -219.8521881, 211.9787750, -426.8898926, 427.0751648
7: -234.8669281, 196.6575928, -240.1766357, 201.0883026, -435.9552307, 436.8342285
8: -282.7800598, 192.8340759, -289.2331848, 197.1266174, -479.9066467, 482.0672607
9: -213.0522919, 210.1563110, -217.8293915, 214.8981171, -427.9504089, 427.9857178

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1817360, upper bound: 385.1809741
time: 10.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1837364, upper bound: 385.1827322
time: 8.26 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -237.5785980, 188.1260834, -235.0505981, 186.1899567, -423.7685547, 423.1766357
1: -199.4214020, 167.0109253, -197.2579956, 165.2008362, -364.6222229, 364.2689209
2: -261.8223267, 169.4930878, -259.0479431, 167.6212616, -429.4434814, 428.5410156
3: -278.8170166, 146.6810913, -275.7994385, 145.1157227, -423.9327393, 422.4804993
4: -254.8795471, 195.1088715, -252.1463470, 192.9314423, -447.8109741, 447.2552185
5: -228.1727142, 177.1310883, -225.7425232, 175.1535187, -403.3262024, 402.8735962
6: -218.2875519, 210.4626465, -215.9430695, 208.2210846, -426.5086365, 426.4057007
7: -238.4990692, 199.6945343, -235.9456177, 197.5567017, -436.0557861, 435.6401367
8: -287.1939697, 195.8099365, -284.1164551, 193.6714935, -480.8654785, 479.9263916
9: -216.3509674, 213.4246216, -213.9953918, 211.1070557, -427.4580078, 427.4199219

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1811008, upper bound: 385.1800894
time: 10.29 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1836442, upper bound: 385.1821292
time: 10.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -237.5785980, 188.1260834, -239.2889404, 189.5350037, -427.1135864, 427.4150391
1: -199.4214020, 167.0109253, -200.7943878, 168.1681061, -367.5894470, 367.8052979
2: -261.8223267, 169.4930878, -263.7323914, 170.6133423, -432.4355774, 433.2254333
3: -278.8170166, 146.6810913, -280.7873535, 147.6848602, -426.5018921, 427.4684448
4: -254.8795471, 195.1088715, -256.6813049, 196.3964844, -451.2760010, 451.7901306
5: -228.1727142, 177.1310883, -229.8008270, 178.2671509, -406.4398499, 406.9319153
6: -218.2875519, 210.4626465, -219.8521881, 211.9787750, -430.2663269, 430.3148193
7: -238.4990692, 199.6945343, -240.1766357, 201.0883026, -439.5873718, 439.8711548
8: -287.1939697, 195.8099365, -289.2331848, 197.1266174, -484.3205566, 485.0431213
9: -216.3509674, 213.4246216, -217.8293915, 214.8981171, -431.2490845, 431.2539673

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1811008, upper bound: 385.1803586
time: 9.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1836442, upper bound: 385.1827062
time: 10.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -233.9247894, 185.2384186, -420.2890015, 420.1147156
1: -197.2579956, 165.2008362, -196.3794403, 164.4563293, -361.7143250, 361.5802307
2: -259.0479431, 167.6212616, -257.7795410, 166.9246521, -425.9725647, 425.4007263
3: -275.7994385, 145.1157227, -274.5066223, 144.4709167, -420.2703247, 419.6223450
4: -252.1463470, 192.9314423, -250.9762573, 192.1235657, -444.2698975, 443.9077148
5: -225.7425232, 175.1535187, -224.6685638, 174.4661713, -400.2086182, 399.8220825
6: -215.9430695, 208.2210846, -214.9111786, 207.2229767, -423.1660461, 423.1322327
7: -235.9456177, 197.5567017, -234.8669281, 196.6575928, -432.6031799, 432.4236450
8: -284.1164551, 193.6714935, -282.7800598, 192.8340759, -476.9505310, 476.4515381
9: -213.9953918, 211.1070557, -213.0522919, 210.1563110, -424.1516724, 424.1593628

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1736362, upper bound: 385.1754931
time: 9.15 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1801243, upper bound: 385.1801243
time: 9.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -235.0505981, 186.1899567, -421.2405396, 421.2405396
1: -197.2579956, 165.2008362, -197.2579956, 165.2008362, -362.4588318, 362.4588318
2: -259.0479431, 167.6212616, -259.0479431, 167.6212616, -426.6691284, 426.6691284
3: -275.7994385, 145.1157227, -275.7994385, 145.1157227, -420.9151306, 420.9151306
4: -252.1463470, 192.9314423, -252.1463470, 192.9314423, -445.0777893, 445.0777893
5: -225.7425232, 175.1535187, -225.7425232, 175.1535187, -400.8959961, 400.8959961
6: -215.9430695, 208.2210846, -215.9430695, 208.2210846, -424.1641541, 424.1641541
7: -235.9456177, 197.5567017, -235.9456177, 197.5567017, -433.5023193, 433.5023193
8: -284.1164551, 193.6714935, -284.1164551, 193.6714935, -477.7879639, 477.7879639
9: -213.9953918, 211.1070557, -213.9953918, 211.1070557, -425.1024475, 425.1024475

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1736362, upper bound: 385.1754931
time: 9.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1801243, upper bound: 385.1801243
time: 9.16 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -233.9247894, 185.2384186, -424.5273438, 423.4597473
1: -200.7943878, 168.1681061, -196.3794403, 164.4563293, -365.2507324, 364.5474548
2: -263.7323914, 170.6133423, -257.7795410, 166.9246521, -430.6569824, 428.3928223
3: -280.7873535, 147.6848602, -274.5066223, 144.4709167, -425.2582397, 422.1914673
4: -256.6813049, 196.3964844, -250.9762573, 192.1235657, -448.8048096, 447.3727417
5: -229.8008270, 178.2671509, -224.6685638, 174.4661713, -404.2669983, 402.9356995
6: -219.8521881, 211.9787750, -214.9111786, 207.2229767, -427.0751648, 426.8898926
7: -240.1766357, 201.0883026, -234.8669281, 196.6575928, -436.8342285, 435.9552307
8: -289.2331848, 197.1266174, -282.7800598, 192.8340759, -482.0672607, 479.9066467
9: -217.8293915, 214.8981171, -213.0522919, 210.1563110, -427.9857178, 427.9504089

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 64

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1736436, upper bound: 385.1753524
time: 8.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806370, upper bound: 385.1802207
time: 8.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -235.0505981, 186.1899567, -425.4788818, 424.5856018
1: -200.7943878, 168.1681061, -197.2579956, 165.2008362, -365.9952393, 365.4260864
2: -263.7323914, 170.6133423, -259.0479431, 167.6212616, -431.3535461, 429.6612244
3: -280.7873535, 147.6848602, -275.7994385, 145.1157227, -425.9030762, 423.4842834
4: -256.6813049, 196.3964844, -252.1463470, 192.9314423, -449.6127319, 448.5428467
5: -229.8008270, 178.2671509, -225.7425232, 175.1535187, -404.9543457, 404.0096130
6: -219.8521881, 211.9787750, -215.9430695, 208.2210846, -428.0732727, 427.9218445
7: -240.1766357, 201.0883026, -235.9456177, 197.5567017, -437.7333374, 437.0339355
8: -289.2331848, 197.1266174, -284.1164551, 193.6714935, -482.9046631, 481.2430115
9: -217.8293915, 214.8981171, -213.9953918, 211.1070557, -428.9364624, 428.8934937

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 64

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1736436, upper bound: 385.1753524
time: 8.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806370, upper bound: 385.1802207
time: 8.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -237.5785980, 188.1260834, -423.1766357, 423.7685547
1: -197.2579956, 165.2008362, -199.4214020, 167.0109253, -364.2689209, 364.6222229
2: -259.0479431, 167.6212616, -261.8223267, 169.4930878, -428.5410156, 429.4434814
3: -275.7994385, 145.1157227, -278.8170166, 146.6810913, -422.4804993, 423.9327393
4: -252.1463470, 192.9314423, -254.8795471, 195.1088715, -447.2552185, 447.8109741
5: -225.7425232, 175.1535187, -228.1727142, 177.1310883, -402.8735962, 403.3262024
6: -215.9430695, 208.2210846, -218.2875519, 210.4626465, -426.4057007, 426.5086365
7: -235.9456177, 197.5567017, -238.4990692, 199.6945343, -435.6401367, 436.0557861
8: -284.1164551, 193.6714935, -287.1939697, 195.8099365, -479.9263916, 480.8654785
9: -213.9953918, 211.1070557, -216.3509674, 213.4246216, -427.4199219, 427.4580078

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1737104, upper bound: 385.1757348
time: 10.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1802207, upper bound: 385.1806370
time: 9.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -239.2889404, 189.5350037, -424.5856018, 425.4788818
1: -197.2579956, 165.2008362, -200.7943878, 168.1681061, -365.4260864, 365.9952393
2: -259.0479431, 167.6212616, -263.7323914, 170.6133423, -429.6612244, 431.3535461
3: -275.7994385, 145.1157227, -280.7873535, 147.6848602, -423.4842834, 425.9030762
4: -252.1463470, 192.9314423, -256.6813049, 196.3964844, -448.5428467, 449.6127319
5: -225.7425232, 175.1535187, -229.8008270, 178.2671509, -404.0096130, 404.9543457
6: -215.9430695, 208.2210846, -219.8521881, 211.9787750, -427.9218445, 428.0732727
7: -235.9456177, 197.5567017, -240.1766357, 201.0883026, -437.0339355, 437.7333374
8: -284.1164551, 193.6714935, -289.2331848, 197.1266174, -481.2430115, 482.9046631
9: -213.9953918, 211.1070557, -217.8293915, 214.8981171, -428.8934937, 428.9364624

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1737104, upper bound: 385.1757348
time: 9.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1802207, upper bound: 385.1806370
time: 8.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -237.5785980, 188.1260834, -427.4150391, 427.1135864
1: -200.7943878, 168.1681061, -199.4214020, 167.0109253, -367.8052979, 367.5894470
2: -263.7323914, 170.6133423, -261.8223267, 169.4930878, -433.2254333, 432.4355774
3: -280.7873535, 147.6848602, -278.8170166, 146.6810913, -427.4684448, 426.5018921
4: -256.6813049, 196.3964844, -254.8795471, 195.1088715, -451.7901306, 451.2760010
5: -229.8008270, 178.2671509, -228.1727142, 177.1310883, -406.9319153, 406.4398499
6: -219.8521881, 211.9787750, -218.2875519, 210.4626465, -430.3148193, 430.2663269
7: -240.1766357, 201.0883026, -238.4990692, 199.6945343, -439.8711548, 439.5873718
8: -289.2331848, 197.1266174, -287.1939697, 195.8099365, -485.0431213, 484.3205566
9: -217.8293915, 214.8981171, -216.3509674, 213.4246216, -431.2539673, 431.2490845

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 64

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1738676, upper bound: 385.1756662
time: 9.50 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1809423, upper bound: 385.1809110
time: 8.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -239.2889404, 189.5350037, -428.8239441, 428.8239441
1: -200.7943878, 168.1681061, -200.7943878, 168.1681061, -368.9624939, 368.9624939
2: -263.7323914, 170.6133423, -263.7323914, 170.6133423, -434.3456421, 434.3456421
3: -280.7873535, 147.6848602, -280.7873535, 147.6848602, -428.4722290, 428.4722290
4: -256.6813049, 196.3964844, -256.6813049, 196.3964844, -453.0777283, 453.0777283
5: -229.8008270, 178.2671509, -229.8008270, 178.2671509, -408.0679932, 408.0679932
6: -219.8521881, 211.9787750, -219.8521881, 211.9787750, -431.8309631, 431.8309631
7: -240.1766357, 201.0883026, -240.1766357, 201.0883026, -441.2649536, 441.2649536
8: -289.2331848, 197.1266174, -289.2331848, 197.1266174, -486.3597717, 486.3597717
9: -217.8293915, 214.8981171, -217.8293915, 214.8981171, -432.7275085, 432.7275085

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 64

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1738676, upper bound: 385.1756662
time: 10.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1809423, upper bound: 385.1809110
time: 9.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.96 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1878840, upper bound: 385.1881327
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1890187, upper bound: 385.1889195
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1878840, upper bound: 385.1881327
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1890187, upper bound: 385.1889267
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1872868, upper bound: 385.1877910
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1888443, upper bound: 385.1888091
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1872868, upper bound: 385.1877934
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1888443, upper bound: 385.1888560
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1817360, upper bound: 385.1806246
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1837364, upper bound: 385.1822363
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1817360, upper bound: 385.1809741
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1837364, upper bound: 385.1827322
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1811008, upper bound: 385.1800894
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1836442, upper bound: 385.1821292
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1811008, upper bound: 385.1803586
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1836442, upper bound: 385.1827062
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1736362, upper bound: 385.1754931
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1801243, upper bound: 385.1801243
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1736362, upper bound: 385.1754931
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1801243, upper bound: 385.1801243
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1736436, upper bound: 385.1753524
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1806370, upper bound: 385.1802207
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1736436, upper bound: 385.1753524
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1806370, upper bound: 385.1802207
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1737104, upper bound: 385.1757348
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1802207, upper bound: 385.1806370
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1737104, upper bound: 385.1757348
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1802207, upper bound: 385.1806370
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1738676, upper bound: 385.1756662
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1809423, upper bound: 385.1809110
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1738676, upper bound: 385.1756662
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 20.96
Output dim: 1, lower bound: -385.1809423, upper bound: 385.1809110
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=387.1527099609375
rel_dist={1: [-385.1965121761082, 385.19651217610806]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1871435, upper bound: 385.1861071
time: 9.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1848791, upper bound: 385.1848790
time: 8.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.28
Output dim: 1, lower bound: -385.1871435, upper bound: 385.1861071
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.28
Output dim: 1, lower bound: -385.1848791, upper bound: 385.1848790

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -249.9593658, 197.8691711, -441.4149475, 442.7714844
1: -204.4127197, 171.1914368, -209.8140106, 175.7198639, -380.1325684, 381.0054321
2: -268.3244019, 173.8122711, -275.3788452, 178.4451294, -446.7695007, 449.1911011
3: -285.7283020, 150.3652191, -293.3136292, 154.3358002, -440.0640869, 443.6788330
4: -261.2940063, 200.0122070, -268.1935425, 205.2961731, -466.5901794, 468.2057495
5: -233.8536224, 181.6165619, -240.0374908, 186.4508667, -420.3045044, 421.6539307
6: -223.7100830, 215.6901703, -229.6011658, 221.3856506, -445.0957336, 445.2913208
7: -244.5041962, 204.6670990, -250.9410553, 210.0859985, -454.5901489, 455.6081543
8: -294.3796692, 200.7644348, -302.1096497, 206.0651550, -500.4448242, 502.8740845
9: -221.7438049, 218.7707672, -227.6000519, 224.5565796, -446.3003845, 446.3708191

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1827560, upper bound: 385.1816520
time: 12.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1831463, upper bound: 385.1823467
time: 9.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -244.8681488, 193.9198761, -244.7326050, 193.7545624, -438.6227112, 438.6524658
1: -205.4636383, 172.0779266, -205.4111786, 172.0250702, -377.4886780, 377.4890747
2: -269.8109131, 174.6556702, -269.6354065, 174.6739349, -444.4848633, 444.2910461
3: -287.2557373, 151.1343994, -287.1409607, 151.1102905, -438.3660278, 438.2753296
4: -262.6797791, 200.9863586, -262.5484619, 200.9785919, -463.6583862, 463.5348206
5: -235.1186066, 182.4603882, -235.0185699, 182.5112152, -417.6298218, 417.4789429
6: -224.9225464, 216.8661041, -224.7862854, 216.7490234, -441.6715393, 441.6523743
7: -245.7872772, 205.7355347, -245.6802216, 205.6748047, -451.4620667, 451.4157410
8: -295.9555969, 201.7678680, -295.7971191, 201.7437897, -497.6994019, 497.5650024
9: -222.8706055, 219.9002075, -222.8046875, 219.8343658, -442.7049255, 442.7048950

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1807924, upper bound: 385.1805519
time: 8.48 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1812414, upper bound: 385.1812414
time: 10.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 20.69 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 20.69
Output dim: 1, lower bound: -385.1827560, upper bound: 385.1816520
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.69
Output dim: 1, lower bound: -385.1831463, upper bound: 385.1823467
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.69
Output dim: 1, lower bound: -385.1807924, upper bound: 385.1805519
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.69
Output dim: 1, lower bound: -385.1812414, upper bound: 385.1812414

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -240.3940582, 190.3311157, -240.2392273, 190.2170563, -430.6110229, 430.5703430
1: -201.7808685, 168.9853210, -201.6957397, 168.9138794, -370.6947632, 370.6810608
2: -264.8701477, 171.5556183, -264.7230225, 171.4863892, -436.3564453, 436.2786255
3: -282.0526733, 148.4347229, -281.9743042, 148.3790894, -430.4317627, 430.4090271
4: -257.9142456, 197.4281616, -257.7682800, 197.3242645, -455.2385254, 455.1964111
5: -230.8450317, 179.2743530, -230.7578583, 179.2253571, -410.0703735, 410.0322266
6: -220.8276825, 212.9164886, -220.7104340, 212.8289337, -433.6566162, 433.6269226
7: -241.3474884, 202.0435333, -241.2026825, 201.9924469, -443.3399048, 443.2461853
8: -290.5809021, 198.1663971, -290.3897705, 198.0525055, -488.6333923, 488.5561523
9: -218.8969727, 215.9486389, -218.8174286, 215.8515625, -434.7485352, 434.7660522

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1778120, upper bound: 385.1772747
time: 9.92 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1778120, upper bound: 385.1810995
time: 10.23 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -239.1815186, 189.3722382, -244.0522614, 193.2302856, -432.4117737, 433.4244995
1: -200.7687988, 168.1326752, -204.8750000, 171.5811310, -372.3499146, 373.0076904
2: -263.5404358, 170.6901703, -268.9419556, 174.1694794, -437.7098694, 439.6320496
3: -280.6383057, 147.6920471, -286.4703979, 150.6894531, -431.3277588, 434.1624146
4: -256.6026611, 196.4332428, -261.8431091, 200.4423676, -457.0450439, 458.2763367
5: -229.6833344, 178.3646698, -234.4127808, 182.0105286, -411.6938477, 412.7774048
6: -219.7196045, 211.8506927, -224.2324066, 216.2109680, -435.9305115, 436.0830688
7: -240.1265259, 201.0267639, -244.9982910, 205.1631775, -445.2897034, 446.0249939
8: -289.1264038, 197.1779938, -294.9960632, 201.1598206, -490.2862244, 492.1740723
9: -217.7904816, 214.8621521, -222.2618408, 219.2660217, -437.0565186, 437.1239624

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 103

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1781634, upper bound: 385.1778184
time: 12.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1825221, upper bound: 385.1819207
time: 11.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -241.6534729, 191.3897705, -234.9154663, 186.0251617, -427.6786194, 426.3052063
1: -202.7768555, 169.8269196, -197.2088318, 165.1502075, -367.9270630, 367.0357666
2: -266.2875061, 172.3529205, -258.8730469, 167.6459808, -433.9334412, 431.2259216
3: -283.5056152, 149.1642914, -275.6864929, 145.0932312, -428.5988159, 424.8507690
4: -259.2314148, 198.3496246, -252.0187378, 192.9259644, -452.1573792, 450.3682861
5: -232.0497131, 180.0683746, -225.6460114, 175.2109985, -407.2606812, 405.7143555
6: -221.9831390, 214.0359344, -215.8065796, 208.1055603, -430.0886536, 429.8425293
7: -242.5649261, 203.0583954, -235.8417206, 197.4996796, -440.0646057, 438.9000549
8: -292.0803833, 199.1169128, -283.9588928, 193.6512146, -485.7315979, 483.0757751
9: -219.9653931, 217.0211639, -213.9339905, 211.0413666, -431.0067749, 430.9551392

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1727662, upper bound: 385.1738614
time: 9.73 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1803558, upper bound: 385.1800770
time: 12.18 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -240.6739349, 190.6122284, -238.9278870, 189.1945953, -429.8685303, 429.5401001
1: -201.9618530, 169.1383972, -200.5573273, 167.9592896, -369.9210815, 369.6957397
2: -265.2131348, 171.6556549, -263.3098145, 170.4732666, -435.6863403, 434.9654236
3: -282.3621216, 148.5655518, -280.4153442, 147.5268250, -429.8889465, 428.9808960
4: -258.1701355, 197.5460510, -256.3086853, 196.2069092, -454.3770447, 453.8547363
5: -231.1084747, 179.3374939, -229.4885406, 178.1489563, -409.2574463, 408.8260193
6: -221.0866241, 213.1762695, -219.5121613, 211.6640472, -432.7506714, 432.6883850
7: -241.5828094, 202.2382660, -239.8426971, 200.8400116, -442.4227905, 442.0808716
8: -290.9065247, 198.3184662, -288.8059387, 196.9222107, -487.8287354, 487.1243896
9: -219.0718079, 216.1427612, -217.5608063, 214.6332245, -433.7050171, 433.7035522

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1730491, upper bound: 385.1742021
time: 11.44 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1808643, upper bound: 385.1808643
time: 11.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.06 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 24.06
Output dim: 1, lower bound: -385.1778120, upper bound: 385.1772747
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 1, lower bound: -385.1778120, upper bound: 385.1810995
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 24.06
Output dim: 1, lower bound: -385.1781634, upper bound: 385.1778184
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 1, lower bound: -385.1825221, upper bound: 385.1819207
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 24.06
Output dim: 1, lower bound: -385.1727662, upper bound: 385.1738614
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 1, lower bound: -385.1803558, upper bound: 385.1800770
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 24.06
Output dim: 1, lower bound: -385.1730491, upper bound: 385.1742021
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 1, lower bound: -385.1808643, upper bound: 385.1808643

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -236.5041809, 187.2378693, -239.1556244, 189.3555145, -425.8596802, 426.3934937
1: -198.5205688, 166.2641754, -200.7875671, 168.1559448, -366.6764832, 367.0517578
2: -260.5673523, 168.7626190, -263.5245667, 170.7086029, -431.2759094, 432.2871704
3: -277.5006714, 146.0738525, -280.7062378, 147.7214203, -425.2221069, 426.7800903
4: -253.7057953, 194.2357025, -256.5961914, 196.4349213, -450.1407166, 450.8318787
5: -227.0999908, 176.3810272, -229.7145538, 178.4195251, -405.5195007, 406.0955505
6: -217.2023773, 209.4715881, -219.7004242, 211.8694611, -429.0718079, 429.1719971
7: -237.4553223, 198.8045197, -240.1186676, 201.0903473, -438.5456543, 438.9231262
8: -285.8446655, 194.9147797, -289.0706482, 197.1465302, -482.9911804, 483.9854126
9: -215.3836060, 212.4635620, -217.8387451, 214.8810120, -430.2645874, 430.3023071

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1820490, upper bound: 385.1810995
time: 11.50 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1820490, upper bound: 385.1810995
time: 9.69 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -235.3302155, 186.3101654, -242.9982452, 192.3921661, -427.7223816, 429.3084106
1: -197.5411072, 165.4381561, -203.9915161, 170.8435059, -368.3845520, 369.4296570
2: -259.2804565, 167.9253082, -267.7758179, 173.4130707, -432.6935120, 435.7011108
3: -276.1312561, 145.3548889, -285.2369080, 150.0494995, -426.1807556, 430.5917969
4: -252.4357147, 193.2728882, -260.7029724, 199.5772400, -452.0129395, 453.9758301
5: -225.9752960, 175.5002747, -233.3975677, 181.2265778, -407.2018738, 408.8978271
6: -216.1306763, 208.4402008, -223.2500305, 215.2773438, -431.4080200, 431.6902466
7: -236.2733154, 197.8198242, -243.9438629, 204.2855072, -440.5587158, 441.7636719
8: -284.4381714, 193.9585876, -293.7123413, 200.2783966, -484.7165527, 487.6709290
9: -214.3119812, 211.4121857, -221.3096619, 218.3217163, -432.6336670, 432.7218323

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1781634, upper bound: 385.1819207
time: 12.03 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1781634, upper bound: 385.1819207
time: 11.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -238.0094147, 188.4917297, -233.8571320, 185.1834717, -423.1928711, 422.3488770
1: -199.7221222, 167.2756653, -196.3219147, 164.4099884, -364.1320801, 363.5975952
2: -262.2544861, 169.7379761, -257.7023315, 166.8862305, -429.1407166, 427.4403076
3: -279.2405701, 146.9519958, -274.4483643, 144.4509430, -423.6915283, 421.4003601
4: -255.2910767, 195.3572693, -250.8741302, 192.0574493, -447.3485107, 446.2313538
5: -228.5413055, 177.3584290, -224.6268158, 174.4239044, -402.9652100, 401.9852295
6: -218.5880432, 210.8086853, -214.8202057, 207.1683655, -425.7564087, 425.6289062
7: -238.9193115, 200.0236816, -234.7831879, 196.6185150, -435.5378418, 434.8067932
8: -287.6416016, 196.0689697, -282.6703491, 192.7659302, -480.4075012, 478.7393188
9: -216.6724854, 213.7581482, -212.9780426, 210.0936890, -426.7661743, 426.7362061

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1715526, upper bound: 385.1715007
time: 9.84 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1716501, upper bound: 385.1713895
time: 11.08 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -237.0626678, 187.7404480, -237.8962250, 188.3743286, -425.4370117, 425.6366577
1: -198.9340668, 166.6100769, -199.6924744, 167.2374115, -366.1714172, 366.3025513
2: -261.2169189, 169.0640869, -262.1685791, 169.7329559, -430.9498596, 431.2326355
3: -278.1356812, 146.3729248, -279.2080994, 146.9004822, -425.0361633, 425.5810242
4: -254.2652130, 194.5812073, -255.1929932, 195.3601379, -449.6253662, 449.7742004
5: -227.6310883, 176.6520081, -228.4948578, 177.3817902, -405.0128174, 405.1468506
6: -217.7218933, 209.9780273, -218.5507660, 210.7505188, -428.4724121, 428.5287476
7: -237.9709625, 199.2309418, -238.8108368, 199.9809875, -437.9519653, 438.0417786
8: -286.5071411, 195.2973633, -287.5497437, 196.0595093, -482.5666504, 482.8471069
9: -215.8088989, 212.9086456, -216.6290741, 213.7091370, -429.5179749, 429.5377197

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1719517, upper bound: 385.1721639
time: 9.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1719231, upper bound: 385.1719231
time: 9.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.32 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.32
Output dim: 1, lower bound: -385.1820490, upper bound: 385.1810995
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.32
Output dim: 1, lower bound: -385.1820490, upper bound: 385.1810995
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 20.32
Output dim: 1, lower bound: -385.1781634, upper bound: 385.1819207
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.32
Output dim: 1, lower bound: -385.1781634, upper bound: 385.1819207
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.32
Output dim: 1, lower bound: -385.1715526, upper bound: 385.1715007
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 20.32
Output dim: 1, lower bound: -385.1716501, upper bound: 385.1713895
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.32
Output dim: 1, lower bound: -385.1719517, upper bound: 385.1721639
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 20.32
Output dim: 1, lower bound: -385.1719231, upper bound: 385.1719231

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -236.5041809, 187.2378693, -232.8202667, 184.3598938, -420.8640747, 420.0581055
1: -198.5205688, 166.2641754, -195.4533997, 163.6836548, -362.2041931, 361.7175903
2: -260.5673523, 168.7626190, -256.5574341, 166.1316681, -426.6990356, 425.3200684
3: -277.5006714, 146.0738525, -273.2141418, 143.8002014, -421.3008728, 419.2879944
4: -253.7057953, 194.2357025, -249.7814178, 191.2168579, -444.9226685, 444.0170593
5: -227.0999908, 176.3810272, -223.6051788, 173.6446533, -400.7446289, 399.9861450
6: -217.2023773, 209.4715881, -213.8817444, 206.2447052, -423.4470825, 423.3533325
7: -237.4553223, 198.8045197, -233.7616272, 195.7376862, -433.1929932, 432.5661621
8: -285.8446655, 194.9147797, -281.4350281, 191.9104462, -477.7551270, 476.3497925
9: -215.3836060, 212.4635620, -212.0544281, 209.1665497, -424.5501404, 424.5180054

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1736184, upper bound: 385.1724318
time: 10.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1734711, upper bound: 385.1724873
time: 10.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -236.5041809, 187.2378693, -234.0178528, 185.3689728, -421.8731689, 421.2557373
1: -198.5205688, 166.2641754, -196.3922577, 164.4779358, -362.9985046, 362.6564331
2: -260.5673523, 168.7626190, -257.9051208, 166.8803253, -427.4476929, 426.6677246
3: -277.5006714, 146.0738525, -274.5909119, 144.4886627, -421.9893188, 420.6647644
4: -253.7057953, 194.2357025, -251.0298004, 192.0835419, -445.7893372, 445.2654724
5: -227.0999908, 176.3810272, -224.7484131, 174.3856049, -401.4855957, 401.1294556
6: -217.2023773, 209.4715881, -214.9810333, 207.3065948, -424.5089722, 424.4526367
7: -237.4553223, 198.8045197, -234.9124756, 196.6967773, -434.1520996, 433.7169800
8: -285.8446655, 194.9147797, -282.8586731, 192.8074951, -478.6521301, 477.7734375
9: -215.3836060, 212.4635620, -213.0621643, 210.1823578, -425.5659790, 425.5257263

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1736184, upper bound: 385.1724318
time: 9.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1734711, upper bound: 385.1724873
time: 12.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -235.3302155, 186.3101654, -236.5046082, 187.2722168, -422.6024170, 422.8147583
1: -197.5411072, 165.4381561, -198.5209961, 166.2592773, -363.8003845, 363.9591370
2: -259.2804565, 167.9253082, -260.6340942, 168.7221680, -428.0025635, 428.5593262
3: -276.1312561, 145.3548889, -277.5600891, 146.0288239, -422.1600952, 422.9149780
4: -252.4357147, 193.2728882, -253.7177277, 194.2274475, -446.6631165, 446.9905701
5: -225.9752960, 175.5002747, -227.1386261, 176.3322449, -402.3074951, 402.6389160
6: -216.1306763, 208.4402008, -217.2867737, 209.5114136, -425.6420593, 425.7269897
7: -236.2733154, 197.8198242, -237.4244537, 198.8002777, -435.0735474, 435.2442627
8: -284.4381714, 193.9585876, -285.8862000, 194.9118347, -479.3500061, 479.8447876
9: -214.3119812, 211.4121857, -215.3806763, 212.4622803, -426.7742310, 426.7927856

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1739812, upper bound: 385.1730075
time: 11.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1737486, upper bound: 385.1730049
time: 10.19 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -235.3302155, 186.3101654, -238.2789459, 188.7321472, -424.0622559, 424.5890198
1: -197.5411072, 165.4381561, -199.9472961, 167.4609222, -365.0020142, 365.3854370
2: -259.2804565, 167.9253082, -262.6145935, 169.8886719, -429.1691284, 430.5398254
3: -276.1312561, 145.3548889, -279.6047974, 147.0715332, -423.2027893, 424.9596863
4: -252.4357147, 193.2728882, -255.5891724, 195.5671844, -448.0028687, 448.8620300
5: -225.9752960, 175.5002747, -228.8283081, 177.5161438, -403.4914551, 404.3285828
6: -216.1306763, 208.4402008, -218.9113312, 211.0843201, -427.2149353, 427.3515320
7: -236.2733154, 197.8198242, -239.1663361, 200.2471771, -436.5204773, 436.9861450
8: -284.4381714, 193.9585876, -288.0024414, 196.2816620, -480.7198486, 481.9610291
9: -214.3119812, 211.4121857, -216.9166260, 213.9933167, -428.3052368, 428.3287964

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1739812, upper bound: 385.1730075
time: 9.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1737486, upper bound: 385.1730049
time: 9.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.52 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 1, lower bound: -385.1736184, upper bound: 385.1724318
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 1, lower bound: -385.1734711, upper bound: 385.1724873
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 1, lower bound: -385.1736184, upper bound: 385.1724318
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 1, lower bound: -385.1734711, upper bound: 385.1724873
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 1, lower bound: -385.1739812, upper bound: 385.1730075
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 1, lower bound: -385.1737486, upper bound: 385.1730049
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 1, lower bound: -385.1739812, upper bound: 385.1730075
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 1, lower bound: -385.1737486, upper bound: 385.1730049
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=387.1527099609375
rel_dist={1: [-385.1964844738563, 385.19648447385634]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1878300, upper bound: 385.1865194
time: 10.23 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849453, upper bound: 385.1849453
time: 8.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.67
Output dim: 1, lower bound: -385.1878300, upper bound: 385.1865194
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.67
Output dim: 1, lower bound: -385.1849453, upper bound: 385.1849453

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

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849453, upper bound: 385.1849453
time: 8.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1849453, upper bound: 385.1849453
time: 8.01 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -244.8681488, 193.9198761, -246.3611603, 195.0366516, -439.9047852, 440.2810364
1: -205.4636383, 172.0779266, -206.7829285, 173.1754303, -378.6390686, 378.8608398
2: -269.8109131, 174.6556702, -271.4250488, 175.8490448, -445.6599731, 446.0806885
3: -287.2557373, 151.1343994, -289.0644836, 152.1156158, -439.3713379, 440.1988831
4: -262.6797791, 200.9863586, -264.3057556, 202.3233643, -465.0031128, 465.2921143
5: -235.1186066, 182.4603882, -236.5834198, 183.7382050, -418.8567505, 419.0437927
6: -224.9225464, 216.8661041, -226.2856140, 218.1936035, -443.1161499, 443.1516418
7: -245.7872772, 205.7355347, -247.3180695, 207.0492859, -452.8365479, 453.0535889
8: -295.9555969, 201.7678680, -297.7632751, 203.0897675, -499.0453491, 499.5311279
9: -222.8706055, 219.9002075, -224.2971802, 221.3053894, -444.1759949, 444.1973572

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1809295, upper bound: 385.1806446
time: 8.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1813382, upper bound: 385.1813382
time: 10.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.14 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.14
Output dim: 1, lower bound: -385.1849453, upper bound: 385.1849453
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.14
Output dim: 1, lower bound: -385.1849453, upper bound: 385.1849453
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.14
Output dim: 1, lower bound: -385.1809295, upper bound: 385.1806446
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.14
Output dim: 1, lower bound: -385.1813382, upper bound: 385.1813382

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -243.5457764, 192.8121338, -436.3579102, 436.3579102
1: -204.4127197, 171.1914368, -204.4127197, 171.1914368, -375.6041565, 375.6041565
2: -268.3244019, 173.8122711, -268.3244019, 173.8122711, -442.1365967, 442.1365967
3: -285.7283020, 150.3652191, -285.7283020, 150.3652191, -436.0935059, 436.0935059
4: -261.2940063, 200.0122070, -261.2940063, 200.0122070, -461.3062134, 461.3062134
5: -233.8536224, 181.6165619, -233.8536224, 181.6165619, -415.4701843, 415.4701843
6: -223.7100830, 215.6901703, -223.7100830, 215.6901703, -439.4002075, 439.4002075
7: -244.5041962, 204.6670990, -244.5041962, 204.6670990, -449.1712646, 449.1712646
8: -294.3796692, 200.7644348, -294.3796692, 200.7644348, -495.1441040, 495.1441040
9: -221.7438049, 218.7707672, -221.7438049, 218.7707672, -440.5145874, 440.5145874

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1836166, upper bound: 385.1825930
time: 10.45 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1837717, upper bound: 385.1827260
time: 10.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -243.5457764, 192.8121338, -244.8681488, 193.9198761, -437.4656372, 437.6802979
1: -204.4127197, 171.1914368, -205.4636383, 172.0779266, -376.4906311, 376.6550598
2: -268.3244019, 173.8122711, -269.8109131, 174.6556702, -442.9800110, 443.6231689
3: -285.7283020, 150.3652191, -287.2557373, 151.1343994, -436.8626709, 437.6209717
4: -261.2940063, 200.0122070, -262.6797791, 200.9863586, -462.2803650, 462.6919861
5: -233.8536224, 181.6165619, -235.1186066, 182.4603882, -416.3140259, 416.7350769
6: -223.7100830, 215.6901703, -224.9225464, 216.8661041, -440.5761108, 440.6127014
7: -244.5041962, 204.6670990, -245.7872772, 205.7355347, -450.2397156, 450.4543457
8: -294.3796692, 200.7644348, -295.9555969, 201.7678680, -496.1475220, 496.7200012
9: -221.7438049, 218.7707672, -222.8706055, 219.9002075, -441.6440125, 441.6413574

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1836166, upper bound: 385.1825930
time: 9.46 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1837717, upper bound: 385.1827260
time: 11.22 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -243.1446075, 192.5635223, -236.5627136, 187.3223267, -430.4669189, 429.1262207
1: -204.0231628, 170.8710785, -198.5966644, 166.3143768, -370.3375244, 369.4677429
2: -267.9220276, 173.4210510, -260.6834412, 168.8347168, -436.7566833, 434.1044922
3: -285.2453308, 150.0782776, -277.6318359, 146.1102600, -431.3555603, 427.7101135
4: -260.8312378, 199.5727539, -253.7964172, 194.2861633, -455.1174011, 453.3691711
5: -233.4734802, 181.1779633, -227.2288971, 176.4532013, -409.9266357, 408.4068604
6: -223.3467560, 215.3488159, -217.3229218, 209.5671387, -432.9138489, 432.6716919
7: -244.0595703, 204.3003387, -237.4990997, 198.8900604, -442.9496460, 441.7994385
8: -293.8780823, 200.3467407, -285.9476624, 195.0133362, -488.8913879, 486.2944031
9: -221.3131256, 218.3568268, -215.4440308, 212.5290222, -433.8421631, 433.8008423

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1805623, upper bound: 385.1805623
time: 8.76 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1805623, upper bound: 385.1806446
time: 8.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -241.8932495, 191.5739594, -240.5326843, 190.4582062, -432.3514099, 432.1066284
1: -202.9802246, 169.9932098, -201.9095306, 169.0930786, -372.0732117, 371.9027405
2: -266.5496521, 172.5279236, -265.0738220, 171.6305695, -438.1801758, 437.6017151
3: -283.7850952, 149.3127289, -282.3113708, 148.5177002, -432.3027954, 431.6240845
4: -259.4815369, 198.5462799, -258.0402222, 197.5327454, -457.0142822, 456.5864868
5: -232.2743378, 180.2456665, -231.0310211, 179.3579407, -411.6322632, 411.2766724
6: -222.2018890, 214.2491913, -220.9893341, 213.0879364, -435.2897644, 435.2385254
7: -242.8054352, 203.2551880, -241.4564972, 202.1940613, -444.9994507, 444.7116699
8: -292.3745422, 199.3216858, -290.7435608, 198.2489014, -490.6234436, 490.0652466
9: -220.1765137, 217.2354431, -219.0315094, 216.0835724, -436.2600708, 436.2669678

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1733633, upper bound: 385.1748140
time: 9.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1808999, upper bound: 385.1808999
time: 10.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.76 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 1, lower bound: -385.1836166, upper bound: 385.1825930
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 1, lower bound: -385.1837717, upper bound: 385.1827260
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 1, lower bound: -385.1836166, upper bound: 385.1825930
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 1, lower bound: -385.1837717, upper bound: 385.1827260
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 1, lower bound: -385.1805623, upper bound: 385.1805623
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 1, lower bound: -385.1805623, upper bound: 385.1806446
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 20.76
Output dim: 1, lower bound: -385.1733633, upper bound: 385.1748140
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 20.76
Output dim: 1, lower bound: -385.1808999, upper bound: 385.1808999

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -241.8562469, 191.4823761, -425.4071350, 427.0946655
1: -196.3794403, 164.4563293, -203.0020294, 170.0089569, -366.3883057, 367.4583435
2: -257.7795410, 166.9246521, -266.4728699, 172.6027069, -430.3822632, 433.3974609
3: -274.5066223, 144.4709167, -283.7580872, 149.3304749, -423.8370972, 428.2290039
4: -250.9762573, 192.1235657, -259.4823303, 198.6271057, -449.6033325, 451.6058960
5: -224.6685638, 174.4661713, -232.2410278, 180.3609314, -405.0294495, 406.7071533
6: -214.9111786, 207.2229767, -222.1649933, 214.2034454, -429.1146240, 429.3879700
7: -234.8669281, 196.6575928, -242.8120880, 203.2608490, -438.1277771, 439.4696350
8: -282.7800598, 192.8340759, -292.3435364, 199.3718414, -482.1518555, 485.1776123
9: -213.0522919, 210.1563110, -220.2179260, 217.2581024, -430.3103943, 430.3742065

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 92

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1899780, upper bound: 385.1899780
time: 10.10 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1899780, upper bound: 385.1900037
time: 8.48 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -237.5785980, 188.1260834, -240.4509583, 190.3723602, -427.9509583, 428.5770264
1: -199.4214020, 167.0109253, -201.8283539, 169.0223389, -368.4436646, 368.8392334
2: -261.8223267, 169.4930878, -264.9316101, 171.5980530, -433.4203186, 434.4246826
3: -278.8170166, 146.6810913, -282.1187134, 148.4694977, -427.2864990, 428.7997742
4: -254.8795471, 195.1088715, -257.9674377, 197.4738922, -452.3533630, 453.0762939
5: -228.1727142, 177.1310883, -230.8959351, 179.3105621, -407.4832764, 408.0270386
6: -218.2875519, 210.4626465, -220.8804321, 212.9672852, -431.2548218, 431.3430786
7: -238.4990692, 199.6945343, -241.3995819, 202.0855408, -440.5845947, 441.0941162
8: -287.1939697, 195.8099365, -290.6537781, 198.2213135, -485.4152527, 486.4637146
9: -216.3509674, 213.4246216, -218.9403229, 215.9988861, -432.3498230, 432.3648987

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1875887, upper bound: 385.1871431
time: 9.17 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1888077, upper bound: 385.1888077
time: 7.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -243.1446075, 192.5635223, -426.4882812, 428.3830261
1: -196.3794403, 164.4563293, -204.0231628, 170.8710785, -367.2504578, 368.4794922
2: -257.7795410, 166.9246521, -267.9220276, 173.4210510, -431.2005920, 434.8466492
3: -274.5066223, 144.4709167, -285.2453308, 150.0782776, -424.5848999, 429.7162170
4: -250.9762573, 192.1235657, -260.8312378, 199.5727539, -450.5490112, 452.9548035
5: -224.6685638, 174.4661713, -233.4734802, 181.1779633, -405.8465271, 407.9396057
6: -214.9111786, 207.2229767, -223.3467560, 215.3488159, -430.2599182, 430.5697327
7: -234.8669281, 196.6575928, -244.0595703, 204.3003387, -439.1672668, 440.7171021
8: -282.7800598, 192.8340759, -293.8780823, 200.3467407, -483.1268005, 486.7121582
9: -213.0522919, 210.1563110, -221.3131256, 218.3568268, -431.4091187, 431.4694214

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 131

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1833915, upper bound: 385.1820652
time: 10.56 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1833915, upper bound: 385.1825930
time: 9.98 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -237.5785980, 188.1260834, -241.8932495, 191.5739594, -429.1525269, 430.0193481
1: -199.4214020, 167.0109253, -202.9802246, 169.9932098, -369.4145813, 369.9911499
2: -261.8223267, 169.4930878, -266.5496521, 172.5279236, -434.3501892, 436.0426941
3: -278.8170166, 146.6810913, -283.7850952, 149.3127289, -428.1297302, 430.4661865
4: -254.8795471, 195.1088715, -259.4815369, 198.5462799, -453.4257507, 454.5903931
5: -228.1727142, 177.1310883, -232.2743378, 180.2456665, -408.4183655, 409.4054260
6: -218.2875519, 210.4626465, -222.2018890, 214.2491913, -432.5367432, 432.6645508
7: -238.4990692, 199.6945343, -242.8054352, 203.2551880, -441.7542725, 442.4999390
8: -287.1939697, 195.8099365, -292.3745422, 199.3216858, -486.5156555, 488.1844788
9: -216.3509674, 213.4246216, -220.1765137, 217.2354431, -433.5863953, 433.6011047

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1766534, upper bound: 385.1744377
time: 10.42 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1830139, upper bound: 385.1822654
time: 11.59 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -236.5627136, 187.3223267, -422.3729248, 422.7526855
1: -197.2579956, 165.2008362, -198.5966644, 166.3143768, -363.5723877, 363.7974854
2: -259.0479431, 167.6212616, -260.6834412, 168.8347168, -427.8825989, 428.3046265
3: -275.7994385, 145.1157227, -277.6318359, 146.1102600, -421.9096680, 422.7475586
4: -252.1463470, 192.9314423, -253.7964172, 194.2861633, -446.4324951, 446.7278442
5: -225.7425232, 175.1535187, -227.2288971, 176.4532013, -402.1956482, 402.3824158
6: -215.9430695, 208.2210846, -217.3229218, 209.5671387, -425.5101624, 425.5440063
7: -235.9456177, 197.5567017, -237.4990997, 198.8900604, -434.8356934, 435.0557861
8: -284.1164551, 193.6714935, -285.9476624, 195.0133362, -479.1297302, 479.6191406
9: -213.9953918, 211.1070557, -215.4440308, 212.5290222, -426.5243530, 426.5510864

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1805654, upper bound: 385.1805632
time: 11.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1805654, upper bound: 385.1805632
time: 9.30 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -236.5627136, 187.3223267, -426.6112671, 426.0977173
1: -200.7943878, 168.1681061, -198.5966644, 166.3143768, -367.1087646, 366.7647705
2: -263.7323914, 170.6133423, -260.6834412, 168.8347168, -432.5670166, 431.2967529
3: -280.7873535, 147.6848602, -277.6318359, 146.1102600, -426.8975830, 425.3167114
4: -256.6813049, 196.3964844, -253.7964172, 194.2861633, -450.9674377, 450.1929016
5: -229.8008270, 178.2671509, -227.2288971, 176.4532013, -406.2540283, 405.4960327
6: -219.8521881, 211.9787750, -217.3229218, 209.5671387, -429.4192810, 429.3016968
7: -240.1766357, 201.0883026, -237.4990997, 198.8900604, -439.0667114, 438.5874023
8: -289.2331848, 197.1266174, -285.9476624, 195.0133362, -484.2464905, 483.0742798
9: -217.8293915, 214.8981171, -215.4440308, 212.5290222, -430.3583984, 430.3421631

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 103

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1805654, upper bound: 385.1806446
time: 8.42 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1805654, upper bound: 385.1806446
time: 10.69 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -238.2761688, 188.6972961, -240.1015320, 190.1154480, -428.3916016, 428.7988281
1: -199.9474182, 167.4607544, -201.5481415, 168.7914581, -368.7388611, 369.0088806
2: -262.5466919, 169.9321747, -264.5969238, 171.3211823, -433.8678589, 434.5291138
3: -279.5519409, 147.1165466, -281.8068542, 148.2559814, -427.8079224, 428.9234009
4: -255.5701141, 195.5766449, -257.5740356, 197.1789398, -452.7490234, 453.1506653
5: -228.7912140, 177.5557098, -230.6157684, 179.0373840, -407.8285522, 408.1714783
6: -218.8318634, 211.0455933, -220.5875854, 212.7061462, -431.5379333, 431.6331787
7: -239.1875305, 200.2427521, -241.0252991, 201.8350525, -441.0225830, 441.2680359
8: -287.9680481, 196.2958069, -290.2186584, 197.8883820, -485.8564148, 486.5144653
9: -216.9081268, 213.9960175, -218.6421509, 215.6974182, -432.6055298, 432.6381531

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1808999, upper bound: 385.1808999
time: 9.40 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1808999, upper bound: 385.1808999
time: 10.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1899780, upper bound: 385.1899780
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1899780, upper bound: 385.1900037
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1875887, upper bound: 385.1871431
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1888077, upper bound: 385.1888077
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1833915, upper bound: 385.1820652
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1833915, upper bound: 385.1825930
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1766534, upper bound: 385.1744377
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1830139, upper bound: 385.1822654
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1805654, upper bound: 385.1805632
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1805654, upper bound: 385.1805632
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1805654, upper bound: 385.1806446
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1805654, upper bound: 385.1806446
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1808999, upper bound: 385.1808999
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -385.1808999, upper bound: 385.1808999

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -233.9247894, 185.2384186, -419.1631775, 419.1631775
1: -196.3794403, 164.4563293, -196.3794403, 164.4563293, -360.8357544, 360.8357544
2: -257.7795410, 166.9246521, -257.7795410, 166.9246521, -424.7041626, 424.7041626
3: -274.5066223, 144.4709167, -274.5066223, 144.4709167, -418.9775085, 418.9775085
4: -250.9762573, 192.1235657, -250.9762573, 192.1235657, -443.0997925, 443.0997925
5: -224.6685638, 174.4661713, -224.6685638, 174.4661713, -399.1347046, 399.1347046
6: -214.9111786, 207.2229767, -214.9111786, 207.2229767, -422.1341248, 422.1341248
7: -234.8669281, 196.6575928, -234.8669281, 196.6575928, -431.5245056, 431.5245056
8: -282.7800598, 192.8340759, -282.7800598, 192.8340759, -475.6141357, 475.6141357
9: -213.0522919, 210.1563110, -213.0522919, 210.1563110, -423.2086182, 423.2086182

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1876285, upper bound: 385.1878420
time: 10.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1888960, upper bound: 385.1888198
time: 8.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -237.5785980, 188.1260834, -422.0508118, 422.8170166
1: -196.3794403, 164.4563293, -199.4214020, 167.0109253, -363.3903198, 363.8777466
2: -257.7795410, 166.9246521, -261.8223267, 169.4930878, -427.2726135, 428.7469177
3: -274.5066223, 144.4709167, -278.8170166, 146.6810913, -421.1877136, 423.2879333
4: -250.9762573, 192.1235657, -254.8795471, 195.1088715, -446.0851135, 447.0030518
5: -224.6685638, 174.4661713, -228.1727142, 177.1310883, -401.7996521, 402.6388550
6: -214.9111786, 207.2229767, -218.2875519, 210.4626465, -425.3738098, 425.5105286
7: -234.8669281, 196.6575928, -238.4990692, 199.6945343, -434.5614624, 435.1566467
8: -282.7800598, 192.8340759, -287.1939697, 195.8099365, -478.5899963, 480.0280457
9: -213.0522919, 210.1563110, -216.3509674, 213.4246216, -426.4768982, 426.5072632

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1876285, upper bound: 385.1878420
time: 8.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1888960, upper bound: 385.1888247
time: 7.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -235.3010101, 186.3142548, -233.4652100, 184.8034363, -420.1044006, 419.7793884
1: -197.5102234, 165.4161682, -195.9745941, 164.1397858, -361.6499634, 361.3907471
2: -259.3016357, 167.8585205, -257.2029419, 166.5842133, -425.8857727, 425.0614624
3: -276.1455994, 145.2936859, -273.9354248, 144.2233276, -420.3689270, 419.2291260
4: -252.4179230, 193.2378387, -250.4080505, 191.7421570, -444.1600342, 443.6458435
5: -225.9799347, 175.4369812, -224.1658020, 174.1134338, -400.0933533, 399.6027832
6: -216.1686707, 208.4425507, -214.3680725, 206.7732086, -422.9418945, 422.8106079
7: -236.2190247, 197.7937317, -234.4132843, 196.2665710, -432.4855957, 432.2069702
8: -284.4167786, 193.9037170, -282.1362000, 192.3556366, -476.7723999, 476.0399170
9: -214.2881622, 211.3800354, -212.6262817, 209.7309418, -424.0190430, 424.0063171

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 131

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1793270, upper bound: 385.1786698
time: 9.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1791958, upper bound: 385.1788299
time: 9.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -237.1316833, 187.7707825, -236.5918884, 187.3038940, -424.4355469, 424.3626099
1: -199.0467529, 166.6981812, -198.5941162, 166.3224335, -365.3692017, 365.2922974
2: -261.3278809, 169.1723328, -260.6630554, 168.8275146, -430.1553650, 429.8353271
3: -278.2939758, 146.4096527, -277.6027527, 146.1277924, -424.4217529, 424.0123901
4: -254.3960876, 194.7421112, -253.7921906, 194.3069763, -448.7030640, 448.5342712
5: -227.7423859, 176.7986908, -227.1803284, 176.4401703, -404.1825256, 403.9790039
6: -217.8711243, 210.0668793, -217.2840576, 209.5499573, -427.4210815, 427.3509216
7: -238.0519257, 199.3224182, -237.5386200, 198.8720551, -436.9239807, 436.8610229
8: -286.6498108, 195.4362183, -285.9559326, 194.9955597, -481.6453857, 481.3921509
9: -215.9471893, 213.0241852, -215.4548340, 212.5420074, -428.4891357, 428.4790039

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 92

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1807580, upper bound: 385.1804965
time: 9.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1806477, upper bound: 385.1806477
time: 9.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -235.0505981, 186.1899567, -420.1147156, 420.2890015
1: -196.3794403, 164.4563293, -197.2579956, 165.2008362, -361.5802307, 361.7143250
2: -257.7795410, 166.9246521, -259.0479431, 167.6212616, -425.4007263, 425.9725647
3: -274.5066223, 144.4709167, -275.7994385, 145.1157227, -419.6223450, 420.2703247
4: -250.9762573, 192.1235657, -252.1463470, 192.9314423, -443.9077148, 444.2698975
5: -224.6685638, 174.4661713, -225.7425232, 175.1535187, -399.8220825, 400.2086182
6: -214.9111786, 207.2229767, -215.9430695, 208.2210846, -423.1322327, 423.1660461
7: -234.8669281, 196.6575928, -235.9456177, 197.5567017, -432.4236450, 432.6031799
8: -282.7800598, 192.8340759, -284.1164551, 193.6714935, -476.4515381, 476.9505310
9: -213.0522919, 210.1563110, -213.9953918, 211.1070557, -424.1593628, 424.1516724

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1797227, upper bound: 385.1788833
time: 11.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1826684, upper bound: 385.1815290
time: 10.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -233.9247894, 185.2384186, -239.2889404, 189.5350037, -423.4597473, 424.5273438
1: -196.3794403, 164.4563293, -200.7943878, 168.1681061, -364.5474548, 365.2507324
2: -257.7795410, 166.9246521, -263.7323914, 170.6133423, -428.3928223, 430.6569824
3: -274.5066223, 144.4709167, -280.7873535, 147.6848602, -422.1914673, 425.2582397
4: -250.9762573, 192.1235657, -256.6813049, 196.3964844, -447.3727417, 448.8048096
5: -224.6685638, 174.4661713, -229.8008270, 178.2671509, -402.9356995, 404.2669983
6: -214.9111786, 207.2229767, -219.8521881, 211.9787750, -426.8898926, 427.0751648
7: -234.8669281, 196.6575928, -240.1766357, 201.0883026, -435.9552307, 436.8342285
8: -282.7800598, 192.8340759, -289.2331848, 197.1266174, -479.9066467, 482.0672607
9: -213.0522919, 210.1563110, -217.8293915, 214.8981171, -427.9504089, 427.9857178

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1797227, upper bound: 385.1790574
time: 12.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1826684, upper bound: 385.1820097
time: 10.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -237.1316833, 187.7707825, -238.2761688, 188.6972961, -425.8289795, 426.0469360
1: -199.0467529, 166.6981812, -199.9474182, 167.4607544, -366.5075073, 366.6455994
2: -261.3278809, 169.1723328, -262.5466919, 169.9321747, -431.2600708, 431.7189636
3: -278.2939758, 146.4096527, -279.5519409, 147.1165466, -425.4105225, 425.9615784
4: -254.3960876, 194.7421112, -255.5701141, 195.5766449, -449.9727173, 450.3121643
5: -227.7423859, 176.7986908, -228.7912140, 177.5557098, -405.2980957, 405.5899048
6: -217.8711243, 210.0668793, -218.8318634, 211.0455933, -428.9167175, 428.8986511
7: -238.0519257, 199.3224182, -239.1875305, 200.2427521, -438.2946472, 438.5099487
8: -286.6498108, 195.4362183, -287.9680481, 196.2958069, -482.9456177, 483.4042664
9: -215.9471893, 213.0241852, -216.9081268, 213.9960175, -429.9432068, 429.9323120

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 92

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1685328, upper bound: 385.1734121
time: 11.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1742832, upper bound: 385.1734030
time: 11.31 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -233.9247894, 185.2384186, -420.2890015, 420.1147156
1: -197.2579956, 165.2008362, -196.3794403, 164.4563293, -361.7143250, 361.5802307
2: -259.0479431, 167.6212616, -257.7795410, 166.9246521, -425.9725647, 425.4007263
3: -275.7994385, 145.1157227, -274.5066223, 144.4709167, -420.2703247, 419.6223450
4: -252.1463470, 192.9314423, -250.9762573, 192.1235657, -444.2698975, 443.9077148
5: -225.7425232, 175.1535187, -224.6685638, 174.4661713, -400.2086182, 399.8220825
6: -215.9430695, 208.2210846, -214.9111786, 207.2229767, -423.1660461, 423.1322327
7: -235.9456177, 197.5567017, -234.8669281, 196.6575928, -432.6031799, 432.4236450
8: -284.1164551, 193.6714935, -282.7800598, 192.8340759, -476.9505310, 476.4515381
9: -213.9953918, 211.1070557, -213.0522919, 210.1563110, -424.1516724, 424.1593628

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1731909, upper bound: 385.1746356
time: 9.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1800686, upper bound: 385.1800686
time: 9.92 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -235.0505981, 186.1899567, -235.0505981, 186.1899567, -421.2405396, 421.2405396
1: -197.2579956, 165.2008362, -197.2579956, 165.2008362, -362.4588318, 362.4588318
2: -259.0479431, 167.6212616, -259.0479431, 167.6212616, -426.6691284, 426.6691284
3: -275.7994385, 145.1157227, -275.7994385, 145.1157227, -420.9151306, 420.9151306
4: -252.1463470, 192.9314423, -252.1463470, 192.9314423, -445.0777893, 445.0777893
5: -225.7425232, 175.1535187, -225.7425232, 175.1535187, -400.8959961, 400.8959961
6: -215.9430695, 208.2210846, -215.9430695, 208.2210846, -424.1641541, 424.1641541
7: -235.9456177, 197.5567017, -235.9456177, 197.5567017, -433.5023193, 433.5023193
8: -284.1164551, 193.6714935, -284.1164551, 193.6714935, -477.7879639, 477.7879639
9: -213.9953918, 211.1070557, -213.9953918, 211.1070557, -425.1024475, 425.1024475

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1731910, upper bound: 385.1746356
time: 9.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1800686, upper bound: 385.1800686
time: 10.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -233.9247894, 185.2384186, -424.5273438, 423.4597473
1: -200.7943878, 168.1681061, -196.3794403, 164.4563293, -365.2507324, 364.5474548
2: -263.7323914, 170.6133423, -257.7795410, 166.9246521, -430.6569824, 428.3928223
3: -280.7873535, 147.6848602, -274.5066223, 144.4709167, -425.2582397, 422.1914673
4: -256.6813049, 196.3964844, -250.9762573, 192.1235657, -448.8048096, 447.3727417
5: -229.8008270, 178.2671509, -224.6685638, 174.4661713, -404.2669983, 402.9356995
6: -219.8521881, 211.9787750, -214.9111786, 207.2229767, -427.0751648, 426.8898926
7: -240.1766357, 201.0883026, -234.8669281, 196.6575928, -436.8342285, 435.9552307
8: -289.2331848, 197.1266174, -282.7800598, 192.8340759, -482.0672607, 479.9066467
9: -217.8293915, 214.8981171, -213.0522919, 210.1563110, -427.9857178, 427.9504089

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 64

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1730975, upper bound: 385.1744414
time: 12.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1804740, upper bound: 385.1801337
time: 8.23 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -239.2889404, 189.5350037, -235.0505981, 186.1899567, -425.4788818, 424.5856018
1: -200.7943878, 168.1681061, -197.2579956, 165.2008362, -365.9952393, 365.4260864
2: -263.7323914, 170.6133423, -259.0479431, 167.6212616, -431.3535461, 429.6612244
3: -280.7873535, 147.6848602, -275.7994385, 145.1157227, -425.9030762, 423.4842834
4: -256.6813049, 196.3964844, -252.1463470, 192.9314423, -449.6127319, 448.5428467
5: -229.8008270, 178.2671509, -225.7425232, 175.1535187, -404.9543457, 404.0096130
6: -219.8521881, 211.9787750, -215.9430695, 208.2210846, -428.0732727, 427.9218445
7: -240.1766357, 201.0883026, -235.9456177, 197.5567017, -437.7333374, 437.0339355
8: -289.2331848, 197.1266174, -284.1164551, 193.6714935, -482.9046631, 481.2430115
9: -217.8293915, 214.8981171, -213.9953918, 211.1070557, -428.9364624, 428.8934937

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 64

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 250

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1730975, upper bound: 385.1744414
time: 11.06 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1804740, upper bound: 385.1801337
time: 10.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -238.2761688, 188.6972961, -237.1316833, 187.7707825, -426.0469360, 425.8289795
1: -199.9474182, 167.4607544, -199.0467529, 166.6981812, -366.6455994, 366.5075073
2: -262.5466919, 169.9321747, -261.3278809, 169.1723328, -431.7189636, 431.2600708
3: -279.5519409, 147.1165466, -278.2939758, 146.4096527, -425.9615784, 425.4105225
4: -255.5701141, 195.5766449, -254.3960876, 194.7421112, -450.3121338, 449.9727173
5: -228.7912140, 177.5557098, -227.7423859, 176.7986908, -405.5899048, 405.2980957
6: -218.8318634, 211.0455933, -217.8711243, 210.0668793, -428.8986511, 428.9167175
7: -239.1875305, 200.2427521, -238.0519257, 199.3224182, -438.5099487, 438.2946472
8: -287.9680481, 196.2958069, -286.6498108, 195.4362183, -483.4042664, 482.9456177
9: -216.9081268, 213.9960175, -215.9471893, 213.0241852, -429.9323120, 429.9432068

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1804740
time: 9.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1808765
time: 9.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -238.2761688, 188.6972961, -238.8685455, 189.2007751, -427.4769287, 427.5658569
1: -199.9474182, 167.4607544, -200.4418030, 167.8737335, -367.8211365, 367.9025574
2: -262.5466919, 169.9321747, -263.2670593, 170.3117371, -432.8583984, 433.1992188
3: -279.5519409, 147.1165466, -280.2950745, 147.4295502, -426.9815063, 427.4116211
4: -255.5701141, 195.5766449, -256.2267151, 196.0513153, -451.6213684, 451.8033447
5: -228.7912140, 177.5557098, -229.3960419, 177.9545288, -406.7457275, 406.9517517
6: -218.8318634, 211.0455933, -219.4605560, 211.6064606, -430.4382629, 430.5061340
7: -239.1875305, 200.2427521, -239.7561493, 200.7381897, -439.9257202, 439.9989014
8: -287.9680481, 196.2958069, -288.7209473, 196.7748871, -484.7429199, 485.0167542
9: -216.9081268, 213.9960175, -217.4494476, 214.5214539, -431.4295654, 431.4454651

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 103
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 131

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1804740
time: 8.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1808765
time: 11.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.08 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1876285, upper bound: 385.1878420
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1888960, upper bound: 385.1888198
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1876285, upper bound: 385.1878420
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1888960, upper bound: 385.1888247
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1793270, upper bound: 385.1786698
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1791958, upper bound: 385.1788299
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1807580, upper bound: 385.1804965
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1806477, upper bound: 385.1806477
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1797227, upper bound: 385.1788833
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1826684, upper bound: 385.1815290
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1797227, upper bound: 385.1790574
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1826684, upper bound: 385.1820097
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1685328, upper bound: 385.1734121
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1742832, upper bound: 385.1734030
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1731909, upper bound: 385.1746356
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1800686, upper bound: 385.1800686
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1731910, upper bound: 385.1746356
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1800686, upper bound: 385.1800686
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1730975, upper bound: 385.1744414
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1804740, upper bound: 385.1801337
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1730975, upper bound: 385.1744414
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1804740, upper bound: 385.1801337
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1804740
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1808765
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1804740
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.08
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1808765

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -227.0373077, 179.7470093, -231.6963654, 183.4659729, -410.5032959, 411.4433594
1: -190.6080780, 159.6412811, -194.5100861, 162.8960724, -353.5041504, 354.1513672
2: -250.1600800, 161.9821472, -255.3139038, 165.3259277, -415.4860229, 417.2960510
3: -266.4379578, 140.2819366, -271.8931580, 143.1134491, -409.5513916, 412.1751099
4: -243.5213165, 186.4741974, -248.5677185, 190.2939453, -433.8152466, 435.0419006
5: -218.0314331, 169.3416138, -222.5225372, 172.8090668, -390.8404846, 391.8641052
6: -208.4920044, 201.1164093, -212.8378906, 205.2467041, -413.7387085, 413.9542847
7: -227.9802704, 190.9191284, -232.6369934, 194.7982635, -422.7785339, 423.5560913
8: -274.3823242, 187.0534973, -280.0642700, 190.9691925, -465.3515015, 467.1177673
9: -206.8244019, 203.9786835, -211.0342407, 208.1570740, -414.9813843, 415.0129395

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 103
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 193

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -385.1798572, upper bound: 385.1801739
time: 10.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -385.1802412, upper bound: 385.1804843
time: 10.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 21.92 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 21.92
Output dim: 1, lower bound: -385.1798572, upper bound: 385.1801739
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.92
Output dim: 1, lower bound: -385.1802412, upper bound: 385.1804843
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1888960, upper bound: 385.1888198
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1876285, upper bound: 385.1878420
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1888960, upper bound: 385.1888247
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1807580, upper bound: 385.1804965
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1806477, upper bound: 385.1806477
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1826684, upper bound: 385.1815290
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1826684, upper bound: 385.1820097
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1804740, upper bound: 385.1801337
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1804740, upper bound: 385.1801337
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1804740
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1808765
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1804740
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 1, lower bound: -385.1800666, upper bound: 385.1808765
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=387.1527099609375
rel_dist={1: [-385.1964942223473, 385.1964942223473]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1566.93 seconds
