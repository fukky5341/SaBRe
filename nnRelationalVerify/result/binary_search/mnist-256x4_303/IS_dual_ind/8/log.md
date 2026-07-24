## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 264.612307261
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-144.3596649, 115.0605316, -144.3596649, 115.0605316, -259.4201660, 259.4201660)
1: (-120.8221283, 102.0048447, -120.8221283, 102.0048447, -222.8269653, 222.8269653)
2: (-158.9320831, 103.2858582, -158.9320831, 103.2858582, -262.2179565, 262.2179565)
3: (-169.2086487, 90.2014694, -169.2086487, 90.2014694, -259.4100952, 259.4100952)
4: (-154.7666321, 118.7446289, -154.7666321, 118.7446289, -273.5112305, 273.5112305)
5: (-139.1623535, 108.0147781, -139.1623535, 108.0147781, -247.1771240, 247.1771240)
6: (-133.0277405, 128.5737915, -133.0277405, 128.5737915, -261.6014709, 261.6014709)
7: (-144.7083588, 121.6381607, -144.7083588, 121.6381607, -266.3464966, 266.3464966)
8: (-174.5120850, 119.3504868, -174.5120850, 119.3504868, -293.8625793, 293.8625793)
9: (-131.5167694, 129.9706268, -131.5167694, 129.9706268, -261.4873962, 261.4873962)

## BASE Result
execution time: IAR + LP analysis = 1.09 + 11.03 = 12.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -264.6227448, upper bound: 264.6227448


# Binary Search by BASE starts (time budget: 2687.88 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=266.34649658203125
rel_dist={7: [-264.62269597868163, 264.6226959786816]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=266.34649658203125
rel_dist={7: [-264.62266420921173, 264.6226641973501]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=266.34649658203125
rel_dist={7: [-264.6222950859965, 264.6222950868074]}

## Binary Search Result
Binary search time: 40.39 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2647.49 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6213344, upper bound: 264.6212907
time: 7.26 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212889, upper bound: 264.6212889
time: 7.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.19 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.19
Output dim: 7, lower bound: -264.6213344, upper bound: 264.6212907
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.19
Output dim: 7, lower bound: -264.6212889, upper bound: 264.6212889

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -137.8412933, 109.8958969, -144.0007935, 114.7776337, -252.6189117, 253.8966827
1: -115.3550110, 97.4181595, -120.5208511, 101.7524872, -217.1074982, 217.9389801
2: -151.7102203, 98.6460876, -158.5346680, 103.0288239, -254.7390442, 257.1807556
3: -161.5674744, 86.2000275, -168.7920685, 89.9800491, -251.5475006, 254.9920959
4: -147.7318878, 113.3872681, -154.3788910, 118.4491959, -266.1810913, 267.7661743
5: -132.8794250, 103.1506424, -138.8191986, 107.7467270, -240.6261444, 241.9698334
6: -127.0270538, 122.7826920, -132.6981506, 128.2549133, -255.2819672, 255.4808350
7: -138.1151123, 116.1337585, -144.3447113, 121.3333588, -259.4484863, 260.4784241
8: -166.6094513, 114.0110855, -174.0796204, 119.0535812, -285.6630249, 288.0906982
9: -125.5774002, 124.1473160, -131.1901550, 129.6473083, -255.2246704, 255.3374634

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212576
time: 8.67 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212576
time: 7.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -142.3042603, 113.4523087, -144.3450623, 115.0490952, -257.3533325, 257.7973328
1: -119.0911560, 100.5554810, -120.8098068, 101.9945450, -221.0856781, 221.3652954
2: -156.6529388, 101.8062897, -158.9158630, 103.2753372, -259.9281921, 260.7221375
3: -166.8479462, 88.9353714, -169.1918640, 90.1924438, -257.0404053, 258.1271973
4: -152.5530548, 117.0456924, -154.7508698, 118.7325439, -271.2855835, 271.7965698
5: -137.2094727, 106.4746170, -139.1484375, 108.0038300, -245.2133026, 245.6230469
6: -131.1432648, 126.7484055, -133.0143433, 128.5607910, -259.7040405, 259.7627258
7: -142.6210785, 119.8799820, -144.6934967, 121.6256409, -264.2467041, 264.5733948
8: -172.0416260, 117.6300964, -174.4945068, 119.3382721, -291.3798828, 292.1246033
9: -129.6566162, 128.1073914, -131.5035248, 129.9573517, -259.6139526, 259.6109009

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212889
time: 8.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212889
time: 8.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.65
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212576
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.65
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212576
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.65
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212889
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.65
Output dim: 7, lower bound: -264.6212575, upper bound: 264.6212889

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -137.8412933, 109.8958969, -137.8412933, 109.8958969, -247.7371826, 247.7371826
1: -115.3550110, 97.4181595, -115.3550110, 97.4181595, -212.7731476, 212.7731476
2: -151.7102203, 98.6460876, -151.7102203, 98.6460876, -250.3563080, 250.3563080
3: -161.5674744, 86.2000275, -161.5674744, 86.2000275, -247.7675018, 247.7675018
4: -147.7318878, 113.3872681, -147.7318878, 113.3872681, -261.1191406, 261.1191406
5: -132.8794250, 103.1506424, -132.8794250, 103.1506424, -236.0300598, 236.0300598
6: -127.0270538, 122.7826920, -127.0270538, 122.7826920, -249.8097534, 249.8097534
7: -138.1151123, 116.1337585, -138.1151123, 116.1337585, -254.2488708, 254.2488708
8: -166.6094513, 114.0110855, -166.6094513, 114.0110855, -280.6205444, 280.6205444
9: -125.5774002, 124.1473160, -125.5774002, 124.1473160, -249.7246857, 249.7246857

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6158601, upper bound: 264.6159605
time: 7.56 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6163035, upper bound: 264.6162954
time: 7.94 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -137.8412933, 109.8958969, -142.3042603, 113.4523087, -251.2935791, 252.2001495
1: -115.3550110, 97.4181595, -119.0911560, 100.5554810, -215.9104919, 216.5092926
2: -151.7102203, 98.6460876, -156.6529388, 101.8062897, -253.5165100, 255.2990265
3: -161.5674744, 86.2000275, -166.8479462, 88.9353714, -250.5028381, 253.0479736
4: -147.7318878, 113.3872681, -152.5530548, 117.0456924, -264.7775879, 265.9403076
5: -132.8794250, 103.1506424, -137.2094727, 106.4746170, -239.3540344, 240.3601074
6: -127.0270538, 122.7826920, -131.1432648, 126.7484055, -253.7754517, 253.9259644
7: -138.1151123, 116.1337585, -142.6210785, 119.8799820, -257.9950562, 258.7548218
8: -166.6094513, 114.0110855, -172.0416260, 117.6300964, -284.2395325, 286.0527039
9: -125.5774002, 124.1473160, -129.6566162, 128.1073914, -253.6847534, 253.8039093

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6158601, upper bound: 264.6159605
time: 10.00 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6163035, upper bound: 264.6162954
time: 7.63 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -142.3042603, 113.4523087, -137.8412933, 109.8958969, -252.2001495, 251.2935791
1: -119.0911560, 100.5554810, -115.3550110, 97.4181595, -216.5092926, 215.9104919
2: -156.6529388, 101.8062897, -151.7102203, 98.6460876, -255.2990265, 253.5165100
3: -166.8479462, 88.9353714, -161.5674744, 86.2000275, -253.0479736, 250.5028381
4: -152.5530548, 117.0456924, -147.7318878, 113.3872681, -265.9403076, 264.7775879
5: -137.2094727, 106.4746170, -132.8794250, 103.1506424, -240.3601074, 239.3540344
6: -131.1432648, 126.7484055, -127.0270538, 122.7826920, -253.9259644, 253.7754517
7: -142.6210785, 119.8799820, -138.1151123, 116.1337585, -258.7548218, 257.9950562
8: -172.0416260, 117.6300964, -166.6094513, 114.0110855, -286.0527039, 284.2395325
9: -129.6566162, 128.1073914, -125.5774002, 124.1473160, -253.8039093, 253.6847534

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157481, upper bound: 264.6158030
time: 7.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6162762, upper bound: 264.6163397
time: 7.93 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -142.3042603, 113.4523087, -142.3042603, 113.4523087, -255.7565308, 255.7565613
1: -119.0911560, 100.5554810, -119.0911560, 100.5554810, -219.6466370, 219.6466370
2: -156.6529388, 101.8062897, -156.6529388, 101.8062897, -258.4591370, 258.4591370
3: -166.8479462, 88.9353714, -166.8479462, 88.9353714, -255.7833099, 255.7833252
4: -152.5530548, 117.0456924, -152.5530548, 117.0456924, -269.5987549, 269.5987549
5: -137.2094727, 106.4746170, -137.2094727, 106.4746170, -243.6840820, 243.6840820
6: -131.1432648, 126.7484055, -131.1432648, 126.7484055, -257.8916626, 257.8916626
7: -142.6210785, 119.8799820, -142.6210785, 119.8799820, -262.5009766, 262.5009766
8: -172.0416260, 117.6300964, -172.0416260, 117.6300964, -289.6717224, 289.6717224
9: -129.6566162, 128.1073914, -129.6566162, 128.1073914, -257.7639771, 257.7639771

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157481, upper bound: 264.6158030
time: 7.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6162762, upper bound: 264.6163397
time: 7.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -264.6158601, upper bound: 264.6159605
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -264.6163035, upper bound: 264.6162954
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -264.6158601, upper bound: 264.6159605
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -264.6163035, upper bound: 264.6162954
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -264.6157481, upper bound: 264.6158030
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -264.6162762, upper bound: 264.6163397
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -264.6157481, upper bound: 264.6158030
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.72
Output dim: 7, lower bound: -264.6162762, upper bound: 264.6163397

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -128.7110443, 102.6759796, -137.8412933, 109.8958969, -238.6069336, 240.5172577
1: -107.6105347, 90.8847733, -115.3550110, 97.4181595, -205.0286713, 206.2397766
2: -141.6197968, 92.0970001, -151.7102203, 98.6460876, -240.2658844, 243.8072205
3: -150.7785187, 80.4096222, -161.5674744, 86.2000275, -236.9785461, 241.9770966
4: -137.9255219, 105.8342514, -147.7318878, 113.3872681, -251.3127899, 253.5661316
5: -124.1200485, 96.2677765, -132.8794250, 103.1506424, -227.2706909, 229.1472015
6: -118.6563950, 114.6190338, -127.0270538, 122.7826920, -241.4390869, 241.6460876
7: -128.9323425, 108.4165039, -138.1151123, 116.1337585, -245.0661011, 246.5316010
8: -155.6531372, 106.4151306, -166.6094513, 114.0110855, -269.6642151, 273.0245361
9: -117.1985931, 115.8351593, -125.5774002, 124.1473160, -241.3458862, 241.4125061

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157989, upper bound: 264.6157989
time: 7.72 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157989, upper bound: 264.6160327
time: 7.53 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -132.0236664, 105.2966156, -136.7031403, 108.9952011, -241.0188599, 241.9997253
1: -110.4031601, 93.2120895, -114.3916779, 96.6043777, -207.0075378, 207.6037598
2: -145.2716827, 94.4220505, -150.4527130, 97.8295364, -243.1011963, 244.8747559
3: -154.6574097, 82.4426422, -160.2217560, 85.4796066, -240.1370087, 242.6643829
4: -141.5063934, 108.5333481, -146.5117340, 112.4461899, -253.9525757, 255.0450592
5: -127.2928391, 98.7439041, -131.7856903, 102.2946320, -229.5874634, 230.5295715
6: -121.6913681, 117.5601883, -125.9833603, 121.7658463, -243.4572144, 243.5435486
7: -132.2469788, 111.2113724, -136.9710236, 115.1736298, -247.4206085, 248.1823883
8: -159.6582336, 109.1145325, -165.2444305, 113.0651855, -272.7234192, 274.3589478
9: -120.2265167, 118.7983475, -124.5348816, 123.1142654, -243.3407898, 243.3332214

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157027, upper bound: 264.6158082
time: 7.83 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6159515, upper bound: 264.6159515
time: 8.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -128.7110443, 102.6759796, -142.3042603, 113.4523087, -242.1633301, 244.9802246
1: -107.6105347, 90.8847733, -119.0911560, 100.5554810, -208.1660156, 209.9759216
2: -141.6197968, 92.0970001, -156.6529388, 101.8062897, -243.4260864, 248.7499390
3: -150.7785187, 80.4096222, -166.8479462, 88.9353714, -239.7138824, 247.2575684
4: -137.9255219, 105.8342514, -152.5530548, 117.0456924, -254.9711761, 258.3872986
5: -124.1200485, 96.2677765, -137.2094727, 106.4746170, -230.5946655, 233.4772491
6: -118.6563950, 114.6190338, -131.1432648, 126.7484055, -245.4048004, 245.7622986
7: -128.9323425, 108.4165039, -142.6210785, 119.8799820, -248.8123016, 251.0375671
8: -155.6531372, 106.4151306, -172.0416260, 117.6300964, -273.2832031, 278.4567566
9: -117.1985931, 115.8351593, -129.6566162, 128.1073914, -245.3059692, 245.4917297

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6155964, upper bound: 264.6156036
time: 7.55 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6155964, upper bound: 264.6159373
time: 7.36 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -132.0236664, 105.2966156, -141.1721344, 112.5560837, -244.5797424, 246.4687347
1: -110.4031601, 93.2120895, -118.1325912, 99.7458649, -210.1490021, 211.3446808
2: -145.2716827, 94.4220505, -155.4023285, 100.9940720, -246.2657318, 249.8243713
3: -154.6574097, 82.4426422, -165.5083313, 88.2183533, -242.8757629, 247.9509583
4: -141.5063934, 108.5333481, -151.3392029, 116.1095581, -257.6159363, 259.8725281
5: -127.2928391, 98.7439041, -136.1209717, 105.6231995, -232.9160004, 234.8648682
6: -121.6913681, 117.5601883, -130.1048126, 125.7368927, -247.4282227, 247.6650085
7: -132.2469788, 111.2113724, -141.4833221, 118.9249725, -251.1719513, 252.6946869
8: -159.6582336, 109.1145325, -170.6836395, 116.6894226, -276.3476562, 279.7981567
9: -120.2265167, 118.7983475, -128.6191559, 127.0797043, -247.3062134, 247.4175110

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6156015, upper bound: 264.6157071
time: 7.20 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6158498, upper bound: 264.6158783
time: 8.11 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -133.2117767, 106.2656326, -137.8412933, 109.8958969, -243.1076660, 244.1069336
1: -111.3794937, 94.0493698, -115.3550110, 97.4181595, -208.7976227, 209.4043427
2: -146.6026611, 95.2830658, -151.7102203, 98.6460876, -245.2487488, 246.9932709
3: -156.1095276, 83.1680069, -161.5674744, 86.2000275, -242.3095551, 244.7354736
4: -142.7886810, 109.5240860, -147.7318878, 113.3872681, -256.1759338, 257.2559814
5: -128.4900055, 99.6197433, -132.8794250, 103.1506424, -231.6406250, 232.4991760
6: -122.8076019, 118.6185913, -127.0270538, 122.7826920, -245.5902863, 245.6456299
7: -133.4747009, 112.1934509, -138.1151123, 116.1337585, -249.6084595, 250.3085480
8: -161.1326141, 110.0628586, -166.6094513, 114.0110855, -275.1437073, 276.6722717
9: -121.3137970, 119.8276825, -125.5774002, 124.1473160, -245.4610901, 245.4050446

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6156036, upper bound: 264.6155964
time: 8.62 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6156036, upper bound: 264.6158411
time: 7.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -137.3429260, 109.5298615, -136.7031403, 108.9952011, -246.3381195, 246.2329712
1: -114.8522186, 96.9537201, -114.3916779, 96.6043777, -211.4565887, 211.3453979
2: -151.1631165, 98.1891479, -150.4527130, 97.8295364, -248.9926453, 248.6418610
3: -160.9450836, 85.6996307, -160.2217560, 85.4796066, -246.4246826, 245.9213715
4: -147.2469025, 112.8937683, -146.5117340, 112.4461899, -259.6930542, 259.4054565
5: -132.4353638, 102.7049255, -131.7856903, 102.2946320, -234.7299957, 234.4906158
6: -126.5909271, 122.2810898, -125.9833603, 121.7658463, -248.3567810, 248.2644501
7: -137.6180267, 115.6721725, -136.9710236, 115.1736298, -252.7916412, 252.6431885
8: -166.1233368, 113.4352722, -165.2444305, 113.0651855, -279.1885376, 278.6796875
9: -125.0816574, 123.5211411, -124.5348816, 123.1142654, -248.1959229, 248.0560303

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6156126, upper bound: 264.6156882
time: 7.64 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6158783, upper bound: 264.6158498
time: 8.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -133.2117767, 106.2656326, -142.3042603, 113.4523087, -246.6640625, 248.5698853
1: -111.3794937, 94.0493698, -119.0911560, 100.5554810, -211.9349670, 213.1405029
2: -146.6026611, 95.2830658, -156.6529388, 101.8062897, -248.4089508, 251.9359894
3: -156.1095276, 83.1680069, -166.8479462, 88.9353714, -245.0448914, 250.0159607
4: -142.7886810, 109.5240860, -152.5530548, 117.0456924, -259.8343811, 262.0771484
5: -128.4900055, 99.6197433, -137.2094727, 106.4746170, -234.9646149, 236.8292236
6: -122.8076019, 118.6185913, -131.1432648, 126.7484055, -249.5559998, 249.7618256
7: -133.4747009, 112.1934509, -142.6210785, 119.8799820, -253.3546600, 254.8145142
8: -161.1326141, 110.0628586, -172.0416260, 117.6300964, -278.7626953, 282.1044922
9: -121.3137970, 119.8276825, -129.6566162, 128.1073914, -249.4211731, 249.4842529

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6154950, upper bound: 264.6154951
time: 7.43 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6154950, upper bound: 264.6158030
time: 7.08 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -137.3429260, 109.5298615, -141.1721344, 112.5560837, -249.8990021, 250.7019806
1: -114.8522186, 96.9537201, -118.1325912, 99.7458649, -214.5980377, 215.0863037
2: -151.1631165, 98.1891479, -155.4023285, 100.9940720, -252.1571808, 253.5914612
3: -160.9450836, 85.6996307, -165.5083313, 88.2183533, -249.1634369, 251.2079468
4: -147.2469025, 112.8937683, -151.3392029, 116.1095581, -263.3564148, 264.2329407
5: -132.4353638, 102.7049255, -136.1209717, 105.6231995, -238.0585327, 238.8258972
6: -126.5909271, 122.2810898, -130.1048126, 125.7368927, -252.3278046, 252.3858948
7: -137.6180267, 115.6721725, -141.4833221, 118.9249725, -256.5429993, 257.1554565
8: -166.1233368, 113.4352722, -170.6836395, 116.6894226, -282.8127441, 284.1188965
9: -125.0816574, 123.5211411, -128.6191559, 127.0797043, -252.1613617, 252.1402893

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6155616, upper bound: 264.6156599
time: 8.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6158126, upper bound: 264.6158225
time: 8.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.12 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6157989, upper bound: 264.6157989
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6157989, upper bound: 264.6160327
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6157027, upper bound: 264.6158082
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6159515, upper bound: 264.6159515
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6155964, upper bound: 264.6156036
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6155964, upper bound: 264.6159373
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6156015, upper bound: 264.6157071
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6158498, upper bound: 264.6158783
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6156036, upper bound: 264.6155964
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6156036, upper bound: 264.6158411
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6156126, upper bound: 264.6156882
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6158783, upper bound: 264.6158498
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6154950, upper bound: 264.6154951
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6154950, upper bound: 264.6158030
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6155616, upper bound: 264.6156599
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.12
Output dim: 7, lower bound: -264.6158126, upper bound: 264.6158225

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -128.7110443, 102.6759796, -128.7110443, 102.6759796, -231.3870087, 231.3870087
1: -107.6105347, 90.8847733, -107.6105347, 90.8847733, -198.4953003, 198.4953003
2: -141.6197968, 92.0970001, -141.6197968, 92.0970001, -233.7167969, 233.7167969
3: -150.7785187, 80.4096222, -150.7785187, 80.4096222, -231.1881409, 231.1881409
4: -137.9255219, 105.8342514, -137.9255219, 105.8342514, -243.7597656, 243.7597656
5: -124.1200485, 96.2677765, -124.1200485, 96.2677765, -220.3878174, 220.3878174
6: -118.6563950, 114.6190338, -118.6563950, 114.6190338, -233.2754211, 233.2754211
7: -128.9323425, 108.4165039, -128.9323425, 108.4165039, -237.3488312, 237.3488312
8: -155.6531372, 106.4151306, -155.6531372, 106.4151306, -262.0682373, 262.0682678
9: -117.1985931, 115.8351593, -117.1985931, 115.8351593, -233.0337067, 233.0337067

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153769, upper bound: 264.6152966
time: 8.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153769, upper bound: 264.6155092
time: 7.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -128.7110443, 102.6759796, -132.0236664, 105.2966156, -234.0076447, 234.6996307
1: -107.6105347, 90.8847733, -110.4031601, 93.2120895, -200.8226166, 201.2879333
2: -141.6197968, 92.0970001, -145.2716827, 94.4220505, -236.0418396, 237.3686829
3: -150.7785187, 80.4096222, -154.6574097, 82.4426422, -233.2211456, 235.0670319
4: -137.9255219, 105.8342514, -141.5063934, 108.5333481, -246.4588318, 247.3406372
5: -124.1200485, 96.2677765, -127.2928391, 98.7439041, -222.8639526, 223.5606079
6: -118.6563950, 114.6190338, -121.6913681, 117.5601883, -236.2165833, 236.3103943
7: -128.9323425, 108.4165039, -132.2469788, 111.2113724, -240.1437073, 240.6634827
8: -155.6531372, 106.4151306, -159.6582336, 109.1145325, -264.7676392, 266.0733643
9: -117.1985931, 115.8351593, -120.2265167, 118.7983475, -235.9969482, 236.0616455

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153769, upper bound: 264.6154679
time: 7.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153769, upper bound: 264.6157022
time: 8.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -129.4062805, 103.2073441, -118.1674347, 94.2206268, -223.6269073, 221.3747559
1: -108.2248077, 91.3628998, -98.9682007, 83.5308456, -191.7556305, 190.3311005
2: -142.3971405, 92.5618286, -130.1066437, 84.6620941, -227.0592346, 222.6684723
3: -151.5896759, 80.8317032, -138.5239105, 74.0830917, -225.6727600, 219.3556213
4: -138.7149963, 106.3886566, -126.7254944, 97.2588730, -235.9738770, 233.1141510
5: -124.7651215, 96.7891006, -113.9036255, 88.4639053, -213.2290192, 210.6927185
6: -119.2729492, 115.2450714, -108.8709946, 105.3724670, -224.6453705, 224.1160583
7: -129.6325531, 109.0313644, -118.4532471, 99.7369308, -229.3694763, 227.4845886
8: -156.5016022, 106.9544449, -142.9004211, 97.7810059, -254.2826080, 249.8548584
9: -117.8516006, 116.4590378, -107.7146988, 106.5532532, -224.4048462, 224.1736908

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6060420, upper bound: 264.6059122
time: 8.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6143647, upper bound: 264.6144675
time: 8.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -132.0236664, 105.2966156, -130.1301117, 103.7567291, -235.7803955, 235.4267120
1: -110.4031601, 93.2120895, -108.8887634, 91.9553833, -202.3585358, 202.1008606
2: -145.2716827, 94.4220505, -143.2196655, 93.1609802, -238.4326477, 237.6417236
3: -154.6574097, 82.4426422, -152.5095520, 81.4191284, -236.0765381, 234.9521790
4: -141.5063934, 108.5333481, -139.4625092, 107.0565186, -248.5629120, 247.9958191
5: -127.2928391, 98.7439041, -125.4488144, 97.3813019, -224.6741333, 224.1927185
6: -121.6913681, 117.5601883, -119.9088745, 115.9408875, -237.6322479, 237.4690552
7: -132.2469788, 111.2113724, -130.3873291, 109.6867523, -241.9337311, 241.5986938
8: -159.6582336, 109.1145325, -157.2913818, 107.6252060, -267.2834167, 266.4058838
9: -120.2265167, 118.7983475, -118.5542297, 117.2256699, -237.4521637, 237.3525696

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6158083, upper bound: 264.6157027
time: 8.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6158083, upper bound: 264.6159514
time: 7.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -128.7110443, 102.6759796, -133.2117767, 106.2656326, -234.9766846, 235.8877411
1: -107.6105347, 90.8847733, -111.3794937, 94.0493698, -201.6598663, 202.2642670
2: -141.6197968, 92.0970001, -146.6026611, 95.2830658, -236.9028625, 238.6996613
3: -150.7785187, 80.4096222, -156.1095276, 83.1680069, -233.9465179, 236.5191498
4: -137.9255219, 105.8342514, -142.7886810, 109.5240860, -247.4496155, 248.6229095
5: -124.1200485, 96.2677765, -128.4900055, 99.6197433, -223.7397919, 224.7577820
6: -118.6563950, 114.6190338, -122.8076019, 118.6185913, -237.2749481, 237.4266357
7: -128.9323425, 108.4165039, -133.4747009, 112.1934509, -241.1257782, 241.8911896
8: -155.6531372, 106.4151306, -161.1326141, 110.0628586, -265.7159729, 267.5477295
9: -117.1985931, 115.8351593, -121.3137970, 119.8276825, -237.0262451, 237.1489105

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6150600, upper bound: 264.6151013
time: 7.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6151965, upper bound: 264.6153149
time: 7.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -128.7110443, 102.6759796, -137.3429260, 109.5298615, -238.2408905, 240.0188904
1: -107.6105347, 90.8847733, -114.8522186, 96.9537201, -204.5642548, 205.7369843
2: -141.6197968, 92.0970001, -151.1631165, 98.1891479, -239.8089447, 243.2601166
3: -150.7785187, 80.4096222, -160.9450836, 85.6996307, -236.4781494, 241.3547058
4: -137.9255219, 105.8342514, -147.2469025, 112.8937683, -250.8192749, 253.0811462
5: -124.1200485, 96.2677765, -132.4353638, 102.7049255, -226.8249817, 228.7031403
6: -118.6563950, 114.6190338, -126.5909271, 122.2810898, -240.9374847, 241.2099609
7: -128.9323425, 108.4165039, -137.6180267, 115.6721725, -244.6045227, 246.0345154
8: -155.6531372, 106.4151306, -166.1233368, 113.4352722, -269.0884094, 272.5384521
9: -117.1985931, 115.8351593, -125.0816574, 123.5211411, -240.7197113, 240.9168091

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6150600, upper bound: 264.6153775
time: 7.53 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6151965, upper bound: 264.6155833
time: 6.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -129.4062805, 103.2073441, -122.5869446, 97.7365570, -227.1428375, 225.7942810
1: -108.2248077, 91.3628998, -102.6653976, 86.6339340, -194.8587341, 194.0282898
2: -142.3971405, 92.5618286, -134.9971619, 87.7941513, -230.1912842, 227.5589905
3: -151.5896759, 80.8317032, -143.7406921, 76.7933426, -228.3830261, 224.5723877
4: -138.7149963, 106.3886566, -131.4915771, 100.8778687, -239.5928650, 237.8802338
5: -124.7651215, 96.7891006, -118.1885529, 91.7580872, -216.5231934, 214.9776459
6: -119.2729492, 115.2450714, -112.9453812, 109.2922745, -228.5652161, 228.1904449
7: -129.6325531, 109.0313644, -122.9141312, 103.4475708, -233.0801086, 231.9454956
8: -156.5016022, 106.9544449, -148.2730408, 101.3585587, -257.8601685, 255.2274780
9: -117.8516006, 116.4590378, -111.7460480, 110.4731903, -228.3247681, 228.2050629

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6059286, upper bound: 264.6058451
time: 8.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6143054, upper bound: 264.6144179
time: 8.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -132.0236664, 105.2966156, -134.4788513, 107.2189713, -239.2426300, 239.7754669
1: -110.4031601, 93.2120895, -112.5275955, 95.0110779, -205.4142303, 205.7396851
2: -145.2716827, 94.4220505, -148.0343933, 96.2406845, -241.5123596, 242.4564362
3: -154.6574097, 82.4426422, -157.6525269, 84.0846481, -238.7420654, 240.0951385
4: -141.5063934, 108.5333481, -144.1576538, 110.6204071, -252.1267853, 252.6909637
5: -127.2928391, 98.7439041, -129.6666565, 100.6190948, -227.9119263, 228.4105530
6: -121.6913681, 117.5601883, -123.9194260, 119.8024673, -241.4938049, 241.4796143
7: -132.2469788, 111.2113724, -134.7769318, 113.3366394, -245.5836182, 245.9882965
8: -159.6582336, 109.1145325, -162.5822144, 111.1485901, -270.8068237, 271.6967468
9: -120.2265167, 118.7983475, -122.5270996, 121.0826263, -241.3091431, 241.3254395

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6156882, upper bound: 264.6156126
time: 7.53 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6156882, upper bound: 264.6158783
time: 7.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -133.2117767, 106.2656326, -128.7110443, 102.6759796, -235.8877411, 234.9766846
1: -111.3794937, 94.0493698, -107.6105347, 90.8847733, -202.2642670, 201.6598663
2: -146.6026611, 95.2830658, -141.6197968, 92.0970001, -238.6996613, 236.9028625
3: -156.1095276, 83.1680069, -150.7785187, 80.4096222, -236.5191498, 233.9465179
4: -142.7886810, 109.5240860, -137.9255219, 105.8342514, -248.6229095, 247.4496155
5: -128.4900055, 99.6197433, -124.1200485, 96.2677765, -224.7577820, 223.7397919
6: -122.8076019, 118.6185913, -118.6563950, 114.6190338, -237.4266357, 237.2749481
7: -133.4747009, 112.1934509, -128.9323425, 108.4165039, -241.8911896, 241.1257782
8: -161.1326141, 110.0628586, -155.6531372, 106.4151306, -267.5477295, 265.7159729
9: -121.3137970, 119.8276825, -117.1985931, 115.8351593, -237.1489105, 237.0262451

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6151410, upper bound: 264.6149538
time: 8.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6152496, upper bound: 264.6151364
time: 7.86 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -133.2117767, 106.2656326, -132.0236664, 105.2966156, -238.5083771, 238.2893066
1: -111.3794937, 94.0493698, -110.4031601, 93.2120895, -204.5915680, 204.4524994
2: -146.6026611, 95.2830658, -145.2716827, 94.4220505, -241.0247192, 240.5547180
3: -156.1095276, 83.1680069, -154.6574097, 82.4426422, -238.5521545, 237.8254089
4: -142.7886810, 109.5240860, -141.5063934, 108.5333481, -251.3219757, 251.0304871
5: -128.4900055, 99.6197433, -127.2928391, 98.7439041, -227.2338867, 226.9125824
6: -122.8076019, 118.6185913, -121.6913681, 117.5601883, -240.3677979, 240.3099060
7: -133.4747009, 112.1934509, -132.2469788, 111.2113724, -244.6860657, 244.4404297
8: -161.1326141, 110.0628586, -159.6582336, 109.1145325, -270.2471313, 269.7210999
9: -121.3137970, 119.8276825, -120.2265167, 118.7983475, -240.1121521, 240.0541687

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6151410, upper bound: 264.6151812
time: 7.88 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6152496, upper bound: 264.6153550
time: 7.15 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -134.7030029, 107.4218292, -118.1674347, 94.2206268, -228.9236298, 225.5892639
1: -112.6545563, 95.0880356, -98.9682007, 83.5308456, -196.1853638, 194.0562439
2: -148.2630157, 96.3127823, -130.1066437, 84.6620941, -232.9250946, 226.4194336
3: -157.8498230, 84.0746918, -138.5239105, 74.0830917, -231.9329071, 222.5986023
4: -144.4297333, 110.7300873, -126.7254944, 97.2588730, -241.6885986, 237.4555817
5: -129.8856354, 100.7329865, -113.9036255, 88.4639053, -218.3495483, 214.6366119
6: -124.1511459, 119.9449921, -108.8709946, 105.3724670, -229.5235901, 228.8159637
7: -134.9799194, 113.4732513, -118.4532471, 99.7369308, -234.7168579, 231.9264526
8: -162.9385986, 111.2554703, -142.9004211, 97.7810059, -260.7195435, 254.1558838
9: -122.6849899, 121.1612854, -107.7146988, 106.5532532, -229.2382507, 228.8759308

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6057832, upper bound: 264.6054879
time: 8.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6143028, upper bound: 264.6143619
time: 6.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -137.3429260, 109.5298615, -130.1301117, 103.7567291, -241.0996552, 239.6599579
1: -114.8522186, 96.9537201, -108.8887634, 91.9553833, -206.8075714, 205.8424835
2: -151.1631165, 98.1891479, -143.2196655, 93.1609802, -244.3240967, 241.4088135
3: -160.9450836, 85.6996307, -152.5095520, 81.4191284, -242.3642120, 238.2091675
4: -147.2469025, 112.8937683, -139.4625092, 107.0565186, -254.3034210, 252.3562622
5: -132.4353638, 102.7049255, -125.4488144, 97.3813019, -229.8166656, 228.1537476
6: -126.5909271, 122.2810898, -119.9088745, 115.9408875, -242.5318146, 242.1899567
7: -137.6180267, 115.6721725, -130.3873291, 109.6867523, -247.3047485, 246.0595093
8: -166.1233368, 113.4352722, -157.2913818, 107.6252060, -273.7485352, 270.7266235
9: -125.0816574, 123.5211411, -118.5542297, 117.2256699, -242.3073273, 242.0753479

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157071, upper bound: 264.6156014
time: 7.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157071, upper bound: 264.6158498
time: 8.10 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -133.2117767, 106.2656326, -133.2117767, 106.2656326, -239.4774170, 239.4774170
1: -111.3794937, 94.0493698, -111.3794937, 94.0493698, -205.4288177, 205.4288177
2: -146.6026611, 95.2830658, -146.6026611, 95.2830658, -241.8857117, 241.8857117
3: -156.1095276, 83.1680069, -156.1095276, 83.1680069, -239.2775269, 239.2775269
4: -142.7886810, 109.5240860, -142.7886810, 109.5240860, -252.3127747, 252.3127747
5: -128.4900055, 99.6197433, -128.4900055, 99.6197433, -228.1097412, 228.1097412
6: -122.8076019, 118.6185913, -122.8076019, 118.6185913, -241.4261627, 241.4261627
7: -133.4747009, 112.1934509, -133.4747009, 112.1934509, -245.6681366, 245.6681366
8: -161.1326141, 110.0628586, -161.1326141, 110.0628586, -271.1954651, 271.1954651
9: -121.3137970, 119.8276825, -121.3137970, 119.8276825, -241.1414337, 241.1414337

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6149359, upper bound: 264.6148672
time: 9.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6150496, upper bound: 264.6150493
time: 8.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -133.2117767, 106.2656326, -137.3429260, 109.5298615, -242.7416229, 243.6085510
1: -111.3794937, 94.0493698, -114.8522186, 96.9537201, -208.3332062, 208.9015350
2: -146.6026611, 95.2830658, -151.1631165, 98.1891479, -244.7918091, 246.4461670
3: -156.1095276, 83.1680069, -160.9450836, 85.6996307, -241.8091431, 244.1130829
4: -142.7886810, 109.5240860, -147.2469025, 112.8937683, -255.6824341, 256.7709961
5: -128.4900055, 99.6197433, -132.4353638, 102.7049255, -231.1949310, 232.0551147
6: -122.8076019, 118.6185913, -126.5909271, 122.2810898, -245.0886841, 245.2094879
7: -133.4747009, 112.1934509, -137.6180267, 115.6721725, -249.1468811, 249.8114624
8: -161.1326141, 110.0628586, -166.1233368, 113.4352722, -274.5678711, 276.1861877
9: -121.3137970, 119.8276825, -125.0816574, 123.5211411, -244.8349152, 244.9093323

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6149359, upper bound: 264.6151499
time: 9.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6150496, upper bound: 264.6153239
time: 8.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -134.7030029, 107.4218292, -122.5869446, 97.7365570, -232.4395599, 230.0087738
1: -112.6545563, 95.0880356, -102.6653976, 86.6339340, -199.2884827, 197.7534180
2: -148.2630157, 96.3127823, -134.9971619, 87.7941513, -236.0571442, 231.3099365
3: -157.8498230, 84.0746918, -143.7406921, 76.7933426, -234.6431580, 227.8153839
4: -144.4297333, 110.7300873, -131.4915771, 100.8778687, -245.3076019, 242.2216644
5: -129.8856354, 100.7329865, -118.1885529, 91.7580872, -221.6437225, 218.9215393
6: -124.1511459, 119.9449921, -112.9453812, 109.2922745, -233.4434204, 232.8903351
7: -134.9799194, 113.4732513, -122.9141312, 103.4475708, -238.4274597, 236.3873749
8: -162.9385986, 111.2554703, -148.2730408, 101.3585587, -264.2971497, 259.5285034
9: -122.6849899, 121.1612854, -111.7460480, 110.4731903, -233.1581726, 232.9073029

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6057560, upper bound: 264.6054792
time: 8.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6142732, upper bound: 264.6143551
time: 7.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -137.3429260, 109.5298615, -134.4788513, 107.2189713, -244.5618744, 244.0087128
1: -114.8522186, 96.9537201, -112.5275955, 95.0110779, -209.8632812, 209.4813232
2: -151.1631165, 98.1891479, -148.0343933, 96.2406845, -247.4038086, 246.2235260
3: -160.9450836, 85.6996307, -157.6525269, 84.0846481, -245.0297241, 243.3521271
4: -147.2469025, 112.8937683, -144.1576538, 110.6204071, -257.8673096, 257.0514221
5: -132.4353638, 102.7049255, -129.6666565, 100.6190948, -233.0544434, 232.3715820
6: -126.5909271, 122.2810898, -123.9194260, 119.8024673, -246.3933868, 246.2005157
7: -137.6180267, 115.6721725, -134.7769318, 113.3366394, -250.9546661, 250.4490967
8: -166.1233368, 113.4352722, -162.5822144, 111.1485901, -277.2719116, 276.0174866
9: -125.0816574, 123.5211411, -122.5270996, 121.0826263, -246.1642761, 246.0482330

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6156387, upper bound: 264.6155665
time: 7.27 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6156387, upper bound: 264.6158225
time: 7.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.06 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6153769, upper bound: 264.6152966
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6153769, upper bound: 264.6155092
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6153769, upper bound: 264.6154679
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6153769, upper bound: 264.6157022
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6060420, upper bound: 264.6059122
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6143647, upper bound: 264.6144675
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6158083, upper bound: 264.6157027
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6158083, upper bound: 264.6159514
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6150600, upper bound: 264.6151013
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6151965, upper bound: 264.6153149
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6150600, upper bound: 264.6153775
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6151965, upper bound: 264.6155833
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6059286, upper bound: 264.6058451
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6143054, upper bound: 264.6144179
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6156882, upper bound: 264.6156126
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6156882, upper bound: 264.6158783
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6151410, upper bound: 264.6149538
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6152496, upper bound: 264.6151364
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6151410, upper bound: 264.6151812
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6152496, upper bound: 264.6153550
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6057832, upper bound: 264.6054879
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6143028, upper bound: 264.6143619
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6157071, upper bound: 264.6156014
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6157071, upper bound: 264.6158498
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6149359, upper bound: 264.6148672
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6150496, upper bound: 264.6150493
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6149359, upper bound: 264.6151499
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6150496, upper bound: 264.6153239
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6057560, upper bound: 264.6054792
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6142732, upper bound: 264.6143551
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6156387, upper bound: 264.6155665
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.06
Output dim: 7, lower bound: -264.6156387, upper bound: 264.6158225

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -110.1871033, 87.9133148, -126.0856018, 100.5806351, -210.7677307, 213.9989166
1: -92.2007370, 77.8242493, -105.4254913, 89.0300140, -181.2307434, 183.2497406
2: -121.2882309, 78.9399490, -138.7360382, 90.2313156, -211.5195312, 217.6759796
3: -129.0992584, 69.0220337, -147.7030334, 78.7929153, -207.8921509, 216.7250671
4: -118.1508102, 90.6569672, -135.1249084, 103.6828232, -221.8336334, 225.7818756
5: -106.2512512, 82.4439545, -121.5853500, 94.3061752, -200.5574341, 204.0292969
6: -101.5581436, 98.2387085, -116.2312088, 112.2965012, -213.8546448, 214.4699097
7: -110.4279709, 92.9889755, -126.3085556, 106.2296219, -216.6575928, 219.2975311
8: -133.3260803, 91.1390381, -152.4873352, 104.2475586, -237.5736389, 243.6263733
9: -100.3855743, 99.2826996, -114.8157578, 113.4879837, -213.8735352, 214.0984497

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6133629, upper bound: 264.6133433
time: 8.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6129850, upper bound: 264.6129335
time: 7.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -122.1439133, 97.4420929, -128.7110443, 102.6759796, -224.8198700, 226.1531219
1: -102.1135635, 86.2390823, -107.6105347, 90.8847733, -192.9983368, 193.8496094
2: -134.3940887, 87.4328537, -141.6197968, 92.0970001, -226.4910889, 229.0526428
3: -143.0731964, 76.3528214, -150.7785187, 80.4096222, -223.4828186, 227.1313171
4: -130.8843536, 100.4496765, -137.9255219, 105.8342514, -236.7185974, 238.3751984
5: -117.7896271, 91.3581619, -124.1200485, 96.2677765, -214.0574036, 215.4782104
6: -112.5868454, 108.8003769, -118.6563950, 114.6190338, -227.2058716, 227.4567719
7: -122.3562317, 102.9359436, -128.9323425, 108.4165039, -230.7727356, 231.8682861
8: -147.7095490, 100.9783859, -155.6531372, 106.4151306, -254.1246490, 256.6315002
9: -111.2249527, 109.9530106, -117.1985931, 115.8351593, -227.0600586, 227.1516113

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6155601, upper bound: 264.6156635
time: 7.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6155601, upper bound: 264.6157753
time: 7.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -110.1871033, 87.9133148, -129.4062805, 103.2073441, -213.3944397, 217.3195953
1: -92.2007370, 77.8242493, -108.2248077, 91.3628998, -183.5636292, 186.0490570
2: -121.2882309, 78.9399490, -142.3971405, 92.5618286, -213.8500671, 221.3370819
3: -129.0992584, 69.0220337, -151.5896759, 80.8317032, -209.9309692, 220.6117096
4: -118.1508102, 90.6569672, -138.7149963, 106.3886566, -224.5394592, 229.3719635
5: -106.2512512, 82.4439545, -124.7651215, 96.7891006, -203.0403442, 207.2090759
6: -101.5581436, 98.2387085, -119.2729492, 115.2450714, -216.8032227, 217.5116425
7: -110.4279709, 92.9889755, -129.6325531, 109.0313644, -219.4593353, 222.6215210
8: -133.3260803, 91.1390381, -156.5016022, 106.9544449, -240.2805176, 247.6406403
9: -100.3855743, 99.2826996, -117.8516006, 116.4590378, -216.8446045, 217.1342773

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6058033, upper bound: 264.6059923
time: 7.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6141284, upper bound: 264.6142239
time: 8.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -122.1439133, 97.4420929, -132.0236664, 105.2966156, -227.4405060, 229.4657440
1: -102.1135635, 86.2390823, -110.4031601, 93.2120895, -195.3256531, 196.6422424
2: -134.3940887, 87.4328537, -145.2716827, 94.4220505, -228.8161316, 232.7045135
3: -143.0731964, 76.3528214, -154.6574097, 82.4426422, -225.5158234, 231.0102081
4: -130.8843536, 100.4496765, -141.5063934, 108.5333481, -239.4176636, 241.9560699
5: -117.7896271, 91.3581619, -127.2928391, 98.7439041, -216.5335083, 218.6510010
6: -112.5868454, 108.8003769, -121.6913681, 117.5601883, -230.1470337, 230.4917450
7: -122.3562317, 102.9359436, -132.2469788, 111.2113724, -233.5675964, 235.1829224
8: -147.7095490, 100.9783859, -159.6582336, 109.1145325, -256.8240967, 260.6366272
9: -111.2249527, 109.9530106, -120.2265167, 118.7983475, -230.0233002, 230.1795349

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153187, upper bound: 264.6155267
time: 7.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153187, upper bound: 264.6157022
time: 7.17 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 16.20 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 7, lower bound: -264.6133629, upper bound: 264.6133433
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 7, lower bound: -264.6129850, upper bound: 264.6129335
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 7, lower bound: -264.6155601, upper bound: 264.6156635
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 7, lower bound: -264.6155601, upper bound: 264.6157753
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.20
Output dim: 7, lower bound: -264.6058033, upper bound: 264.6059923
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 7, lower bound: -264.6141284, upper bound: 264.6142239
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 7, lower bound: -264.6153187, upper bound: 264.6155267
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 7, lower bound: -264.6153187, upper bound: 264.6157022
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6143647, upper bound: 264.6144675
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6158083, upper bound: 264.6157027
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6158083, upper bound: 264.6159514
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6150600, upper bound: 264.6151013
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6151965, upper bound: 264.6153149
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6150600, upper bound: 264.6153775
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6151965, upper bound: 264.6155833
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6143054, upper bound: 264.6144179
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6156882, upper bound: 264.6156126
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6156882, upper bound: 264.6158783
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6151410, upper bound: 264.6149538
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6152496, upper bound: 264.6151364
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6151410, upper bound: 264.6151812
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6152496, upper bound: 264.6153550
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6143028, upper bound: 264.6143619
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6157071, upper bound: 264.6156014
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6157071, upper bound: 264.6158498
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6149359, upper bound: 264.6148672
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6150496, upper bound: 264.6150493
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6149359, upper bound: 264.6151499
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6150496, upper bound: 264.6153239
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6142732, upper bound: 264.6143551
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6156387, upper bound: 264.6155665
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.20
Output dim: 7, lower bound: -264.6156387, upper bound: 264.6158225
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=266.34649658203125
rel_dist={7: [-264.62269597868163, 264.6226959786816]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6210894, upper bound: 264.6210911
time: 9.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6210762, upper bound: 264.6210762
time: 8.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.62
Output dim: 7, lower bound: -264.6210894, upper bound: 264.6210911
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.62
Output dim: 7, lower bound: -264.6210762, upper bound: 264.6210762

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -137.8412933, 109.8958969, -142.9869385, 113.9779510, -251.8192444, 252.8828278
1: -115.3550110, 97.4181595, -119.6691360, 101.0392990, -216.3943176, 217.0872650
2: -151.7102203, 98.6460876, -157.4117889, 102.3023453, -254.0125732, 256.0578613
3: -161.5674744, 86.2000275, -167.6147461, 89.3541946, -250.9216614, 253.8147736
4: -147.7318878, 113.3872681, -153.2836304, 117.6143951, -265.3462524, 266.6708679
5: -132.8794250, 103.1506424, -137.8493958, 106.9892807, -239.8687134, 241.0000305
6: -127.0270538, 122.7826920, -131.7663422, 127.3540497, -254.3811035, 254.5490417
7: -138.1151123, 116.1337585, -143.3169098, 120.4720535, -258.5871582, 259.4506531
8: -166.6094513, 114.0110855, -172.8577728, 118.2140808, -284.8235168, 286.8688660
9: -125.5774002, 124.1473160, -130.2671051, 128.7333069, -254.3106842, 254.4143982

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153901, upper bound: 264.6153782
time: 9.02 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6160173, upper bound: 264.6159814
time: 9.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -142.3042603, 113.4523087, -143.5935516, 114.4609909, -256.7652588, 257.0458679
1: -119.0911560, 100.5554810, -120.1770477, 101.4646530, -220.5558167, 220.7325287
2: -156.6529388, 101.8062897, -158.0825806, 102.7344284, -259.3872681, 259.8888550
3: -166.8479462, 88.9353714, -168.3286285, 89.7295456, -256.5774841, 257.2639771
4: -152.5530548, 117.0456924, -153.9416504, 118.1114731, -270.6645203, 270.9873352
5: -137.2094727, 106.4746170, -138.4344177, 107.4407578, -244.6502380, 244.9090271
6: -131.1432648, 126.7484055, -132.3254242, 127.8934402, -259.0366821, 259.0738220
7: -142.6210785, 119.8799820, -143.9304657, 120.9829636, -263.6040344, 263.8103943
8: -172.0416260, 117.6300964, -173.5911560, 118.7095490, -290.7511597, 291.2211609
9: -129.6566162, 128.1073914, -130.8234406, 129.2762604, -258.9328613, 258.9307556

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153660, upper bound: 264.6153909
time: 9.19 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6160183, upper bound: 264.6160183
time: 8.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.93 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.93
Output dim: 7, lower bound: -264.6153901, upper bound: 264.6153782
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.93
Output dim: 7, lower bound: -264.6160173, upper bound: 264.6159814
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.93
Output dim: 7, lower bound: -264.6153660, upper bound: 264.6153909
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.93
Output dim: 7, lower bound: -264.6160183, upper bound: 264.6160183

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -134.4368591, 107.2041016, -133.9681396, 106.8482666, -241.2851105, 241.1722412
1: -112.4666672, 94.9820862, -112.0184097, 94.5854492, -207.0521240, 207.0004730
2: -147.9477234, 96.2041779, -147.4440765, 95.8312607, -243.7789764, 243.6482544
3: -157.5433502, 84.0398712, -156.9615631, 83.6318970, -241.1752319, 241.0014038
4: -144.0753937, 110.5706863, -143.5986633, 110.1534424, -254.2288055, 254.1693420
5: -129.6129761, 100.5843811, -129.1991882, 100.1894913, -229.8024597, 229.7835541
6: -123.9059677, 119.7381744, -123.4979172, 119.2902527, -243.1961670, 243.2360840
7: -134.6915741, 113.2554474, -134.2463531, 112.8479538, -247.5395203, 247.5017700
8: -162.5248413, 111.1792908, -162.0373535, 110.7077255, -273.2325745, 273.2166443
9: -122.4528809, 121.0480270, -121.9910507, 120.5196609, -242.9725342, 243.0390778

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6148520, upper bound: 264.6148026
time: 10.38 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6149335, upper bound: 264.6149882
time: 10.33 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -133.8258972, 106.7179794, -137.9519043, 109.9987793, -243.8246765, 244.6698608
1: -111.9554825, 94.5474930, -115.3674316, 97.3843765, -209.3398590, 209.9149170
2: -147.2739258, 95.7647171, -151.8386688, 98.6308136, -245.9047394, 247.6033936
3: -156.8212280, 83.6589127, -161.6291504, 86.0719681, -242.8931885, 245.2880402
4: -143.4263763, 110.0670929, -147.8988800, 113.4015045, -256.8278809, 257.9659424
5: -129.0205688, 100.1306152, -133.0061340, 103.1625443, -232.1831055, 233.1367493
6: -123.3450928, 119.1956711, -127.1470032, 122.8203735, -246.1654663, 246.3426666
7: -134.0789642, 112.7459869, -138.2373810, 116.2002335, -250.2791748, 250.9833679
8: -161.7943115, 110.6736450, -166.8506165, 113.9557495, -275.7500000, 277.5242615
9: -121.8994980, 120.5025253, -125.6261902, 124.0795898, -245.9790802, 246.1286926

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6154094, upper bound: 264.6153490
time: 7.38 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6155386, upper bound: 264.6155714
time: 8.22 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -138.9165039, 110.7750015, -134.5254059, 107.2927933, -246.2092896, 245.3004150
1: -116.2175598, 98.1314392, -112.4849243, 94.9754791, -211.1930389, 210.6163635
2: -152.9084625, 99.3760910, -148.0593414, 96.2280807, -249.1365356, 247.4354248
3: -162.8459015, 86.7855988, -157.6181183, 83.9766769, -246.8225708, 244.4037170
4: -148.9150085, 114.2430420, -144.2034912, 110.6096649, -259.5246582, 258.4465332
5: -133.9606171, 103.9206619, -129.7372131, 100.6038818, -234.5644989, 233.6578674
6: -128.0380859, 123.7189636, -124.0118179, 119.7853928, -247.8234863, 247.7307434
7: -139.2136383, 117.0154495, -134.8092651, 113.3166275, -252.5302734, 251.8247070
8: -167.9778900, 114.8112106, -162.7112732, 111.1621323, -279.1400146, 277.5224609
9: -126.5478134, 125.0226898, -122.5023422, 121.0181580, -247.5659790, 247.5250092

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6148192, upper bound: 264.6147585
time: 9.39 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6149171, upper bound: 264.6149276
time: 8.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -138.3100128, 110.2897186, -138.6251526, 110.5329666, -248.8429871, 248.9148560
1: -115.7083511, 97.6993484, -115.9303284, 97.8564835, -213.5648346, 213.6296692
2: -152.2405701, 98.9401703, -152.5838928, 99.1103668, -251.3509369, 251.5240173
3: -162.1233368, 86.4065323, -162.4183807, 86.4876404, -248.6109772, 248.8249207
4: -148.2698364, 113.7427139, -148.6284485, 113.9528656, -262.2227173, 262.3711548
5: -133.3687134, 103.4710007, -133.6531982, 103.6642303, -237.0329437, 237.1241913
6: -127.4795761, 123.1797409, -127.7662277, 123.4190903, -250.8986664, 250.9459229
7: -138.6068573, 116.5104446, -138.9187012, 116.7674179, -255.3742523, 255.4291382
8: -167.2509155, 114.3108826, -167.6636963, 114.5061951, -281.7570801, 281.9745178
9: -125.9966125, 124.4814301, -126.2420807, 124.6819382, -250.6785431, 250.7235107

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153814, upper bound: 264.6152929
time: 8.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6155158, upper bound: 264.6155158
time: 8.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.12 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.12
Output dim: 7, lower bound: -264.6148520, upper bound: 264.6148026
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.12
Output dim: 7, lower bound: -264.6149335, upper bound: 264.6149882
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.12
Output dim: 7, lower bound: -264.6154094, upper bound: 264.6153490
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.12
Output dim: 7, lower bound: -264.6155386, upper bound: 264.6155714
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.12
Output dim: 7, lower bound: -264.6148192, upper bound: 264.6147585
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.12
Output dim: 7, lower bound: -264.6149171, upper bound: 264.6149276
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.12
Output dim: 7, lower bound: -264.6153814, upper bound: 264.6152929
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.12
Output dim: 7, lower bound: -264.6155158, upper bound: 264.6155158

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -115.8925018, 92.4237976, -126.2687454, 100.7030792, -216.5955811, 218.6925354
1: -97.0372391, 81.9040070, -105.6103363, 89.1462631, -186.1835022, 187.5143433
2: -127.5921326, 83.0309982, -138.9849243, 90.3592758, -217.9513702, 222.0159149
3: -135.8376312, 72.6387024, -147.9403534, 78.8921127, -214.7297211, 220.5790558
4: -124.2786407, 95.3766708, -135.3810425, 103.8432312, -228.1218719, 230.7577209
5: -111.7238998, 86.7457657, -121.7654800, 94.4376831, -206.1615906, 208.5112457
6: -106.7869339, 103.3377304, -116.3858566, 112.4775009, -219.2644348, 219.7235870
7: -116.1643753, 97.8106537, -126.5510101, 106.4336853, -222.5980530, 224.3616486
8: -140.1707458, 95.8868027, -152.7503967, 104.3488541, -244.5195618, 248.6371918
9: -105.6228256, 104.4773865, -114.9999924, 113.6367264, -219.2595520, 219.4773560

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6037355, upper bound: 264.6036074
time: 9.63 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6135468, upper bound: 264.6135011
time: 8.92 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -127.8719406, 101.9718399, -131.5763855, 104.9409866, -232.8128967, 233.5482178
1: -106.9715576, 90.3385620, -110.0155334, 92.8928299, -199.8643646, 200.3540955
2: -140.7237549, 91.5413284, -144.8109436, 94.1326904, -234.8564301, 236.3522644
3: -149.8412018, 79.9847183, -154.1535645, 82.1544113, -231.9956055, 234.1382751
4: -137.0355072, 105.1878586, -141.0325775, 108.1917648, -245.2272644, 246.2204285
5: -123.2842789, 95.6767883, -126.8929825, 98.4013519, -221.6855927, 222.5697632
6: -117.8380737, 113.9208145, -121.2869644, 117.1692734, -235.0073547, 235.2077789
7: -128.1165314, 107.7761536, -131.8505554, 110.8511658, -238.9676971, 239.6267090
8: -154.5815430, 105.7450562, -159.1420135, 108.7265854, -263.3081360, 264.8870239
9: -116.4808350, 115.1672897, -119.8142090, 118.3764572, -234.8572998, 234.9815063

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6041757, upper bound: 264.6043984
time: 9.73 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6136867, upper bound: 264.6137438
time: 9.07 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -115.3476410, 91.9894562, -130.2995148, 103.8903351, -219.2379608, 222.2889557
1: -96.5819473, 81.5150299, -108.9968262, 91.9773102, -188.5592346, 190.5118561
2: -126.9908752, 82.6393356, -143.4316254, 93.1919403, -220.1828156, 226.0709534
3: -135.1900787, 72.2984238, -152.6582031, 81.3622360, -216.5523071, 224.9566345
4: -123.7032166, 94.9270554, -139.7341003, 107.1290817, -230.8322906, 234.6611328
5: -111.1945648, 86.3444595, -125.6155472, 97.4489899, -208.6435547, 211.9600067
6: -106.2849579, 102.8540192, -120.0750809, 116.0490875, -222.3340454, 222.9290924
7: -115.6197662, 97.3591614, -130.5912018, 109.8268509, -225.4466095, 227.9503632
8: -139.5199890, 95.4365311, -157.6199951, 107.6376266, -247.1576233, 253.0565186
9: -105.1333542, 103.9948807, -118.6803436, 117.2414322, -222.3747864, 222.6752319

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 188

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6038531, upper bound: 264.6037680
time: 10.24 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6140672, upper bound: 264.6139987
time: 8.34 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -127.2770920, 101.4989395, -135.5607147, 108.0922928, -235.3693848, 237.0596466
1: -106.4734955, 89.9151001, -113.3644714, 95.6918182, -202.1653137, 203.2795715
2: -140.0672607, 91.1135330, -149.2053528, 96.9319916, -236.9992523, 240.3188782
3: -149.1379852, 79.6131516, -158.8220367, 84.5937424, -233.7317200, 238.4351807
4: -136.4030609, 104.6970444, -145.3321991, 111.4398651, -247.8429108, 250.0292358
5: -122.7082977, 95.2344971, -130.7006531, 101.3743820, -224.0826721, 225.9351501
6: -117.2931747, 113.3920059, -124.9365387, 120.6990509, -237.9922180, 238.3285370
7: -127.5190964, 107.2793655, -135.8413849, 114.2020798, -241.7211761, 243.1207581
8: -153.8701477, 105.2521362, -163.9555054, 111.9739227, -265.8440552, 269.2076416
9: -115.9401398, 114.6355133, -123.4484787, 121.9356613, -237.8757935, 238.0839844

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6042943, upper bound: 264.6045367
time: 30.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6142710, upper bound: 264.6143128
time: 8.69 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -120.3103638, 95.9397812, -126.8261108, 101.1475601, -221.4579163, 222.7658997
1: -100.7337799, 85.0060196, -106.0767365, 89.5362778, -190.2700500, 191.0827484
2: -132.4792480, 86.1608582, -139.6003265, 90.7561188, -223.2353668, 225.7611847
3: -141.0557556, 75.3483658, -148.5967712, 79.2369614, -220.2926941, 223.9451294
4: -129.0436249, 98.9941101, -135.9859924, 104.2993698, -233.3429718, 234.9801025
5: -116.0094757, 90.0386047, -122.3035202, 94.8521957, -210.8616638, 212.3421173
6: -110.8603745, 107.2555771, -116.8997955, 112.9725800, -223.8329468, 224.1553650
7: -120.6221237, 101.5190735, -127.1140137, 106.9025192, -227.5246429, 228.6330719
8: -145.5411377, 99.4613190, -153.4243927, 104.8032913, -250.3444214, 252.8857117
9: -109.6533127, 108.3946457, -115.5114822, 114.1352921, -223.7886047, 223.9061279

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6037266, upper bound: 264.6036218
time: 10.93 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6135313, upper bound: 264.6134654
time: 8.83 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -132.2233124, 105.4375763, -132.1312256, 105.3834000, -237.6067047, 237.5688019
1: -110.6131134, 93.3964920, -110.4798126, 93.2810059, -203.8941193, 203.8763123
2: -145.5408478, 94.6226349, -145.4234009, 94.5277252, -240.0685730, 240.0460205
3: -154.9898987, 82.6518936, -154.8069916, 82.4975510, -237.4874573, 237.4588623
4: -141.7340851, 108.7539978, -141.6347351, 108.6459045, -250.3799744, 250.3887329
5: -127.5063400, 98.9162903, -127.4285660, 98.8138580, -226.3201752, 226.3448486
6: -121.8513641, 117.7849274, -121.7985764, 117.6621170, -239.5134583, 239.5834808
7: -132.5082397, 111.4275208, -132.4109497, 111.3177032, -243.8259125, 243.8384705
8: -159.8760834, 109.2699890, -159.8127289, 109.1788177, -269.0549011, 269.0827026
9: -120.4565430, 119.0260468, -120.3232117, 118.8725586, -239.3291016, 239.3492584

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6041764, upper bound: 264.6043510
time: 9.50 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6136837, upper bound: 264.6137142
time: 8.53 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -119.7777481, 95.5131378, -130.9752350, 104.4266052, -224.2043304, 226.4883575
1: -100.2868347, 84.6250381, -109.5618591, 92.4513092, -192.7381287, 194.1868896
2: -131.8942719, 85.7797012, -144.1801453, 93.6733322, -225.5675964, 229.9598236
3: -140.4165192, 75.0147171, -153.4500732, 81.7797165, -222.1962280, 228.4647675
4: -128.4803619, 98.5542450, -140.4665833, 107.6824265, -236.1627808, 239.0208282
5: -115.4878311, 89.6469727, -126.2648468, 97.9528198, -213.4406281, 215.9118195
6: -110.3684998, 106.7827911, -120.6966782, 116.6501923, -227.0186920, 227.4794617
7: -120.0919952, 101.0794296, -131.2756042, 110.3964233, -230.4884186, 232.3550415
8: -144.9051971, 99.0237198, -158.4363708, 108.1902695, -253.0954590, 257.4600830
9: -109.1732254, 107.9248199, -119.2985764, 117.8464432, -227.0196381, 227.2233734

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6038477, upper bound: 264.6037849
time: 10.51 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6140656, upper bound: 264.6139690
time: 8.96 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -131.6348877, 104.9674835, -136.2310181, 108.6241531, -240.2590332, 241.1985016
1: -110.1188431, 92.9770737, -113.9247894, 96.1618118, -206.2806549, 206.9018555
2: -144.8923187, 94.2003174, -149.9474640, 97.4095459, -242.3018646, 244.1477814
3: -154.2894440, 82.2835007, -159.6077881, 85.0075760, -239.2970276, 241.8912964
4: -141.1074524, 108.2682877, -146.0585938, 111.9888840, -253.0963440, 254.3268738
5: -126.9329453, 98.4798965, -131.3448181, 101.8739166, -228.8068542, 229.8247070
6: -121.3112183, 117.2612152, -125.5530701, 121.2952652, -242.6064453, 242.8142853
7: -131.9183960, 110.9372025, -136.5198822, 114.7667465, -246.6851044, 247.4570770
8: -159.1711121, 108.7837906, -164.7649994, 112.5219879, -271.6930542, 273.5487976
9: -119.9203339, 118.5008392, -124.0616531, 122.5354233, -242.4557495, 242.5625000

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6042926, upper bound: 264.6045076
time: 9.33 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6142710, upper bound: 264.6142710
time: 9.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.46 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6037355, upper bound: 264.6036074
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6135468, upper bound: 264.6135011
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6041757, upper bound: 264.6043984
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6136867, upper bound: 264.6137438
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6038531, upper bound: 264.6037680
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6140672, upper bound: 264.6139987
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6042943, upper bound: 264.6045367
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6142710, upper bound: 264.6143128
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6037266, upper bound: 264.6036218
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6135313, upper bound: 264.6134654
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6041764, upper bound: 264.6043510
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6136837, upper bound: 264.6137142
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6038477, upper bound: 264.6037849
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6140656, upper bound: 264.6139690
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6042926, upper bound: 264.6045076
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.46
Output dim: 7, lower bound: -264.6142710, upper bound: 264.6142710

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -115.6924973, 92.2650452, -124.6475906, 99.4170074, -215.1094971, 216.9125977
1: -96.8697357, 81.7605057, -104.2515488, 87.9823151, -184.8520508, 186.0120392
2: -127.3721313, 82.8861160, -137.2033691, 89.1858902, -216.5580139, 220.0894775
3: -135.5995789, 72.5156403, -146.0117950, 77.8956757, -213.4952240, 218.5274353
4: -124.0627518, 95.2095108, -133.6321716, 102.4897995, -226.5525513, 228.8416748
5: -111.5315475, 86.5940781, -120.2069473, 93.2079773, -204.7395325, 206.8009796
6: -106.6009216, 103.1608200, -114.8795929, 111.0442123, -217.6451263, 218.0404053
7: -115.9610138, 97.6416931, -124.9040451, 105.0657272, -221.0267334, 222.5457153
8: -139.9295197, 95.7208557, -150.7980804, 103.0050278, -242.9345398, 246.5189362
9: -105.4379349, 104.2960281, -113.5030441, 112.1682434, -217.6061401, 217.7990723

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6117301, upper bound: 264.6115587
time: 9.26 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6114931, upper bound: 264.6113111
time: 8.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -127.6688309, 101.8107452, -129.9600677, 103.6589355, -231.3277588, 231.7708130
1: -106.8013687, 90.1927567, -108.6609955, 91.7326355, -198.5339966, 198.8537598
2: -140.5005951, 91.3944550, -143.0352020, 92.9631500, -233.4637451, 234.4296417
3: -149.5995636, 79.8598251, -152.2310944, 81.1610718, -230.7605896, 232.0909119
4: -136.8163757, 105.0183792, -139.2889709, 106.8426285, -243.6589813, 244.3073425
5: -123.0890274, 95.5228500, -125.3391876, 97.1753311, -220.2643585, 220.8620300
6: -117.6494217, 113.7412643, -119.7854538, 115.7406006, -233.3900146, 233.5267181
7: -127.9102859, 107.6048050, -130.2088776, 109.4876251, -237.3978729, 237.8136749
8: -154.3370209, 105.5767136, -157.1958160, 107.3870621, -261.7240601, 262.7724915
9: -116.2933350, 114.9833984, -118.3222275, 116.9128952, -233.2061768, 233.3056183

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6118770, upper bound: 264.6117872
time: 8.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6116270, upper bound: 264.6115462
time: 9.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -115.1490707, 91.8318100, -128.6957245, 102.6185150, -217.7675476, 220.5275269
1: -96.4156113, 81.3725891, -107.6528549, 90.8269730, -187.2425842, 189.0254364
2: -126.7724609, 82.4955139, -141.6694489, 92.0328445, -218.8052979, 224.1649475
3: -134.9537354, 72.1762695, -150.7520447, 80.3762207, -215.3299561, 222.9283142
4: -123.4888687, 94.7610779, -138.0041656, 105.7910843, -229.2799530, 232.7652435
5: -111.0035706, 86.1938553, -124.0738373, 96.2340393, -207.2376099, 210.2676849
6: -106.1002884, 102.6783981, -118.5858002, 114.6315155, -220.7318115, 221.2641907
7: -115.4178696, 97.1913986, -128.9637146, 108.4742737, -223.8921204, 226.1551056
8: -139.2804565, 95.2717056, -155.6889954, 106.3081131, -245.5885620, 250.9606781
9: -104.9497375, 103.8148346, -117.1996841, 115.7896729, -220.7394104, 221.0144958

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6121534, upper bound: 264.6119828
time: 9.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6119373, upper bound: 264.6117668
time: 9.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -127.0753555, 101.3388977, -133.9597015, 106.8227615, -233.8981018, 235.2985840
1: -106.3044357, 89.7703018, -112.0229340, 94.5436249, -200.8480377, 201.7932434
2: -139.8455353, 90.9675751, -147.4463959, 95.7749176, -235.6204376, 238.4139404
3: -148.8979645, 79.4890823, -156.9192352, 83.6094208, -232.5073853, 236.4082794
4: -136.1853333, 104.5286636, -143.6052704, 110.1044540, -246.2897797, 248.1339417
5: -122.5143585, 95.0815735, -129.1614838, 100.1614304, -222.6757660, 224.2430267
6: -117.1057281, 113.2136230, -123.4495316, 119.2840271, -236.3897552, 236.6631165
7: -127.3141174, 107.1091232, -134.2166443, 112.8517990, -240.1659241, 241.3257751
8: -153.6272278, 105.0848770, -162.0278168, 110.6470871, -264.2743225, 267.1127014
9: -115.7538071, 114.4528046, -121.9709778, 120.4868622, -236.2406616, 236.4237518

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6124465, upper bound: 264.6123131
time: 9.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6121845, upper bound: 264.6121019
time: 8.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -120.1122131, 95.7824173, -125.2139664, 99.8684006, -219.9806061, 220.9963837
1: -100.5678024, 84.8638840, -104.7252808, 88.3786621, -188.9464722, 189.5891418
2: -132.2613678, 86.0174103, -137.8289795, 89.5894089, -221.8507690, 223.8463898
3: -140.8199005, 75.2264862, -146.6784058, 78.2459641, -219.0658569, 221.9048767
4: -128.8298340, 98.8284531, -134.2469330, 102.9530792, -231.7829132, 233.0753784
5: -115.8187637, 89.8883133, -120.7531738, 93.6292038, -209.4479523, 210.6414795
6: -110.6761017, 107.0803757, -115.4017410, 111.5473175, -222.2234192, 222.4821167
7: -120.4208221, 101.3517532, -125.4764557, 105.5420761, -225.9628906, 226.8282166
8: -145.3022156, 99.2968826, -151.4830780, 103.4668045, -248.7690125, 250.7799377
9: -109.4701385, 108.2150421, -114.0225601, 112.6748657, -222.1450043, 222.2375793

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 163

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6117431, upper bound: 264.6116150
time: 8.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6115367, upper bound: 264.6114323
time: 8.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -132.0219269, 105.2778397, -130.5235901, 104.1080856, -236.1300049, 235.8014221
1: -110.4443817, 93.2519836, -109.1323853, 92.1268845, -202.5712585, 202.3843689
2: -145.3197021, 94.4770660, -143.6574097, 93.3644943, -238.6841736, 238.1344757
3: -154.7503357, 82.5281067, -152.8943787, 81.5095062, -236.2598267, 235.4224854
4: -141.5168762, 108.5859680, -139.9005432, 107.3036499, -248.8205261, 248.4865112
5: -127.3127060, 98.7636490, -125.8828201, 97.5942230, -224.9069214, 224.6464691
6: -121.6643295, 117.6069565, -120.3049469, 116.2411041, -237.9054260, 237.9118652
7: -132.3038483, 111.2576828, -130.7781372, 109.9613342, -242.2651520, 242.0357971
8: -159.6336212, 109.1031418, -157.8770294, 107.8463745, -267.4799805, 266.9801636
9: -120.2706528, 118.8437881, -118.8389587, 117.4167862, -237.6874390, 237.6827393

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6118966, upper bound: 264.6118317
time: 9.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6116856, upper bound: 264.6116738
time: 9.07 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -119.5806503, 95.3566589, -129.3806763, 103.1619949, -222.7426453, 224.7373047
1: -100.1217422, 84.4837189, -108.2255020, 91.3073349, -191.4290771, 192.7091980
2: -131.6775818, 85.6370468, -142.4281006, 92.5208130, -224.1983948, 228.0651398
3: -140.1819458, 74.8935089, -151.5544739, 80.7993774, -220.9813232, 226.4479675
4: -128.2677002, 98.3894806, -138.7466278, 106.3519745, -234.6196747, 237.1361084
5: -115.2981720, 89.4975204, -124.7318192, 96.7446747, -212.0428467, 214.2293396
6: -110.1852188, 106.6085129, -119.2157669, 115.2407455, -225.4259491, 225.8242798
7: -119.8917847, 100.9130096, -129.6573334, 109.0514297, -228.9432068, 230.5703430
8: -144.6675568, 98.8601685, -156.5164337, 106.8683243, -251.5358734, 255.3766022
9: -108.9910355, 107.7461853, -117.8262482, 116.4029083, -225.3939209, 225.5724182

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6122808, upper bound: 264.6121549
time: 8.47 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6120745, upper bound: 264.6119910
time: 8.27 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -131.4348907, 104.8088379, -134.6394043, 107.3619690, -238.7968597, 239.4482422
1: -109.9512558, 92.8335800, -112.5911179, 95.0201645, -204.9714203, 205.4246979
2: -144.6726837, 94.0557632, -148.1989594, 96.2592850, -240.9319763, 242.2547150
3: -154.0515289, 82.1605682, -157.7157745, 84.0290833, -238.0805969, 239.8763428
4: -140.8917542, 108.1013794, -144.3419037, 110.6611176, -251.5528717, 252.4432831
5: -126.7406540, 98.3283386, -129.8144684, 100.6679306, -227.4085541, 228.1428070
6: -121.1254349, 117.0844955, -124.0746613, 119.8885727, -241.0139771, 241.1591492
7: -131.7154083, 110.7685318, -134.9046326, 113.4243164, -245.1396942, 245.6731567
8: -158.9303894, 108.6180649, -162.8486786, 111.2029343, -270.1332397, 271.4667358
9: -119.7357254, 118.3198242, -122.5926132, 121.0950546, -240.8307800, 240.9124298

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6126095, upper bound: 264.6125523
time: 7.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6123921, upper bound: 264.6123921
time: 7.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.75 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6117301, upper bound: 264.6115587
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6114931, upper bound: 264.6113111
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6118770, upper bound: 264.6117872
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6116270, upper bound: 264.6115462
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6121534, upper bound: 264.6119828
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6119373, upper bound: 264.6117668
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6124465, upper bound: 264.6123131
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6121845, upper bound: 264.6121019
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6117431, upper bound: 264.6116150
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6115367, upper bound: 264.6114323
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6118966, upper bound: 264.6118317
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6116856, upper bound: 264.6116738
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6122808, upper bound: 264.6121549
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6120745, upper bound: 264.6119910
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6126095, upper bound: 264.6125523
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.75
Output dim: 7, lower bound: -264.6123921, upper bound: 264.6123921

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -119.8562241, 95.5586090, -131.5272522, 104.8747101, -224.7309265, 227.0858459
1: -100.2327271, 84.6960678, -109.9767151, 92.8324814, -193.0652161, 194.6727753
2: -131.8569031, 85.7825775, -144.7528992, 94.0261230, -225.8830261, 230.5354767
3: -140.4355011, 74.9286118, -154.0675507, 82.0710678, -222.5065613, 228.9961548
4: -128.4861755, 98.5839615, -141.0104980, 108.1010895, -236.5872498, 239.5944519
5: -115.5788879, 89.6973877, -126.8246841, 98.3458176, -213.9247131, 216.5220642
6: -110.5011597, 106.7532349, -121.2234955, 117.1064072, -227.6075745, 227.9767303
7: -120.1245956, 101.0351028, -131.7929993, 110.8037567, -230.9283295, 232.8280792
8: -144.8708954, 99.0115280, -159.0760803, 108.5988922, -253.4697876, 258.0875244
9: -109.2514725, 107.9228287, -119.7802429, 118.2846603, -227.5361328, 227.7030640

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6122875, upper bound: 264.6122917
time: 9.22 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6122875, upper bound: 264.6123131
time: 9.22 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -124.2052612, 99.0205765, -132.1951599, 105.4047852, -229.6100464, 231.2157288
1: -103.8689499, 87.7497864, -110.5354462, 93.3009109, -197.1698608, 198.2852325
2: -136.6713867, 88.8617020, -145.4925232, 94.5023270, -231.1736908, 234.3542175
3: -145.5756683, 77.5915222, -154.8507080, 82.4836884, -228.0593567, 232.4422150
4: -133.1809082, 102.1476974, -141.7343750, 108.6480637, -241.8289795, 243.8820801
5: -119.7959518, 92.9353867, -127.4666672, 98.8436432, -218.6395874, 220.4020386
6: -114.5099411, 110.6142426, -121.8380432, 117.7004242, -232.2103424, 232.4522858
7: -124.5131073, 104.6837692, -132.4692993, 111.3664093, -235.8795013, 237.1530457
8: -150.1608429, 102.5352325, -159.8827515, 109.1451569, -259.3059998, 262.4179688
9: -113.2240372, 111.7785339, -120.3914337, 118.8826523, -232.1066895, 232.1699371

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6122810, upper bound: 264.6123290
time: 8.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6122810, upper bound: 264.6125523
time: 8.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -126.2991791, 100.6304855, -130.4022369, 103.9702988, -230.2694702, 231.0326843
1: -105.6309891, 89.2669601, -109.0241699, 92.0602341, -197.6912079, 198.2911377
2: -138.9960022, 90.3215408, -143.5253906, 93.2236023, -232.2195587, 233.8469238
3: -148.0853271, 78.8713989, -152.7733307, 81.3745956, -229.4599152, 231.6447296
4: -135.4845428, 103.8733063, -139.8317871, 107.1833725, -242.6679077, 243.7050934
5: -121.8183899, 94.4903793, -125.7492599, 97.5182343, -219.3366241, 220.2396240
6: -116.4331360, 112.4850159, -120.2008362, 116.1128159, -232.5459595, 232.6858521
7: -126.6623611, 106.4652405, -130.6995544, 109.8748932, -236.5372620, 237.1647797
8: -152.6924896, 104.2021942, -157.7198944, 107.6538239, -260.3462830, 261.9219971
9: -115.1690903, 113.6501389, -118.7855835, 117.2719879, -232.4410706, 232.4357147

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 213

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6121019, upper bound: 264.6121845
time: 8.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6121019, upper bound: 264.6123921
time: 9.19 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.43 seconds
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 18.43
Output dim: 7, lower bound: -264.6122875, upper bound: 264.6122917
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.43
Output dim: 7, lower bound: -264.6122875, upper bound: 264.6123131
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.43
Output dim: 7, lower bound: -264.6122810, upper bound: 264.6123290
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.43
Output dim: 7, lower bound: -264.6122810, upper bound: 264.6125523
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.43
Output dim: 7, lower bound: -264.6121019, upper bound: 264.6121845
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.43
Output dim: 7, lower bound: -264.6121019, upper bound: 264.6123921

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -119.8562241, 95.5586090, -130.9310608, 104.4161072, -224.2723236, 226.4896393
1: -100.2327271, 84.6960678, -109.4719543, 92.4104843, -192.6432037, 194.1679993
2: -131.8569031, 85.7825775, -144.0916443, 93.5942535, -225.4511566, 229.8742218
3: -140.4355011, 74.9286118, -153.3974152, 81.7066040, -222.1421051, 228.3260193
4: -128.4861755, 98.5839615, -140.3726654, 107.6032867, -236.0894470, 238.9566193
5: -115.5788879, 89.6973877, -126.2659531, 97.8976135, -213.4765015, 215.9633484
6: -110.5011597, 106.7532349, -120.6791687, 116.5782013, -227.0793610, 227.4324036
7: -120.1245956, 101.0351028, -131.1860199, 110.2862167, -230.4108124, 232.2210999
8: -144.8708954, 99.0115280, -158.3633575, 108.0889893, -252.9598694, 257.3748169
9: -109.2514725, 107.9228287, -119.2470856, 117.7379990, -226.9894714, 227.1699066

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6115779, upper bound: 264.6115481
time: 8.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6120645, upper bound: 264.6120828
time: 8.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -124.2052612, 99.0205765, -125.7971420, 100.3302536, -224.5354919, 224.8177185
1: -103.8689499, 87.7497864, -105.1803284, 88.8023453, -192.6712952, 192.9300995
2: -136.6713867, 88.8617020, -138.4045563, 89.9588852, -226.6302643, 227.2662659
3: -145.5756683, 77.5915222, -147.3327637, 78.5654221, -224.1410828, 224.9242706
4: -133.1809082, 102.1476974, -134.8309021, 103.3951721, -236.5760803, 236.9786072
5: -119.7959518, 92.9353867, -121.3001328, 94.0745239, -213.8704681, 214.2355194
6: -114.5099411, 110.6142426, -115.9511948, 112.0205002, -226.5304260, 226.5654144
7: -124.5131073, 104.6837692, -126.0042419, 105.9795914, -230.4926910, 230.6880188
8: -150.1608429, 102.5352325, -152.1228943, 103.9192963, -254.0801392, 254.6581116
9: -113.2240372, 111.7785339, -114.5591278, 113.1809921, -226.4050293, 226.3376312

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6115707, upper bound: 264.6116128
time: 8.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6120535, upper bound: 264.6121505
time: 8.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -124.2052612, 99.0205765, -130.9310608, 104.4161072, -228.6213684, 229.9516296
1: -103.8689499, 87.7497864, -109.4719543, 92.4104843, -196.2794342, 197.2217102
2: -136.6713867, 88.8617020, -144.0916443, 93.5942535, -230.2656403, 232.9533386
3: -145.5756683, 77.5915222, -153.3974152, 81.7066040, -227.2822723, 230.9889221
4: -133.1809082, 102.1476974, -140.3726654, 107.6032867, -240.7841644, 242.5203552
5: -119.7959518, 92.9353867, -126.2659531, 97.8976135, -217.6935730, 219.2013397
6: -114.5099411, 110.6142426, -120.6791687, 116.5782013, -231.0881348, 231.2934113
7: -124.5131073, 104.6837692, -131.1860199, 110.2862167, -234.7992859, 235.8697662
8: -150.1608429, 102.5352325, -158.3633575, 108.0889893, -258.2498169, 260.8985901
9: -113.2240372, 111.7785339, -119.2470856, 117.7379990, -230.9620361, 231.0255890

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6115707, upper bound: 264.6119293
time: 8.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6120535, upper bound: 264.6124051
time: 8.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -126.2991791, 100.6304855, -129.1711578, 103.0074387, -229.3066101, 229.8016205
1: -105.6309891, 89.2669601, -107.9875259, 91.1923599, -196.8233337, 197.2544708
2: -138.9960022, 90.3215408, -142.1612854, 92.3391495, -231.3351440, 232.4828186
3: -148.0853271, 78.8713989, -151.3568726, 80.6171951, -228.7025146, 230.2282715
4: -135.4845428, 103.8733063, -138.5061188, 106.1651230, -241.6496582, 242.3794250
5: -121.8183899, 94.4903793, -124.5793610, 96.5966721, -218.4150391, 219.0697327
6: -116.4331360, 112.4850159, -119.0720062, 115.0198517, -231.4529877, 231.5570221
7: -126.6623611, 106.4652405, -129.4492035, 108.8226929, -235.4850311, 235.9144287
8: -152.6924896, 104.2021942, -156.2399139, 106.6248474, -259.3172607, 260.4420471
9: -115.1690903, 113.6501389, -117.6705933, 116.1567154, -231.3258057, 231.3206940

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6113945, upper bound: 264.6117784
time: 9.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6118816, upper bound: 264.6122554
time: 9.05 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 19.59 seconds
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 19.59
Output dim: 7, lower bound: -264.6115779, upper bound: 264.6115481
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 19.59
Output dim: 7, lower bound: -264.6120645, upper bound: 264.6120828
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 19.59
Output dim: 7, lower bound: -264.6115707, upper bound: 264.6116128
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 19.59
Output dim: 7, lower bound: -264.6120535, upper bound: 264.6121505
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 19.59
Output dim: 7, lower bound: -264.6115707, upper bound: 264.6119293
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 19.59
Output dim: 7, lower bound: -264.6120535, upper bound: 264.6124051
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 19.59
Output dim: 7, lower bound: -264.6113945, upper bound: 264.6117784
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 19.59
Output dim: 7, lower bound: -264.6118816, upper bound: 264.6122554
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=266.34649658203125
rel_dist={7: [-264.62266420921173, 264.6226641973501]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6204928, upper bound: 264.6204989
time: 10.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6204874, upper bound: 264.6204874
time: 10.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 21.00 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 21.00
Output dim: 7, lower bound: -264.6204928, upper bound: 264.6204989
IS_A2, status: Status.UNKNOWN, split count: 1, time: 21.00
Output dim: 7, lower bound: -264.6204874, upper bound: 264.6204874

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -137.8412933, 109.8958969, -141.8700256, 113.0970154, -250.9382782, 251.7659149
1: -115.3550110, 97.4181595, -118.7306442, 100.2537842, -215.6087799, 216.1488037
2: -151.7102203, 98.6460876, -156.1749420, 101.5017471, -253.2119751, 254.8210297
3: -161.5674744, 86.2000275, -166.3176727, 88.6648026, -250.2322693, 252.5177002
4: -147.7318878, 113.3872681, -152.0771027, 116.6947098, -264.4265442, 265.4643555
5: -132.8794250, 103.1506424, -136.7808685, 106.1549301, -239.0343475, 239.9315033
6: -127.0270538, 122.7826920, -130.7394257, 126.3617325, -253.3887939, 253.5221100
7: -138.1151123, 116.1337585, -142.1844635, 119.5232162, -257.6383362, 258.3182068
8: -166.6094513, 114.0110855, -171.5119324, 117.2887268, -283.8981934, 285.5229797
9: -125.5774002, 124.1473160, -129.2500610, 127.7262192, -253.3035736, 253.3973541

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6146110, upper bound: 264.6146031
time: 14.85 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153783, upper bound: 264.6153596
time: 9.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -142.3042603, 113.4523087, -142.8218384, 113.8571777, -256.1614380, 256.2741089
1: -119.0911560, 100.5554810, -119.5271835, 100.9204941, -220.0116577, 220.0826721
2: -156.6529388, 101.8062897, -157.2268219, 102.1790390, -258.8318481, 259.0330811
3: -166.8479462, 88.9353714, -167.4423065, 89.2542114, -256.1021729, 256.3776855
4: -152.5530548, 117.0456924, -153.1104736, 117.4736099, -270.0266724, 270.1561584
5: -137.2094727, 106.4746170, -137.7011871, 106.8624878, -244.0719604, 244.1757965
6: -131.1432648, 126.7484055, -131.6179504, 127.2080688, -258.3512878, 258.3663330
7: -142.6210785, 119.8799820, -143.1468811, 120.3228836, -262.9439087, 263.0267944
8: -172.0416260, 117.6300964, -172.6636658, 118.0637360, -290.1053467, 290.2937622
9: -129.6566162, 128.1073914, -130.1250000, 128.5767822, -258.2333984, 258.2323608

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6146010, upper bound: 264.6146075
time: 11.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6154069, upper bound: 264.6154069
time: 11.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.83 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.83
Output dim: 7, lower bound: -264.6146110, upper bound: 264.6146031
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.83
Output dim: 7, lower bound: -264.6153783, upper bound: 264.6153596
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.83
Output dim: 7, lower bound: -264.6146010, upper bound: 264.6146075
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.83
Output dim: 7, lower bound: -264.6154069, upper bound: 264.6154069

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -130.9954376, 104.4827728, -132.8746796, 105.9856491, -236.9810791, 237.3574219
1: -109.5478897, 92.5193558, -111.0998459, 93.8166885, -203.3645782, 203.6192017
2: -144.1445618, 93.7357941, -146.2335815, 95.0475006, -239.1920624, 239.9693756
3: -153.4769440, 81.8576126, -155.6913452, 82.9571915, -236.4341431, 237.5489502
4: -140.3791199, 107.7239075, -142.4171753, 109.2531815, -249.6322784, 250.1410370
5: -126.3115921, 97.9900665, -128.1532745, 99.3726959, -225.6842651, 226.1433411
6: -120.7509079, 116.6611786, -122.4922867, 118.3189468, -239.0698547, 239.1534576
7: -131.2301636, 110.3469696, -133.1377563, 111.9193115, -243.1494751, 243.4847260
8: -158.3950348, 108.3162613, -160.7196503, 109.8020706, -268.1970825, 269.0359192
9: -119.2948914, 117.9148712, -120.9952850, 119.5337448, -238.8286285, 238.9101562

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6141347, upper bound: 264.6141504
time: 10.72 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6142449, upper bound: 264.6142823
time: 10.68 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -130.7225037, 104.2610474, -136.7925873, 109.0848312, -239.8072968, 241.0535889
1: -109.3285828, 92.3298416, -114.3939362, 96.5690765, -205.8976593, 206.7237701
2: -143.8455200, 93.5379257, -150.5544434, 97.8005600, -241.6460876, 244.0923615
3: -153.1558533, 81.6958618, -160.2826843, 85.3571014, -238.5129547, 241.9785461
4: -140.0988617, 107.5005112, -146.6465607, 112.4472275, -252.5460815, 254.1470642
5: -126.0388870, 97.7962875, -131.8977051, 102.2965012, -228.3353882, 229.6940002
6: -120.4994431, 116.4241257, -126.0815430, 121.7904663, -242.2899170, 242.5056763
7: -130.9595642, 110.1289749, -137.0615845, 115.2158279, -246.1753845, 247.1905518
8: -158.0730591, 108.0932846, -165.4536591, 112.9959259, -271.0689697, 273.5469360
9: -119.0575867, 117.6858215, -124.5705338, 123.0348434, -242.0924072, 242.2563477

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6141347, upper bound: 264.6147722
time: 11.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6149993, upper bound: 264.6150155
time: 11.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -135.4895020, 108.0663300, -133.7403564, 106.6789246, -242.1684265, 241.8066711
1: -113.3110504, 95.6791306, -111.8244781, 94.4220886, -207.7331390, 207.5035858
2: -149.1206360, 96.9175110, -147.1888580, 95.6633987, -244.7840271, 244.1063690
3: -158.7984619, 84.6121979, -156.7166290, 83.4934921, -242.2919617, 241.3288269
4: -145.2350159, 111.4080811, -143.3579712, 109.9610367, -255.1960449, 254.7660522
5: -130.6741791, 101.3372498, -128.9919281, 100.0158234, -230.6900024, 230.3291779
6: -124.8961868, 120.6548920, -123.2923050, 119.0880814, -243.9842682, 243.9472046
7: -135.7664642, 114.1185303, -134.0118408, 112.6455383, -248.4120026, 248.1303711
8: -163.8661041, 111.9592056, -161.7678680, 110.5054779, -274.3715820, 273.7270813
9: -123.4035034, 121.9019012, -121.7921143, 120.3069000, -243.7103729, 243.6940155

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6141300, upper bound: 264.6141317
time: 11.82 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6142419, upper bound: 264.6142665
time: 12.19 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -135.2216339, 107.8437805, -137.8717194, 109.9435654, -245.1651917, 245.7154999
1: -113.0933304, 95.4918060, -115.2965164, 97.3257751, -210.4190979, 210.7883148
2: -148.8296509, 96.7248764, -151.7489319, 98.5689850, -247.3986359, 248.4738007
3: -158.4725952, 84.4522171, -161.5521698, 86.0243988, -244.4969940, 246.0043945
4: -144.9582062, 111.1881866, -147.8168030, 113.3301392, -258.2882690, 259.0050049
5: -130.3995514, 101.1485672, -132.9374390, 103.1003494, -233.4999084, 234.0859985
6: -124.6473541, 120.4211349, -127.0755463, 122.7502670, -247.3976135, 247.4966736
7: -135.5033569, 113.9066010, -138.1539764, 116.1236725, -251.6270294, 252.0605774
8: -163.5471649, 111.7440643, -166.7582092, 113.8767471, -277.4239197, 278.5022583
9: -123.1676483, 121.6787491, -125.5600357, 123.9996948, -247.1673431, 247.2387695

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6147786, upper bound: 264.6147477
time: 13.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6149928, upper bound: 264.6149928
time: 9.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.75 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 7, lower bound: -264.6141347, upper bound: 264.6141504
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 7, lower bound: -264.6142449, upper bound: 264.6142823
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 7, lower bound: -264.6141347, upper bound: 264.6147722
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 7, lower bound: -264.6149993, upper bound: 264.6150155
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 7, lower bound: -264.6141300, upper bound: 264.6141317
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 7, lower bound: -264.6142419, upper bound: 264.6142665
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 7, lower bound: -264.6147786, upper bound: 264.6147477
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.75
Output dim: 7, lower bound: -264.6149928, upper bound: 264.6149928

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -112.4649200, 89.7143784, -119.6256943, 95.4126892, -207.8776093, 209.3400574
1: -94.1315994, 79.4529724, -100.0730438, 84.4581757, -178.5897827, 179.5259857
2: -123.8047485, 80.5733719, -131.6798553, 85.6315994, -209.4363403, 212.2532349
3: -131.7894592, 70.4656372, -140.1688385, 74.8016586, -206.5910950, 210.6344604
4: -120.5974884, 92.5412598, -128.2790833, 98.3944702, -218.9919586, 220.8203278
5: -108.4363022, 84.1613846, -115.3619995, 89.4773407, -197.9136200, 199.5233765
6: -103.6458054, 100.2742310, -110.2563705, 106.5971069, -210.2428741, 210.5305786
7: -112.7181091, 94.9138031, -119.8988800, 100.8823700, -213.6004791, 214.8126678
8: -136.0589142, 93.0345154, -144.7413330, 98.8603516, -234.9192352, 237.7758484
9: -102.4763489, 101.3568192, -108.9670868, 107.6920700, -210.1684113, 210.3239136

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6015444, upper bound: 264.6014762
time: 11.78 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6127289, upper bound: 264.6127544
time: 9.20 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -124.4360275, 99.2550278, -127.8750458, 101.9986115, -226.4346313, 227.1300659
1: -104.0576172, 87.8794022, -106.9137268, 90.2789078, -194.3365173, 194.7931061
2: -136.9269409, 89.0770035, -140.7301331, 91.4970093, -228.4239197, 229.8071289
3: -145.7810974, 77.8057327, -149.8223877, 79.8690109, -225.6501007, 227.6280975
4: -133.3460541, 102.3456650, -137.0536957, 105.1530075, -238.4990540, 239.3993530
5: -119.9884872, 93.0865097, -123.3321991, 95.6347351, -215.6231842, 216.4187012
6: -114.6879654, 110.8492813, -117.8706131, 113.8863068, -228.5742493, 228.7198639
7: -124.6614456, 104.8727570, -128.1296844, 107.7459106, -232.4073486, 233.0024261
8: -150.4596710, 102.8856659, -154.6688843, 105.6619110, -256.1215820, 257.5545044
9: -113.3283463, 112.0396652, -116.4450378, 115.0546417, -228.3829803, 228.4847107

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6023077, upper bound: 264.6023760
time: 13.65 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6128195, upper bound: 264.6128619
time: 11.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -112.3029099, 89.5792618, -123.6282501, 98.5781631, -210.8810730, 213.2075195
1: -94.0047913, 79.3385620, -103.4356537, 87.2676620, -181.2724304, 182.7742157
2: -123.6264801, 80.4543991, -136.0920715, 88.4443588, -212.0708313, 216.5464783
3: -131.5919037, 70.3715134, -144.8507385, 77.2533951, -208.8453064, 215.2222443
4: -120.4394608, 92.4086456, -132.6013489, 101.6573792, -222.0968323, 225.0099945
5: -108.2692108, 84.0558167, -119.1853027, 92.4696503, -200.7388611, 203.2410889
6: -103.4923096, 100.1345215, -113.9160919, 110.1425552, -213.6348572, 214.0506134
7: -112.5599518, 94.7916031, -123.9079590, 104.2523422, -216.8122864, 218.6995392
8: -135.8703003, 92.9053497, -149.5754547, 102.1275558, -237.9978638, 242.4807892
9: -102.3461380, 101.2319107, -112.6238022, 111.2733917, -213.6195374, 213.8557129

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6017951, upper bound: 264.6017097
time: 11.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6133255, upper bound: 264.6132946
time: 13.24 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -124.1799698, 99.0474167, -131.7997131, 105.1039734, -229.2839355, 230.8471375
1: -103.8515015, 87.7010574, -110.2119217, 93.0355377, -196.8870239, 197.9129791
2: -136.6447144, 88.8901596, -145.0565491, 94.2534561, -230.8981628, 233.9467163
3: -145.4795380, 77.6531601, -154.4215393, 82.2716370, -227.7511749, 232.0746765
4: -133.0814819, 102.1348343, -141.2874451, 108.3516006, -241.4330750, 243.4222717
5: -119.7335968, 92.9032135, -127.0839539, 98.5619965, -218.2955933, 219.9871674
6: -114.4537430, 110.6250992, -121.4665375, 117.3620682, -231.8158112, 232.0916443
7: -124.4058304, 104.6660614, -132.0589294, 111.0446396, -235.4504700, 236.7249908
8: -150.1560669, 102.6753387, -159.4096680, 108.8592529, -259.0153198, 262.0849915
9: -113.1023636, 111.8230743, -120.0233231, 118.5589600, -231.6613159, 231.8464050

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6025459, upper bound: 264.6025874
time: 12.17 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6135659, upper bound: 264.6135729
time: 11.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -116.8903427, 93.2371597, -120.4915237, 96.1057968, -212.9961395, 213.7286377
1: -97.8339233, 82.5600357, -100.7977371, 85.0638275, -182.8977356, 183.3577271
2: -128.6988373, 83.7074280, -132.6359100, 86.2475510, -214.9463501, 216.3433380
3: -137.0176849, 73.1799088, -141.1934357, 75.3384857, -212.3561707, 214.3733521
4: -125.3703766, 96.1649857, -129.2206268, 99.1021271, -224.4724884, 225.3856049
5: -112.7303085, 87.4598312, -116.2003021, 90.1212387, -202.8515320, 203.6601257
6: -107.7259216, 104.1977463, -111.0565338, 107.3663635, -215.0922852, 215.2542725
7: -117.1820602, 98.6275711, -120.7737045, 101.6091461, -218.7911682, 219.4012604
8: -141.4377136, 96.6138458, -145.7899628, 99.5642090, -241.0019226, 242.4038086
9: -106.5139771, 105.2795792, -109.7644806, 108.4657898, -214.9797668, 215.0440674

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6015532, upper bound: 264.6014717
time: 12.92 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6127290, upper bound: 264.6127377
time: 11.15 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -128.7952271, 102.7281570, -128.7269745, 102.6809158, -231.4761200, 231.4551392
1: -107.7060547, 90.9431992, -107.6266174, 90.8745117, -198.5805359, 198.5697937
2: -141.7521820, 92.1637650, -141.6704559, 92.1032181, -233.8554077, 233.8341980
3: -150.9413605, 80.4775848, -150.8313141, 80.3963928, -231.3377533, 231.3088989
4: -138.0531158, 105.9185715, -137.9796906, 105.8498001, -243.9029083, 243.8982239
5: -124.2194824, 96.3321686, -124.1576614, 96.2675171, -220.4869995, 220.4898376
6: -118.7082367, 114.7202301, -118.6579208, 114.6431885, -233.3514252, 233.3781433
7: -129.0605927, 108.5303421, -128.9903259, 108.4604416, -237.5210266, 237.5206604
8: -155.7642975, 106.4161453, -155.7004395, 106.3539505, -262.1181641, 262.1165771
9: -117.3118362, 115.9049377, -117.2295456, 115.8153076, -233.1271362, 233.1344910

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6023098, upper bound: 264.6023671
time: 14.42 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6128210, upper bound: 264.6128501
time: 12.11 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -116.7421722, 93.1092529, -124.7157974, 99.4435349, -216.1856842, 217.8250427
1: -97.7162857, 82.4550171, -104.3451996, 88.0310745, -185.7473450, 186.8002167
2: -128.5411072, 83.6020203, -137.2980652, 89.2195053, -217.7606049, 220.9000854
3: -136.8261719, 73.0934219, -146.1292419, 77.9267502, -214.7529297, 219.2226562
4: -125.2261047, 96.0431442, -133.7817841, 102.5471725, -227.7732544, 229.8249207
5: -112.5697021, 87.3659821, -120.2321091, 93.2810516, -205.8507538, 207.5980530
6: -107.5835876, 104.0712662, -114.9181442, 111.1108017, -218.6943665, 218.9894104
7: -117.0425873, 98.5203552, -125.0114136, 105.1681519, -222.2107086, 223.5317535
8: -141.2654266, 96.5010071, -150.8916168, 103.0158234, -244.2812500, 247.3925781
9: -106.3932190, 105.1706772, -113.6213913, 112.2471771, -218.6403961, 218.7920685

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6017998, upper bound: 264.6017266
time: 12.36 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6133388, upper bound: 264.6132891
time: 10.44 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -128.5479126, 102.5229568, -132.8645935, 105.9513550, -234.4992676, 235.3875427
1: -107.5042648, 90.7698746, -111.1023178, 93.7821655, -201.2864380, 201.8721924
2: -141.4822083, 91.9853134, -146.2354431, 95.0120316, -236.4942322, 238.2207336
3: -150.6401062, 80.3291016, -155.6746368, 82.9300842, -233.5701752, 236.0037079
4: -137.7964935, 105.7142487, -142.4426880, 109.2229309, -247.0194244, 248.1569214
5: -123.9655380, 96.1568451, -128.1097717, 99.3551254, -223.3206635, 224.2666016
6: -118.4803772, 114.5031204, -122.4476166, 118.3093719, -236.7897339, 236.9507446
7: -128.8163147, 108.3328781, -133.1371002, 111.9403763, -240.7566833, 241.4699707
8: -155.4688110, 106.2167511, -160.6966400, 109.7283859, -265.1971741, 266.9133911
9: -117.0907593, 115.6980743, -120.9999390, 119.5111542, -236.6019135, 236.6979980

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6025448, upper bound: 264.6025908
time: 12.79 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6135775, upper bound: 264.6135775
time: 11.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.91 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6015444, upper bound: 264.6014762
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6127289, upper bound: 264.6127544
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6023077, upper bound: 264.6023760
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6128195, upper bound: 264.6128619
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6017951, upper bound: 264.6017097
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6133255, upper bound: 264.6132946
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6025459, upper bound: 264.6025874
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6135659, upper bound: 264.6135729
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6015532, upper bound: 264.6014717
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6127290, upper bound: 264.6127377
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6023098, upper bound: 264.6023671
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6128210, upper bound: 264.6128501
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6017998, upper bound: 264.6017266
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6133388, upper bound: 264.6132891
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6025448, upper bound: 264.6025908
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.91
Output dim: 7, lower bound: -264.6135775, upper bound: 264.6135775

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -111.4059982, 88.8738785, -118.0287247, 94.1452942, -205.5513000, 206.9025726
1: -93.2450104, 78.6931992, -98.7358170, 83.3129120, -176.5579224, 177.4290161
2: -122.6398392, 79.8060303, -129.9236145, 84.4749222, -207.1147308, 209.7296448
3: -130.5292816, 69.8143692, -138.2693481, 73.8199615, -204.3492432, 208.0836945
4: -119.4545898, 91.6562958, -126.5558548, 97.0600510, -216.5146484, 218.2121429
5: -107.4181213, 83.3580322, -113.8262024, 88.2656174, -195.6837158, 197.1842346
6: -102.6610260, 99.3376923, -108.7716064, 105.1849213, -207.8459473, 208.1092987
7: -111.6412506, 94.0193634, -118.2757339, 99.5339737, -211.1752319, 212.2951050
8: -134.7819214, 92.1559372, -142.8160553, 97.5356827, -232.3175964, 234.9719696
9: -101.4972763, 100.3967133, -107.4915924, 106.2446213, -207.7418976, 207.8882751

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6108637, upper bound: 264.6108021
time: 13.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6107767, upper bound: 264.6107175
time: 9.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -123.3693390, 98.4090424, -126.2544937, 100.7134399, -224.0827789, 224.6635437
1: -103.1635361, 87.1136093, -105.5558090, 89.1159286, -192.2794647, 192.6693878
2: -135.7549133, 88.3052063, -138.9498444, 90.3247375, -226.0796509, 227.2550201
3: -144.5119934, 77.1500549, -147.8952789, 78.8733215, -223.3853149, 225.0453339
4: -132.1949921, 101.4552917, -135.3054504, 103.8006973, -235.9956665, 236.7607422
5: -118.9632111, 92.2774887, -121.7744827, 94.4057846, -213.3689880, 214.0519562
6: -113.6971359, 109.9062195, -116.3655014, 112.4540329, -226.1511688, 226.2717285
7: -123.5778427, 103.9727325, -126.4839859, 106.3791428, -229.9569855, 230.4567261
8: -149.1754150, 102.0015411, -152.7179565, 104.3191528, -253.4945679, 254.7194977
9: -112.3435059, 111.0737762, -114.9493713, 113.5877151, -225.9311829, 226.0231323

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6110112, upper bound: 264.6109725
time: 10.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6109132, upper bound: 264.6108929
time: 11.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -111.2600937, 88.7513580, -122.0534439, 97.3290253, -208.5890961, 210.8048096
1: -93.1311798, 78.5904617, -102.1168823, 86.1390533, -179.2702179, 180.7073364
2: -122.4793015, 79.6989670, -134.3597412, 87.3048553, -209.7841492, 214.0587158
3: -130.3507080, 69.7298279, -142.9783478, 76.2847977, -206.6354980, 212.7081604
4: -119.3135834, 91.5369415, -130.9015198, 100.3420486, -219.6556091, 222.4384613
5: -107.2662964, 83.2646179, -117.6714020, 91.2758560, -198.5421448, 200.9360046
6: -102.5225601, 99.2120514, -112.4520721, 108.7500076, -211.2725372, 211.6641083
7: -111.4996109, 93.9105072, -122.3084946, 102.9227905, -214.4223938, 216.2189941
8: -134.6126251, 92.0397186, -147.6761322, 100.8209991, -235.4336243, 239.7158356
9: -101.3816452, 100.2861710, -111.1687241, 109.8462219, -211.2278595, 211.4548492

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6113892, upper bound: 264.6113088
time: 10.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6113095, upper bound: 264.6112257
time: 11.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -123.1277084, 98.2127838, -130.1933594, 103.8304749, -226.9581604, 228.4061432
1: -102.9692688, 86.9456635, -108.8661499, 91.8835297, -194.8527985, 195.8117981
2: -135.4884949, 88.1287079, -143.2915192, 93.0926208, -228.5811157, 231.4202271
3: -144.2276154, 77.0061493, -152.5128021, 81.2841492, -225.5117493, 229.5189514
4: -131.9459534, 101.2562561, -139.5547180, 107.0119171, -238.9578705, 240.8109589
5: -118.7221298, 92.1051254, -125.5398712, 97.3451462, -216.0672455, 217.6449738
6: -113.4762192, 109.6946716, -119.9748077, 115.9422302, -229.4184570, 229.6694489
7: -123.3367538, 103.7779922, -130.4288635, 109.6900253, -233.0267487, 234.2068481
8: -148.8891449, 101.8030319, -157.4755402, 107.5280457, -256.4171753, 259.2785645
9: -112.1306229, 110.8701324, -118.5409622, 117.1055527, -229.2361755, 229.4110870

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6117472, upper bound: 264.6116655
time: 12.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6116540, upper bound: 264.6115892
time: 10.13 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -115.8462296, 92.4079132, -118.9115524, 94.8514938, -210.6977234, 211.3194580
1: -96.9595566, 81.8109512, -99.4744034, 83.9304581, -180.8899689, 181.2853546
2: -127.5507050, 82.9512634, -130.8985748, 85.1032333, -212.6539307, 213.8498383
3: -135.7748260, 72.5375519, -139.3132324, 74.3670731, -210.1419067, 211.8507385
4: -124.2437439, 95.2922058, -127.5157318, 97.7813034, -222.0250549, 222.8079376
5: -111.7253647, 86.6675491, -114.6802139, 88.9220047, -200.6473694, 201.3477478
6: -106.7547379, 103.2744980, -109.5871887, 105.9692001, -212.7239380, 212.8616638
7: -116.1211472, 97.7457352, -119.1679153, 100.2747879, -216.3959351, 216.9136505
8: -140.1786652, 95.7474060, -143.8851471, 98.2534866, -238.4321442, 239.6325531
9: -105.5487442, 104.3328629, -108.3041687, 107.0335312, -212.5822754, 212.6370239

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6109330, upper bound: 264.6109058
time: 11.28 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6108628, upper bound: 264.6108406
time: 11.26 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -127.7370148, 101.8887558, -127.1244507, 101.4096909, -229.1466980, 229.0131531
1: -106.8191147, 90.1836014, -106.2834778, 89.7240448, -196.5431519, 196.4670715
2: -140.5899353, 91.3984680, -139.9101868, 90.9439240, -231.5338593, 231.3086548
3: -149.6824188, 79.8273392, -148.9246674, 79.4115906, -229.0940094, 228.7519684
4: -136.9116516, 105.0352173, -136.2509613, 104.5119019, -241.4235535, 241.2861786
5: -123.2019272, 95.5295029, -122.6166916, 95.0517883, -218.2537231, 218.1461945
6: -117.7253036, 113.7849655, -117.1691742, 113.2267990, -230.9521027, 230.9541321
7: -127.9860687, 107.6377869, -127.3629227, 107.1086044, -235.0946655, 235.0007019
8: -154.4904480, 105.5393295, -153.7711639, 105.0259781, -259.5163574, 259.3104858
9: -116.3349380, 114.9470673, -115.7500000, 114.3644791, -230.6994171, 230.6970520

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6110662, upper bound: 264.6110706
time: 11.59 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6109967, upper bound: 264.6110274
time: 12.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -115.7132263, 92.2920532, -123.1604614, 98.2096252, -213.9228363, 215.4525146
1: -96.8543015, 81.7168427, -103.0426483, 86.9159851, -183.7702942, 184.7594910
2: -127.4095230, 82.8567581, -135.5873413, 88.0940628, -215.5035706, 218.4440918
3: -135.6013184, 72.4603424, -144.2791138, 76.9701691, -212.5714874, 216.7394562
4: -124.1153564, 95.1829987, -132.1031036, 101.2477036, -225.3630676, 227.2861023
5: -111.5795212, 86.5850906, -118.7364731, 92.1016922, -203.6812134, 205.3215485
6: -106.6265411, 103.1612396, -113.4719162, 109.7354889, -216.3620300, 216.6331482
7: -115.9968262, 97.6510162, -123.4316483, 103.8547821, -219.8516083, 221.0826416
8: -140.0243835, 95.6467285, -149.0157166, 101.7253723, -241.7497406, 244.6624451
9: -105.4416885, 104.2374954, -112.1839294, 110.8374557, -216.2791443, 216.4214172

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6116501, upper bound: 264.6116202
time: 12.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6115818, upper bound: 264.6115474
time: 12.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -127.5048828, 101.6956558, -131.2762756, 104.6920013, -232.1968842, 232.9718781
1: -106.6298065, 90.0212479, -109.7715836, 92.6428223, -199.2726135, 199.7928162
2: -140.3367310, 91.2310104, -144.4906769, 93.8643265, -234.2010498, 235.7216797
3: -149.3990784, 79.6879501, -153.7865601, 81.9536591, -231.3527222, 233.4745178
4: -136.6712952, 104.8434677, -140.7296753, 107.8979645, -244.5692596, 245.5731506
5: -122.9624863, 95.3658295, -126.5826416, 98.1517105, -221.1141968, 221.9484406
6: -117.5113754, 113.5812073, -120.9724045, 116.9056244, -234.4169922, 234.5536194
7: -127.7573166, 107.4529800, -131.5254669, 110.6008835, -238.3582001, 238.9784241
8: -154.2132721, 105.3523178, -158.7844391, 108.4121933, -262.6254578, 264.1367493
9: -116.1277618, 114.7539597, -119.5339661, 118.0740891, -234.2018433, 234.2879333

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6120962, upper bound: 264.6120850
time: 11.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6120263, upper bound: 264.6120263
time: 11.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.29 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6108637, upper bound: 264.6108021
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6107767, upper bound: 264.6107175
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6110112, upper bound: 264.6109725
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6109132, upper bound: 264.6108929
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6113892, upper bound: 264.6113088
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6113095, upper bound: 264.6112257
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6117472, upper bound: 264.6116655
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6116540, upper bound: 264.6115892
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6109330, upper bound: 264.6109058
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6108628, upper bound: 264.6108406
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6110662, upper bound: 264.6110706
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6109967, upper bound: 264.6110274
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6116501, upper bound: 264.6116202
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6115818, upper bound: 264.6115474
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6120962, upper bound: 264.6120850
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 24.29
Output dim: 7, lower bound: -264.6120263, upper bound: 264.6120263
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=266.34649658203125
rel_dist={7: [-264.6222950859965, 264.6222950868074]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 213

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6208467, upper bound: 264.6208528
time: 9.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6208322, upper bound: 264.6208322
time: 8.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 18.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 18.61
Output dim: 7, lower bound: -264.6208467, upper bound: 264.6208528
IS_A2, status: Status.UNKNOWN, split count: 1, time: 18.61
Output dim: 7, lower bound: -264.6208322, upper bound: 264.6208322

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -137.8412933, 109.8958969, -142.4951935, 113.5900574, -251.4313507, 252.3910828
1: -115.3550110, 97.4181595, -119.2559280, 100.6934280, -216.0484314, 216.6740875
2: -151.7102203, 98.6460876, -156.8672028, 101.9498138, -253.6600189, 255.5132904
3: -161.5674744, 86.2000275, -167.0437164, 89.0506439, -250.6181183, 253.2437439
4: -147.7318878, 113.3872681, -152.7524109, 117.2094650, -264.9413452, 266.1396790
5: -132.8794250, 103.1506424, -137.3789215, 106.6219177, -239.5013428, 240.5295410
6: -127.0270538, 122.7826920, -131.3142242, 126.9171219, -253.9441833, 254.0969086
7: -138.1151123, 116.1337585, -142.8182983, 120.0542831, -258.1694031, 258.9520264
8: -166.6094513, 114.0110855, -172.2652130, 117.8066330, -284.4160767, 286.2763062
9: -125.5774002, 124.1473160, -129.8193512, 128.2898407, -253.8672333, 253.9666443

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6150567, upper bound: 264.6150479
time: 8.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157589, upper bound: 264.6157344
time: 9.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -142.3042603, 113.4523087, -143.2448425, 114.1881027, -256.4923401, 256.6970825
1: -119.0911560, 100.5554810, -119.8833847, 101.2187347, -220.3098907, 220.4388733
2: -156.6529388, 101.8062897, -157.6958466, 102.4834442, -259.1362915, 259.5021362
3: -166.8479462, 88.9353714, -167.9280701, 89.5147400, -256.3626709, 256.8634338
4: -152.5530548, 117.0456924, -153.5660400, 117.8232193, -270.3762817, 270.6117249
5: -137.2094727, 106.4746170, -138.1030579, 107.1794357, -244.3889160, 244.5776520
6: -131.1432648, 126.7484055, -132.0057220, 127.5837402, -258.7269897, 258.7541199
7: -142.6210785, 119.8799820, -143.5764008, 120.6846848, -263.3056946, 263.4562378
8: -172.0416260, 117.6300964, -173.1719818, 118.4177475, -290.4593811, 290.8020325
9: -129.6566162, 128.1073914, -130.5077820, 128.9601746, -258.6167603, 258.6151733

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 163

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6150424, upper bound: 264.6150601
time: 8.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6157711, upper bound: 264.6157711
time: 9.20 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.19 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.19
Output dim: 7, lower bound: -264.6150567, upper bound: 264.6150479
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.19
Output dim: 7, lower bound: -264.6157589, upper bound: 264.6157344
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.19
Output dim: 7, lower bound: -264.6150424, upper bound: 264.6150601
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.19
Output dim: 7, lower bound: -264.6157711, upper bound: 264.6157711

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -132.8679810, 105.9634781, -133.4868774, 106.4686050, -239.3365784, 239.4503479
1: -111.1358414, 93.8593979, -111.6140823, 94.2470779, -205.3829193, 205.4734802
2: -146.2139740, 95.0789108, -146.9113007, 95.4862518, -241.7002106, 241.9901581
3: -155.6892700, 83.0448456, -156.4025116, 83.3349152, -239.0241852, 239.4473572
4: -142.3905182, 109.2729034, -143.0786285, 109.7571945, -252.1477051, 252.3515015
5: -128.1077423, 99.4017944, -128.7388153, 99.8299942, -227.9377136, 228.1406097
6: -122.4676743, 118.3354111, -123.0552902, 118.8627090, -241.3303528, 241.3907013
7: -133.1138306, 111.9293518, -133.7583923, 112.4391785, -245.5530090, 245.6877441
8: -160.6423035, 109.8743744, -161.4573975, 110.3090820, -270.9513245, 271.3317871
9: -121.0132065, 119.6196899, -121.5527878, 120.0856781, -241.0988770, 241.1724854

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6145367, upper bound: 264.6145229
time: 9.53 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6146263, upper bound: 264.6146956
time: 9.08 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -132.4436646, 105.6237030, -137.4418793, 109.5966949, -242.0403595, 243.0655823
1: -110.7853775, 93.5597153, -114.9391556, 97.0256577, -207.8110352, 208.4988708
2: -145.7468262, 94.7729416, -151.2736816, 98.2654724, -244.0122986, 246.0466309
3: -155.1885681, 82.7845917, -161.0368195, 85.7574463, -240.9460144, 243.8213959
4: -141.9443817, 108.9239807, -147.3478699, 112.9816284, -254.9260101, 256.2718506
5: -127.6923447, 99.0909653, -132.5184937, 102.7814941, -230.4738464, 231.6094666
6: -122.0776138, 117.9611816, -126.6782303, 122.3672485, -244.4448547, 244.6393890
7: -132.6894684, 111.5802612, -137.7200623, 115.7671127, -248.4565582, 249.3003235
8: -160.1368103, 109.5246048, -166.2360077, 113.5334320, -273.6701965, 275.7605591
9: -120.6336975, 119.2479095, -125.1617432, 123.6199188, -244.2536011, 244.4096527

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6151526, upper bound: 264.6151154
time: 8.25 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6153018, upper bound: 264.6153353
time: 9.76 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -137.3573456, 109.5426254, -134.1711578, 107.0157928, -244.3731384, 243.7137756
1: -114.8949966, 97.0156784, -112.1869202, 94.7257767, -209.6207733, 209.2025604
2: -151.1852112, 98.2575760, -147.6665649, 95.9732513, -247.1584320, 245.9241333
3: -161.0041199, 85.7965317, -157.2113190, 83.7586441, -244.7627563, 243.0078430
4: -147.2408905, 112.9531937, -143.8219757, 110.3170013, -257.5578308, 256.7751770
5: -132.4652557, 102.7454071, -129.4009247, 100.3385315, -232.8037872, 232.1463318
6: -126.6087036, 122.3248672, -123.6871719, 119.4707413, -246.0794373, 246.0120392
7: -137.6455231, 115.6973114, -134.4494476, 113.0138168, -250.6593323, 250.1467590
8: -166.1073761, 113.5137634, -162.2855530, 110.8658524, -276.9732361, 275.7993164
9: -125.1172180, 123.6028366, -122.1818771, 120.6972198, -245.8144379, 245.7846832

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6145213, upper bound: 264.6144906
time: 8.53 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6146199, upper bound: 264.6146536
time: 10.00 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -136.9354706, 109.2012177, -138.2852783, 110.2670670, -247.2024994, 247.4864655
1: -114.5442886, 96.7166748, -115.6443939, 97.6170731, -212.1613617, 212.3610687
2: -150.7223816, 97.9541550, -152.2072296, 98.8660965, -249.5884705, 250.1613770
3: -160.4984436, 85.5365753, -162.0275726, 86.2786484, -246.7770996, 247.5641479
4: -146.7960358, 112.6057663, -148.2622681, 113.6719360, -260.4678955, 260.8680420
5: -132.0471802, 102.4372406, -133.3302917, 103.4098282, -235.4570007, 235.7675171
6: -126.2189941, 121.9518967, -127.4546356, 123.1173630, -249.3363647, 249.4065247
7: -137.2255249, 115.3513718, -138.5737000, 116.4770050, -253.7025299, 253.9250488
8: -165.6025543, 113.1685486, -167.2552032, 114.2222443, -279.8247986, 280.4237366
9: -124.7374191, 123.2339783, -125.9343796, 124.3741684, -249.1115875, 249.1683655

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6151340, upper bound: 264.6150687
time: 9.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6152853, upper bound: 264.6152853
time: 9.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.95 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.95
Output dim: 7, lower bound: -264.6145367, upper bound: 264.6145229
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.95
Output dim: 7, lower bound: -264.6146263, upper bound: 264.6146956
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.95
Output dim: 7, lower bound: -264.6151526, upper bound: 264.6151154
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.95
Output dim: 7, lower bound: -264.6153018, upper bound: 264.6153353
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.95
Output dim: 7, lower bound: -264.6145213, upper bound: 264.6144906
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.95
Output dim: 7, lower bound: -264.6146199, upper bound: 264.6146536
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.95
Output dim: 7, lower bound: -264.6151340, upper bound: 264.6150687
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.95
Output dim: 7, lower bound: -264.6152853, upper bound: 264.6152853

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -114.3337936, 91.1917343, -123.3261490, 98.3595428, -212.6933136, 214.5178833
1: -95.7157822, 80.7893448, -103.1573944, 87.0693130, -182.7850952, 183.9467468
2: -125.8697662, 81.9133301, -135.7490387, 88.2649536, -214.1347198, 217.6623688
3: -133.9967194, 71.6502914, -144.4975433, 77.0803070, -211.0770264, 216.1478271
4: -122.6045227, 94.0873032, -132.2350006, 101.4297180, -224.0342102, 226.3222809
5: -110.2289047, 85.5704193, -118.9289703, 92.2402878, -202.4691620, 204.4993744
6: -105.3584518, 101.9444809, -113.6705856, 109.8726807, -215.2311401, 215.6150665
7: -114.5972061, 96.4931870, -123.6042786, 103.9747925, -218.5719910, 220.0974731
8: -138.3009796, 94.5896835, -149.2025757, 101.9174347, -240.2184143, 243.7922516
9: -104.1919327, 103.0583420, -112.3275604, 111.0033875, -215.1953125, 215.3858948

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6027879, upper bound: 264.6026415
time: 10.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6131742, upper bound: 264.6132006
time: 8.96 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -126.3109970, 100.7376251, -129.8962555, 103.6051331, -229.9161377, 230.6338654
1: -105.6476440, 89.2213745, -108.6074219, 91.7062225, -197.3538666, 197.8287964
2: -138.9989014, 90.4218063, -142.9585876, 92.9362793, -231.9351807, 233.3803864
3: -147.9966583, 78.9946823, -152.1871490, 81.1169052, -229.1135559, 231.1818237
4: -135.3594513, 103.8966370, -139.2265320, 106.8125992, -242.1720428, 243.1231689
5: -121.7869339, 94.5001984, -125.2764587, 97.1455154, -218.9324493, 219.7766571
6: -116.4068909, 112.5253067, -119.7362289, 115.6789093, -232.0857849, 232.2615356
7: -126.5469437, 106.4571381, -130.1616821, 109.4417648, -235.9886627, 236.6188202
8: -152.7089691, 104.4460907, -157.1113434, 107.3355103, -260.0444946, 261.5574341
9: -115.0487213, 113.7464905, -118.2849045, 116.8686218, -231.9173126, 232.0314026

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6033051, upper bound: 264.6034542
time: 10.40 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6132939, upper bound: 264.6133632
time: 9.06 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -113.9928436, 90.9171677, -127.3444443, 101.5369797, -215.5298157, 218.2616119
1: -95.4352951, 80.5464554, -106.5332336, 89.8909836, -185.3262787, 187.0796814
2: -125.4938049, 81.6671448, -140.1804199, 91.0890198, -216.5827942, 221.8475647
3: -133.5888519, 71.4409714, -149.2001801, 79.5426178, -213.1314545, 220.6411438
4: -122.2510452, 93.8064423, -136.5745850, 104.7048111, -226.9558563, 230.3810272
5: -109.8928909, 85.3261566, -122.7670441, 95.2435455, -205.1364441, 208.0932007
6: -105.0423355, 101.6438675, -117.3470993, 113.4327011, -218.4750366, 218.9909668
7: -114.2581558, 96.2167206, -127.6307526, 107.3575745, -221.6157074, 223.8474731
8: -137.8958435, 94.3104477, -154.0563354, 105.1968155, -243.0926514, 248.3667908
9: -103.8932190, 102.7654419, -115.9973984, 114.5974350, -218.4906464, 218.7628021

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6029495, upper bound: 264.6028206
time: 9.22 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6137661, upper bound: 264.6137249
time: 8.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -125.9021072, 100.4106522, -133.8567505, 106.7384186, -232.6405334, 234.2673950
1: -105.3094864, 88.9320297, -111.9361572, 94.4883957, -199.7978821, 200.8681793
2: -138.5477753, 90.1264420, -147.3257751, 95.7185440, -234.2662964, 237.4522095
3: -147.5139313, 78.7428360, -156.8284149, 83.5415344, -231.0554657, 235.5712433
4: -134.9284210, 103.5596008, -143.4999084, 110.0407867, -244.9692078, 247.0595093
5: -121.3877106, 94.1994934, -129.0621185, 100.1003342, -221.4880371, 223.2615814
6: -116.0326233, 112.1635590, -123.3644562, 119.1872253, -235.2198181, 235.5280151
7: -126.1368790, 106.1191101, -134.1279755, 112.7718353, -238.9087219, 240.2470703
8: -152.2213745, 104.1081772, -161.8960266, 110.5626068, -262.7839355, 266.0041199
9: -114.6802216, 113.3869858, -121.8968735, 120.4058609, -235.0860901, 235.2838593

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6034858, upper bound: 264.6036243
time: 10.05 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6139733, upper bound: 264.6140001
time: 9.53 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -118.7543564, 94.7102890, -124.0102539, 98.9063492, -217.6607056, 218.7205200
1: -99.4143295, 83.8932037, -103.7298584, 87.5478134, -186.9621124, 187.6230621
2: -130.7591400, 85.0446396, -136.5042114, 88.7517242, -219.5108337, 221.5488586
3: -139.2185364, 74.3615341, -145.3055725, 77.5040741, -216.7226105, 219.6671143
4: -127.3723450, 97.7069778, -132.9782715, 101.9890366, -229.3613892, 230.6852417
5: -114.5175705, 88.8651962, -119.5906906, 92.7488480, -207.2664185, 208.4558868
6: -109.4343338, 105.8642273, -114.3022003, 110.4804077, -219.9147186, 220.1664276
7: -119.0569611, 100.2033615, -124.2953339, 104.5493088, -223.6062622, 224.4986877
8: -143.6743469, 98.1656647, -150.0305634, 102.4741516, -246.1484985, 248.1962280
9: -108.2249374, 106.9773560, -112.9565887, 111.6147919, -219.8397217, 219.9339294

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6027869, upper bound: 264.6026458
time: 10.60 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6131668, upper bound: 264.6131670
time: 9.86 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -130.6663055, 104.2070312, -130.5740967, 104.1470261, -234.8133240, 234.7810974
1: -109.2925873, 92.2822189, -109.1744843, 92.1801834, -201.4727631, 201.4566956
2: -143.8200226, 93.5058746, -143.7066345, 93.4185638, -237.2385864, 237.2124939
3: -153.1509552, 81.6641159, -152.9880676, 81.5363159, -234.6872711, 234.6521912
4: -140.0622406, 107.4661636, -139.9627228, 107.3670502, -247.4292755, 247.4288940
5: -126.0134430, 97.7427368, -125.9322433, 97.6491165, -223.6625519, 223.6749878
6: -120.4237137, 116.3928299, -120.3619690, 116.2809982, -236.7047119, 236.7547607
7: -130.9424438, 110.1115646, -130.8461761, 110.0107956, -240.9532166, 240.9577332
8: -158.0083923, 107.9738693, -157.9314270, 107.8866959, -265.8950806, 265.9053040
9: -119.0282211, 117.6085434, -118.9079971, 117.4740677, -236.5022583, 236.5165253

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6033062, upper bound: 264.6034267
time: 10.47 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6132965, upper bound: 264.6133521
time: 9.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -118.4270859, 94.4436264, -128.1924896, 102.2111435, -220.6382141, 222.6361084
1: -99.1430054, 83.6593475, -107.2424469, 90.4861145, -189.6291199, 190.9017792
2: -130.4021759, 84.8108902, -141.1203613, 91.6932755, -222.0954590, 225.9312439
3: -138.8189240, 74.1598129, -150.1959991, 80.0671997, -218.8861237, 224.3558044
4: -127.0325775, 97.4368362, -137.4947357, 105.3987961, -232.4313660, 234.9315491
5: -114.1893387, 88.6321335, -123.5829086, 95.8760071, -210.0653381, 212.2150116
6: -109.1293030, 105.5762024, -118.1278458, 114.1875458, -223.3168488, 223.7040405
7: -118.7350922, 99.9407578, -128.4904480, 108.0718613, -226.8069458, 228.4312134
8: -143.2856598, 97.9012604, -155.0819702, 105.8898926, -249.1755524, 252.9832153
9: -107.9363403, 106.6993256, -116.7744751, 115.3567734, -223.2931061, 223.4737854

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6029447, upper bound: 264.6028325
time: 9.07 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6137708, upper bound: 264.6136912
time: 9.57 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -130.2640381, 103.8820724, -134.6928406, 107.4029694, -237.6669922, 238.5749207
1: -108.9578094, 91.9967575, -112.6351547, 95.0746307, -204.0324402, 204.6319122
2: -143.3780365, 93.2168274, -148.2514496, 96.3142090, -239.6922455, 241.4682770
3: -152.6689301, 81.4155121, -157.8106384, 84.0582809, -236.7272034, 239.2261505
4: -139.6371765, 107.1341934, -144.4066620, 110.7252045, -250.3623810, 251.5408020
5: -125.6153030, 97.4483566, -129.8668365, 100.7233276, -226.3386230, 227.3151855
6: -120.0541611, 116.0364456, -124.1342850, 119.9309998, -239.9851685, 240.1707001
7: -130.5408630, 109.7806931, -134.9744873, 113.4755249, -244.0163727, 244.7551880
8: -157.5271912, 107.6438904, -162.9062805, 111.2454758, -268.7726440, 270.5501404
9: -118.6637344, 117.2562561, -122.6629028, 121.1536636, -239.8173523, 239.9191589

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6034841, upper bound: 264.6036288
time: 8.81 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6139778, upper bound: 264.6139778
time: 10.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 20.04 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6027879, upper bound: 264.6026415
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6131742, upper bound: 264.6132006
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6033051, upper bound: 264.6034542
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6132939, upper bound: 264.6133632
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6029495, upper bound: 264.6028206
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6137661, upper bound: 264.6137249
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6034858, upper bound: 264.6036243
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6139733, upper bound: 264.6140001
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6027869, upper bound: 264.6026458
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6131668, upper bound: 264.6131670
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6033062, upper bound: 264.6034267
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6132965, upper bound: 264.6133521
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6029447, upper bound: 264.6028325
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6137708, upper bound: 264.6136912
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6034841, upper bound: 264.6036288
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 20.04
Output dim: 7, lower bound: -264.6139778, upper bound: 264.6139778

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -113.7352524, 90.7166138, -121.7030106, 97.0718842, -210.8071289, 212.4196167
1: -95.2145767, 80.3598862, -101.7970734, 85.9040680, -181.1186523, 182.1569519
2: -125.2113190, 81.4796829, -133.9653473, 87.0901871, -212.3014984, 215.4450378
3: -133.2843323, 71.2820969, -142.5669861, 76.0827560, -209.3670959, 213.8490906
4: -121.9585037, 93.5870895, -130.4840546, 100.0747528, -222.0332031, 224.0711365
5: -109.6532516, 85.1163864, -117.3684998, 91.0092850, -200.6625366, 202.4848785
6: -104.8017960, 101.4151001, -112.1626434, 108.4376526, -213.2394409, 213.5777130
7: -113.9885635, 95.9876022, -121.9555435, 102.6052628, -216.5937805, 217.9431305
8: -137.5791321, 94.0930634, -147.2480011, 100.5720139, -238.1511383, 241.3410645
9: -103.6385193, 102.5156860, -110.8287201, 109.5331726, -213.1716919, 213.3444061

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6113235, upper bound: 264.6112190
time: 9.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6111593, upper bound: 264.6110426
time: 9.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -125.7078094, 100.2592163, -128.2783051, 102.3219147, -228.0297241, 228.5375214
1: -105.1421051, 88.7883530, -107.2515640, 90.5449448, -195.6870270, 196.0398865
2: -138.3360748, 89.9854507, -141.1809845, 91.7656403, -230.1017151, 231.1664276
3: -147.2790222, 78.6238556, -150.2628937, 80.1226425, -227.4016571, 228.8867493
4: -134.7085724, 103.3932037, -137.4810486, 105.4622726, -240.1708221, 240.8742371
5: -121.2071686, 94.0428314, -123.7211609, 95.9184036, -217.1255798, 217.7639618
6: -115.8466263, 111.9920349, -118.2333374, 114.2487869, -230.0954132, 230.2253723
7: -125.9341965, 105.9481888, -128.5184021, 108.0769577, -234.0111084, 234.4665833
8: -151.9827271, 103.9461060, -155.1632843, 105.9947662, -257.9774780, 259.1093750
9: -114.4918213, 113.2003403, -116.7915039, 115.4037704, -229.8955994, 229.9918365

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6114691, upper bound: 264.6114293
time: 9.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6112954, upper bound: 264.6112570
time: 8.21 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -113.4011459, 90.4474411, -125.7396622, 100.2645187, -213.6656647, 216.1871033
1: -94.9396439, 80.1220093, -105.1884537, 88.7400818, -183.6796875, 185.3104553
2: -124.8428955, 81.2385712, -138.4168549, 89.9291153, -214.7720032, 219.6554260
3: -132.8846130, 71.0769272, -147.2929840, 78.5559540, -211.4405670, 218.3699036
4: -121.6122742, 93.3118668, -134.8434448, 103.3659592, -224.9782257, 228.1552887
5: -109.3238068, 84.8773499, -121.2244644, 94.0278778, -203.3516541, 206.1018066
6: -104.4920807, 101.1204758, -115.8567200, 112.0141525, -216.5062103, 216.9772034
7: -113.6565323, 95.7168198, -126.0021667, 106.0039902, -219.6605225, 221.7189789
8: -137.1821899, 93.8192749, -152.1238708, 103.8664398, -241.0486298, 245.9431458
9: -103.3459778, 102.2288437, -114.5157318, 113.1446838, -216.4906616, 216.7445679

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6118189, upper bound: 264.6116760
time: 7.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6116756, upper bound: 264.6115242
time: 10.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -125.3040695, 99.9363022, -132.2536163, 105.4673004, -230.7713623, 232.1899109
1: -104.8081818, 88.5027466, -110.5929642, 93.3386688, -198.1468353, 199.0956879
2: -137.8905945, 89.6937180, -145.5642242, 94.5598526, -232.4504395, 235.2579346
3: -146.8023987, 78.3750687, -154.9232330, 82.5559387, -229.3583374, 233.2983093
4: -134.2830353, 103.0603867, -141.7706299, 108.7036743, -242.9866943, 244.8310242
5: -120.8128662, 93.7459793, -127.5210114, 98.8857727, -219.6986237, 221.2669983
6: -115.4770660, 111.6347427, -121.8755646, 117.7702332, -233.2472992, 233.5102997
7: -125.5292435, 105.6144257, -132.5010071, 111.4197922, -236.9490204, 238.1154327
8: -151.5012665, 103.6124191, -159.9656830, 109.2340164, -260.7352905, 263.5780640
9: -114.1279449, 112.8453827, -120.4173737, 118.9551697, -233.0831146, 233.2627563

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6121446, upper bound: 264.6120550
time: 8.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6119742, upper bound: 264.6119062
time: 8.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -118.1620255, 94.2399139, -122.3993988, 97.6281281, -215.7901459, 216.6392975
1: -98.9181900, 83.4683304, -102.3795013, 86.3911514, -185.3093262, 185.8478241
2: -130.1078339, 84.6158218, -134.7342834, 87.5859222, -217.6937561, 219.3500977
3: -138.5134735, 73.9971848, -143.3889160, 76.5139389, -215.0274048, 217.3861084
4: -126.7332458, 97.2118073, -131.2406311, 100.6438370, -227.3770752, 228.4524384
5: -113.9474945, 88.4159012, -118.0414505, 91.5268936, -205.4743958, 206.4573517
6: -108.8834305, 105.3404922, -112.8053741, 109.0562439, -217.9396667, 218.1458588
7: -118.4552536, 99.7031860, -122.6592102, 103.1899872, -221.6452332, 222.3623810
8: -142.9601135, 97.6741486, -148.0907593, 101.1386795, -244.0987854, 245.7649078
9: -107.6773758, 106.4403839, -111.4687195, 110.1555099, -217.8328705, 217.9090729

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6113584, upper bound: 264.6113013
time: 9.80 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6112191, upper bound: 264.6111719
time: 9.15 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -130.0675354, 103.7320557, -128.9689026, 102.8736343, -232.9411621, 232.7009277
1: -108.7907867, 91.8524246, -107.8291092, 91.0278015, -199.8185883, 199.6815186
2: -143.1623230, 93.0729141, -141.9432983, 92.2571945, -235.4194946, 235.0162048
3: -152.4385986, 81.2961273, -151.0782776, 80.5497818, -232.9883728, 232.3743896
4: -139.4163666, 106.9664001, -138.2312012, 106.0269012, -245.4432678, 245.1975861
5: -125.4376373, 97.2886658, -124.3887405, 96.4313354, -221.8689728, 221.6773987
6: -119.8675690, 115.8635864, -118.8706818, 114.8621445, -234.7296906, 234.7342682
7: -130.3345184, 109.6065598, -129.2159729, 108.6565552, -238.9910583, 238.8225250
8: -157.2875519, 107.4777527, -155.9987793, 106.5563965, -263.8439331, 263.4765320
9: -118.4755249, 117.0665741, -117.4259796, 116.0206680, -234.4961853, 234.4925537

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6115079, upper bound: 264.6115067
time: 9.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6113655, upper bound: 264.6114055
time: 9.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -117.8407898, 93.9780960, -126.6010284, 100.9490814, -218.7898712, 220.5791321
1: -98.6518402, 83.2388306, -105.9087601, 89.3443909, -187.9962311, 189.1475830
2: -129.7574005, 84.3864212, -139.3714905, 90.5429306, -220.3003235, 223.7579041
3: -138.1210632, 73.7991943, -148.3040314, 79.0887604, -217.2098236, 222.1032257
4: -126.3997879, 96.9467468, -135.7780151, 104.0707932, -230.4705811, 232.7247467
5: -113.6251984, 88.1874084, -122.0528412, 94.6701660, -208.2953339, 210.2402496
6: -108.5839691, 105.0577087, -116.6496506, 112.7807159, -221.3646851, 221.7073669
7: -118.1393204, 99.4455261, -126.8752823, 106.7292938, -224.8686218, 226.3208008
8: -142.5785370, 97.4146118, -153.1654663, 104.5704346, -247.1489716, 250.5800781
9: -107.3942413, 106.1677856, -115.3048477, 113.9159012, -221.3101196, 221.4726257

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 163

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6119830, upper bound: 264.6119178
time: 8.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6118435, upper bound: 264.6117896
time: 9.48 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -129.6713867, 103.4119873, -133.1029205, 106.1422043, -235.8135986, 236.5149078
1: -108.4610214, 91.5714264, -111.3029633, 93.9341736, -202.3952026, 202.8743896
2: -142.7271576, 92.7883148, -146.5047760, 95.1651840, -237.8923340, 239.2930756
3: -151.9638062, 81.0512009, -155.9206085, 83.0808487, -235.0446472, 236.9718018
4: -138.9979095, 106.6394577, -142.6918182, 109.3988419, -248.3967590, 249.3312683
5: -125.0453873, 96.9989700, -128.3381195, 99.5186081, -224.5639954, 225.3370972
6: -119.5036163, 115.5126495, -122.6574783, 118.5257721, -238.0293884, 238.1701202
7: -129.9392242, 109.2807617, -133.3610992, 112.1345825, -242.0737762, 242.6418610
8: -156.8137817, 107.1527176, -160.9919891, 109.9278564, -266.7416382, 268.1446533
9: -118.1165695, 116.7197876, -121.1953735, 119.7149277, -237.8314972, 237.9151611

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 226
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 188

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -264.6123726, upper bound: 264.6123507
time: 11.06 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6113655, upper bound: 264.6122308
time: 10.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.29 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6113235, upper bound: 264.6112190
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6111593, upper bound: 264.6110426
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6114691, upper bound: 264.6114293
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6112954, upper bound: 264.6112570
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6118189, upper bound: 264.6116760
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6116756, upper bound: 264.6115242
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6121446, upper bound: 264.6120550
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6119742, upper bound: 264.6119062
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6113584, upper bound: 264.6113013
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6112191, upper bound: 264.6111719
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6115079, upper bound: 264.6115067
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6113655, upper bound: 264.6114055
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6119830, upper bound: 264.6119178
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6118435, upper bound: 264.6117896
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6123726, upper bound: 264.6123507
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 22.29
Output dim: 7, lower bound: -264.6113655, upper bound: 264.6122308

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -122.4372406, 97.6204834, -129.3144379, 103.1093063, -225.5465393, 226.9349213
1: -102.3747635, 86.4843063, -108.1165848, 91.2697372, -193.6444855, 194.6008911
2: -134.7209167, 87.5910568, -142.3106079, 92.4424591, -227.1633759, 229.9016724
3: -143.4829102, 76.4788971, -151.4803619, 80.6857986, -224.1686859, 227.9592590
4: -131.2824402, 100.6819687, -138.6505280, 106.2790909, -237.5615234, 239.3324890
5: -118.0966797, 91.6025696, -124.6992493, 96.6914902, -214.7881775, 216.3018188
6: -112.8840790, 109.0383606, -119.1909485, 115.1346970, -228.0187531, 228.2292938
7: -122.7322540, 103.1920776, -129.5868073, 108.9451141, -231.6773682, 232.7788849
8: -148.0389709, 101.0656662, -156.3956451, 106.7394180, -254.7783813, 257.4613037
9: -111.6008530, 110.1744843, -117.7839813, 116.2863388, -227.8871918, 227.9584198

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 226
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 253
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6117376, upper bound: 264.6117639
time: 10.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -264.6122633, upper bound: 264.6122309
time: 8.10 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 19.44 seconds
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 19.44
Output dim: 7, lower bound: -264.6117376, upper bound: 264.6117639
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 19.44
Output dim: 7, lower bound: -264.6122633, upper bound: 264.6122309
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=266.34649658203125
rel_dist={7: [-264.62256362911285, 264.62256362911285]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2283.15 seconds
