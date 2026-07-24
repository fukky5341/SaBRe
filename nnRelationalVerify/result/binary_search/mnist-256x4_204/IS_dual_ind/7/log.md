## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 81.1446251145
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-50.3818474, 38.2952919, -50.3818474, 38.2952919, -88.6771393, 88.6771393)
1: (-39.8088417, 35.9474564, -39.8088417, 35.9474564, -75.7563019, 75.7563019)
2: (-53.4981537, 35.4362068, -53.4981537, 35.4362068, -88.9343567, 88.9343567)
3: (-59.6625023, 30.4251728, -59.6625023, 30.4251728, -90.0876770, 90.0876770)
4: (-57.4144287, 38.5423965, -57.4144287, 38.5423965, -95.9568253, 95.9568253)
5: (-50.4630890, 34.7195892, -50.4630890, 34.7195892, -85.1826782, 85.1826782)
6: (-52.5452843, 38.3607750, -52.5452843, 38.3607750, -90.9060516, 90.9060593)
7: (-47.6143074, 43.1696014, -47.6143074, 43.1696014, -90.7839050, 90.7839050)
8: (-63.5914040, 37.7574081, -63.5914040, 37.7574081, -101.3488007, 101.3488007)
9: (-44.2846413, 44.8899040, -44.2846413, 44.8899040, -89.1745377, 89.1745377)

## BASE Result
execution time: IAR + LP analysis = 1.21 + 8.24 = 9.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -81.2973621, upper bound: 81.2973621


# Binary Search by BASE starts (time budget: 2690.55 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=90.90605926513672
rel_dist={6: [-81.29731319725406, 81.29731319725403]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=90.90605926513672
rel_dist={6: [-81.29714583279736, 81.29714583279738]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=90.90605926513672
rel_dist={6: [-81.29696620335748, 81.29696620339547]}

## Binary Search Result
Binary search time: 36.72 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2653.84 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2972358, upper bound: 81.2972487
time: 7.53 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031
time: 6.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.37
Output dim: 6, lower bound: -81.2972358, upper bound: 81.2972487
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.37
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -39.4353180, 29.7706108, -49.6208725, 37.7164841, -77.1518021, 79.3914795
1: -30.7156715, 28.1498985, -39.1971130, 35.4103165, -66.1259842, 67.3470154
2: -41.5748978, 27.6886997, -52.6819572, 34.9052162, -76.4801025, 80.3706360
3: -46.7086945, 23.7850609, -58.7605209, 29.9727154, -76.6814117, 82.5455780
4: -45.2326088, 29.8140793, -56.5600739, 37.9559555, -83.1885681, 86.3741302
5: -39.7052155, 26.8396111, -49.7070465, 34.1866570, -73.8918762, 76.5466461
6: -41.8891563, 29.1658058, -51.7795715, 37.7587051, -79.6478500, 80.9453735
7: -36.8842659, 33.8704376, -46.8805313, 42.5218658, -79.4061203, 80.7509689
8: -50.0938263, 29.0696526, -62.6527672, 37.1842728, -87.2780991, 91.7224197
9: -34.4803238, 34.9603157, -43.6110611, 44.2115746, -78.6918945, 78.5713654

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031
time: 7.96 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031
time: 8.42 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -48.7763367, 37.0808640, -50.3818474, 38.2952919, -87.0716248, 87.4626923
1: -38.5242729, 34.8119087, -39.8088417, 35.9474564, -74.4717255, 74.6207428
2: -51.7800751, 34.3216591, -53.4981537, 35.4362068, -87.2162781, 87.8198090
3: -57.7584953, 29.4703693, -59.6625023, 30.4251728, -88.1836548, 89.1328659
4: -55.6023178, 37.3147583, -57.4144287, 38.5423965, -94.1447144, 94.7291870
5: -48.8602371, 33.6030312, -50.4630890, 34.7195892, -83.5798264, 84.0661087
6: -50.9166870, 37.1101799, -52.5452843, 38.3607750, -89.2774658, 89.6554642
7: -46.0756607, 41.8013268, -47.6143074, 43.1696014, -89.2452621, 89.4156342
8: -61.6037064, 36.5603180, -63.5914040, 37.7574081, -99.3610992, 100.1517181
9: -42.8696823, 43.4593582, -44.2846413, 44.8899040, -87.7595673, 87.7439880

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031
time: 6.54 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031
time: 6.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 6, lower bound: -81.2972031, upper bound: 81.2972031

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -39.4353180, 29.7706108, -39.4353180, 29.7706108, -69.2059326, 69.2059326
1: -30.7156715, 28.1498985, -30.7156715, 28.1498985, -58.8655701, 58.8655701
2: -41.5748978, 27.6886997, -41.5748978, 27.6886997, -69.2635956, 69.2635956
3: -46.7086945, 23.7850609, -46.7086945, 23.7850609, -70.4937592, 70.4937592
4: -45.2326088, 29.8140793, -45.2326088, 29.8140793, -75.0466919, 75.0466919
5: -39.7052155, 26.8396111, -39.7052155, 26.8396111, -66.5448151, 66.5448151
6: -41.8891563, 29.1658058, -41.8891563, 29.1658058, -71.0549622, 71.0549622
7: -36.8842659, 33.8704376, -36.8842659, 33.8704376, -70.7546844, 70.7546844
8: -50.0938263, 29.0696526, -50.0938263, 29.0696526, -79.1634827, 79.1634827
9: -34.4803238, 34.9603157, -34.4803238, 34.9603157, -69.4406433, 69.4406433

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2916206, upper bound: 81.2902702
time: 10.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
time: 8.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -39.4353180, 29.7706108, -48.7763367, 37.0808640, -76.5161667, 78.5469513
1: -30.7156715, 28.1498985, -38.5242729, 34.8119087, -65.5275803, 66.6741714
2: -41.5748978, 27.6886997, -51.7800751, 34.3216591, -75.8965607, 79.4687653
3: -46.7086945, 23.7850609, -57.7584953, 29.4703693, -76.1790619, 81.5435410
4: -45.2326088, 29.8140793, -55.6023178, 37.3147583, -82.5473633, 85.4163818
5: -39.7052155, 26.8396111, -48.8602371, 33.6030312, -73.3082352, 75.6998444
6: -41.8891563, 29.1658058, -50.9166870, 37.1101799, -78.9993362, 80.0824890
7: -36.8842659, 33.8704376, -46.0756607, 41.8013268, -78.6855927, 79.9460983
8: -50.0938263, 29.0696526, -61.6037064, 36.5603180, -86.6541443, 90.6733551
9: -34.4803238, 34.9603157, -42.8696823, 43.4593582, -77.9396820, 77.8299789

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2916206, upper bound: 81.2902706
time: 10.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
time: 7.29 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -48.7763367, 37.0808640, -39.4353180, 29.7706108, -78.5469513, 76.5161667
1: -38.5242729, 34.8119087, -30.7156715, 28.1498985, -66.6741714, 65.5275803
2: -51.7800751, 34.3216591, -41.5748978, 27.6886997, -79.4687653, 75.8965607
3: -57.7584953, 29.4703693, -46.7086945, 23.7850609, -81.5435410, 76.1790619
4: -55.6023178, 37.3147583, -45.2326088, 29.8140793, -85.4163895, 82.5473633
5: -48.8602371, 33.6030312, -39.7052155, 26.8396111, -75.6998444, 73.3082352
6: -50.9166870, 37.1101799, -41.8891563, 29.1658058, -80.0824890, 78.9993362
7: -46.0756607, 41.8013268, -36.8842659, 33.8704376, -79.9460983, 78.6855927
8: -61.6037064, 36.5603180, -50.0938263, 29.0696526, -90.6733551, 86.6541443
9: -42.8696823, 43.4593582, -34.4803238, 34.9603157, -77.8299789, 77.9396820

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2915714, upper bound: 81.2902222
time: 9.40 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884447, upper bound: 81.2884448
time: 6.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -48.7763367, 37.0808640, -48.7763367, 37.0808640, -85.8571854, 85.8571854
1: -38.5242729, 34.8119087, -38.5242729, 34.8119087, -73.3361816, 73.3361816
2: -51.7800751, 34.3216591, -51.7800751, 34.3216591, -86.1017303, 86.1017303
3: -57.7584953, 29.4703693, -57.7584953, 29.4703693, -87.2288437, 87.2288437
4: -55.6023178, 37.3147583, -55.6023178, 37.3147583, -92.9170761, 92.9170685
5: -48.8602371, 33.6030312, -48.8602371, 33.6030312, -82.4632568, 82.4632568
6: -50.9166870, 37.1101799, -50.9166870, 37.1101799, -88.0268707, 88.0268707
7: -46.0756607, 41.8013268, -46.0756607, 41.8013268, -87.8769836, 87.8769836
8: -61.6037064, 36.5603180, -61.6037064, 36.5603180, -98.1640244, 98.1640244
9: -42.8696823, 43.4593582, -42.8696823, 43.4593582, -86.3290176, 86.3290176

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2915714, upper bound: 81.2902222
time: 7.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884448
time: 6.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.28
Output dim: 6, lower bound: -81.2916206, upper bound: 81.2902702
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.28
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.28
Output dim: 6, lower bound: -81.2916206, upper bound: 81.2902706
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.28
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.28
Output dim: 6, lower bound: -81.2915714, upper bound: 81.2902222
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.28
Output dim: 6, lower bound: -81.2884447, upper bound: 81.2884448
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.28
Output dim: 6, lower bound: -81.2915714, upper bound: 81.2902222
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.28
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884448

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -36.7280846, 27.6646996, -39.4353180, 29.7706108, -66.4986954, 67.1000137
1: -28.4664154, 26.2053585, -30.7156715, 28.1498985, -56.6163139, 56.9210281
2: -38.6123466, 25.7999229, -41.5748978, 27.6886997, -66.3010406, 67.3748169
3: -43.4643974, 22.0956497, -46.7086945, 23.7850609, -67.2494583, 68.8043442
4: -42.1785355, 27.6544113, -45.2326088, 29.8140793, -71.9925919, 72.8870239
5: -37.0389252, 24.8807831, -39.7052155, 26.8396111, -63.8785248, 64.5859909
6: -39.2171059, 26.8512917, -41.8891563, 29.1658058, -68.3829117, 68.7404480
7: -34.2506599, 31.5664749, -36.8842659, 33.8704376, -68.1210938, 68.4507294
8: -46.7221947, 26.9137268, -50.0938263, 29.0696526, -75.7918472, 77.0075531
9: -32.0445175, 32.4881477, -34.4803238, 34.9603157, -67.0048370, 66.9684677

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
time: 8.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
time: 8.92 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -42.9759293, 32.1972351, -38.5315018, 29.0648708, -72.0408020, 70.7287292
1: -33.1128654, 30.5186424, -29.9557381, 27.4951553, -60.6080170, 60.4743805
2: -45.0236664, 30.0465298, -40.5807228, 27.0575504, -72.0812149, 70.6272507
3: -50.7500114, 25.6291046, -45.6251984, 23.2155190, -73.9655304, 71.2543030
4: -49.3081818, 32.0276031, -44.2149887, 29.0884056, -78.3965759, 76.2425919
5: -43.4173050, 28.9366531, -38.8163719, 26.1815624, -69.5988617, 67.7530212
6: -45.8698502, 30.9571648, -41.0013771, 28.3818073, -74.2516556, 71.9585419
7: -39.8950768, 36.8984833, -36.0012321, 33.1005325, -72.9956055, 72.8996964
8: -54.5415649, 31.0423737, -48.9703064, 28.3430672, -82.8846283, 80.0126724
9: -37.3058853, 37.8575325, -33.6636543, 34.1303635, -71.4362488, 71.5211639

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2850843, upper bound: 81.2860111
time: 7.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 6.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -36.7280846, 27.6646996, -48.7763367, 37.0808640, -73.8089294, 76.4410400
1: -28.4664154, 26.2053585, -38.5242729, 34.8119087, -63.2783203, 64.7296295
2: -38.6123466, 25.7999229, -51.7800751, 34.3216591, -72.9340057, 77.5800018
3: -43.4643974, 22.0956497, -57.7584953, 29.4703693, -72.9347687, 79.8541336
4: -42.1785355, 27.6544113, -55.6023178, 37.3147583, -79.4932861, 83.2567291
5: -37.0389252, 24.8807831, -48.8602371, 33.6030312, -70.6419449, 73.7410126
6: -39.2171059, 26.8512917, -50.9166870, 37.1101799, -76.3272781, 77.7679749
7: -34.2506599, 31.5664749, -46.0756607, 41.8013268, -76.0519867, 77.6421356
8: -46.7221947, 26.9137268, -61.6037064, 36.5603180, -83.2825165, 88.5174255
9: -32.0445175, 32.4881477, -42.8696823, 43.4593582, -75.5038605, 75.3578033

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
time: 8.17 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
time: 6.89 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -42.9759293, 32.1972351, -47.7339668, 36.2588921, -79.2348175, 79.9311752
1: -33.1128654, 30.5186424, -37.6407318, 34.0504150, -67.1632843, 68.1593704
2: -45.0236664, 30.0465298, -50.6269684, 33.5915413, -78.6152039, 80.6734924
3: -50.7500114, 25.6291046, -56.5051689, 28.8143215, -79.5643311, 82.1342621
4: -49.3081818, 32.0276031, -54.4308167, 36.4719048, -85.7800751, 86.4584198
5: -43.4173050, 28.9366531, -47.8389893, 32.8366127, -76.2539215, 76.7756424
6: -45.8698502, 30.9571648, -49.8985023, 36.1944962, -82.0643463, 80.8556671
7: -39.8950768, 36.8984833, -45.0432549, 40.9171371, -80.8122101, 81.9417343
8: -54.5415649, 31.0423737, -60.3117638, 35.7104797, -90.2520447, 91.3541412
9: -37.3058853, 37.8575325, -41.9216080, 42.4953384, -79.8012238, 79.7791290

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2850843, upper bound: 81.2860101
time: 9.00 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.81 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -45.8831482, 34.8190536, -39.4353180, 29.7706108, -75.6537628, 74.2543640
1: -36.1050339, 32.7193146, -30.7156715, 28.1498985, -64.2549286, 63.4349861
2: -48.6059608, 32.3003120, -41.5748978, 27.6886997, -76.2946472, 73.8752060
3: -54.2904091, 27.6711502, -46.7086945, 23.7850609, -78.0754700, 74.3798370
4: -52.3488693, 35.0008011, -45.2326088, 29.8140793, -82.1629486, 80.2334061
5: -46.0224190, 31.4999313, -39.7052155, 26.8396111, -72.8620300, 71.2051392
6: -48.0753975, 34.6196404, -41.8891563, 29.1658058, -77.2411957, 76.5087967
7: -43.2348099, 39.3503342, -36.8842659, 33.8704376, -77.1052475, 76.2346039
8: -58.0194626, 34.2389603, -50.0938263, 29.0696526, -87.0891113, 84.3327866
9: -40.2577782, 40.8078308, -34.4803238, 34.9603157, -75.2180786, 75.2881546

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884448, upper bound: 81.2884448
time: 8.46 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884448, upper bound: 81.2884447
time: 6.79 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -52.8844795, 39.9176064, -38.5315018, 29.0648708, -81.9493484, 78.4491119
1: -41.3286514, 37.5414619, -29.9557381, 27.4951553, -68.8238068, 67.4972000
2: -55.8145599, 37.0655212, -40.5807228, 27.0575504, -82.8721085, 77.6462402
3: -62.4288521, 31.6118279, -45.6251984, 23.2155190, -85.6443710, 77.2370148
4: -60.3328934, 39.9323578, -44.2149887, 29.0884056, -89.4212799, 84.1473465
5: -53.1586914, 36.0457268, -38.8163719, 26.1815624, -79.3402405, 74.8620911
6: -55.5044479, 39.2829056, -41.0013771, 28.3818073, -83.8862534, 80.2842865
7: -49.5798950, 45.3233414, -36.0012321, 33.1005325, -82.6804276, 81.3245621
8: -66.7850571, 38.9231262, -48.9703064, 28.3430672, -95.1281052, 87.8934326
9: -46.1535759, 46.8444901, -33.6636543, 34.1303635, -80.2839355, 80.5081406

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2850843, upper bound: 81.2860111
time: 8.03 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 6.62 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -45.8831482, 34.8190536, -48.7763367, 37.0808640, -82.9640045, 83.5953827
1: -36.1050339, 32.7193146, -38.5242729, 34.8119087, -70.9169388, 71.2435913
2: -48.6059608, 32.3003120, -51.7800751, 34.3216591, -82.9276199, 84.0803833
3: -54.2904091, 27.6711502, -57.7584953, 29.4703693, -83.7607803, 85.4296341
4: -52.3488693, 35.0008011, -55.6023178, 37.3147583, -89.6636276, 90.6031036
5: -46.0224190, 31.4999313, -48.8602371, 33.6030312, -79.6254425, 80.3601685
6: -48.0753975, 34.6196404, -50.9166870, 37.1101799, -85.1855698, 85.5363312
7: -43.2348099, 39.3503342, -46.0756607, 41.8013268, -85.0361328, 85.4259949
8: -58.0194626, 34.2389603, -61.6037064, 36.5603180, -94.5797806, 95.8426666
9: -40.2577782, 40.8078308, -42.8696823, 43.4593582, -83.7171173, 83.6774979

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
time: 8.03 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884448
time: 6.73 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -52.8844795, 39.9176064, -47.7339668, 36.2588921, -89.1433716, 87.6515732
1: -41.3286514, 37.5414619, -37.6407318, 34.0504150, -75.3790665, 75.1821823
2: -55.8145599, 37.0655212, -50.6269684, 33.5915413, -89.4060974, 87.6924896
3: -62.4288521, 31.6118279, -56.5051689, 28.8143215, -91.2431717, 88.1169739
4: -60.3328934, 39.9323578, -54.4308167, 36.4719048, -96.8047943, 94.3631744
5: -53.1586914, 36.0457268, -47.8389893, 32.8366127, -85.9953003, 83.8847198
6: -55.5044479, 39.2829056, -49.8985023, 36.1944962, -91.6989441, 89.1813965
7: -49.5798950, 45.3233414, -45.0432549, 40.9171371, -90.4970322, 90.3665924
8: -66.7850571, 38.9231262, -60.3117638, 35.7104797, -102.4955368, 99.2348938
9: -46.1535759, 46.8444901, -41.9216080, 42.4953384, -88.6489105, 88.7660980

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2850843, upper bound: 81.2860111
time: 7.85 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 6.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.14 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2850843, upper bound: 81.2860111
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2850843, upper bound: 81.2860101
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2884448, upper bound: 81.2884448
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2884448, upper bound: 81.2884447
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2850843, upper bound: 81.2860111
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884447
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2884451, upper bound: 81.2884448
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2850843, upper bound: 81.2860111
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.14
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -36.7280846, 27.6646996, -36.7280846, 27.6646996, -64.3927841, 64.3927841
1: -28.4664154, 26.2053585, -28.4664154, 26.2053585, -54.6717682, 54.6717644
2: -38.6123466, 25.7999229, -38.6123466, 25.7999229, -64.4122620, 64.4122696
3: -43.4643974, 22.0956497, -43.4643974, 22.0956497, -65.5600433, 65.5600433
4: -42.1785355, 27.6544113, -42.1785355, 27.6544113, -69.8329391, 69.8329468
5: -37.0389252, 24.8807831, -37.0389252, 24.8807831, -61.9197083, 61.9197044
6: -39.2171059, 26.8512917, -39.2171059, 26.8512917, -66.0683975, 66.0683975
7: -34.2506599, 31.5664749, -34.2506599, 31.5664749, -65.8171387, 65.8171387
8: -46.7221947, 26.9137268, -46.7221947, 26.9137268, -73.6359100, 73.6359100
9: -32.0445175, 32.4881477, -32.0445175, 32.4881477, -64.5326691, 64.5326614

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
time: 10.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
time: 10.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -36.7280846, 27.6646996, -42.9759293, 32.1972351, -68.9253082, 70.6406250
1: -28.4664154, 26.2053585, -33.1128654, 30.5186424, -58.9850578, 59.3182182
2: -38.6123466, 25.7999229, -45.0236664, 30.0465298, -68.6588593, 70.8235931
3: -43.4643974, 22.0956497, -50.7500114, 25.6291046, -69.0935059, 72.8456573
4: -42.1785355, 27.6544113, -49.3081818, 32.0276031, -74.2061310, 76.9625931
5: -37.0389252, 24.8807831, -43.4173050, 28.9366531, -65.9755783, 68.2980881
6: -39.2171059, 26.8512917, -45.8698502, 30.9571648, -70.1742706, 72.7211456
7: -34.2506599, 31.5664749, -39.8950768, 36.8984833, -71.1491394, 71.4615479
8: -46.7221947, 26.9137268, -54.5415649, 31.0423737, -77.7645645, 81.4552917
9: -32.0445175, 32.4881477, -37.3058853, 37.8575325, -69.9020462, 69.7940369

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
time: 8.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
time: 8.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -42.9759293, 32.1972351, -35.5111732, 26.7373600, -69.7132874, 67.7083893
1: -33.1128654, 30.5186424, -27.5455341, 25.3689556, -58.4818192, 58.0641785
2: -45.0236664, 30.0465298, -37.3420296, 24.9512329, -69.9748993, 67.3885498
3: -50.7500114, 25.6291046, -42.0624390, 21.3974552, -72.1474609, 67.6915436
4: -49.3081818, 32.0276031, -40.8259277, 26.7342663, -76.0424347, 72.8535309
5: -43.4173050, 28.9366531, -35.8248940, 24.0434818, -67.4607849, 64.7615509
6: -45.8698502, 30.9571648, -37.9853401, 25.9367695, -71.8066177, 68.9425049
7: -39.8950768, 36.8984833, -33.1252594, 30.5387230, -70.4337997, 70.0237350
8: -54.5415649, 31.0423737, -45.2404480, 26.0466270, -80.5881958, 76.2828217
9: -37.3058853, 37.8575325, -30.9900475, 31.4443703, -68.7502594, 68.8475723

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -42.1647034, 31.5661430, -40.1694260, 30.1456108, -72.3103180, 71.7355652
1: -32.4632950, 29.9484825, -31.1567230, 28.6339207, -61.0972023, 61.1052055
2: -44.1515884, 29.4783173, -42.2568970, 28.1376476, -72.2892380, 71.7352142
3: -49.7989883, 25.1401424, -47.5994568, 24.1382256, -73.9372101, 72.7396011
4: -48.4082527, 31.3890495, -46.2917938, 30.1309242, -78.5391617, 77.6808472
5: -42.6205902, 28.3523064, -40.6001511, 27.0523834, -69.6729736, 68.9524536
6: -45.0777931, 30.2834148, -43.0602341, 29.1610546, -74.2388458, 73.3436508
7: -39.1218719, 36.2116814, -37.4557037, 34.5758972, -73.6977692, 73.6673737
8: -53.5448074, 30.4193726, -51.2469406, 29.3568306, -82.9016266, 81.6663132
9: -36.5868301, 37.1389771, -35.0368271, 35.5746040, -72.1614380, 72.1758041

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.09 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 8.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -36.7280846, 27.6646996, -45.8831482, 34.8190536, -71.5471344, 73.5478516
1: -28.4664154, 26.2053585, -36.1050339, 32.7193146, -61.1857300, 62.3103867
2: -38.6123466, 25.7999229, -48.6059608, 32.3003120, -70.9126587, 74.4058838
3: -43.4643974, 22.0956497, -54.2904091, 27.6711502, -71.1355438, 76.3860550
4: -42.1785355, 27.6544113, -52.3488693, 35.0008011, -77.1793365, 80.0032806
5: -37.0389252, 24.8807831, -46.0224190, 31.4999313, -68.5388489, 70.9031982
6: -39.2171059, 26.8512917, -48.0753975, 34.6196404, -73.8367462, 74.9266891
7: -34.2506599, 31.5664749, -43.2348099, 39.3503342, -73.6009979, 74.8012848
8: -46.7221947, 26.9137268, -58.0194626, 34.2389603, -80.9611435, 84.9331894
9: -32.0445175, 32.4881477, -40.2577782, 40.8078308, -72.8523483, 72.7459106

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
time: 10.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
time: 9.24 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -36.7280846, 27.6646996, -52.8844795, 39.9176064, -76.6456909, 80.5491791
1: -28.4664154, 26.2053585, -41.3286514, 37.5414619, -66.0078735, 67.5340118
2: -38.6123466, 25.7999229, -55.8145599, 37.0655212, -75.6778641, 81.6144867
3: -43.4643974, 22.0956497, -62.4288521, 31.6118279, -75.0762253, 84.5244980
4: -42.1785355, 27.6544113, -60.3328934, 39.9323578, -82.1108932, 87.9873047
5: -37.0389252, 24.8807831, -53.1586914, 36.0457268, -73.0846405, 78.0394592
6: -39.2171059, 26.8512917, -55.5044479, 39.2829056, -78.5000000, 82.3557434
7: -34.2506599, 31.5664749, -49.5798950, 45.3233414, -79.5739899, 81.1463699
8: -46.7221947, 26.9137268, -66.7850571, 38.9231262, -85.6453094, 93.6987762
9: -32.0445175, 32.4881477, -46.1535759, 46.8444901, -78.8890076, 78.6417160

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
time: 9.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
time: 9.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -42.9759293, 32.1972351, -44.3326950, 33.6230125, -76.5989380, 76.5299301
1: -33.1128654, 30.5186424, -34.8999710, 31.6394730, -64.7523346, 65.4186096
2: -45.0236664, 30.0465298, -46.9655647, 31.2172318, -76.2408905, 77.0120850
3: -50.7500114, 25.6291046, -52.4934998, 26.7619858, -77.5119934, 78.1226044
4: -49.3081818, 32.0276031, -50.6241150, 33.8063774, -83.1145477, 82.6517181
5: -43.4173050, 28.9366531, -44.4777985, 30.4115887, -73.8288956, 73.4144363
6: -45.8698502, 30.9571648, -46.5249214, 33.4067535, -79.2766037, 77.4820862
7: -39.8950768, 36.8984833, -41.7798500, 38.0330276, -77.9281006, 78.6783142
8: -54.5415649, 31.0423737, -56.1293030, 33.0994377, -87.6410065, 87.1716766
9: -37.3058853, 37.8575325, -38.8961182, 39.4593658, -76.7652512, 76.7536392

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 6.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 6.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -42.1647034, 31.5661430, -49.3820000, 37.3544426, -79.5191422, 80.9481201
1: -32.4632950, 29.9484825, -38.8528214, 35.1975937, -67.6608810, 68.8013000
2: -44.1515884, 29.4783173, -52.3069611, 34.7022667, -78.8538513, 81.7852783
3: -49.7989883, 25.1401424, -58.4690132, 29.7291794, -79.5281677, 83.6091537
4: -48.4082527, 31.3890495, -56.5116196, 37.5103531, -85.9185944, 87.9006653
5: -42.6205902, 28.3523064, -49.6108818, 33.7230606, -76.3436508, 77.9631805
6: -45.0777931, 30.2834148, -51.9689980, 36.9944382, -82.0722122, 82.2524109
7: -39.1218719, 36.2116814, -46.5093956, 42.3902016, -81.5120697, 82.7210770
8: -53.5448074, 30.4193726, -62.6037560, 36.7547836, -90.2995758, 93.0231323
9: -36.5868301, 37.1389771, -43.3039169, 43.9615135, -80.5483398, 80.4428940

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 10.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -45.8831482, 34.8190536, -36.7280846, 27.6646996, -73.5478516, 71.5471344
1: -36.1050339, 32.7193146, -28.4664154, 26.2053585, -62.3103867, 61.1857300
2: -48.6059608, 32.3003120, -38.6123466, 25.7999229, -74.4058838, 70.9126511
3: -54.2904091, 27.6711502, -43.4643974, 22.0956497, -76.3860550, 71.1355438
4: -52.3488693, 35.0008011, -42.1785355, 27.6544113, -80.0032806, 77.1793213
5: -46.0224190, 31.4999313, -37.0389252, 24.8807831, -70.9031982, 68.5388565
6: -48.0753975, 34.6196404, -39.2171059, 26.8512917, -74.9266891, 73.8367462
7: -43.2348099, 39.3503342, -34.2506599, 31.5664749, -74.8012848, 73.6009979
8: -58.0194626, 34.2389603, -46.7221947, 26.9137268, -84.9331894, 80.9611435
9: -40.2577782, 40.8078308, -32.0445175, 32.4881477, -72.7459106, 72.8523407

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
time: 9.06 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
time: 9.26 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -45.8831482, 34.8190536, -42.9759293, 32.1972351, -78.0803833, 77.7949829
1: -36.1050339, 32.7193146, -33.1128654, 30.5186424, -66.6236725, 65.8321838
2: -48.6059608, 32.3003120, -45.0236664, 30.0465298, -78.6524811, 77.3239746
3: -54.2904091, 27.6711502, -50.7500114, 25.6291046, -79.9195099, 78.4211502
4: -52.3488693, 35.0008011, -49.3081818, 32.0276031, -84.3764725, 84.3089676
5: -46.0224190, 31.4999313, -43.4173050, 28.9366531, -74.9590759, 74.9172363
6: -48.0753975, 34.6196404, -45.8698502, 30.9571648, -79.0325546, 80.4894867
7: -43.2348099, 39.3503342, -39.8950768, 36.8984833, -80.1332855, 79.2454071
8: -58.0194626, 34.2389603, -54.5415649, 31.0423737, -89.0618362, 88.7805252
9: -40.2577782, 40.8078308, -37.3058853, 37.8575325, -78.1152878, 78.1137161

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
time: 10.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
time: 9.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -52.8844795, 39.9176064, -35.5111732, 26.7373600, -79.6218414, 75.4287796
1: -41.3286514, 37.5414619, -27.5455341, 25.3689556, -66.6976089, 65.0869980
2: -55.8145599, 37.0655212, -37.3420296, 24.9512329, -80.7657928, 74.4075470
3: -62.4288521, 31.6118279, -42.0624390, 21.3974552, -83.8263092, 73.6742554
4: -60.3328934, 39.9323578, -40.8259277, 26.7342663, -87.0671463, 80.7582855
5: -53.1586914, 36.0457268, -35.8248940, 24.0434818, -77.2021637, 71.8706055
6: -55.5044479, 39.2829056, -37.9853401, 25.9367695, -81.4412155, 77.2682495
7: -49.5798950, 45.3233414, -33.1252594, 30.5387230, -80.1186218, 78.4486008
8: -66.7850571, 38.9231262, -45.2404480, 26.0466270, -92.8316803, 84.1635742
9: -46.1535759, 46.8444901, -30.9900475, 31.4443703, -77.5979462, 77.8345337

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -51.9948158, 39.2261734, -40.1694260, 30.1456108, -82.1404266, 79.3955994
1: -40.6089859, 36.9119568, -31.1567230, 28.6339207, -69.2429047, 68.0686798
2: -54.8555756, 36.4451294, -42.2568970, 28.1376476, -82.9932251, 78.7020111
3: -61.3828278, 31.0756607, -47.5994568, 24.1382256, -85.5210419, 78.6751175
4: -59.3465080, 39.2279015, -46.2917938, 30.1309242, -89.4774323, 85.5196991
5: -52.2839317, 35.4048615, -40.6001511, 27.0523834, -79.3363037, 76.0050125
6: -54.6363182, 38.5392647, -43.0602341, 29.1610546, -83.7973709, 81.5994797
7: -48.7269402, 44.5717545, -37.4557037, 34.5758972, -83.3028412, 82.0274506
8: -65.6975174, 38.2340851, -51.2469406, 29.3568306, -95.0543365, 89.4810181
9: -45.3620911, 46.0515404, -35.0368271, 35.5746040, -80.9366913, 81.0883636

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 6.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -45.8831482, 34.8190536, -45.8831482, 34.8190536, -80.7022018, 80.7022018
1: -36.1050339, 32.7193146, -36.1050339, 32.7193146, -68.8243484, 68.8243484
2: -48.6059608, 32.3003120, -48.6059608, 32.3003120, -80.9062653, 80.9062653
3: -54.2904091, 27.6711502, -54.2904091, 27.6711502, -81.9615555, 81.9615555
4: -52.3488693, 35.0008011, -52.3488693, 35.0008011, -87.3496704, 87.3496704
5: -46.0224190, 31.4999313, -46.0224190, 31.4999313, -77.5223541, 77.5223541
6: -48.0753975, 34.6196404, -48.0753975, 34.6196404, -82.6950378, 82.6950302
7: -43.2348099, 39.3503342, -43.2348099, 39.3503342, -82.5851440, 82.5851440
8: -58.0194626, 34.2389603, -58.0194626, 34.2389603, -92.2584229, 92.2584229
9: -40.2577782, 40.8078308, -40.2577782, 40.8078308, -81.0655899, 81.0655975

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
time: 10.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
time: 9.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -45.8831482, 34.8190536, -52.8844795, 39.9176064, -85.8007507, 87.7035294
1: -36.1050339, 32.7193146, -41.3286514, 37.5414619, -73.6464920, 74.0479660
2: -48.6059608, 32.3003120, -55.8145599, 37.0655212, -85.6714783, 88.1148682
3: -54.2904091, 27.6711502, -62.4288521, 31.6118279, -85.9022293, 90.0999985
4: -52.3488693, 35.0008011, -60.3328934, 39.9323578, -92.2812271, 95.3336792
5: -46.0224190, 31.4999313, -53.1586914, 36.0457268, -82.0681458, 84.6586227
6: -48.0753975, 34.6196404, -55.5044479, 39.2829056, -87.3582916, 90.1240845
7: -43.2348099, 39.3503342, -49.5798950, 45.3233414, -88.5581360, 88.9302292
8: -58.0194626, 34.2389603, -66.7850571, 38.9231262, -96.9425888, 101.0240173
9: -40.2577782, 40.8078308, -46.1535759, 46.8444901, -87.1022491, 86.9614029

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
time: 10.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
time: 11.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -52.8844795, 39.9176064, -44.3326950, 33.6230125, -86.5074921, 84.2503052
1: -41.3286514, 37.5414619, -34.8999710, 31.6394730, -72.9681244, 72.4414215
2: -55.8145599, 37.0655212, -46.9655647, 31.2172318, -87.0317841, 84.0310822
3: -62.4288521, 31.6118279, -52.4934998, 26.7619858, -89.1908417, 84.1053314
4: -60.3328934, 39.9323578, -50.6241150, 33.8063774, -94.1392670, 90.5564728
5: -53.1586914, 36.0457268, -44.4777985, 30.4115887, -83.5702744, 80.5235062
6: -55.5044479, 39.2829056, -46.5249214, 33.4067535, -88.9112015, 85.8078308
7: -49.5798950, 45.3233414, -41.7798500, 38.0330276, -87.6129227, 87.1031799
8: -66.7850571, 38.9231262, -56.1293030, 33.0994377, -99.8844833, 95.0524292
9: -46.1535759, 46.8444901, -38.8961182, 39.4593658, -85.6129456, 85.7406082

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 6.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.25 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -51.9948158, 39.2261734, -49.3820000, 37.3544426, -89.3492584, 88.6081543
1: -40.6089859, 36.9119568, -38.8528214, 35.1975937, -75.8065796, 75.7647781
2: -54.8555756, 36.4451294, -52.3069611, 34.7022667, -89.5578461, 88.7520905
3: -61.3828278, 31.0756607, -58.4690132, 29.7291794, -91.1119995, 89.5446548
4: -59.3465080, 39.2279015, -56.5116196, 37.5103531, -96.8568573, 95.7395172
5: -52.2839317, 35.4048615, -49.6108818, 33.7230606, -86.0069885, 85.0157394
6: -54.6363182, 38.5392647, -51.9689980, 36.9944382, -91.6307449, 90.5082550
7: -48.7269402, 44.5717545, -46.5093956, 42.3902016, -91.1171417, 91.0811462
8: -65.6975174, 38.2340851, -62.6037560, 36.7547836, -102.4522858, 100.8378296
9: -45.3620911, 46.0515404, -43.3039169, 43.9615135, -89.3235931, 89.3554535

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 7.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
time: 6.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.74 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.74
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -33.7678146, 25.3830261, -36.7280846, 27.6646996, -61.4325066, 62.1111107
1: -26.1093979, 24.1229401, -28.4664154, 26.2053585, -52.3147507, 52.5893517
2: -35.4354095, 23.7361279, -38.6123466, 25.7999229, -61.2353325, 62.3484726
3: -39.9777794, 20.3199406, -43.4643974, 22.0956497, -62.0734253, 63.7843399
4: -38.8525925, 25.3513680, -42.1785355, 27.6544113, -66.5070038, 67.5299072
5: -34.1002693, 22.7953033, -37.0389252, 24.8807831, -58.9810524, 59.8342247
6: -36.2500801, 24.4688549, -39.2171059, 26.8512917, -63.1013603, 63.6859589
7: -31.4317207, 29.0526142, -34.2506599, 31.5664749, -62.9981956, 63.3032761
8: -43.0566597, 24.6713142, -46.7221947, 26.9137268, -69.9703827, 71.3935089
9: -29.4266415, 29.8577557, -32.0445175, 32.4881477, -61.9147873, 61.9022713

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2937972, upper bound: 81.2937972
time: 8.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2937972, upper bound: 81.2937972
time: 8.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -38.1350021, 28.5710354, -35.9208221, 27.0376282, -65.1726303, 64.4918518
1: -29.4735260, 27.1868324, -27.8192787, 25.6372490, -55.1107750, 55.0061111
2: -40.0413857, 26.7278824, -37.7432976, 25.2356415, -65.2770233, 64.4711685
3: -45.1914978, 22.8951454, -42.5159721, 21.6089687, -66.8004684, 65.4111176
4: -44.0099106, 28.5249786, -41.2829323, 27.0186520, -71.0285645, 69.8078918
5: -38.6059990, 25.5964947, -36.2439423, 24.3014851, -62.9074593, 61.8404312
6: -41.0689697, 27.4315929, -38.4274597, 26.1812325, -67.2501984, 65.8590546
7: -35.4992714, 32.8524094, -33.4792099, 30.8852043, -66.3844757, 66.3316193
8: -48.7374268, 27.7502899, -45.7304344, 26.2917862, -75.0292053, 73.4807129
9: -33.2247086, 33.7363853, -31.3280830, 31.7711143, -64.9958115, 65.0644608

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2921351, upper bound: 81.2924189
time: 7.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2916497, upper bound: 81.2916497
time: 7.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -33.7678146, 25.3830261, -42.9759293, 32.1972351, -65.9650421, 68.3589554
1: -26.1093979, 24.1229401, -33.1128654, 30.5186424, -56.6280403, 57.2358055
2: -35.4354095, 23.7361279, -45.0236664, 30.0465298, -65.4819336, 68.7597961
3: -39.9777794, 20.3199406, -50.7500114, 25.6291046, -65.6068802, 71.0699539
4: -38.8525925, 25.3513680, -49.3081818, 32.0276031, -70.8801880, 74.6595459
5: -34.1002693, 22.7953033, -43.4173050, 28.9366531, -63.0369225, 66.2126083
6: -36.2500801, 24.4688549, -45.8698502, 30.9571648, -67.2072449, 70.3387070
7: -31.4317207, 29.0526142, -39.8950768, 36.8984833, -68.3302002, 68.9476929
8: -43.0566597, 24.6713142, -54.5415649, 31.0423737, -74.0990295, 79.2128754
9: -29.4266415, 29.8577557, -37.3058853, 37.8575325, -67.2841721, 67.1636429

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
time: 10.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
time: 10.06 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 21.52 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -81.2937972, upper bound: 81.2937972
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -81.2937972, upper bound: 81.2937972
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -81.2921351, upper bound: 81.2924189
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -81.2916497, upper bound: 81.2916497
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.52
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2890581, upper bound: 81.2869851
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2877232, upper bound: 81.2863322
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2889515, upper bound: 81.2869351
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2876257, upper bound: 81.2862585
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.52
Output dim: 6, lower bound: -81.2844304, upper bound: 81.2844304
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=90.90605926513672
rel_dist={6: [-81.29731319725406, 81.29731319725403]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2970560, upper bound: 81.2970610
time: 8.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2970382, upper bound: 81.2970382
time: 7.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 16.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 16.72
Output dim: 6, lower bound: -81.2970560, upper bound: 81.2970610
IS_A2, status: Status.UNKNOWN, split count: 1, time: 16.72
Output dim: 6, lower bound: -81.2970382, upper bound: 81.2970382

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -39.4353180, 29.7706108, -48.3500061, 36.7482758, -76.1835938, 78.1206207
1: -30.7156715, 28.1498985, -38.1755028, 34.5130424, -65.2287140, 66.3254013
2: -41.5748978, 27.6886997, -51.3174629, 34.0195160, -75.5944138, 79.0061646
3: -46.7086945, 23.7850609, -57.2524452, 29.2160530, -75.9247437, 81.0374908
4: -45.2326088, 29.8140793, -55.1307907, 36.9761887, -82.2087936, 84.9448700
5: -39.7052155, 26.8396111, -48.4430389, 33.2963943, -73.0016098, 75.2826462
6: -41.8891563, 29.1658058, -50.5001144, 36.7519073, -78.6410675, 79.6659241
7: -36.8842659, 33.8704376, -45.6548004, 41.4399605, -78.3242188, 79.5252304
8: -50.0938263, 29.0696526, -61.0850372, 36.2256279, -86.3194427, 90.1546936
9: -34.4803238, 34.9603157, -42.4856110, 43.0785141, -77.5588379, 77.4459152

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2892520, upper bound: 81.2900854
time: 9.91 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2883224, upper bound: 81.2883224
time: 8.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -48.7763367, 37.0808640, -49.8580780, 37.8991852, -86.6755219, 86.9389191
1: -38.5242729, 34.8119087, -39.3895531, 35.5770187, -74.1012878, 74.2014542
2: -51.7800751, 34.3216591, -52.9377670, 35.0723801, -86.8524475, 87.2594147
3: -57.7584953, 29.4703693, -59.0415802, 30.1137619, -87.8722458, 88.5119400
4: -55.6023178, 37.3147583, -56.8241348, 38.1415443, -93.7438660, 94.1388931
5: -48.8602371, 33.6030312, -49.9406586, 34.3550301, -83.2152557, 83.5436783
6: -50.9166870, 37.1101799, -52.0146790, 37.9520340, -88.8686981, 89.1248627
7: -46.0756607, 41.8013268, -47.1120796, 42.7232666, -88.7989273, 88.9134064
8: -61.6037064, 36.5603180, -62.9433784, 37.3667450, -98.9704437, 99.5036926
9: -42.8696823, 43.4593582, -43.8228493, 44.4231415, -87.2928238, 87.2821884

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2892519, upper bound: 81.2900854
time: 8.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2883228, upper bound: 81.2883224
time: 7.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 17.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 17.43
Output dim: 6, lower bound: -81.2892520, upper bound: 81.2900854
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 17.43
Output dim: 6, lower bound: -81.2883224, upper bound: 81.2883224
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 17.43
Output dim: 6, lower bound: -81.2892519, upper bound: 81.2900854
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 17.43
Output dim: 6, lower bound: -81.2883228, upper bound: 81.2883224

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -38.9453735, 29.3895073, -45.4891968, 34.5105896, -73.4559631, 74.8787079
1: -30.3070889, 27.7972946, -35.7824745, 32.4428482, -62.7499390, 63.5797691
2: -41.0386276, 27.3468456, -48.1788292, 32.0202103, -73.0588379, 75.5256577
3: -46.1214600, 23.4791126, -53.8212891, 27.4356823, -73.5571442, 77.3003922
4: -44.6807213, 29.4230614, -51.9130249, 34.6872063, -79.3679276, 81.3360901
5: -39.2231941, 26.4848518, -45.6361923, 31.2166214, -70.4398117, 72.1210327
6: -41.4065781, 28.7453575, -47.6901855, 34.2878342, -75.6944122, 76.4355392
7: -36.4076729, 33.4534950, -42.8460579, 39.0150146, -75.4226837, 76.2995529
8: -49.4849091, 28.6789169, -57.5398560, 33.9293098, -83.4142151, 86.2187729
9: -34.0395508, 34.5123100, -39.9026489, 40.4551430, -74.4946899, 74.4149551

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
time: 10.24 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860917
time: 8.96 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -37.2631569, 28.0743217, -52.4761009, 39.5996170, -76.8627701, 80.5504227
1: -28.8964138, 26.5790710, -40.9959526, 37.2558365, -66.1522522, 67.5750198
2: -39.1857758, 26.1716270, -55.3732605, 36.7756653, -75.9614258, 81.5448761
3: -44.1044006, 22.4163933, -61.9442825, 31.3685646, -75.4729614, 84.3606720
4: -42.7844505, 28.0700378, -59.8817101, 39.6087685, -82.3932114, 87.9517517
5: -37.5687828, 25.2572498, -52.7572060, 35.7531624, -73.3219452, 78.0144501
6: -39.7538948, 27.2845459, -55.1023254, 38.9423065, -78.6961975, 82.3868637
7: -34.7617416, 32.0194054, -49.1801758, 44.9757538, -79.7374954, 81.1995773
8: -47.3899155, 27.3244820, -66.2866592, 38.6043777, -85.9942780, 93.6111374
9: -32.5162697, 32.9678612, -45.7873268, 46.4796867, -78.9959564, 78.7551880

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
time: 7.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
time: 7.90 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -48.2528763, 36.6717491, -46.9612503, 35.6361313, -83.8890076, 83.6329956
1: -38.0863495, 34.4332123, -36.9670639, 33.4824753, -71.5688248, 71.4002762
2: -51.2054977, 33.9561691, -49.7595978, 33.0489807, -84.2544785, 83.7157593
3: -57.1308899, 29.1448727, -55.5691681, 28.3135643, -85.4444351, 84.7140427
4: -55.0143890, 36.8956680, -53.5687790, 35.8238983, -90.8382874, 90.4644470
5: -48.3471794, 33.2222977, -47.1003418, 32.2492790, -80.5964432, 80.3226395
6: -50.4036903, 36.6585655, -49.1719437, 35.4576492, -85.8613281, 85.8305054
7: -45.5614548, 41.3583183, -44.2673759, 40.2704849, -85.8319397, 85.6256943
8: -60.9557953, 36.1393318, -59.3556938, 35.0418167, -95.9976120, 95.4950180
9: -42.3968086, 42.9796181, -41.2076187, 41.7689667, -84.1657715, 84.1872253

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866315
time: 10.09 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 8.93 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -46.2781410, 35.1114731, -53.9740372, 40.7430878, -87.0212250, 89.0854950
1: -36.4073334, 32.9871902, -42.2013321, 38.3125114, -74.7198486, 75.1885223
2: -49.0173264, 32.5716934, -56.9791374, 37.8261070, -86.8434296, 89.5508270
3: -54.7557259, 27.8968391, -63.7181702, 32.2612801, -87.0169907, 91.6150055
4: -52.7936935, 35.2961922, -61.5625305, 40.7646637, -93.5583572, 96.8587189
5: -46.4131660, 31.7663689, -54.2456818, 36.8029900, -83.2161484, 86.0120544
6: -48.4752579, 34.9171906, -56.6078873, 40.1333046, -88.6085663, 91.5250778
7: -43.6013947, 39.6818657, -50.6227074, 46.2529755, -89.8543701, 90.3045578
8: -58.5076180, 34.5260468, -68.1351013, 39.7351189, -98.2427368, 102.6611481
9: -40.5975609, 41.1495819, -47.1130180, 47.8159981, -88.4135590, 88.2626038

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
time: 10.09 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
time: 8.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.57 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.57
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.57
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860917
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.57
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.57
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.57
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866315
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.57
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.57
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.57
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -35.9119606, 27.0518684, -44.3907433, 33.6592941, -69.5712509, 71.4426041
1: -27.8819427, 25.6612568, -34.8966064, 31.6638260, -59.5457687, 60.5578613
2: -37.7864151, 25.2318325, -46.9970551, 31.2528954, -69.0393066, 72.2288895
3: -42.5447235, 21.6518593, -52.5242729, 26.7717514, -69.3164749, 74.1761322
4: -41.2787971, 27.0579681, -50.6839905, 33.8257561, -75.1045532, 77.7419586
5: -36.2189484, 24.3372993, -44.5504074, 30.4335823, -66.6525116, 68.8877106
6: -38.3798752, 26.2869072, -46.6003036, 33.3874245, -71.7672958, 72.8872070
7: -33.5202293, 30.8819275, -41.7925072, 38.0830803, -71.6033096, 72.6744385
8: -45.7424278, 26.3710480, -56.1886520, 33.0866394, -78.8290482, 82.5596924
9: -31.3555813, 31.8137512, -38.9259262, 39.4743881, -70.8299637, 70.7396698

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862971, upper bound: 81.2866315
time: 11.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862971, upper bound: 81.2866306
time: 8.48 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -40.5749664, 30.4639149, -43.3592453, 32.8465805, -73.4215393, 73.8231583
1: -31.5004234, 28.9309387, -34.0533714, 30.9286118, -62.4290352, 62.9843102
2: -42.7060280, 28.4213371, -45.8797264, 30.5287285, -73.2347565, 74.3010635
3: -48.0870323, 24.3957005, -51.3079529, 26.1438847, -74.2309189, 75.7036514
4: -46.7478371, 30.4593105, -49.5458603, 32.9994164, -79.7472534, 80.0051727
5: -40.9983139, 27.3494663, -43.5409546, 29.6739712, -70.6722870, 70.8904114
6: -43.4582710, 29.5174408, -45.6128960, 32.5008926, -75.9591675, 75.1303329
7: -37.8543510, 34.9223328, -40.7953949, 37.2139130, -75.0682526, 75.7177277
8: -51.7524567, 29.6857338, -54.9351463, 32.2712708, -84.0237274, 84.6208801
9: -35.4056435, 35.9493408, -38.0025291, 38.5513039, -73.9569473, 73.9518738

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 10.31 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 11.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -34.2901573, 25.7833252, -51.3980827, 38.7681160, -73.0582733, 77.1814117
1: -26.5288410, 24.4888325, -40.1303711, 36.4950867, -63.0239182, 64.6192017
2: -35.9953079, 24.0989132, -54.2149887, 36.0265236, -72.0218277, 78.3138885
3: -40.6023712, 20.6324310, -60.6767807, 30.7219772, -71.3243484, 81.3092117
4: -39.4458694, 25.7558594, -58.6783333, 38.7650299, -78.2108841, 84.4341888
5: -34.6197586, 23.1607475, -51.6914597, 34.9887733, -69.6085281, 74.8522034
6: -36.7776642, 24.8890457, -54.0324440, 38.0624771, -74.8401413, 78.9214935
7: -31.9298515, 29.4958725, -48.1510239, 44.0619698, -75.9918213, 77.6468964
8: -43.7105827, 25.0709801, -64.9611053, 37.7825394, -81.4931030, 90.0320740
9: -29.8861389, 30.3261337, -44.8319511, 45.5203018, -75.4064407, 75.1580811

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
time: 11.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
time: 9.30 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -38.8842316, 29.1443443, -50.3638191, 37.9565239, -76.8407516, 79.5081635
1: -30.0759830, 27.7023106, -39.2872963, 35.7613487, -65.8373184, 66.9896088
2: -40.8417587, 27.2391644, -53.0942535, 35.3017464, -76.1435089, 80.3334198
3: -46.0632477, 23.3330879, -59.4626427, 30.0951672, -76.1584167, 82.7957306
4: -44.8520012, 29.0983276, -57.5394592, 37.9362411, -82.7882309, 86.6377792
5: -39.3365364, 26.1234474, -50.6782990, 34.2320747, -73.5686111, 76.8017426
6: -41.7964516, 28.0467491, -53.0399704, 37.1758842, -78.9723206, 81.0867004
7: -36.2017326, 33.4785652, -47.1539459, 43.1906891, -79.3924255, 80.6325073
8: -49.6509247, 28.3280487, -63.7023773, 36.9672699, -86.6181946, 92.0304260
9: -33.8782921, 34.3939705, -43.9087486, 44.5958023, -78.4740753, 78.3027191

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
time: 7.93 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
time: 7.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -44.8475266, 34.0331192, -45.8453674, 34.7727547, -79.6202850, 79.8784866
1: -35.3435402, 32.0194778, -36.0683784, 32.6918449, -68.0353851, 68.0878601
2: -47.5396004, 31.5788803, -48.5587769, 32.2707176, -79.8103180, 80.1376572
3: -53.1152573, 27.0910015, -54.2540398, 27.6404037, -80.7556458, 81.3450394
4: -51.2031593, 34.2275162, -52.3218307, 34.9494705, -86.1526337, 86.5493469
5: -44.9825287, 30.7941589, -45.9987144, 31.4538040, -76.4363174, 76.7928619
6: -47.0261192, 33.8669014, -48.0669250, 34.5425072, -81.5686264, 81.9338226
7: -42.2948875, 38.4709702, -43.1976662, 39.3251114, -81.6199799, 81.6686401
8: -56.7681465, 33.5234947, -57.9845810, 34.1853104, -90.9534531, 91.5080719
9: -39.3674850, 39.9395981, -40.2154732, 40.7735901, -80.1410751, 80.1550751

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
time: 10.96 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866315
time: 10.73 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -50.0301933, 37.8697701, -44.8025475, 33.9513931, -83.9815826, 82.6723175
1: -39.4081802, 35.6772385, -35.2157364, 31.9483242, -71.3565063, 70.8929749
2: -53.0269241, 35.1567230, -47.4288635, 31.5380421, -84.5649643, 82.5855789
3: -59.2480240, 30.1390991, -53.0244598, 27.0055542, -86.2535782, 83.1635437
4: -57.2312813, 38.0417976, -51.1707916, 34.1135368, -91.3448181, 89.2125854
5: -50.2399559, 34.2049370, -44.9783211, 30.6861382, -80.9260941, 79.1832581
6: -52.5886536, 37.5804634, -47.0670815, 33.6470032, -86.2356567, 84.6475449
7: -47.1526718, 42.9352417, -42.1891479, 38.4459419, -85.5986176, 85.1243896
8: -63.4006996, 37.3005905, -56.7155342, 33.3613510, -96.7620392, 94.0161285
9: -43.8925362, 44.5628548, -39.2818222, 39.8401031, -83.7326355, 83.8446503

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860917
time: 11.13 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 11.03 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -42.8998184, 32.4943771, -52.8848114, 39.9014282, -82.8012390, 85.3791885
1: -33.6842499, 30.5927620, -41.3262062, 37.5408516, -71.2250900, 71.9189682
2: -45.3839455, 30.2120819, -55.8075714, 37.0669136, -82.4508591, 86.0196304
3: -50.7684708, 25.8556442, -62.4344025, 31.6066895, -82.3751602, 88.2900467
4: -49.0130157, 32.6498566, -60.3449707, 39.9125824, -88.9255981, 92.9948273
5: -43.0739632, 29.3595829, -53.1685600, 36.0289841, -79.1029510, 82.5281219
6: -45.1217651, 32.1538696, -55.5261688, 39.2425041, -84.3642578, 87.6800385
7: -40.3612442, 36.8152657, -49.5801926, 45.3280563, -85.6893005, 86.3954620
8: -54.3528748, 31.9379654, -66.7939453, 38.9028244, -93.2556992, 98.7319107
9: -37.5939941, 38.1360092, -46.1445999, 46.8445244, -84.4385223, 84.2806091

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
time: 11.06 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
time: 10.68 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -47.9212074, 36.2034378, -51.8415146, 39.0841331, -87.0053406, 88.0449524
1: -37.6178322, 34.1298294, -40.4760628, 36.8006973, -74.4185333, 74.6058731
2: -50.6920624, 33.6741486, -54.6776886, 36.3350258, -87.0270844, 88.3518372
3: -56.7160721, 28.8117523, -61.2116928, 30.9749489, -87.6910095, 90.0234375
4: -54.8668594, 36.3320732, -59.1970787, 39.0774651, -93.9443207, 95.5291443
5: -48.1840820, 32.6455879, -52.1468887, 35.2663040, -83.4503784, 84.7924805
6: -50.5352211, 35.7130127, -54.5262489, 38.3492279, -88.8844452, 90.2392578
7: -45.0660858, 41.1495399, -48.5745697, 44.4490280, -89.5151062, 89.7241058
8: -60.7887726, 35.5586319, -65.5240707, 38.0811462, -98.8699112, 101.0827026
9: -41.9772224, 42.6092682, -45.2134933, 45.9119797, -87.8891983, 87.8227463

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
time: 7.47 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
time: 8.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.70 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2862971, upper bound: 81.2866315
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2862971, upper bound: 81.2866306
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866315
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860917
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2852284, upper bound: 81.2846559
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 17.70
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -35.9119606, 27.0518684, -35.7458038, 26.9073963, -62.8193398, 62.7976646
1: -27.8819427, 25.6612568, -27.6841202, 25.5152588, -53.3972015, 53.3453751
2: -37.7864151, 25.2318325, -37.5584564, 25.1151505, -62.9015656, 62.7902870
3: -42.5447235, 21.6518593, -42.3090591, 21.5056343, -64.0503540, 63.9609184
4: -41.2787971, 27.0579681, -41.0785904, 26.8894939, -68.1682892, 68.1365509
5: -36.2189484, 24.3372993, -36.0657387, 24.1875343, -60.4064751, 60.4030380
6: -38.3798752, 26.2869072, -38.2358398, 26.0569344, -64.4368134, 64.5227509
7: -33.5202293, 30.8819275, -33.3163376, 30.7338696, -64.2540970, 64.1982574
8: -45.7424278, 26.3710480, -45.5074196, 26.1685181, -71.9109344, 71.8784637
9: -31.3555813, 31.8137512, -31.1757545, 31.6162891, -62.9718513, 62.9895058

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
time: 10.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866315
time: 11.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -35.9119606, 27.0518684, -44.7679901, 33.9553909, -69.8673477, 71.8198547
1: -27.8819427, 25.6612568, -35.2060318, 31.9286270, -59.8105698, 60.8672867
2: -37.7864151, 25.2318325, -47.4064789, 31.5209618, -69.3073730, 72.6383133
3: -42.5447235, 21.6518593, -52.9739609, 26.9976444, -69.5423660, 74.6258240
4: -41.2787971, 27.0579681, -51.1002884, 34.1273918, -75.4061890, 78.1582565
5: -36.2189484, 24.3372993, -44.9193192, 30.7059689, -66.9249115, 69.2566223
6: -38.3798752, 26.2869072, -46.9669037, 33.7083855, -72.0882568, 73.2537918
7: -33.5202293, 30.8819275, -42.1655579, 38.4033813, -71.9236145, 73.0474854
8: -45.7424278, 26.3710480, -56.6471481, 33.3850594, -79.1274643, 83.0181808
9: -31.3555813, 31.8137512, -39.2663269, 39.8122025, -71.1677704, 71.0800781

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
time: 10.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866315
time: 12.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -40.5749664, 30.4639149, -34.8297234, 26.1898537, -66.7648087, 65.2936401
1: -31.5004234, 28.9309387, -26.9441795, 24.8694477, -56.3698730, 55.8751144
2: -42.7060280, 28.4213371, -36.5679626, 24.4728203, -67.1788483, 64.9892960
3: -48.0870323, 24.3957005, -41.2321548, 20.9521217, -69.0391541, 65.6278534
4: -46.7478371, 30.4593105, -40.0706177, 26.1591511, -72.9069901, 70.5299225
5: -40.9983139, 27.3494663, -35.1681328, 23.5187855, -64.5170898, 62.5175858
6: -43.4582710, 29.5174408, -37.3595352, 25.2775116, -68.7357712, 66.8769684
7: -37.8543510, 34.9223328, -32.4356537, 29.9632988, -67.8176422, 67.3579865
8: -51.7524567, 29.6857338, -44.3890953, 25.4511356, -77.2035904, 74.0748291
9: -35.4056435, 35.9493408, -30.3592567, 30.8006134, -66.2062531, 66.3085938

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 11.29 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860917
time: 9.56 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -40.5749664, 30.4639149, -43.7236786, 33.1326523, -73.7076111, 74.1875839
1: -31.5004234, 28.9309387, -34.3522186, 31.1839848, -62.6844025, 63.2831497
2: -42.7060280, 28.4213371, -46.2749519, 30.7872391, -73.4932556, 74.6962891
3: -48.0870323, 24.3957005, -51.7423172, 26.3619251, -74.4489594, 76.1380157
4: -46.7478371, 30.4593105, -49.9471893, 33.2909851, -80.0388184, 80.4064865
5: -40.9983139, 27.3494663, -43.8970718, 29.9373016, -70.9356079, 71.2465363
6: -43.4582710, 29.5174408, -45.9658051, 32.8117828, -76.2700424, 75.4832458
7: -37.8543510, 34.9223328, -41.1558571, 37.5228691, -75.3772202, 76.0781860
8: -51.7524567, 29.6857338, -55.3765259, 32.5602455, -84.3126984, 85.0622559
9: -35.4056435, 35.9493408, -38.3315353, 38.8772278, -74.2828674, 74.2808762

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 9.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 9.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -34.2901573, 25.7833252, -42.3002129, 31.6695137, -65.9596634, 68.0835419
1: -26.5288410, 24.4888325, -32.5683365, 30.0463104, -56.5751495, 57.0571671
2: -35.9953079, 24.0989132, -44.2972870, 29.5725632, -65.5678711, 68.3961945
3: -40.6023712, 20.6324310, -49.9515457, 25.2169476, -65.8193207, 70.5839767
4: -39.4458694, 25.7558594, -48.5764923, 31.4857140, -70.9315796, 74.3323441
5: -34.6197586, 23.1607475, -42.7600327, 28.4430180, -63.0627747, 65.9207764
6: -36.7776642, 24.8890457, -45.2247734, 30.3801517, -67.1578140, 70.1138153
7: -31.9298515, 29.4958725, -39.2435760, 36.3293686, -68.2592163, 68.7394333
8: -43.7105827, 25.0709801, -53.7201843, 30.5141106, -74.2246704, 78.7911606
9: -29.8861389, 30.3261337, -36.7074394, 37.2627296, -67.1488647, 67.0335693

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2838479, upper bound: 81.2830839
time: 10.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2832789, upper bound: 81.2827626
time: 10.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -34.2901573, 25.7833252, -51.7761650, 39.0634995, -73.3536530, 77.5594940
1: -26.5288410, 24.4888325, -40.4395370, 36.7584953, -63.2873230, 64.9283676
2: -35.9953079, 24.0989132, -54.6241646, 36.2945480, -72.2898483, 78.7230759
3: -40.6023712, 20.6324310, -61.1240997, 30.9475403, -71.5499115, 81.7565308
4: -39.4458694, 25.7558594, -59.0933380, 39.0663452, -78.5121994, 84.8491821
5: -34.6197586, 23.1607475, -52.0618439, 35.2599678, -69.8797302, 75.2225952
6: -36.7776642, 24.8890457, -54.4009705, 38.3815613, -75.1592178, 79.2900162
7: -31.9298515, 29.4958725, -48.5220108, 44.3826218, -76.3124695, 78.0178833
8: -43.7105827, 25.0709801, -65.4211273, 38.0797806, -81.7903519, 90.4920959
9: -29.8861389, 30.3261337, -45.1708755, 45.8580399, -75.7441559, 75.4970016

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2838479, upper bound: 81.2830839
time: 10.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2832789, upper bound: 81.2827626
time: 10.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -38.8842316, 29.1443443, -41.3338547, 30.9113503, -69.7955780, 70.4781952
1: -30.0759830, 27.7023106, -31.7902546, 29.3675575, -59.4435310, 59.4925652
2: -40.8417587, 27.2391644, -43.2549210, 28.8948727, -69.7366180, 70.4940643
3: -46.0632477, 23.3330879, -48.8228951, 24.6332664, -70.6965179, 72.1559753
4: -44.8520012, 29.0983276, -47.5139084, 30.7191486, -75.5711517, 76.6122131
5: -39.3365364, 26.1234474, -41.8177757, 27.7363968, -67.0729218, 67.9412231
6: -41.7964516, 28.0467491, -44.3020401, 29.5573921, -71.3538361, 72.3487778
7: -36.2017326, 33.4785652, -38.3194504, 35.5147552, -71.7164917, 71.7979965
8: -49.6509247, 28.3280487, -52.5423393, 29.7622147, -79.4131393, 80.8703690
9: -33.8782921, 34.3939705, -35.8493881, 36.4073334, -70.2856216, 70.2433548

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2696539, upper bound: 81.2717054
time: 9.51 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
time: 8.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -38.8842316, 29.1443443, -50.7349434, 38.2465935, -77.1308289, 79.8792877
1: -30.0759830, 27.7023106, -39.5908890, 36.0198288, -66.0958099, 67.2931976
2: -40.8417587, 27.2391644, -53.4959183, 35.5649719, -76.4067307, 80.7350616
3: -46.0632477, 23.3330879, -59.9021759, 30.3164291, -76.3796768, 83.2352600
4: -44.8520012, 29.0983276, -57.9468613, 38.2322006, -83.0841827, 87.0451813
5: -39.3365364, 26.1234474, -51.0416412, 34.4984779, -73.8350143, 77.1650620
6: -41.7964516, 28.0467491, -53.4010582, 37.4897232, -79.2861786, 81.4477921
7: -36.2017326, 33.4785652, -47.5181274, 43.5053558, -79.7070770, 80.9966736
8: -49.6509247, 28.3280487, -64.1534348, 37.2591476, -86.9100571, 92.4814682
9: -33.8782921, 34.3939705, -44.2413025, 44.9271851, -78.8054810, 78.6352692

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2696539, upper bound: 81.2717054
time: 11.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
time: 9.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -44.8475266, 34.0331192, -35.7458038, 26.9073963, -71.7549133, 69.7789154
1: -35.3435402, 32.0194778, -27.6841202, 25.5152588, -60.8587990, 59.7035751
2: -47.5396004, 31.5788803, -37.5584564, 25.1151505, -72.6547470, 69.1373367
3: -53.1152573, 27.0910015, -42.3090591, 21.5056343, -74.6208878, 69.4000549
4: -51.2031593, 34.2275162, -41.0785904, 26.8894939, -78.0926514, 75.3061066
5: -44.9825287, 30.7941589, -36.0657387, 24.1875343, -69.1700592, 66.8598938
6: -47.0261192, 33.8669014, -38.2358398, 26.0569344, -73.0830536, 72.1027298
7: -42.2948875, 38.4709702, -33.3163376, 30.7338696, -73.0287476, 71.7873001
8: -56.7681465, 33.5234947, -45.5074196, 26.1685181, -82.9366455, 79.0309143
9: -39.3674850, 39.9395981, -31.1757545, 31.6162891, -70.9837646, 71.1153488

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862971, upper bound: 81.2866315
time: 9.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862971, upper bound: 81.2866315
time: 9.93 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -44.8475266, 34.0331192, -44.7828751, 33.9662666, -78.8137970, 78.8159790
1: -35.3435402, 32.0194778, -35.2175598, 31.9391251, -67.2826691, 67.2370300
2: -47.5396004, 31.5788803, -47.4222984, 31.5315971, -79.0711975, 79.0011749
3: -53.1152573, 27.0910015, -52.9914627, 27.0060177, -80.1212769, 80.0824661
4: -51.2031593, 34.2275162, -51.1179123, 34.1378975, -85.3410568, 85.3454285
5: -44.9825287, 30.7941589, -44.9347763, 30.7155876, -75.6981125, 75.7289352
6: -47.0261192, 33.8669014, -46.9837952, 33.7180405, -80.7441483, 80.8506851
7: -42.2948875, 38.4709702, -42.1794548, 38.4166336, -80.7115097, 80.6504135
8: -56.7681465, 33.5234947, -56.6661453, 33.3949356, -90.1630707, 90.1896362
9: -39.3674850, 39.9395981, -39.2792740, 39.8254623, -79.1929474, 79.2188721

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
time: 11.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
time: 10.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -50.0301933, 37.8697701, -34.8297234, 26.1898537, -76.2200394, 72.6994858
1: -39.4081802, 35.6772385, -26.9441795, 24.8694477, -64.2776108, 62.6214180
2: -53.0269241, 35.1567230, -36.5679626, 24.4728203, -77.4997406, 71.7246857
3: -59.2480240, 30.1390991, -41.2321548, 20.9521217, -80.2001495, 71.3712540
4: -57.2312813, 38.0417976, -40.0706177, 26.1591511, -83.3904343, 78.1124115
5: -50.2399559, 34.2049370, -35.1681328, 23.5187855, -73.7587433, 69.3730698
6: -52.5886536, 37.5804634, -37.3595352, 25.2775116, -77.8661575, 74.9400024
7: -47.1526718, 42.9352417, -32.4356537, 29.9632988, -77.1159668, 75.3708954
8: -63.4006996, 37.3005905, -44.3890953, 25.4511356, -88.8518219, 81.6896820
9: -43.8925362, 44.5628548, -30.3592567, 30.8006134, -74.6931458, 74.9220963

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860917
time: 8.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 9.63 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -50.0301933, 37.8697701, -43.7384796, 33.1434669, -83.1736450, 81.6082458
1: -39.4081802, 35.6772385, -34.3636780, 31.1944008, -70.6025620, 70.0409164
2: -53.0269241, 35.1567230, -46.2906723, 30.7977638, -83.8246918, 81.4473877
3: -59.2480240, 30.1390991, -51.7597122, 26.3702564, -85.6182785, 81.8988113
4: -57.2312813, 38.0417976, -49.9647141, 33.3014259, -90.5327072, 88.0065155
5: -50.2399559, 34.2049370, -43.9124374, 29.9468765, -80.1868286, 78.1173706
6: -52.5886536, 37.5804634, -45.9825974, 32.8213577, -85.4100113, 83.5630646
7: -47.1526718, 42.9352417, -41.1696892, 37.5360298, -84.6887054, 84.1049347
8: -63.4006996, 37.3005905, -55.3954124, 32.5700569, -95.9707565, 92.6959991
9: -43.8925362, 44.5628548, -38.3444176, 38.8904037, -82.7829361, 82.9072647

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 10.99 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
time: 9.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -42.8998184, 32.4943771, -42.3002129, 31.6695137, -74.5693207, 74.7945862
1: -33.6842499, 30.5927620, -32.5683365, 30.0463104, -63.7305603, 63.1610870
2: -45.3839455, 30.2120819, -44.2972870, 29.5725632, -74.9565048, 74.5093536
3: -50.7684708, 25.8556442, -49.9515457, 25.2169476, -75.9854202, 75.8071899
4: -49.0130157, 32.6498566, -48.5764923, 31.4857140, -80.4987259, 81.2263336
5: -43.0739632, 29.3595829, -42.7600327, 28.4430180, -71.5169830, 72.1196136
6: -45.1217651, 32.1538696, -45.2247734, 30.3801517, -75.5019150, 77.3786469
7: -40.3612442, 36.8152657, -39.2435760, 36.3293686, -76.6906128, 76.0588379
8: -54.3528748, 31.9379654, -53.7201843, 30.5141106, -84.8669739, 85.6581497
9: -37.5939941, 38.1360092, -36.7074394, 37.2627296, -74.8567200, 74.8434448

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2838479, upper bound: 81.2830839
time: 9.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2832789, upper bound: 81.2827626
time: 12.31 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -42.8998184, 32.4943771, -51.8053284, 39.0849342, -81.9847488, 84.2997055
1: -33.6842499, 30.5927620, -40.4619789, 36.7792740, -70.4635239, 71.0547409
2: -45.3839455, 30.2120819, -54.6541519, 36.3164635, -81.7004089, 84.8662338
3: -50.7684708, 25.8556442, -61.1569977, 30.9637184, -81.7321930, 87.0126419
4: -49.0130157, 32.6498566, -59.1271629, 39.0871124, -88.1001129, 91.7770157
5: -43.0739632, 29.3595829, -52.0910721, 35.2799339, -78.3538895, 81.4506378
6: -45.1217651, 32.1538696, -54.4325409, 38.4019890, -83.5237503, 86.5864105
7: -40.3612442, 36.8152657, -48.5489845, 44.4081535, -84.7693939, 85.3642502
8: -54.3528748, 31.9379654, -65.4579468, 38.0997810, -92.4526520, 97.3959122
9: -37.5939941, 38.1360092, -45.1959915, 45.8839836, -83.4779816, 83.3320007

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2838479, upper bound: 81.2830839
time: 9.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2832789, upper bound: 81.2827626
time: 34.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 45.82 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866315
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866315
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860917
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2838479, upper bound: 81.2830839
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2832789, upper bound: 81.2827626
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2838479, upper bound: 81.2830839
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2832789, upper bound: 81.2827626
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2696539, upper bound: 81.2717054
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2696539, upper bound: 81.2717054
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2862971, upper bound: 81.2866315
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2862971, upper bound: 81.2866315
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2862979, upper bound: 81.2866306
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860917
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2852822, upper bound: 81.2860908
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2838479, upper bound: 81.2830839
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2832789, upper bound: 81.2827626
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2838479, upper bound: 81.2830839
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 45.82
Output dim: 6, lower bound: -81.2832789, upper bound: 81.2827626
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 45.82
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 45.82
Output dim: 6, lower bound: -81.2843132, upper bound: 81.2843132
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=90.90605926513672
rel_dist={6: [-81.29714583279736, 81.29714583279738]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2968566, upper bound: 81.2968535
time: 16.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2968432, upper bound: 81.2968432
time: 14.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 30.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 30.59
Output dim: 6, lower bound: -81.2968566, upper bound: 81.2968535
IS_A2, status: Status.UNKNOWN, split count: 1, time: 30.59
Output dim: 6, lower bound: -81.2968432, upper bound: 81.2968432

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -39.4353180, 29.7706108, -46.9520149, 35.6792068, -75.1145172, 76.7226257
1: -30.7156715, 28.1498985, -37.0507011, 33.5261192, -64.2417908, 65.2005997
2: -41.5748978, 27.6886997, -49.8155746, 33.0447159, -74.6196060, 77.5042725
3: -46.7086945, 23.7850609, -55.5943260, 28.3807602, -75.0894470, 79.3793869
4: -45.2326088, 29.8140793, -53.5605392, 35.8951302, -81.1277390, 83.3745956
5: -39.7052155, 26.8396111, -47.0554390, 32.3145142, -72.0197296, 73.8950424
6: -41.8891563, 29.1658058, -49.0986176, 35.6366043, -77.5257492, 78.2644196
7: -36.8842659, 33.8704376, -44.3033638, 40.2508850, -77.1351471, 78.1737976
8: -50.0938263, 29.0696526, -59.3626556, 35.1665115, -85.2603378, 88.4323120
9: -34.4803238, 34.9603157, -41.2455597, 41.8315125, -76.3118362, 76.2058716

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2885335, upper bound: 81.2888504
time: 8.49 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2882181, upper bound: 81.2882181
time: 8.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -48.7763367, 37.0808640, -49.1974602, 37.3996735, -86.1760101, 86.2783127
1: -38.5242729, 34.8119087, -38.8610725, 35.1099205, -73.6341934, 73.6729660
2: -51.7800751, 34.3216591, -52.2309456, 34.6137733, -86.3938446, 86.5526047
3: -57.7584953, 29.4703693, -58.2582932, 29.7210732, -87.4795609, 87.7286606
4: -55.6023178, 37.3147583, -56.0785294, 37.6365585, -93.2388763, 93.3932877
5: -48.8602371, 33.6030312, -49.2812576, 33.8957443, -82.7559814, 82.8842926
6: -50.9166870, 37.1101799, -51.3445435, 37.4378586, -88.3545380, 88.4547272
7: -46.0756607, 41.8013268, -46.4791412, 42.1604385, -88.2360992, 88.2804718
8: -61.6037064, 36.5603180, -62.1254311, 36.8743286, -98.4780197, 98.6857452
9: -42.8696823, 43.4593582, -43.2409248, 43.8346558, -86.7043304, 86.7002640

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2885342, upper bound: 81.2888504
time: 10.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2882181, upper bound: 81.2882174
time: 10.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.80 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 22.80
Output dim: 6, lower bound: -81.2885335, upper bound: 81.2888504
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 22.80
Output dim: 6, lower bound: -81.2882181, upper bound: 81.2882181
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 22.80
Output dim: 6, lower bound: -81.2885342, upper bound: 81.2888504
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 22.80
Output dim: 6, lower bound: -81.2882181, upper bound: 81.2882174

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -37.5738220, 28.3226185, -44.1194458, 33.4626274, -71.0364304, 72.4420624
1: -29.1655807, 26.8109398, -34.6798401, 31.4755402, -60.6411133, 61.4907799
2: -39.5381966, 26.3898067, -46.7081146, 31.0649853, -70.6031799, 73.0979233
3: -44.4776230, 22.6227741, -52.1920433, 26.6157112, -71.0933380, 74.8148193
4: -43.1346207, 28.3288975, -50.3723831, 33.6278915, -76.7625122, 78.7012711
5: -37.8729744, 25.4918861, -44.2724876, 30.2567234, -68.1296997, 69.7643738
6: -40.0542679, 27.5711861, -46.3130417, 33.1979866, -73.2522583, 73.8842316
7: -35.0740204, 32.2861404, -41.5227699, 37.8467102, -72.9207306, 73.8088913
8: -47.7787476, 27.5862389, -55.8501167, 32.8937225, -80.6724701, 83.4363556
9: -32.8056068, 33.2592697, -38.6881409, 39.2313385, -72.0369415, 71.9474106

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2849449, upper bound: 81.2850869
time: 9.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2845505, upper bound: 81.2848841
time: 10.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -35.6973877, 26.8504753, -51.0360489, 38.5042267, -74.2016144, 77.8865204
1: -27.5921841, 25.4519119, -39.8385048, 36.2451973, -63.8373795, 65.2904129
2: -37.4598999, 25.0791702, -53.8237839, 35.7742081, -73.2341080, 78.9029465
3: -42.2320251, 21.4330730, -60.2422829, 30.5124493, -72.7444687, 81.6753540
4: -41.0186844, 26.8143253, -58.2764816, 38.4936638, -79.5123444, 85.0908051
5: -36.0254288, 24.1226158, -51.3279724, 34.7464218, -70.7718353, 75.4505920
6: -38.2087517, 25.9371262, -53.6687889, 37.7890434, -75.9977951, 79.6059113
7: -33.2310867, 30.6842213, -47.7966194, 43.7528954, -76.9839783, 78.4808350
8: -45.4303627, 26.0709724, -64.5135040, 37.5133133, -82.9436798, 90.5844727
9: -31.1003342, 31.5349522, -44.5157928, 45.1987648, -76.2990952, 76.0507431

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
time: 11.06 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143
time: 10.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -46.7968216, 35.5332832, -46.3022232, 35.1369667, -81.9337921, 81.8355103
1: -36.8689384, 33.3802948, -36.4402962, 33.0161133, -69.8850327, 69.8205872
2: -49.6080780, 32.9389687, -49.0542984, 32.5914230, -82.1995010, 81.9932709
3: -55.3861923, 28.2396660, -54.7878265, 27.9211845, -83.3073654, 83.0274963
4: -53.3770294, 35.7312622, -52.8235703, 35.3207436, -88.6977463, 88.5548248
5: -46.9192200, 32.1637573, -46.4418640, 31.7910805, -78.7102966, 78.6056061
6: -48.9740524, 35.4047813, -48.5020866, 34.9450836, -83.9191360, 83.9068604
7: -44.1318359, 40.1249352, -43.6361809, 39.7084122, -83.8402405, 83.7611160
8: -59.1521606, 34.9711075, -58.5393066, 34.5507812, -93.7029266, 93.5104141
9: -41.0826378, 41.6454163, -40.6270676, 41.1816940, -82.2643280, 82.2724838

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2849449, upper bound: 81.2850869
time: 11.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2845505, upper bound: 81.2848841
time: 8.45 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -44.4552307, 33.6750565, -53.3051262, 40.2363243, -84.6915359, 86.9801788
1: -34.8592529, 31.6549683, -41.6655197, 37.8390579, -72.6983032, 73.3204880
2: -47.0053444, 31.2930584, -56.2649536, 37.3580284, -84.3633728, 87.5580139
3: -52.5588989, 26.7415581, -62.9281693, 31.8631516, -84.4220505, 89.6697235
4: -50.7434692, 33.8243141, -60.8081627, 40.2543221, -90.9977722, 94.6324768
5: -44.6243057, 30.4294682, -53.5793533, 36.3380127, -80.9623184, 84.0087967
6: -46.6889420, 33.3235474, -55.9315834, 39.6108780, -86.2998047, 89.2551270
7: -41.7990112, 38.1299057, -49.9826202, 45.6825638, -87.4815750, 88.1125107
8: -56.2455597, 33.0484009, -67.3064117, 39.2369766, -95.4825363, 100.3548050
9: -38.9404068, 39.4639473, -46.5244141, 47.2195320, -86.1599274, 85.9883575

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
time: 9.63 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143
time: 11.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.99 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 6, lower bound: -81.2849449, upper bound: 81.2850869
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 6, lower bound: -81.2845505, upper bound: 81.2848841
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 6, lower bound: -81.2849449, upper bound: 81.2850869
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 6, lower bound: -81.2845505, upper bound: 81.2848841
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.99
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -34.5845795, 26.0189934, -41.6562004, 31.5552216, -66.1398010, 67.6751938
1: -26.7848568, 24.7084236, -32.6956215, 29.7298222, -56.5146790, 57.4040451
2: -36.3307610, 24.3055058, -44.0609245, 29.3454704, -65.6762314, 68.3664246
3: -40.9548378, 20.8268738, -49.2834053, 25.1279774, -66.0828018, 70.1102753
4: -39.7778282, 25.9999924, -47.6160736, 31.6992950, -71.4771118, 73.6160660
5: -34.9094467, 23.3806896, -41.8383827, 28.5016632, -63.4111099, 65.2190628
6: -37.0636024, 25.1586742, -43.8672485, 31.1826706, -68.2462692, 69.0259247
7: -32.2264938, 29.7494926, -39.1627159, 35.7579231, -67.9844055, 68.9122086
8: -44.0806694, 25.3174572, -52.8220215, 31.0075741, -75.0882339, 78.1394806
9: -30.1599102, 30.6022243, -36.4985352, 37.0346832, -67.1945953, 67.1007538

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2831157, upper bound: 81.2832962
time: 8.06 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2830323, upper bound: 81.2831641
time: 11.23 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -39.1932411, 29.3908882, -40.5978775, 30.7137165, -69.9069519, 69.9887695
1: -30.3426170, 27.9311600, -31.8245201, 28.9753532, -59.3179703, 59.7556801
2: -41.1919937, 27.4555397, -42.9108543, 28.5999680, -69.7919617, 70.3663940
3: -46.4331512, 23.5356693, -48.0421829, 24.4817924, -70.9149475, 71.5778427
4: -45.1984863, 29.3543148, -46.4620857, 30.8397121, -76.0382004, 75.8163834
5: -39.6385689, 26.3545971, -40.8095551, 27.7080307, -67.3465881, 67.1641541
6: -42.0959625, 28.3298550, -42.8808975, 30.2466774, -72.3426361, 71.2107468
7: -36.5110207, 33.7441750, -38.1365929, 34.8723221, -71.3833389, 71.8807678
8: -50.0376854, 28.5851765, -51.5474777, 30.1565094, -80.1941986, 80.1326447
9: -34.1644821, 34.6824112, -35.5492859, 36.0885925, -70.2530746, 70.2316895

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2827714, upper bound: 81.2831385
time: 12.65 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2826791, upper bound: 81.2829839
time: 10.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -32.7911606, 24.6106606, -48.6250877, 36.6391869, -69.4303436, 73.2357483
1: -25.2791061, 23.4084740, -37.9006882, 34.5405731, -59.8196793, 61.3091621
2: -34.3443451, 23.0521603, -51.2317123, 34.0943756, -68.4387207, 74.2838745
3: -38.8079872, 19.6942177, -57.4041595, 29.0614624, -67.8694458, 77.0983734
4: -37.7470665, 24.5607224, -55.5792923, 36.6053925, -74.3524628, 80.1400070
5: -33.1361771, 22.0798912, -48.9414520, 33.0351219, -66.1712952, 71.0213318
6: -35.2900734, 23.6067085, -51.2684250, 35.8218880, -71.1119614, 74.8751221
7: -30.4687424, 28.2137566, -45.4898682, 41.7062073, -72.1749420, 73.7036133
8: -41.8327484, 23.8752460, -61.5434952, 35.6725311, -77.5052795, 85.4187393
9: -28.5364399, 28.9520645, -42.3767090, 43.0485611, -71.5849991, 71.3287735

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2693498, upper bound: 81.2685737
time: 12.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
time: 10.38 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -37.3622322, 27.9542694, -47.5408554, 35.7787476, -73.1409760, 75.4951172
1: -28.8063316, 26.6039047, -37.0082512, 33.7695808, -62.5759048, 63.6121521
2: -39.1583862, 26.1758385, -50.0528488, 33.3305168, -72.4889069, 76.2286835
3: -44.2412910, 22.3793030, -56.1350403, 28.4005165, -72.6418076, 78.5143433
4: -43.1333008, 27.8763466, -54.3962708, 35.7236328, -78.8569336, 82.2726135
5: -37.8363914, 25.0219460, -47.8878479, 32.2249603, -70.0613480, 72.9097900
6: -40.2915878, 26.7350922, -50.2524834, 34.8650475, -75.1566238, 76.9875717
7: -34.7096252, 32.1780548, -44.4388123, 40.7975388, -75.5071640, 76.6168671
8: -47.7435989, 27.1093674, -60.2341270, 34.8010139, -82.5446167, 87.3434906
9: -32.5046959, 33.0004845, -41.4063835, 42.0777969, -74.5824814, 74.4068680

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2690235, upper bound: 81.2683687
time: 8.74 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143
time: 10.46 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -43.3999214, 32.9005775, -43.8054390, 33.2018204, -76.6017227, 76.7060165
1: -34.1306877, 30.9719677, -34.4269409, 31.2456951, -65.3763809, 65.3989105
2: -45.9523773, 30.5668793, -46.3680458, 30.8474293, -76.7998047, 76.9349136
3: -51.3778305, 26.1884861, -51.8397713, 26.4119549, -77.7897797, 78.0282440
4: -49.5752869, 33.0687828, -50.0289993, 33.3634338, -82.9387207, 83.0977783
5: -43.5623245, 29.7417393, -43.9732513, 30.0116615, -73.5739822, 73.7149887
6: -45.6036186, 32.6220055, -46.0234871, 32.9010849, -78.5046997, 78.6454926
7: -40.8723946, 37.2440224, -41.2416000, 37.5889702, -78.4613647, 78.4856262
8: -54.9750519, 32.3650665, -55.4672661, 32.6370506, -87.6120987, 87.8323364
9: -38.0612411, 38.6138458, -38.4067841, 38.9527054, -77.0139465, 77.0206299

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2831157, upper bound: 81.2832966
time: 10.17 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2830323, upper bound: 81.2831643
time: 12.66 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -48.4600487, 36.6401367, -42.7207031, 32.3388939, -80.7989426, 79.3608398
1: -38.0927925, 34.5372238, -33.5332184, 30.4702396, -68.5630341, 68.0704269
2: -51.3051567, 34.0584717, -45.1881866, 30.0836830, -81.3888321, 79.2466583
3: -57.3676338, 29.1624527, -50.5621033, 25.7490044, -83.1166382, 79.7245560
4: -55.4748611, 36.7806320, -48.8425217, 32.4830627, -87.9579163, 85.6231537
5: -48.7056961, 33.0605392, -42.9182129, 29.1973572, -77.9030533, 75.9787521
6: -51.0568008, 36.2183990, -45.0088463, 31.9422684, -82.9990692, 81.2272491
7: -45.6126289, 41.6097641, -40.1878891, 36.6790886, -82.2916946, 81.7976456
8: -61.4610291, 36.0271187, -54.1587563, 31.7638168, -93.2248459, 90.1858749
9: -42.4790688, 43.1245804, -37.4317894, 37.9808731, -80.4599457, 80.5563660

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2827714, upper bound: 81.2831411
time: 10.87 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2826791, upper bound: 81.2829848
time: 11.07 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -41.1019669, 31.0812683, -50.8545151, 38.3445625, -79.4465179, 81.9357834
1: -32.1609650, 29.2854881, -39.6967545, 36.1050262, -68.2659912, 68.9822388
2: -43.3990669, 28.9554024, -53.6274490, 35.6514893, -79.0505524, 82.5828476
3: -48.6120949, 24.7200928, -60.0437164, 30.3897457, -79.0018311, 84.7638092
4: -46.9927597, 31.1970901, -58.0702591, 38.3357468, -85.3284912, 89.2673492
5: -41.3058929, 28.0468616, -51.1535835, 34.5996361, -75.9055176, 79.2004471
6: -43.3590698, 30.5865726, -53.4973679, 37.6099586, -80.9690247, 84.0839386
7: -38.5908813, 35.2856064, -47.6377258, 43.6026421, -82.1935120, 82.9233322
8: -52.1199074, 30.4820118, -64.2884674, 37.3644867, -89.4843903, 94.7704773
9: -35.9630775, 36.4770546, -44.3492279, 45.0345268, -80.9975967, 80.8262787

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2693498, upper bound: 81.2685737
time: 11.25 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
time: 10.41 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -46.0801620, 34.7559433, -49.7488708, 37.4694595, -83.5496216, 84.5048141
1: -36.0593109, 32.7868500, -38.7877808, 35.3195457, -71.3788605, 71.5746231
2: -48.6585693, 32.3780899, -52.4257126, 34.8737602, -83.5323334, 84.8037872
3: -54.5149269, 27.6506996, -58.7489548, 29.7165604, -84.2314911, 86.3996582
4: -52.8064651, 34.8470078, -56.8633575, 37.4377785, -90.2442474, 91.7103653
5: -46.3868484, 31.2957878, -50.0783882, 33.7757416, -80.1625900, 81.3741760
6: -48.7389984, 34.1018372, -52.4598694, 36.6366615, -85.3756561, 86.5616913
7: -43.2482948, 39.5868759, -46.5676193, 42.6760902, -85.9243851, 86.1544952
8: -58.5066643, 34.0680542, -62.9527588, 36.4772720, -94.9839325, 97.0208130
9: -40.3093033, 40.9137878, -43.3600540, 44.0455246, -84.3548126, 84.2738190

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 147

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2690235, upper bound: 81.2683687
time: 10.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143
time: 9.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.66 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2831157, upper bound: 81.2832962
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2830323, upper bound: 81.2831641
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2827714, upper bound: 81.2831385
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2826791, upper bound: 81.2829839
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2693498, upper bound: 81.2685737
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2690235, upper bound: 81.2683687
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2831157, upper bound: 81.2832966
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2830323, upper bound: 81.2831643
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2827714, upper bound: 81.2831411
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2826791, upper bound: 81.2829848
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2693498, upper bound: 81.2685737
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2690235, upper bound: 81.2683687
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.66
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -33.0722733, 24.8609295, -39.3860588, 29.8145981, -62.8868713, 64.2469711
1: -25.5741482, 23.6441097, -30.8613434, 28.1279621, -53.7021103, 54.5054436
2: -34.7042122, 23.2557182, -41.6234818, 27.7674198, -62.4716339, 64.8791962
3: -39.1602669, 19.9288692, -46.5973434, 23.7753944, -62.9356613, 66.5262146
4: -38.0598869, 24.8368359, -45.0577888, 29.9445019, -68.0043869, 69.8946228
5: -33.3954697, 22.3365555, -39.5817299, 26.9227867, -60.3182564, 61.9182663
6: -35.5213432, 23.9732246, -41.5884895, 29.3761768, -64.8975220, 65.5617142
7: -30.7793350, 28.4573498, -36.9798851, 33.8284416, -64.6077728, 65.4372253
8: -42.1938782, 24.1809464, -50.0125313, 29.2867203, -71.4805984, 74.1934662
9: -28.8224144, 29.2387581, -34.4854355, 34.9908104, -63.8132248, 63.7241898

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 170

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2824874, upper bound: 81.2826599
time: 10.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2823823, upper bound: 81.2826143
time: 10.49 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -33.0002556, 24.8045044, -47.4291573, 36.1230850, -69.1233368, 72.2336578
1: -25.5171242, 23.5938091, -37.5201530, 33.7833138, -59.3004379, 61.1139603
2: -34.6268311, 23.2055969, -50.2848320, 33.4456062, -68.0724182, 73.4904327
3: -39.0769577, 19.8882256, -56.0780563, 28.6105919, -67.6875458, 75.9662781
4: -37.9810257, 24.7820549, -53.8674622, 36.3435974, -74.3246231, 78.6495056
5: -33.3226891, 22.2855492, -47.4292183, 32.8039207, -66.1266022, 69.7147675
6: -35.4499092, 23.9173183, -49.2400246, 36.1705475, -71.6204453, 73.1573334
7: -30.7138271, 28.3970871, -44.7790833, 40.5704765, -71.2843018, 73.1761703
8: -42.1044540, 24.1296692, -59.8057404, 35.7752342, -77.8796692, 83.9354019
9: -28.7624283, 29.1749954, -41.6446533, 42.1854019, -70.9478302, 70.8196487

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 170

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2824296, upper bound: 81.2825320
time: 10.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2823055, upper bound: 81.2824775
time: 11.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -37.6713753, 28.2267704, -38.3625870, 28.9995766, -66.6709518, 66.5893478
1: -29.1224442, 26.8618927, -30.0167408, 27.4005108, -56.5229530, 56.8786316
2: -39.5559769, 26.3988647, -40.5096092, 27.0464706, -66.6024323, 66.9084778
3: -44.6275406, 22.6339607, -45.4000854, 23.1517696, -67.7793121, 68.0340424
4: -43.4753723, 28.1783829, -43.9434586, 29.1108971, -72.5862732, 72.1218414
5: -38.1199150, 25.3008709, -38.5870552, 26.1543503, -64.2742615, 63.8879242
6: -40.5519066, 27.1309242, -40.6388893, 28.4671021, -69.0189819, 67.7698059
7: -35.0533638, 32.4452705, -35.9880371, 32.9721298, -68.0254898, 68.4333038
8: -48.1406059, 27.4395351, -48.7792892, 28.4607029, -76.6012955, 76.2188187
9: -32.8169022, 33.3142242, -33.5688744, 34.0753860, -66.8922882, 66.8830872

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2644413, upper bound: 81.2652922
time: 10.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2827714, upper bound: 81.2831388
time: 10.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -37.5345688, 28.1231384, -46.1453133, 35.0865135, -72.6210785, 74.2684479
1: -29.0145645, 26.7665195, -36.4474258, 32.8664246, -61.8809891, 63.2139397
2: -39.4101982, 26.3058929, -48.8701782, 32.5453796, -71.9555740, 75.1760712
3: -44.4676437, 22.5544186, -54.5447540, 27.8229713, -72.2906189, 77.0991669
4: -43.3228073, 28.0754776, -52.4470139, 35.2927551, -78.6155624, 80.5224915
5: -37.9832191, 25.2073727, -46.1779060, 31.8361969, -69.8194122, 71.3852768
6: -40.4147491, 27.0264893, -48.0405617, 35.0134010, -75.4281464, 75.0670471
7: -34.9269714, 32.3305397, -43.5197525, 39.4915276, -74.4185028, 75.8502884
8: -47.9711914, 27.3419418, -58.2417870, 34.7202148, -82.6914062, 85.5837250
9: -32.7009468, 33.1929512, -40.4818726, 41.0156937, -73.7166443, 73.6748199

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2643568, upper bound: 81.2651801
time: 11.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2826791, upper bound: 81.2829839
time: 11.56 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -28.8846512, 21.6392212, -45.0756607, 33.9400673, -62.8247147, 66.7148743
1: -22.1839409, 20.6843796, -35.0871010, 32.1425400, -54.3264809, 55.7714767
2: -30.1992950, 20.3917904, -47.5260773, 31.7565765, -61.9558678, 67.9178696
3: -34.1478729, 17.2769585, -53.2530594, 26.8250675, -60.9729271, 70.5300140
4: -33.3867264, 21.5861092, -51.7553711, 33.8569031, -67.2436295, 73.3414764
5: -29.2712669, 19.3687935, -45.4930420, 30.5153618, -59.7865982, 64.8618240
6: -31.3927307, 20.4956455, -47.8860703, 32.8774109, -64.2701416, 68.3817139
7: -26.7487278, 24.9445705, -42.1357803, 38.7777023, -65.5264282, 67.0803528
8: -37.0007324, 20.9858131, -57.2553635, 33.0572968, -70.0580139, 78.2411804
9: -25.1432457, 25.4908199, -39.3242035, 39.9835129, -65.1267548, 64.8150253

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2650872, upper bound: 81.2641972
time: 12.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2648967, upper bound: 81.2640660
time: 10.43 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -31.6918850, 23.7684498, -46.7768097, 35.2162285, -66.9081116, 70.5452576
1: -24.4015961, 22.6439800, -36.4102592, 33.2589569, -57.6605530, 59.0542374
2: -33.1772995, 22.3044090, -49.2645683, 32.8377686, -66.0150681, 71.5689774
3: -37.4962044, 19.0083504, -55.2235184, 27.9154797, -65.4116821, 74.2318726
4: -36.5311775, 23.7186203, -53.5530548, 35.1630363, -71.6942139, 77.2716751
5: -32.0528679, 21.3037300, -47.1309662, 31.7148533, -63.7677231, 68.4346848
6: -34.2129669, 22.7077923, -49.4841843, 34.2815704, -68.4945374, 72.1919708
7: -29.4200096, 27.2968731, -43.7135277, 40.1586113, -69.5786209, 71.0103912
8: -40.4832077, 23.0529900, -59.2812004, 34.2710228, -74.7542114, 82.3341827
9: -27.5828705, 27.9780693, -40.7587967, 41.4123611, -68.9952240, 68.7368622

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2828713, upper bound: 81.2825752
time: 10.26 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2826583, upper bound: 81.2824633
time: 9.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -33.3809013, 24.9254646, -44.0319633, 33.1111145, -66.4920044, 68.9574280
1: -25.6361828, 23.8329353, -34.2318573, 31.4022808, -57.0384598, 58.0647888
2: -34.9275665, 23.4589882, -46.3936539, 31.0215263, -65.9490967, 69.8526459
3: -39.4809265, 19.9216518, -52.0344086, 26.1877823, -65.6687088, 71.9560547
4: -38.6984100, 24.8193226, -50.6156158, 33.0108109, -71.7092209, 75.4349365
5: -33.9050179, 22.2558002, -44.4799652, 29.7340012, -63.6390190, 66.7357483
6: -36.3326340, 23.5305805, -46.9104080, 31.9565811, -68.2892075, 70.4409866
7: -30.9035301, 28.8410721, -41.1257019, 37.9044533, -68.8079834, 69.9667740
8: -42.8081360, 24.1505566, -55.9938431, 32.2180634, -75.0261917, 80.1444016
9: -29.0353508, 29.4787350, -38.3934631, 39.0500107, -68.0853577, 67.8722000

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2647691, upper bound: 81.2639956
time: 12.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2646250, upper bound: 81.2639021
time: 10.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -36.2461624, 27.1018200, -45.7282333, 34.3856125, -70.6317673, 72.8300552
1: -27.9111938, 25.8307209, -35.5488663, 32.5143089, -60.4254951, 61.3795853
2: -37.9755249, 25.4172783, -48.1256218, 32.0991402, -70.0746613, 73.5429001
3: -42.9078064, 21.6865902, -53.9984665, 27.2757359, -70.1835403, 75.6850510
4: -41.9061623, 27.0121822, -52.4080544, 34.3111496, -76.2173157, 79.4202194
5: -36.7380981, 24.2332458, -46.1148071, 30.9310150, -67.6691132, 70.3480530
6: -39.2085152, 25.8139706, -48.5051117, 33.3548508, -72.5633545, 74.3190842
7: -33.6431656, 31.2449265, -42.6972580, 39.2818069, -72.9249649, 73.9421844
8: -46.3717995, 26.2743282, -58.0165634, 33.4286804, -79.8004761, 84.2908783
9: -31.5330315, 32.0186615, -39.8213654, 40.4747620, -72.0077744, 71.8400192

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2647691, upper bound: 81.2824299
time: 11.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2823442, upper bound: 81.2823440
time: 9.94 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -41.6144066, 31.5299835, -41.4227867, 31.3729019, -72.9873047, 72.9527664
1: -32.6858330, 29.7108154, -32.4995728, 29.5629826, -62.2488098, 62.2103844
2: -44.0335464, 29.3257332, -43.8074989, 29.1909580, -73.2244949, 73.1332245
3: -49.2626686, 25.1225624, -49.0169640, 24.9901009, -74.2527695, 74.1395264
4: -47.5601997, 31.6893044, -47.3393478, 31.5219765, -79.0821686, 79.0286484
5: -41.7868881, 28.4984646, -41.6039734, 28.3526707, -70.1395493, 70.1024246
6: -43.8107147, 31.1994781, -43.6297531, 31.0032978, -74.8140106, 74.8292313
7: -39.1529770, 35.7244339, -38.9468842, 35.5609055, -74.7138824, 74.6713181
8: -52.7640915, 31.0107021, -52.5171547, 30.8291492, -83.5932388, 83.5278397
9: -36.4768295, 37.0039330, -36.2921295, 36.8047333, -73.2815399, 73.2960663

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2665166, upper bound: 81.2670339
time: 10.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2831157, upper bound: 81.2832968
time: 10.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -41.4779167, 31.4244919, -49.7236328, 37.9164085, -79.3943253, 81.1481247
1: -32.5763588, 29.6150627, -39.3979492, 35.4161034, -67.9924545, 69.0130157
2: -43.8875351, 29.2319489, -52.7828445, 35.0530624, -78.9405975, 82.0147934
3: -49.1033401, 25.0430965, -58.8229523, 29.9855957, -79.0889359, 83.8660431
4: -47.4078674, 31.5860825, -56.4686394, 38.1359482, -85.5438156, 88.0547180
5: -41.6494179, 28.4035568, -49.7285309, 34.4229546, -76.0723724, 78.1320877
6: -43.6710930, 31.0938206, -51.5360947, 38.0321693, -81.7032623, 82.6299133
7: -39.0260925, 35.6088371, -47.0236130, 42.5497894, -81.5758820, 82.6324463
8: -52.5960388, 30.9114151, -62.6503563, 37.5513992, -90.1474228, 93.5617676
9: -36.3595581, 36.8807907, -43.6798706, 44.2693176, -80.6288757, 80.5606613

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2664718, upper bound: 81.2669727
time: 10.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2830323, upper bound: 81.2831641
time: 11.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -46.6952591, 35.2818756, -40.3302498, 30.5037270, -77.1989822, 75.6121216
1: -36.6622429, 33.2887573, -31.5994568, 28.7826042, -65.4448471, 64.8882141
2: -49.4051247, 32.8292694, -42.6199875, 28.4211998, -77.8263245, 75.4492569
3: -55.2797089, 28.1100979, -47.7311058, 24.3230286, -79.6027222, 75.8411865
4: -53.4839478, 35.4145699, -46.1442299, 30.6355209, -84.1194611, 81.5587997
5: -46.9524422, 31.8275928, -40.5399437, 27.5335922, -74.4860382, 72.3675308
6: -49.2878113, 34.8060532, -42.6074905, 30.0391006, -79.3269119, 77.4135361
7: -43.9122086, 40.1081467, -37.8861008, 34.6444550, -78.5566559, 77.9942474
8: -59.2736282, 34.6827202, -51.1979980, 29.9501381, -89.2237701, 85.8807220
9: -40.9137917, 41.5321121, -35.3107452, 35.8258667, -76.7396545, 76.8428574

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 147

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2645698, upper bound: 81.2655490
time: 10.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2827714, upper bound: 81.2831411
time: 10.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.89 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2824874, upper bound: 81.2826599
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2823823, upper bound: 81.2826143
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2824296, upper bound: 81.2825320
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2823055, upper bound: 81.2824775
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2644413, upper bound: 81.2652922
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2827714, upper bound: 81.2831388
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2643568, upper bound: 81.2651801
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2826791, upper bound: 81.2829839
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2650872, upper bound: 81.2641972
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2648967, upper bound: 81.2640660
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2828713, upper bound: 81.2825752
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2826583, upper bound: 81.2824633
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2647691, upper bound: 81.2639956
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2646250, upper bound: 81.2639021
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2647691, upper bound: 81.2824299
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2823442, upper bound: 81.2823440
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2665166, upper bound: 81.2670339
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2831157, upper bound: 81.2832968
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2664718, upper bound: 81.2669727
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2830323, upper bound: 81.2831641
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2645698, upper bound: 81.2655490
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.89
Output dim: 6, lower bound: -81.2827714, upper bound: 81.2831411
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.89
Output dim: 6, lower bound: -81.2826791, upper bound: 81.2829848
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.89
Output dim: 6, lower bound: -81.2693498, upper bound: 81.2685737
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.89
Output dim: 6, lower bound: -81.2845386, upper bound: 81.2843373
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.89
Output dim: 6, lower bound: -81.2690235, upper bound: 81.2683687
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.89
Output dim: 6, lower bound: -81.2842143, upper bound: 81.2842143
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=90.90605926513672
rel_dist={6: [-81.29696620335748, 81.29696620339547]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1864.25 seconds
