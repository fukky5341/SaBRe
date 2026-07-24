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
execution time: IAR + LP analysis = 1.20 + 8.29 = 9.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -81.2973621, upper bound: 81.2973621


# Binary Search by BASE starts (time budget: 2690.51 seconds, max iter: 100)

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
Binary search time: 36.75 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 2653.76 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2512796, upper bound: 81.2447099
time: 11.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2293967, upper bound: 81.2293967
time: 8.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 20.15
Output dim: 6, lower bound: -81.2512796, upper bound: 81.2447099
IS_A2, status: Status.UNKNOWN, split count: 1, time: 20.15
Output dim: 6, lower bound: -81.2293967, upper bound: 81.2293967

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -49.8440857, 37.8779831, -50.3818474, 38.2952919, -88.1393738, 88.2598267
1: -39.3679924, 35.5630417, -39.8088417, 35.9474564, -75.3154449, 75.3718872
2: -52.9084244, 35.0605469, -53.4981537, 35.4362068, -88.3446274, 88.5587006
3: -59.0251961, 30.1062145, -59.6625023, 30.4251728, -89.4503708, 89.7687149
4: -56.8039589, 38.1175423, -57.4144287, 38.5423965, -95.3463593, 95.5319672
5: -49.9338379, 34.3365517, -50.4630890, 34.7195892, -84.6534271, 84.7996292
6: -52.0132217, 37.9152527, -52.5452843, 38.3607750, -90.3739929, 90.4605331
7: -47.0919571, 42.7135925, -47.6143074, 43.1696014, -90.2615509, 90.3278961
8: -62.9268799, 37.3348656, -63.5914040, 37.7574081, -100.6842880, 100.9262695
9: -43.8035812, 44.4049301, -44.2846413, 44.8899040, -88.6934814, 88.6895752

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2172491, upper bound: 81.2161769
time: 13.36 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2463396, upper bound: 81.2397563
time: 12.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -58.8373566, 44.5105743, -49.8620148, 37.8897858, -96.7271423, 94.3725891
1: -46.4309464, 41.7663078, -39.3805313, 35.5752144, -82.0061493, 81.1468353
2: -62.2661438, 41.2182922, -52.9255180, 35.0725822, -97.3387299, 94.1438141
3: -69.6919250, 35.5211792, -59.0461655, 30.1167507, -99.8086700, 94.5673447
4: -66.8916702, 44.8040695, -56.8261948, 38.1288681, -105.0205383, 101.6302643
5: -58.9493713, 40.3529358, -49.9526825, 34.3469009, -93.2962646, 90.3056183
6: -61.2510452, 44.6034050, -52.0345383, 37.9240570, -99.1751022, 96.6379395
7: -55.5482101, 50.3803253, -47.1078987, 42.7294273, -98.2776260, 97.4882202
8: -74.2525024, 43.8166504, -62.9503937, 37.3442841, -111.5967712, 106.7670288
9: -51.5999146, 52.3046913, -43.8185883, 44.4197083, -96.0196152, 96.1232758

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2196846, upper bound: 81.2191387
time: 7.29 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2180269, upper bound: 81.2180267
time: 5.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.56 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 16.56
Output dim: 6, lower bound: -81.2172491, upper bound: 81.2161769
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 16.56
Output dim: 6, lower bound: -81.2463396, upper bound: 81.2397563
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 16.56
Output dim: 6, lower bound: -81.2196846, upper bound: 81.2191387
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 16.56
Output dim: 6, lower bound: -81.2180269, upper bound: 81.2180267

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -35.9045639, 26.7125473, -47.2005081, 35.7521286, -71.6566925, 73.9130402
1: -27.6878319, 25.7422466, -37.1611328, 33.6985970, -61.3864288, 62.9033813
2: -37.7113190, 25.1617718, -50.0278130, 33.1885719, -70.8998795, 75.1895828
3: -42.5415573, 21.5803776, -55.9120636, 28.4948082, -71.0363617, 77.4924393
4: -41.7203217, 26.5762215, -53.9515343, 35.9415016, -77.6618195, 80.5277557
5: -36.4486809, 23.8585052, -47.3688736, 32.3330536, -68.7817383, 71.2273788
6: -39.1794662, 25.3612213, -49.5857162, 35.5449638, -74.7244034, 74.9469223
7: -33.3830872, 31.0281811, -44.4958954, 40.5011902, -73.8842621, 75.5240784
8: -46.0960007, 25.6204014, -59.7473106, 35.1166840, -81.2126846, 85.3677063
9: -31.2528496, 31.7461548, -41.4240265, 42.0000687, -73.2529144, 73.1701813

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2061757, upper bound: 81.2053547
time: 15.52 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2019535, upper bound: 81.2010426
time: 11.86 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -44.3987198, 33.4985275, -50.3818474, 38.2952919, -82.6940155, 83.8803711
1: -34.7902985, 31.7020683, -39.8088417, 35.9474564, -70.7377548, 71.5109100
2: -46.9426498, 31.1953392, -53.4981537, 35.4362068, -82.3788452, 84.6934967
3: -52.5901108, 26.7635212, -59.6625023, 30.4251728, -83.0152817, 86.4260178
4: -50.8939896, 33.6281929, -57.4144287, 38.5423965, -89.4363861, 91.0426178
5: -44.6671906, 30.2132645, -50.4630890, 34.7195892, -79.3867798, 80.6763535
6: -46.9968910, 33.0107651, -52.5452843, 38.3607750, -85.3576660, 85.5560455
7: -41.7139816, 38.1485977, -47.6143074, 43.1696014, -84.8835754, 85.7628937
8: -56.3489227, 32.7478638, -63.5914040, 37.7574081, -94.1063232, 96.3392639
9: -38.8786430, 39.4352379, -44.2846413, 44.8899040, -83.7685471, 83.7198639

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2398062, upper bound: 81.2336934
time: 11.04 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2365739, upper bound: 81.2296285
time: 12.24 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -49.0273438, 36.8630219, -49.0862770, 37.2779694, -86.3053131, 85.9492950
1: -38.3755302, 34.8979721, -38.7388344, 35.0325851, -73.4081116, 73.6367874
2: -51.7129631, 34.3068275, -52.0858116, 34.5242767, -86.2372437, 86.3926315
3: -58.1820831, 29.5680656, -58.1383514, 29.6445370, -87.8266220, 87.7064209
4: -56.1867104, 36.9556770, -55.9836197, 37.5011826, -93.6878967, 92.9393005
5: -49.4069023, 33.2131233, -49.2042618, 33.7716408, -83.1785431, 82.4173813
6: -51.9148979, 36.2597809, -51.3107567, 37.2455864, -89.1604767, 87.5705414
7: -46.0335693, 42.1451492, -46.3511810, 42.0814552, -88.1150208, 88.4963150
8: -62.2260361, 35.8195915, -62.0024109, 36.6924515, -98.9184875, 97.8219986
9: -42.8816338, 43.4833908, -43.1260414, 43.7183037, -86.5999222, 86.6094284

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2180269, upper bound: 81.2180269
time: 7.29 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2180269, upper bound: 81.2180267
time: 6.59 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -52.6502304, 39.6232262, -49.2740707, 37.4242706, -90.0745010, 88.8972931
1: -41.2809105, 37.4373016, -38.8913383, 35.1638260, -76.4447327, 76.3286285
2: -55.5754433, 36.8362694, -52.2874870, 34.6566772, -90.2321167, 89.1237411
3: -62.4552994, 31.7266712, -58.3582115, 29.7569981, -92.2122955, 90.0848846
4: -60.2324600, 39.7544861, -56.1898804, 37.6506195, -97.8830719, 95.9443588
5: -53.0027924, 35.7450943, -49.3874054, 33.9085693, -86.9113541, 85.1324997
6: -55.5419540, 39.1334114, -51.4910507, 37.4039612, -92.9459152, 90.6244507
7: -49.4960175, 45.2125092, -46.5323372, 42.2387123, -91.7347260, 91.7448425
8: -66.7239532, 38.5853004, -62.2331772, 36.8458900, -103.5698395, 100.8184814
9: -46.0589638, 46.7126465, -43.2920914, 43.8870163, -89.9459763, 90.0047226

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 240

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1655618, upper bound: 81.1732062
time: 7.82 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126
time: 6.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.20 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 18.20
Output dim: 6, lower bound: -81.2061757, upper bound: 81.2053547
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 18.20
Output dim: 6, lower bound: -81.2019535, upper bound: 81.2010426
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 18.20
Output dim: 6, lower bound: -81.2398062, upper bound: 81.2336934
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 18.20
Output dim: 6, lower bound: -81.2365739, upper bound: 81.2296285
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 18.20
Output dim: 6, lower bound: -81.2180269, upper bound: 81.2180269
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 18.20
Output dim: 6, lower bound: -81.2180269, upper bound: 81.2180267
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 18.20
Output dim: 6, lower bound: -81.1655618, upper bound: 81.1732062
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 18.20
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -35.3682098, 26.2919407, -37.5618248, 28.2174530, -63.5856628, 63.8537521
1: -27.2441444, 25.3677273, -29.2471199, 26.9368629, -54.1810074, 54.6148415
2: -37.1343231, 24.7798920, -39.6650887, 26.3902931, -63.5246124, 64.4449768
3: -41.9031830, 21.2516785, -44.5424042, 22.6168251, -64.5199966, 65.7940826
4: -41.1334724, 26.1434841, -43.3729401, 28.2569351, -69.3904037, 69.5164185
5: -35.9279480, 23.4736252, -37.9738541, 25.3187332, -61.2466812, 61.4474792
6: -38.6694679, 24.8994408, -40.3534393, 27.3828011, -66.0522614, 65.2528839
7: -32.8625412, 30.5764694, -35.1406860, 32.3956032, -65.2581406, 65.7171555
8: -45.4265633, 25.1771355, -47.9167252, 27.3082161, -72.7347794, 73.0938492
9: -30.7783184, 31.2620239, -32.8601646, 33.3233795, -64.1016846, 64.1221848

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2009459, upper bound: 81.2003861
time: 12.11 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2023742, upper bound: 81.2016322
time: 10.84 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -35.4793549, 26.3768272, -41.2190285, 31.0037193, -66.4830704, 67.5958557
1: -27.3327980, 25.4453392, -32.1794548, 29.5158558, -56.8486557, 57.6247940
2: -37.2526855, 24.8578434, -43.5597458, 28.9472828, -66.1999664, 68.4175797
3: -42.0340958, 21.3175163, -48.8859749, 24.8034573, -66.8375320, 70.2034912
4: -41.2589073, 26.2292099, -47.4987411, 31.0611324, -72.3200302, 73.7279510
5: -36.0385666, 23.5500908, -41.6246338, 27.8523140, -63.8908806, 65.1747208
6: -38.7817955, 24.9867001, -44.0679741, 30.2334881, -69.0152817, 69.0546722
7: -32.9686012, 30.6707077, -38.6423836, 35.5076828, -68.4762878, 69.3130951
8: -45.5672112, 25.2620201, -52.4811134, 30.0554428, -75.6226501, 77.7431335
9: -30.8752899, 31.3611660, -36.0671883, 36.5959206, -67.4712067, 67.4283524

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1856963, upper bound: 81.1854187
time: 11.97 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1998713, upper bound: 81.1990081
time: 11.76 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -43.6780281, 32.9320869, -40.3181915, 30.4172039, -74.0952301, 73.2502670
1: -34.1957092, 31.1996994, -31.5374908, 28.8900719, -63.0857811, 62.7371712
2: -46.1671829, 30.6863594, -42.6544800, 28.3411732, -74.5083542, 73.3408203
3: -51.7463989, 26.3222351, -47.8169022, 24.2962513, -76.0426483, 74.1391373
4: -50.1147499, 33.0461578, -46.3762703, 30.5100899, -80.6248322, 79.4224243
5: -43.9746132, 29.6794930, -40.6642838, 27.3700848, -71.3446960, 70.3437805
6: -46.3248291, 32.3822746, -42.9351311, 29.7997303, -76.1245499, 75.3174057
7: -41.0129089, 37.5485878, -37.8388519, 34.7167740, -75.7296753, 75.3874359
8: -55.4730034, 32.1458893, -51.2480316, 29.5721779, -85.0451660, 83.3939056
9: -38.2362747, 38.7879639, -35.3336639, 35.8211441, -74.0574188, 74.1216202

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 78

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2359241, upper bound: 81.2292857
time: 12.21 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2359241, upper bound: 81.2296285
time: 10.36 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -43.8674164, 33.0794487, -44.1498489, 33.3438721, -77.2112732, 77.2292862
1: -34.3492889, 31.3317833, -34.6142731, 31.5838737, -65.9331665, 65.9460602
2: -46.3697052, 30.8198776, -46.7391205, 31.0184593, -77.3881683, 77.5589905
3: -51.9679871, 26.4365082, -52.3483582, 26.5885124, -78.5565033, 78.7848587
4: -50.3217926, 33.1969223, -50.6771927, 33.4592247, -83.7810135, 83.8741150
5: -44.1584663, 29.8176594, -44.4739952, 30.0512447, -74.2097015, 74.2916565
6: -46.5060120, 32.5420303, -46.7903175, 32.8221817, -79.3281937, 79.3323441
7: -41.1956329, 37.7066803, -41.5070114, 37.9649277, -79.1605606, 79.2136917
8: -55.7045670, 32.3003922, -56.0065956, 32.4737358, -88.1782837, 88.3069839
9: -38.4039307, 38.9570847, -38.6945877, 39.2471848, -77.6511078, 77.6516724

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 160

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2359241, upper bound: 81.2292857
time: 9.27 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2359241, upper bound: 81.2296285
time: 10.72 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -49.0273438, 36.8630219, -39.8336258, 30.0397701, -79.0671158, 76.6966476
1: -38.3755302, 34.8979721, -31.1394062, 28.5437050, -66.9192352, 66.0373688
2: -51.7129631, 34.3068275, -42.1233215, 28.0023308, -79.7152939, 76.4301453
3: -58.1820831, 29.5680656, -47.2407837, 24.0078812, -82.1899643, 76.8088531
4: -56.1867104, 36.9556770, -45.8258820, 30.1267624, -86.3134766, 82.7815475
5: -49.4069023, 33.2131233, -40.1852188, 27.0256310, -76.4325256, 73.3983307
6: -51.9148979, 36.2597809, -42.4548492, 29.3990936, -81.3139877, 78.7146301
7: -46.0335693, 42.1451492, -37.3684540, 34.3046799, -80.3382492, 79.5135956
8: -62.2260361, 35.8195915, -50.6504593, 29.1925678, -91.4186020, 86.4700394
9: -42.8816338, 43.4833908, -34.9011574, 35.3840218, -78.2656555, 78.3845520

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1800136, upper bound: 81.1747641
time: 9.90 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1721233, upper bound: 81.1689449
time: 8.30 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -49.0273438, 36.8630219, -43.6575699, 32.9598770, -81.9872131, 80.5205917
1: -38.3755302, 34.8979721, -34.2090836, 31.2324352, -69.6079636, 69.1070557
2: -51.7129631, 34.3068275, -46.1991043, 30.6742172, -82.3871765, 80.5059357
3: -58.1820831, 29.5680656, -51.7641296, 26.2961407, -84.4782257, 81.3321838
4: -56.1867104, 36.9556770, -50.1188965, 33.0693855, -89.2560959, 87.0745697
5: -49.4069023, 33.2131233, -43.9894180, 29.7000046, -79.1068954, 77.2025299
6: -51.9148979, 36.2597809, -46.3047256, 32.4126663, -84.3275604, 82.5645065
7: -46.0335693, 42.1451492, -41.0285301, 37.5475540, -83.5811234, 83.1736679
8: -62.2260361, 35.8195915, -55.4011536, 32.0872612, -94.3132858, 91.2207413
9: -42.8816338, 43.4833908, -38.2544670, 38.8040237, -81.6856461, 81.7378464

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1741413, upper bound: 81.1662918
time: 9.42 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1517812, upper bound: 81.1512455
time: 8.20 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -52.6502304, 39.6232262, -46.4057159, 35.1834030, -87.8336334, 86.0289459
1: -41.2809105, 37.4373016, -36.4920959, 33.0899200, -74.3708344, 73.9293900
2: -55.5754433, 36.8362694, -49.1411400, 32.6527214, -88.2281647, 85.9773941
3: -62.4552994, 31.7266712, -54.9183960, 27.9721451, -90.4274445, 86.6450653
4: -60.2324600, 39.7544861, -52.9667892, 35.3552628, -95.5877228, 92.7212524
5: -53.0027924, 35.7450943, -46.5744247, 31.8236504, -84.8264389, 82.3195190
6: -55.5419540, 39.1334114, -48.6757812, 34.9341774, -90.4761276, 87.8091888
7: -49.4960175, 45.2125092, -43.7150803, 39.8093147, -89.3053284, 88.9275818
8: -66.7239532, 38.5853004, -58.6811066, 34.5461464, -101.2700958, 97.2664032
9: -46.0589638, 46.7126465, -40.7023239, 41.2591553, -87.3181152, 87.4149628

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126
time: 6.21 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126
time: 5.68 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -51.6599197, 38.8480225, -53.4261703, 40.2975845, -91.9574966, 92.2741852
1: -40.4437103, 36.7163200, -41.7340851, 37.9260635, -78.3697739, 78.4504089
2: -54.4856567, 36.1431885, -56.3731232, 37.4330177, -91.9186707, 92.5163040
3: -61.2631645, 31.1029549, -63.0807076, 31.9258785, -93.1890259, 94.1836624
4: -59.1248817, 38.9540710, -60.9719467, 40.3044128, -99.4292908, 99.9260178
5: -52.0338745, 35.0198097, -53.7293930, 36.3836899, -88.4175415, 88.7492065
6: -54.5719070, 38.2693253, -56.1215210, 39.6171188, -94.1890259, 94.3908463
7: -48.5188713, 44.3721657, -50.0795898, 45.7980766, -94.3169403, 94.4517517
8: -65.4982986, 37.7858887, -67.4709930, 39.2487373, -104.7470398, 105.2568817
9: -45.1603241, 45.8014259, -46.6161461, 47.3143425, -92.4746475, 92.4175644

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126
time: 7.39 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126
time: 6.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.77 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.2009459, upper bound: 81.2003861
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.2023742, upper bound: 81.2016322
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1856963, upper bound: 81.1854187
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1998713, upper bound: 81.1990081
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.2359241, upper bound: 81.2292857
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.2359241, upper bound: 81.2296285
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.2359241, upper bound: 81.2292857
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.2359241, upper bound: 81.2296285
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1800136, upper bound: 81.1747641
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1721233, upper bound: 81.1689449
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1741413, upper bound: 81.1662918
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1517812, upper bound: 81.1512455
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 16.77
Output dim: 6, lower bound: -81.1502126, upper bound: 81.1502126

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -35.1293793, 26.1060295, -32.3258781, 24.1345978, -59.2639771, 58.4319077
1: -27.0465965, 25.1977043, -24.9090919, 23.2107849, -50.2573738, 50.1067963
2: -36.8765450, 24.6104660, -34.0197411, 22.6780777, -59.5546227, 58.6302071
3: -41.6147766, 21.1047058, -38.2090454, 19.3965778, -61.0113373, 59.3137512
4: -40.8679810, 25.9517403, -37.5583038, 24.0196247, -64.8876038, 63.5100327
5: -35.6916313, 23.3046799, -32.7826157, 21.5798950, -57.2715225, 56.0872955
6: -38.4353676, 24.6982994, -35.2201462, 22.9606895, -61.3960571, 59.9184456
7: -32.6301346, 30.3711319, -30.0386219, 27.9121075, -60.5422401, 60.4097519
8: -45.1297913, 24.9903202, -41.4604568, 23.1851921, -68.3149796, 66.4507675
9: -30.5649433, 31.0452175, -28.1731186, 28.5629787, -59.1279221, 59.2183380

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2009459, upper bound: 81.2003861
time: 13.30 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2009459, upper bound: 81.2003861
time: 11.35 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -34.8184814, 25.8602791, -35.3874817, 26.3799820, -61.1984634, 61.2477570
1: -26.7865753, 24.9759178, -27.2868862, 25.3484688, -52.1350441, 52.2628021
2: -36.5395660, 24.3879414, -37.2339554, 24.7494907, -61.2890549, 61.6218910
3: -41.2378006, 20.9113140, -41.8024368, 21.1728535, -62.4106522, 62.7137527
4: -40.5243797, 25.6968708, -41.0352173, 26.2528858, -66.7772675, 66.7320862
5: -35.3849792, 23.0811806, -35.8592262, 23.5892448, -58.9742241, 58.9404068
6: -38.1347580, 24.4281387, -38.4220009, 25.1385288, -63.2732849, 62.8501396
7: -32.3253555, 30.1037331, -32.8975563, 30.5040970, -62.8294373, 63.0012894
8: -44.7429886, 24.7409897, -45.2857971, 25.3201065, -70.0630951, 70.0267868
9: -30.2860298, 30.7599068, -30.8124733, 31.2186756, -61.5047073, 61.5723801

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1835884, upper bound: 81.1826669
time: 11.55 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1909485, upper bound: 81.1910040
time: 12.69 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -34.4565430, 25.5724316, -34.3110733, 25.5549393, -60.0114822, 59.8835068
1: -26.4655247, 24.7205601, -26.3691978, 24.6356583, -51.1011810, 51.0897484
2: -36.1383438, 24.1281223, -36.0513344, 24.0186367, -60.1569748, 60.1794510
3: -40.7975693, 20.6682777, -40.5075607, 20.4203815, -61.2179489, 61.1758347
4: -40.1443977, 25.3970165, -39.9842110, 25.3965511, -65.5409393, 65.3812256
5: -35.0503082, 22.8097916, -34.9652176, 22.7605515, -57.8108597, 57.7750092
6: -37.8131332, 24.0761528, -37.5504112, 24.0537567, -61.8668785, 61.6265640
7: -31.9674721, 29.8100739, -31.8391647, 29.7038651, -61.6713371, 61.6492386
8: -44.2931213, 24.4135551, -43.9659958, 24.2919922, -68.5851059, 68.3795471
9: -29.9664116, 30.4283371, -29.8873596, 30.3254890, -60.2919006, 60.3156967

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1856963, upper bound: 81.1854187
time: 13.44 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1856963, upper bound: 81.1854187
time: 11.95 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -35.4793549, 26.3768272, -37.3543282, 27.9409790, -63.4203262, 63.7311554
1: -27.3327980, 25.4453392, -28.9114761, 26.7696247, -54.1024170, 54.3568153
2: -37.2526855, 24.8578434, -39.3534775, 26.1865730, -63.4392548, 64.2113190
3: -42.0340958, 21.3175163, -44.2155228, 22.3400898, -64.3741837, 65.5330276
4: -41.2589073, 26.2292099, -43.2889786, 27.8864861, -69.1453781, 69.5181808
5: -36.0385666, 23.5500908, -37.8893204, 24.9999733, -61.0385399, 61.4394112
6: -38.7817955, 24.9867001, -40.4196739, 26.7665882, -65.5483780, 65.4063721
7: -32.9686012, 30.6707077, -34.8269310, 32.2542839, -65.2228851, 65.4976349
8: -45.5672112, 25.2620201, -47.7276535, 26.8202858, -72.3874893, 72.9896698
9: -30.8752899, 31.3611660, -32.6052094, 33.0651588, -63.9404488, 63.9663773

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1941325, upper bound: 81.1934937
time: 12.36 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1963256, upper bound: 81.1956423
time: 13.21 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -35.1621552, 26.2962742, -40.3181915, 30.4172039, -65.5793304, 66.6144485
1: -27.2188568, 25.2268600, -31.5374908, 28.8900719, -56.1089287, 56.7643509
2: -37.0242958, 24.6768665, -42.6544800, 28.3411732, -65.3654556, 67.3313293
3: -41.6529274, 21.1327591, -47.8169022, 24.2962513, -65.9491806, 68.9496613
4: -40.7492905, 26.2585964, -46.3762703, 30.5100899, -71.2593689, 72.6348648
5: -35.6546440, 23.5396442, -40.6642838, 27.3700848, -63.0247269, 64.2039261
6: -38.1144943, 25.2185898, -42.9351311, 29.7997303, -67.9142227, 68.1537094
7: -32.7675629, 30.3763142, -37.8388519, 34.7167740, -67.4843292, 68.2151642
8: -44.9738045, 25.2816124, -51.2480316, 29.5721779, -74.5459824, 76.5296326
9: -30.6903706, 31.1207428, -35.3336639, 35.8211441, -66.5115128, 66.4544067

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2216476, upper bound: 81.2171131
time: 10.61 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2090357, upper bound: 81.2017996
time: 13.28 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -38.8175278, 29.0760517, -40.3181915, 30.4172039, -69.2347183, 69.3942413
1: -30.1486912, 27.7992096, -31.5374908, 28.8900719, -59.0387650, 59.3366890
2: -40.9173203, 27.2331886, -42.6544800, 28.3411732, -69.2584915, 69.8876648
3: -46.0018883, 23.3171616, -47.8169022, 24.2962513, -70.2981415, 71.1340637
4: -44.8704147, 29.0693531, -46.3762703, 30.5100899, -75.3804932, 75.4456253
5: -39.3036232, 26.0573616, -40.6642838, 27.3700848, -66.6737061, 66.7216492
6: -41.8311539, 28.0699863, -42.9351311, 29.7997303, -71.6308746, 71.0051193
7: -36.2566681, 33.4843788, -37.8388519, 34.7167740, -70.9734421, 71.3232269
8: -49.5462112, 28.0377522, -51.2480316, 29.5721779, -79.1183929, 79.2857666
9: -33.8909149, 34.3879738, -35.3336639, 35.8211441, -69.7120514, 69.7216339

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2256206, upper bound: 81.2213597
time: 11.85 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1924397, upper bound: 81.1798627
time: 9.02 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -35.1621552, 26.2962742, -44.1498489, 33.3438721, -68.5060272, 70.4461060
1: -27.2188568, 25.2268600, -34.6142731, 31.5838737, -58.8027306, 59.8411331
2: -37.0242958, 24.6768665, -46.7391205, 31.0184593, -68.0427475, 71.4159851
3: -41.6529274, 21.1327591, -52.3483582, 26.5885124, -68.2414398, 73.4811020
4: -40.7492905, 26.2585964, -50.6771927, 33.4592247, -74.2085114, 76.9357758
5: -35.6546440, 23.5396442, -44.4739952, 30.0512447, -65.7058868, 68.0136337
6: -38.1144943, 25.2185898, -46.7903175, 32.8221817, -70.9366760, 72.0089111
7: -32.7675629, 30.3763142, -41.5070114, 37.9649277, -70.7324829, 71.8833237
8: -44.9738045, 25.2816124, -56.0065956, 32.4737358, -77.4475250, 81.2882080
9: -30.6903706, 31.1207428, -38.6945877, 39.2471848, -69.9375534, 69.8153305

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=90.90605926513672
rel_dist={6: [-81.29731319725406, 81.29731319725403]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2408941, upper bound: 81.2375369
time: 10.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2292925, upper bound: 81.2292925
time: 7.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 17.48 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 17.48
Output dim: 6, lower bound: -81.2408941, upper bound: 81.2375369
IS_A2, status: Status.UNKNOWN, split count: 1, time: 17.48
Output dim: 6, lower bound: -81.2292925, upper bound: 81.2292925

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -49.8440857, 37.8779831, -50.3818474, 38.2952919, -88.1393738, 88.2598267
1: -39.3679924, 35.5630417, -39.8088417, 35.9474564, -75.3154449, 75.3718872
2: -52.9084244, 35.0605469, -53.4981537, 35.4362068, -88.3446274, 88.5587006
3: -59.0251961, 30.1062145, -59.6625023, 30.4251728, -89.4503708, 89.7687149
4: -56.8039589, 38.1175423, -57.4144287, 38.5423965, -95.3463593, 95.5319672
5: -49.9338379, 34.3365517, -50.4630890, 34.7195892, -84.6534271, 84.7996292
6: -52.0132217, 37.9152527, -52.5452843, 38.3607750, -90.3739929, 90.4605331
7: -47.0919571, 42.7135925, -47.6143074, 43.1696014, -90.2615509, 90.3278961
8: -62.9268799, 37.3348656, -63.5914040, 37.7574081, -100.6842880, 100.9262695
9: -43.8035812, 44.4049301, -44.2846413, 44.8899040, -88.6934814, 88.6895752

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2160837, upper bound: 81.2149875
time: 12.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1872833, upper bound: 81.1811010
time: 11.08 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -58.8373566, 44.5105743, -49.0926895, 37.2890549, -96.1264114, 93.6032639
1: -46.4309464, 41.7663078, -38.7463455, 35.0246887, -81.4556198, 80.5126419
2: -62.2661438, 41.2182922, -52.0776215, 34.5336838, -96.7998276, 93.2959137
3: -69.6919250, 35.5211792, -58.1339569, 29.6601944, -99.3521194, 93.6551361
4: -66.8916702, 44.8040695, -55.9547577, 37.5165291, -104.4082031, 100.7588272
5: -58.9493713, 40.3529358, -49.1966820, 33.7948761, -92.7442398, 89.5496216
6: -61.2510452, 44.6034050, -51.2779808, 37.2779160, -98.5289612, 95.8813858
7: -55.5482101, 50.3803253, -46.3584251, 42.0774117, -97.6256180, 96.7387543
8: -74.2525024, 43.8166504, -62.0009308, 36.7326698, -110.9851608, 105.8175735
9: -51.5999146, 52.3046913, -43.1288757, 43.7233658, -95.3232727, 95.4335632

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1998748, upper bound: 81.1985236
time: 8.67 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1959045, upper bound: 81.1959045
time: 6.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 6, lower bound: -81.2160837, upper bound: 81.2149875
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 6, lower bound: -81.1872833, upper bound: 81.1811010
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 6, lower bound: -81.1998748, upper bound: 81.1985236
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 18.54
Output dim: 6, lower bound: -81.1959045, upper bound: 81.1959045

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -49.3171959, 37.4672012, -47.4820404, 36.0307846, -85.3479691, 84.9492264
1: -38.9270859, 35.1824112, -37.3836594, 33.8509941, -72.7780762, 72.5660629
2: -52.3306961, 34.6928253, -50.3172569, 33.4106827, -85.7413635, 85.0100861
3: -58.3938293, 29.7792892, -56.1867371, 28.6236153, -87.0174408, 85.9660263
4: -56.2135773, 37.6953735, -54.1572418, 36.2219467, -92.4355164, 91.8526001
5: -49.4184647, 33.9532852, -47.6208496, 32.6115379, -82.0299911, 81.5741119
6: -51.4976044, 37.4611740, -49.7008209, 35.8635864, -87.3611908, 87.1619949
7: -46.5744095, 42.2681236, -44.7664032, 40.7147064, -87.2891159, 87.0345078
8: -62.2750778, 36.9117355, -60.0006409, 35.4302025, -97.7052765, 96.9123611
9: -43.3280449, 43.9224777, -41.6665611, 42.2332077, -85.5612488, 85.5890350

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1872833, upper bound: 81.1811010
time: 10.11 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1872833, upper bound: 81.1811010
time: 9.07 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -47.3345222, 35.9021149, -54.4957657, 41.1380005, -88.4725189, 90.3978806
1: -37.2419624, 33.7314529, -42.6196518, 38.6827354, -75.9246826, 76.3510895
2: -50.1342659, 33.3024788, -57.5373764, 38.1886444, -88.3229065, 90.8398438
3: -56.0087967, 28.5271683, -64.3379669, 32.5724144, -88.5812073, 92.8651276
4: -53.9848328, 36.0897942, -62.1506195, 41.1643600, -95.1491928, 98.2403870
5: -47.4765015, 32.4919128, -54.7674561, 37.1652603, -84.6417542, 87.2593536
6: -49.5618973, 35.7137146, -57.1368065, 40.5410919, -90.1029892, 92.8505249
7: -44.6068535, 40.5849686, -51.1229019, 46.6984062, -91.3052597, 91.7078705
8: -59.8173523, 35.2930336, -68.7810745, 40.1242905, -99.9416351, 104.0740967
9: -41.5215759, 42.0855675, -47.5735893, 48.2804146, -89.8019867, 89.6591339

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1783679, upper bound: 81.1718760
time: 9.18 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1738844, upper bound: 81.1669024
time: 11.32 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -53.0421906, 39.9351768, -49.0926895, 37.2890549, -90.3312454, 89.0278625
1: -41.6184235, 37.6579971, -38.7463455, 35.0246887, -76.6431122, 76.4043350
2: -55.9498825, 37.1189423, -52.0776215, 34.5336838, -90.4835663, 89.1965637
3: -62.9150314, 32.0085411, -58.1339569, 29.6601944, -92.5752258, 90.1425018
4: -60.5993118, 40.1413612, -55.9547577, 37.5165291, -98.1158447, 96.0961151
5: -53.3026657, 36.0876465, -49.1966820, 33.7948761, -87.0975418, 85.2843246
6: -55.7960281, 39.6502075, -51.2779808, 37.2779160, -93.0739441, 90.9281845
7: -49.9259949, 45.5058098, -46.3584251, 42.0774117, -92.0034027, 91.8642349
8: -67.1685181, 39.0944138, -62.0009308, 36.7326698, -103.9011765, 101.0953445
9: -46.4396248, 47.0763855, -43.1288757, 43.7233658, -90.1629944, 90.2052536

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1823138, upper bound: 81.1817759
time: 10.38 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
time: 9.39 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -49.6555138, 37.0967331, -42.4184151, 32.0208969, -81.6763992, 79.5151367
1: -38.6175766, 35.2272224, -33.2203789, 30.2946796, -68.9122467, 68.4476013
2: -52.1492996, 34.5932426, -44.8189278, 29.8164196, -81.9657211, 79.4121704
3: -58.9587555, 29.8541145, -50.2974663, 25.6175919, -84.5763397, 80.1515503
4: -57.1173515, 37.1969604, -48.6532364, 32.1852036, -89.3025513, 85.8501892
5: -50.0879898, 33.3399773, -42.6671333, 28.8953152, -78.9833069, 76.0071106
6: -52.9949265, 36.2965202, -44.9321976, 31.6160831, -84.6110077, 81.2287140
7: -46.5464821, 42.6694107, -39.8811188, 36.4601898, -83.0066681, 82.5505142
8: -63.0510063, 35.9152298, -53.8312263, 31.3189087, -94.3699036, 89.7464600
9: -43.4148941, 43.9373932, -37.1836243, 37.7149658, -81.1298599, 81.1210175

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 26

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1677162, upper bound: 81.1660836
time: 9.02 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1629609, upper bound: 81.1629608
time: 6.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 30.84 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.84
Output dim: 6, lower bound: -81.1872833, upper bound: 81.1811010
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.84
Output dim: 6, lower bound: -81.1872833, upper bound: 81.1811010
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 30.84
Output dim: 6, lower bound: -81.1783679, upper bound: 81.1718760
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 30.84
Output dim: 6, lower bound: -81.1738844, upper bound: 81.1669024
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 30.84
Output dim: 6, lower bound: -81.1823138, upper bound: 81.1817759
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 30.84
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 30.84
Output dim: 6, lower bound: -81.1677162, upper bound: 81.1660836
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 30.84
Output dim: 6, lower bound: -81.1629609, upper bound: 81.1629608

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -46.9597359, 35.6255112, -47.4820404, 36.0307846, -82.9905167, 83.1075516
1: -36.9560471, 33.4778137, -37.3836594, 33.8509941, -70.8070374, 70.8614731
2: -49.7445755, 33.0451546, -50.3172569, 33.4106827, -83.1552429, 83.3624115
3: -55.5679855, 28.3134727, -56.1867371, 28.6236153, -84.1915970, 84.5001984
4: -53.5632286, 35.8098869, -54.1572418, 36.2219467, -89.7851410, 89.9671326
5: -47.1064873, 32.2394981, -47.6208496, 32.6115379, -79.7180252, 79.8603516
6: -49.1829376, 35.4319191, -49.7008209, 35.8635864, -85.0465240, 85.1327362
7: -44.2593956, 40.2711983, -44.7664032, 40.7147064, -84.9741058, 85.0375671
8: -59.3546982, 35.0208206, -60.0006409, 35.4302025, -94.7848969, 95.0214386
9: -41.1994629, 41.7621765, -41.6665611, 42.2332077, -83.4326630, 83.4287415

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 185

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2084720, upper bound: 81.2073313
time: 12.63 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2061857, upper bound: 81.2045526
time: 13.41 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -53.9837189, 40.7419930, -47.4820404, 36.0307846, -90.0144806, 88.2240295
1: -42.2009277, 38.3159714, -37.3836594, 33.8509941, -76.0519257, 75.6996307
2: -56.9774475, 37.8316841, -50.3172569, 33.4106827, -90.3881226, 88.1489410
3: -63.7300034, 32.2685242, -56.1867371, 28.6236153, -92.3536072, 88.4552612
4: -61.5702133, 40.7608261, -54.1572418, 36.2219467, -97.7921448, 94.9180450
5: -54.2627792, 36.8019028, -47.6208496, 32.6115379, -86.8743134, 84.4227448
6: -56.6289330, 40.1185112, -49.7008209, 35.8635864, -92.4925156, 89.8193359
7: -50.6256294, 46.2629700, -44.7664032, 40.7147064, -91.3403320, 91.0293732
8: -68.1486435, 39.7254524, -60.0006409, 35.4302025, -103.5788422, 99.7260666
9: -47.1147919, 47.8203201, -41.6665611, 42.2332077, -89.3479996, 89.4868774

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2048141, upper bound: 81.2036702
time: 13.19 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2061857, upper bound: 81.2045526
time: 13.96 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -43.9045486, 33.1897240, -44.7046738, 33.4528694, -77.3574219, 77.8943787
1: -34.4001923, 31.3310432, -34.5574570, 31.8219490, -66.2221375, 65.8884964
2: -46.4277916, 30.8732300, -46.9684296, 31.2843971, -77.7121735, 77.8416595
3: -51.9806023, 26.4252033, -52.8268280, 26.6085815, -78.5891724, 79.2520294
4: -50.2573929, 33.3106842, -51.4280701, 33.3229103, -83.5803070, 84.7387543
5: -44.1637955, 29.9453716, -45.2337646, 30.0098610, -74.1736603, 75.1791382
6: -46.3526115, 32.7140427, -47.8223610, 32.1557236, -78.5083313, 80.5363922
7: -41.2596245, 37.7150536, -41.6131020, 38.4644394, -79.7240524, 79.3281555
8: -55.6279678, 32.4169807, -56.7647629, 32.1177216, -87.7456818, 89.1817474
9: -38.4564972, 38.9873695, -38.8780403, 39.4399109, -77.8963928, 77.8654099

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1664425
time: 12.51 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1669024
time: 10.25 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -44.6757202, 33.7894745, -48.2840042, 36.1886597, -80.8643799, 82.0734787
1: -35.0246353, 31.8703499, -37.4321747, 34.3365326, -69.3611679, 69.3025208
2: -47.2521133, 31.4169884, -50.7913246, 33.7859192, -81.0380020, 82.2082977
3: -52.8865700, 26.8886375, -57.0522003, 28.7466698, -81.6332397, 83.9408417
4: -51.1068497, 33.9225616, -55.4512901, 36.0791512, -87.1859741, 89.3738556
5: -44.9178658, 30.5057888, -48.7889938, 32.5125046, -77.4303589, 79.2947845
6: -47.0988884, 33.3582191, -51.4262848, 34.9860611, -82.0849457, 84.7845001
7: -42.0023270, 38.3616791, -45.0385666, 41.4998016, -83.5021286, 83.4002380
8: -56.5753937, 33.0382385, -61.2155228, 34.8384399, -91.4138336, 94.2537460
9: -39.1374207, 39.6777573, -42.0164185, 42.6422958, -81.7797165, 81.6941757

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1664425
time: 14.10 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1669024
time: 12.75 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -49.6949959, 37.3003426, -39.1190262, 29.4854927, -79.1804886, 76.4193726
1: -38.8549995, 35.3155289, -30.5540543, 28.0329227, -66.8879242, 65.8695831
2: -52.3422775, 34.7513161, -41.3413200, 27.5020046, -79.8442764, 76.0926208
3: -58.9898415, 29.9651985, -46.3915672, 23.5828018, -82.5726242, 76.3567657
4: -56.9786186, 37.4284630, -45.0149307, 29.5625114, -86.5411148, 82.4433899
5: -50.0662727, 33.6136742, -39.4800301, 26.5182037, -76.5844650, 73.0936966
6: -52.6668587, 36.7305107, -41.7464218, 28.8099270, -81.4767838, 78.4769211
7: -46.6708603, 42.7030869, -36.6751099, 33.6975365, -80.3683929, 79.3781967
8: -63.0760536, 36.2923088, -49.7695389, 28.6329708, -91.7090149, 86.0618286
9: -43.4585266, 44.0569382, -34.2637405, 34.7404327, -78.1989594, 78.3206787

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
time: 10.25 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
time: 10.79 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -50.5296173, 37.9500389, -42.9286118, 32.3930283, -82.9226456, 80.8786469
1: -39.5319557, 35.8973846, -33.6107559, 30.7118893, -70.2438431, 69.5081406
2: -53.2333908, 35.3394012, -45.4005127, 30.1640892, -83.3974762, 80.7398987
3: -59.9667892, 30.4663544, -50.8979416, 25.8631935, -85.8299713, 81.3642960
4: -57.8897400, 38.0930634, -49.2926102, 32.4927216, -90.3824615, 87.3856735
5: -50.8803558, 34.2200737, -43.2716217, 29.1808338, -80.0611877, 77.4916992
6: -53.4656830, 37.4326591, -45.5849228, 31.8077965, -85.2734680, 83.0175781
7: -47.4728737, 43.4023285, -40.3204193, 36.9291534, -84.4020233, 83.7227325
8: -64.1023178, 36.9731827, -54.5040970, 31.5149574, -95.6172791, 91.4772568
9: -44.1930809, 44.8043594, -37.6034241, 38.1475639, -82.3406448, 82.4077835

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
time: 9.24 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
time: 9.05 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -41.6290398, 30.8589935, -39.3025131, 29.5728798, -71.2019196, 70.1615067
1: -32.0254517, 29.5949669, -30.6515503, 28.1161327, -60.1415863, 60.2465134
2: -43.4916611, 28.9019547, -41.4650192, 27.6091766, -71.1008377, 70.3669586
3: -49.4159203, 24.9634399, -46.6289177, 23.7059116, -73.1218262, 71.5923538
4: -48.2676620, 30.7900829, -45.2719612, 29.6629295, -77.9305878, 76.0620422
5: -42.2433662, 27.5995541, -39.6559753, 26.5899162, -68.8332825, 67.2555237
6: -45.2276611, 29.5327740, -42.0098000, 28.9062080, -74.1338654, 71.5425720
7: -38.7669983, 35.9020882, -36.8473816, 33.8539772, -72.6209717, 72.7494659
8: -53.0656624, 29.4237175, -50.0221596, 28.7286987, -81.7943573, 79.4458771
9: -36.3114967, 36.6841164, -34.4097862, 34.9070702, -71.2185593, 71.0939026

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 132

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1677162, upper bound: 81.1660836
time: 8.78 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1677162, upper bound: 81.1660836
time: 8.75 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -44.2325478, 32.8161850, -39.9054260, 30.0360394, -74.2685852, 72.7216110
1: -34.0957489, 31.4285107, -31.1349678, 28.5393295, -62.6350708, 62.5634766
2: -46.2682419, 30.7220612, -42.1093903, 28.0335999, -74.3018417, 72.8314514
3: -52.5188828, 26.5112381, -47.3405685, 24.0678139, -76.5866852, 73.8517990
4: -51.2446365, 32.7571182, -45.9432449, 30.1362648, -81.3808823, 78.7003479
5: -44.8564911, 29.3622742, -40.2492447, 27.0206852, -71.8771667, 69.6115189
6: -47.9290199, 31.5060234, -42.6055984, 29.3988304, -77.3278503, 74.1116180
7: -41.2447739, 38.1254959, -37.4262848, 34.3617783, -75.6065445, 75.5517807
8: -56.3546143, 31.3320351, -50.7712402, 29.2006512, -85.5552597, 82.1032715
9: -38.5834961, 39.0071793, -34.9409752, 35.4459076, -74.0294037, 73.9481506

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1629609, upper bound: 81.1629609
time: 8.54 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1629609, upper bound: 81.1629608
time: 6.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 18.46 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.2084720, upper bound: 81.2073313
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.2061857, upper bound: 81.2045526
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.2048141, upper bound: 81.2036702
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.2061857, upper bound: 81.2045526
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1664425
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1669024
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1664425
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1669024
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1677162, upper bound: 81.1660836
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1677162, upper bound: 81.1660836
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1629609, upper bound: 81.1629609
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 18.46
Output dim: 6, lower bound: -81.1629609, upper bound: 81.1629608

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -43.5492325, 32.9286842, -37.6586494, 28.3376770, -71.8869095, 70.5873337
1: -34.1303978, 31.0910015, -29.3135471, 26.9626789, -61.0930786, 60.4045410
2: -46.0598183, 30.6298008, -39.7473106, 26.4831696, -72.5429840, 70.3771133
3: -51.5614395, 26.2223263, -44.6152077, 22.6292591, -74.1906967, 70.8375244
4: -49.8565979, 33.0469360, -43.3811646, 28.3886108, -78.2452087, 76.4281006
5: -43.8107758, 29.7076530, -38.0434303, 25.4425774, -69.2533569, 67.7510834
6: -45.9900169, 32.4503937, -40.3120003, 27.5231056, -73.5131073, 72.7623825
7: -40.9322777, 37.4162292, -35.2373199, 32.4533234, -73.3856049, 72.6535416
8: -55.1884651, 32.1628914, -47.9512482, 27.4537888, -82.6422577, 80.1141357
9: -38.1513748, 38.6811409, -32.9395561, 33.3843422, -71.5357208, 71.6206970

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2303163, upper bound: 81.2266654
time: 9.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2303163, upper bound: 81.2268047
time: 10.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -44.3043365, 33.5144463, -41.4183578, 31.2119904, -75.5163116, 74.9328003
1: -34.7406158, 31.6184425, -32.3314285, 29.6083088, -64.3489151, 63.9498672
2: -46.8656158, 31.1619797, -43.7532120, 29.1116619, -75.9772720, 74.9151917
3: -52.4474640, 26.6759758, -49.0632401, 24.8801994, -77.3276596, 75.7392120
4: -50.6876831, 33.6441727, -47.6044540, 31.2765408, -81.9642258, 81.2486267
5: -44.5491753, 30.2556992, -41.7896614, 28.0723419, -72.6215134, 72.0453644
6: -46.7218246, 33.0789261, -44.1005173, 30.4816475, -77.2034760, 77.1794357
7: -41.6586647, 38.0493546, -38.8336067, 35.6451416, -77.3038025, 76.8829422
8: -56.1150284, 32.7684822, -52.6227493, 30.2996883, -86.4147110, 85.3912354
9: -38.8176193, 39.3558350, -36.2368088, 36.7452431, -75.5628662, 75.5926437

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2303163, upper bound: 81.2266654
time: 11.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2303163, upper bound: 81.2268047
time: 12.49 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -44.2193451, 33.0777245, -44.0582962, 33.3224182, -77.5417633, 77.1360092
1: -34.1614685, 31.4769859, -34.5462379, 31.4550133, -65.6164703, 66.0232162
2: -46.4389191, 30.9457893, -46.6163979, 30.9856663, -77.4245834, 77.5621872
3: -52.2527008, 26.3209076, -52.1650009, 26.5251560, -78.7778549, 78.4859085
4: -50.8766785, 32.9414978, -50.4350891, 33.4469452, -84.3236237, 83.3765869
5: -44.7556648, 29.6660519, -44.3124733, 30.0691624, -74.8248138, 73.9785233
6: -47.3400269, 31.7583694, -46.4956360, 32.8683968, -80.2084198, 78.2539902
7: -41.1436501, 38.0528450, -41.4264030, 37.8489685, -78.9925995, 79.4792480
8: -56.1658897, 31.7409134, -55.8162537, 32.5590477, -88.7249374, 87.5571671
9: -38.4460258, 39.0046921, -38.6065750, 39.1389542, -77.5849762, 77.6112671

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2048141, upper bound: 81.2036702
time: 12.89 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2048141, upper bound: 81.2036702
time: 14.48 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -47.7874603, 35.8046799, -44.8121643, 33.9076653, -81.6950989, 80.6168442
1: -37.0267181, 33.9830360, -35.1557465, 31.9808445, -69.0075455, 69.1387787
2: -50.2487221, 33.4391899, -47.4202652, 31.5171471, -81.7658691, 80.8594513
3: -56.4637985, 28.4519043, -53.0498772, 26.9779549, -83.4417496, 81.5017700
4: -54.8871307, 35.6887245, -51.2649918, 34.0434113, -88.9305420, 86.9537201
5: -48.2992134, 32.1609573, -45.0494537, 30.6162415, -78.9154510, 77.2104111
6: -50.9324112, 34.5783768, -47.2264366, 33.4958916, -84.4282990, 81.8048096
7: -44.5582581, 41.0776672, -42.1508560, 38.4806137, -83.0388718, 83.2285233
8: -60.6026955, 34.4522896, -56.7411880, 33.1638947, -93.7665863, 91.1934433
9: -41.5740623, 42.1961861, -39.2715988, 39.8120384, -81.3861008, 81.4677887

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2061857, upper bound: 81.2045526
time: 10.64 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2061857, upper bound: 81.2045526
time: 12.40 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -37.5512772, 28.2432175, -44.7046738, 33.4528694, -71.0041504, 72.9478912
1: -29.2081413, 26.8775063, -34.5574570, 31.8219490, -61.0300827, 61.4349632
2: -39.6109123, 26.4048901, -46.9684296, 31.2843971, -70.8953018, 73.3733215
3: -44.4875145, 22.5599384, -52.8268280, 26.6085815, -71.0960846, 75.3867569
4: -43.2544212, 28.2902451, -51.4280701, 33.3229103, -76.5773315, 79.7183151
5: -37.9399338, 25.3545647, -45.2337646, 30.0098610, -67.9497986, 70.5883255
6: -40.2144737, 27.4105225, -47.8223610, 32.1557236, -72.3701935, 75.2328796
7: -35.1197166, 32.3604012, -41.6131020, 38.4644394, -73.5841522, 73.9735031
8: -47.8177299, 27.3499031, -56.7647629, 32.1177216, -79.9354401, 84.1146698
9: -32.8335724, 33.2768059, -38.8780403, 39.4399109, -72.2734833, 72.1548309

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1743717, upper bound: 81.1685115
time: 10.15 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1783679, upper bound: 81.1718760
time: 10.11 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -41.2860718, 31.0974426, -44.7046738, 33.4528694, -74.7389374, 75.8020935
1: -32.2052498, 29.5022621, -34.5574570, 31.8219490, -64.0271988, 64.0597076
2: -43.5903854, 29.0142994, -46.9684296, 31.2843971, -74.8747711, 75.9827271
3: -48.9042969, 24.7947044, -52.8268280, 26.6085815, -75.5128632, 77.6215210
4: -47.4489517, 31.1601830, -51.4280701, 33.3229103, -80.7718658, 82.5882568
5: -41.6603928, 27.9659958, -45.2337646, 30.0098610, -71.6702423, 73.1997604
6: -43.9761658, 30.3493214, -47.8223610, 32.1557236, -76.1318817, 78.1716766
7: -38.6904335, 35.5294418, -41.6131020, 38.4644394, -77.1548538, 77.1425400
8: -52.4597511, 30.1789665, -56.7647629, 32.1177216, -84.5774689, 86.9437103
9: -36.1069565, 36.6152000, -38.8780403, 39.4399109, -75.5468445, 75.4932404

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1783679, upper bound: 81.1718760
time: 11.63 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.1783679, upper bound: 81.1718760
time: 12.56 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 29.52 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.2303163, upper bound: 81.2266654
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.2303163, upper bound: 81.2268047
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.2303163, upper bound: 81.2266654
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.2303163, upper bound: 81.2268047
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.2048141, upper bound: 81.2036702
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.2048141, upper bound: 81.2036702
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.2061857, upper bound: 81.2045526
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.2061857, upper bound: 81.2045526
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.1743717, upper bound: 81.1685115
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.1783679, upper bound: 81.1718760
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.1783679, upper bound: 81.1718760
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 29.52
Output dim: 6, lower bound: -81.1783679, upper bound: 81.1718760
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1664425
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1732500, upper bound: 81.1669024
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1718075, upper bound: 81.1689188
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1677162, upper bound: 81.1660836
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1677162, upper bound: 81.1660836
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1629609, upper bound: 81.1629609
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.52
Output dim: 6, lower bound: -81.1629609, upper bound: 81.1629608
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=90.90605926513672
rel_dist={6: [-81.29714583279736, 81.29714583279738]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2968566, upper bound: 81.2968535
time: 15.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2968432, upper bound: 81.2968432
time: 14.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 30.25 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 30.25
Output dim: 6, lower bound: -81.2968566, upper bound: 81.2968535
IS_A2, status: Status.UNKNOWN, split count: 1, time: 30.25
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

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2762954, upper bound: 81.2754564
time: 12.70 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2932031, upper bound: 81.2932059
time: 6.94 seconds

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2968432, upper bound: 81.2968432
time: 9.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2968432, upper bound: 81.2968432
time: 8.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 18.87 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.87
Output dim: 6, lower bound: -81.2762954, upper bound: 81.2754564
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.87
Output dim: 6, lower bound: -81.2932031, upper bound: 81.2932059
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.87
Output dim: 6, lower bound: -81.2968432, upper bound: 81.2968432
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.87
Output dim: 6, lower bound: -81.2968432, upper bound: 81.2968432

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -29.3915272, 21.7937012, -33.4849968, 24.8979721, -54.2894974, 55.2786942
1: -22.3946991, 21.0918446, -25.7618999, 24.0428581, -46.4375534, 46.8537407
2: -30.6798630, 20.5759678, -35.1411743, 23.4810944, -54.1609573, 55.7171402
3: -34.7843857, 17.6569614, -39.6645584, 20.1392059, -54.9235916, 57.3215179
4: -34.2598038, 21.6078339, -38.9818039, 24.7456951, -59.0055008, 60.5896378
5: -29.9091415, 19.4559326, -34.0220299, 22.2182846, -52.1274185, 53.4779625
6: -32.4538918, 20.3509903, -36.6983414, 23.5042744, -55.9581566, 57.0493317
7: -27.1100693, 25.4250221, -31.0707970, 28.9557304, -56.0657997, 56.4958191
8: -37.8267288, 20.7650242, -43.0831413, 23.8494720, -61.6761971, 63.8481674
9: -25.5011082, 25.8601341, -29.1306858, 29.6015873, -55.1026955, 54.9908218

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2590231, upper bound: 81.2584714
time: 9.90 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2575479, upper bound: 81.2566227
time: 13.44 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -36.4206543, 27.3631210, -41.6241608, 31.3999405, -67.8205948, 68.9872818
1: -28.2011566, 26.0309372, -32.5769424, 29.7532558, -57.9544144, 58.6078796
2: -38.2908707, 25.5549793, -43.9881210, 29.2666912, -67.5575562, 69.5430908
3: -43.1503525, 21.9265366, -49.3012199, 25.1064110, -68.2567596, 71.2277527
4: -41.9724045, 27.3310738, -47.7878761, 31.5035362, -73.4759369, 75.1189423
5: -36.7829895, 24.5676193, -41.9095688, 28.2823982, -65.0653839, 66.4771881
6: -39.1143341, 26.4599037, -44.1978989, 30.8379364, -69.9522705, 70.6577988
7: -33.9436417, 31.3453903, -39.0466919, 35.7909470, -69.7345810, 70.3920746
8: -46.4518280, 26.5400963, -52.9411583, 30.6851158, -77.1369476, 79.4812546
9: -31.7708054, 32.2255135, -36.4295883, 36.9773445, -68.7481384, 68.6550980

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2748735, upper bound: 81.2753975
time: 10.29 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2743284, upper bound: 81.2743219
time: 7.79 seconds

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

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2754559, upper bound: 81.2762861
time: 9.77 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2931988, upper bound: 81.2931988
time: 7.00 seconds

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

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2326040, upper bound: 81.2315023
time: 10.06 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2285443, upper bound: 81.2285443
time: 7.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 18.77 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 6, lower bound: -81.2590231, upper bound: 81.2584714
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 6, lower bound: -81.2575479, upper bound: 81.2566227
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 6, lower bound: -81.2748735, upper bound: 81.2753975
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 6, lower bound: -81.2743284, upper bound: 81.2743219
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 6, lower bound: -81.2754559, upper bound: 81.2762861
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 6, lower bound: -81.2931988, upper bound: 81.2931988
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 6, lower bound: -81.2326040, upper bound: 81.2315023
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 18.77
Output dim: 6, lower bound: -81.2285443, upper bound: 81.2285443

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -25.3761444, 18.6902046, -27.6391926, 20.3509102, -45.7270546, 46.3293991
1: -19.1192036, 18.2771568, -20.9542389, 19.9298363, -39.0490341, 39.2313957
2: -26.3849792, 17.7452431, -28.8493385, 19.3411789, -45.7261581, 46.5945816
3: -29.9906082, 15.2067547, -32.7009544, 16.5597382, -46.5503387, 47.9077072
4: -29.7957726, 18.4313374, -32.4745865, 20.0915604, -49.8873329, 50.9059181
5: -25.9778214, 16.6047859, -28.3049583, 18.0801182, -44.0579376, 44.9097443
6: -28.5331841, 17.0550575, -31.0018330, 18.6442337, -47.1774101, 48.0568924
7: -23.2719498, 22.0314369, -25.4526596, 24.0153141, -47.2872620, 47.4840927
8: -32.7970963, 17.6029015, -35.7191315, 19.2090187, -52.0061111, 53.3220329
9: -21.9648037, 22.2574787, -23.9712238, 24.3100739, -46.2748795, 46.2286987

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2561910, upper bound: 81.2565698
time: 9.86 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2561910, upper bound: 81.2584714
time: 12.08 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -23.0220566, 16.8674545, -26.8411407, 19.6299477, -42.6520042, 43.7085953
1: -17.1750088, 16.6070290, -20.1905689, 19.3752174, -36.5502243, 36.7975998
2: -23.8524666, 16.0718765, -27.9315910, 18.7147369, -42.5672035, 44.0034637
3: -27.1387787, 13.7444553, -31.7103596, 16.0040092, -43.1427879, 45.4548149
4: -27.1344490, 16.5821648, -31.7059345, 19.2989960, -46.4334450, 48.2881012
5: -23.6552696, 14.9312458, -27.6317081, 17.3648529, -41.0201225, 42.5629501
6: -26.1899109, 15.1301613, -30.4876385, 17.6365776, -43.8264847, 45.6177979
7: -21.0138130, 20.0206699, -24.6008224, 23.3757458, -44.3895493, 44.6214905
8: -29.8371582, 15.7656002, -34.7819061, 18.3523941, -48.1895523, 50.5475082
9: -19.8665771, 20.1224709, -23.2075386, 23.5317326, -43.3983078, 43.3300095

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2555946, upper bound: 81.2555855
time: 8.03 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2555946, upper bound: 81.2566227
time: 11.14 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -33.7188797, 25.2361507, -36.3183022, 27.2289162, -60.9477959, 61.5544434
1: -26.0017090, 24.1208992, -28.1897202, 25.9887352, -51.9904442, 52.3106194
2: -35.3497734, 23.6359482, -38.2084961, 25.4997005, -60.8494644, 61.8444290
3: -39.9512329, 20.2931004, -43.0504456, 21.8850327, -61.8362656, 63.3435440
4: -38.9951248, 25.1812801, -41.9702797, 27.2603321, -66.2554550, 67.1515579
5: -34.1235123, 22.6170502, -36.7200775, 24.4247360, -58.5482407, 59.3371277
6: -36.5061188, 24.2048016, -39.1318054, 26.3457546, -62.8518715, 63.3366089
7: -31.3407021, 29.0627670, -33.9154701, 31.3117924, -62.6524887, 62.9782295
8: -43.0865135, 24.3651276, -46.3922195, 26.3903694, -69.4768753, 70.7573471
9: -29.3755779, 29.8020229, -31.7183037, 32.1948929, -61.5704651, 61.5203247

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2748735, upper bound: 81.2753975
time: 11.24 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2748735, upper bound: 81.2753975
time: 11.02 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -27.7494736, 20.6020393, -34.1261024, 25.3685188, -53.1179848, 54.7281418
1: -21.1274719, 19.8989601, -26.1901054, 24.3918610, -45.5193329, 46.0890541
2: -28.9124546, 19.3862000, -35.7101479, 23.8072262, -52.7196808, 55.0963440
3: -32.8138046, 16.6749020, -40.4173355, 20.4579468, -53.2717514, 57.0922318
4: -32.3281174, 20.4876728, -39.7199631, 25.2782364, -57.6063499, 60.2076263
5: -28.2140217, 18.3898869, -34.6604424, 22.6595306, -50.8735504, 53.0503311
6: -30.6222858, 19.3400631, -37.3614998, 24.0996017, -54.7218857, 56.7015610
7: -25.6106243, 23.9882317, -31.6845474, 29.4892654, -55.0998840, 55.6727791
8: -35.5850525, 19.6374702, -43.6346817, 24.2418137, -59.8268661, 63.2721405
9: -24.1268559, 24.4283218, -29.7661991, 30.1456642, -54.2725220, 54.1945190

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2621799, upper bound: 81.2622603
time: 9.68 seconds

## Relational analysis of IS_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2620276, upper bound: 81.2620245
time: 8.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -34.9970474, 26.0411758, -29.3915272, 21.7937012, -56.7907486, 55.4327011
1: -26.9723091, 25.1026707, -22.3946991, 21.0918446, -48.0641556, 47.4973640
2: -36.7562790, 24.5352211, -30.6798630, 20.5759678, -57.3322449, 55.2150841
3: -41.4582748, 21.0390530, -34.7843857, 17.6569614, -59.1152191, 55.8234406
4: -40.6851578, 25.9018726, -34.2598038, 21.6078339, -62.2929916, 60.1616745
5: -35.5274010, 23.2523422, -29.9091415, 19.4559326, -54.9833336, 53.1614838
6: -38.2299004, 24.6930637, -32.4538918, 20.3509903, -58.5808907, 57.1469498
7: -32.5218658, 30.2466183, -27.1100693, 25.4250221, -57.9468880, 57.3566895
8: -44.9575005, 24.9748878, -37.8267288, 20.7650242, -65.7225266, 62.8016167
9: -30.4604588, 30.9431515, -25.5011082, 25.8601341, -56.3205948, 56.4442596

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2584714, upper bound: 81.2590231
time: 10.01 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2566227, upper bound: 81.2575479
time: 12.13 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -43.3665695, 32.7303543, -36.4206543, 27.3631210, -70.7296906, 69.1510010
1: -33.9785271, 30.9771557, -28.2011566, 26.0309372, -60.0094643, 59.1783142
2: -45.8563347, 30.4829464, -38.2908707, 25.5549793, -71.4113159, 68.7738190
3: -51.3677521, 26.1472015, -43.1503525, 21.9265366, -73.2942657, 69.2975540
4: -49.7322159, 32.8551407, -41.9724045, 27.3310738, -77.0632935, 74.8275452
5: -43.6299324, 29.5068741, -36.7829895, 24.5676193, -68.1975555, 66.2898636
6: -45.9333496, 32.2383423, -39.1143341, 26.4599037, -72.3932495, 71.3526764
7: -40.7334900, 37.2684021, -33.9436417, 31.3453903, -72.0788803, 71.2120438
8: -55.0729866, 32.0055580, -46.4518280, 26.5400963, -81.6130829, 78.4573822
9: -37.9768867, 38.5250854, -31.7708054, 32.2255135, -70.2024002, 70.2958908

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2753975, upper bound: 81.2748735
time: 11.15 seconds

## Relational analysis of IS_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2743219, upper bound: 81.2743284
time: 8.72 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -48.2463074, 36.6688232, -48.7763367, 37.0808640, -85.3271561, 85.4451599
1: -38.0895386, 34.4329262, -38.5242729, 34.8119087, -72.9014359, 72.9571991
2: -51.1983261, 33.9509583, -51.7800751, 34.3216591, -85.5199890, 85.7310333
3: -57.1302643, 29.1557007, -57.7584953, 29.4703693, -86.6006317, 86.9141922
4: -54.9995575, 36.8959198, -55.6023178, 37.3147583, -92.3143158, 92.4982147
5: -48.3382607, 33.2249908, -48.8602371, 33.6030312, -81.9412689, 82.0852280
6: -50.3920174, 36.6705132, -50.9166870, 37.1101799, -87.5021973, 87.5872040
7: -45.5607758, 41.3516083, -46.0756607, 41.8013268, -87.3621063, 87.4272690
8: -60.9485855, 36.1431122, -61.6037064, 36.5603180, -97.5089035, 97.7468185
9: -42.3953400, 42.9808197, -42.8696823, 43.4593582, -85.8546753, 85.8504868

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2285443, upper bound: 81.2285443
time: 8.22 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2285443, upper bound: 81.2285443
time: 13.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -57.1041260, 43.2097626, -45.6457977, 34.6319695, -91.7360916, 88.8555527
1: -45.0313568, 40.5308495, -35.9437332, 32.5722885, -77.6036453, 76.4745789
2: -60.4206161, 40.0120735, -48.3306885, 32.1288681, -92.5494690, 88.3427582
3: -67.6395264, 34.4929581, -54.0449600, 27.6094837, -95.2489929, 88.5379181
4: -64.9809952, 43.4556885, -52.0544815, 34.8252869, -99.8062592, 95.5101700
5: -57.2335892, 39.1427689, -45.7828865, 31.3586807, -88.5922546, 84.9256363
6: -59.5190201, 43.2223816, -47.8413010, 34.4806061, -93.9996109, 91.0636597
7: -53.8674622, 48.9061584, -43.0236778, 39.1476135, -93.0150681, 91.9298401
8: -72.1163788, 42.5200348, -57.7483368, 34.0769882, -106.1933670, 100.2683716
9: -50.0591583, 50.7619095, -40.0617065, 40.6291656, -90.6883087, 90.8236160

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2135603, upper bound: 81.2133984
time: 9.45 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2131937, upper bound: 81.2131940
time: 9.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.18 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2561910, upper bound: 81.2565698
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2561910, upper bound: 81.2584714
IS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2555946, upper bound: 81.2555855
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2555946, upper bound: 81.2566227
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2748735, upper bound: 81.2753975
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2748735, upper bound: 81.2753975
IS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2621799, upper bound: 81.2622603
IS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2620276, upper bound: 81.2620245
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2584714, upper bound: 81.2590231
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2566227, upper bound: 81.2575479
IS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2753975, upper bound: 81.2748735
IS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2743219, upper bound: 81.2743284
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2285443, upper bound: 81.2285443
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2285443, upper bound: 81.2285443
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2135603, upper bound: 81.2133984
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 22.18
Output dim: 6, lower bound: -81.2131937, upper bound: 81.2131940

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -24.5010796, 17.9425659, -27.6391926, 20.3509102, -44.8519897, 45.5817566
1: -18.3239880, 17.6488533, -20.9542389, 19.9298363, -38.2538223, 38.6030922
2: -25.4048996, 17.0942726, -28.8493385, 19.3411789, -44.7460785, 45.9436111
3: -28.9036827, 14.6319990, -32.7009544, 16.5597382, -45.4634171, 47.3329544
4: -28.8472023, 17.6490955, -32.4745865, 20.0915604, -48.9387627, 50.1236687
5: -25.1528664, 15.8956699, -28.3049583, 18.0801182, -43.2329788, 44.2006302
6: -27.7751312, 16.1429939, -31.0018330, 18.6442337, -46.4193611, 47.1448212
7: -22.3791542, 21.2927952, -25.4526596, 24.0153141, -46.3944702, 46.7454529
8: -31.7226028, 16.7956886, -35.7191315, 19.2090187, -50.9316177, 52.5148125
9: -21.1375599, 21.4323845, -23.9712238, 24.3100739, -45.4476318, 45.4036102

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2305421, upper bound: 81.2311337
time: 10.23 seconds

## Relational analysis of IS_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2297990, upper bound: 81.2300685
time: 9.15 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -29.0404224, 21.5414467, -27.6391926, 20.3509102, -49.3913345, 49.1806412
1: -22.1529903, 20.8768806, -20.9542389, 19.9298363, -42.0828247, 41.8311195
2: -30.3433990, 20.3249855, -28.8493385, 19.3411789, -49.6845779, 49.1743240
3: -34.4133339, 17.4464760, -32.7009544, 16.5597382, -50.9730644, 50.1474304
4: -33.9293823, 21.3572388, -32.4745865, 20.0915604, -54.0209389, 53.8318138
5: -29.5939255, 19.2450180, -28.3049583, 18.0801182, -47.6740417, 47.5499763
6: -32.1761856, 20.1051559, -31.0018330, 18.6442337, -50.8204079, 51.1069870
7: -26.8232613, 25.1458893, -25.4526596, 24.0153141, -50.8385773, 50.5985489
8: -37.3777084, 20.4656429, -35.7191315, 19.2090187, -56.5867081, 56.1847725
9: -25.2527828, 25.5730743, -23.9712238, 24.3100739, -49.5628586, 49.5442963

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2310311, upper bound: 81.2309307
time: 10.41 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2297990, upper bound: 81.2323822
time: 11.19 seconds

## BFS IS instance: IS_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -22.2390404, 16.1983624, -26.8411407, 19.6299477, -41.8689880, 43.0395012
1: -16.4634857, 16.0464573, -20.1905689, 19.3752174, -35.8387032, 36.2370262
2: -22.9864616, 15.4879074, -27.9315910, 18.7147369, -41.7011986, 43.4194984
3: -26.1706676, 13.2272348, -31.7103596, 16.0040092, -42.1746712, 44.9375954
4: -26.2833290, 15.8858500, -31.7059345, 19.2989960, -45.5823250, 47.5917740
5: -22.9209290, 14.3019152, -27.6317081, 17.3648529, -40.2857819, 41.9336243
6: -25.5155869, 14.3118963, -30.4876385, 17.6365776, -43.1521606, 44.7995338
7: -20.2271023, 19.3587341, -24.6008224, 23.3757458, -43.6028366, 43.9595566
8: -28.8827744, 15.0311127, -34.7819061, 18.3523941, -47.2351685, 49.8130188
9: -19.1226387, 19.3886261, -23.2075386, 23.5317326, -42.6543732, 42.5961647

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2299919, upper bound: 81.2303717
time: 12.25 seconds

## Relational analysis of IS_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2292359, upper bound: 81.2291979
time: 8.91 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -26.5433941, 19.6253643, -26.8411407, 19.6299477, -46.1733398, 46.4665031
1: -20.1015053, 19.1147804, -20.1905689, 19.3752174, -39.4767227, 39.3053474
2: -27.6431198, 18.5655594, -27.9315910, 18.7147369, -46.3578529, 46.4971504
3: -31.4044762, 15.9190111, -31.7103596, 16.0040092, -47.4084854, 47.6293716
4: -31.1533756, 19.3869057, -31.7059345, 19.2989960, -50.4523697, 51.0928345
5: -27.1511269, 17.4581585, -27.6317081, 17.3648529, -44.5159760, 45.0898628
6: -29.7299881, 18.0286732, -30.4876385, 17.6365776, -47.3665619, 48.5163116
7: -24.4112244, 23.0439796, -24.6008224, 23.3757458, -47.7869530, 47.6448021
8: -34.2450790, 18.5654583, -34.7819061, 18.3523941, -52.5974731, 53.3473511
9: -23.0582390, 23.3188896, -23.2075386, 23.5317326, -46.5899696, 46.5264206

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2304197, upper bound: 81.2310606
time: 11.29 seconds

## Relational analysis of IS_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2292359, upper bound: 81.2306019
time: 9.81 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -33.7188797, 25.2361507, -30.7293243, 22.8801365, -56.5990143, 55.9654694
1: -26.0017090, 24.1208992, -23.5357399, 22.0044155, -48.0061264, 47.6566391
2: -35.3497734, 23.6359482, -32.0922508, 21.5075111, -56.8572731, 55.7281990
3: -39.9512329, 20.2931004, -36.3713913, 18.4641933, -58.4154205, 56.6644897
4: -38.9951248, 25.1812801, -35.6852112, 22.7787743, -61.7738991, 60.8664856
5: -34.1235123, 22.6170502, -31.1874638, 20.4688778, -54.5923805, 53.8045120
6: -36.5061188, 24.2048016, -33.6236839, 21.6643562, -58.1704750, 57.8284798
7: -31.3407021, 29.0627670, -28.4356441, 26.5330219, -57.8737259, 57.4984131
8: -43.0865135, 24.3651276, -39.3489304, 21.9339886, -65.0204926, 63.7140579
9: -29.3755779, 29.8020229, -26.7261124, 27.0905685, -56.4661484, 56.5281372

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2506460, upper bound: 81.2523931
time: 9.89 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2506460, upper bound: 81.2753746
time: 11.03 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -33.7188797, 25.2361507, -37.7699585, 28.3260880, -62.0449677, 63.0061111
1: -26.0017090, 24.1208992, -29.3516502, 27.0117722, -53.0134811, 53.4725418
2: -35.3497734, 23.6359482, -39.7652664, 26.5107422, -61.8605156, 63.4012108
3: -39.9512329, 20.2931004, -44.7780113, 22.7524376, -62.7036705, 65.0711060
4: -38.9951248, 25.1812801, -43.6132050, 28.3839760, -67.3791046, 68.7944870
5: -34.1235123, 22.6170502, -38.1707306, 25.4263020, -59.5498085, 60.7877655
6: -36.5061188, 24.2048016, -40.6080780, 27.4879036, -63.9940224, 64.8128815
7: -31.3407021, 29.0627670, -35.3157578, 32.5544434, -63.8951378, 64.3785172
8: -43.0865135, 24.3651276, -48.1935120, 27.4757118, -70.5622177, 72.5586395
9: -29.3755779, 29.8020229, -33.0034790, 33.4904022, -62.8659744, 62.8055038

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2627800, upper bound: 81.2632258
time: 10.94 seconds

## Relational analysis of IS_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2626198, upper bound: 81.2629961
time: 9.75 seconds

## BFS IS instance: IS_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -23.6408691, 17.4244328, -27.8220596, 20.4971161, -44.1379852, 45.2464905
1: -17.7512665, 17.0136299, -21.0188942, 19.9941959, -37.7454529, 38.0325241
2: -24.5350533, 16.4683704, -28.9838829, 19.3446884, -43.8797417, 45.4522476
3: -27.9157085, 14.1834040, -32.9058228, 16.6364422, -44.5521507, 47.0892258
4: -27.7630939, 17.2358055, -32.7282791, 20.2746372, -48.0377312, 49.9640808
5: -24.1714287, 15.4636116, -28.4604073, 18.1889381, -42.3603668, 43.9240189
6: -26.6156082, 15.9719343, -31.2372608, 18.8903351, -45.5059433, 47.2091904
7: -21.6940918, 20.5069389, -25.6590157, 24.1580048, -45.8520966, 46.1659393
8: -30.4718323, 16.4074936, -35.7834473, 19.2850838, -49.7569122, 52.1909409
9: -20.4980545, 20.7318535, -24.2049026, 24.4853210, -44.9833755, 44.9367523

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2297988, upper bound: 81.2317385
time: 10.77 seconds

## Relational analysis of IS_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2297988, upper bound: 81.2622603
time: 9.82 seconds

## BFS IS instance: IS_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -21.7107849, 15.9412022, -27.6812592, 20.3157063, -42.0264816, 43.6224594
1: -16.1687050, 15.6465130, -20.8090096, 19.9112701, -36.0799713, 36.4555206
2: -22.4659901, 15.1021996, -28.8002701, 19.1914845, -41.6574554, 43.9024658
3: -25.5908031, 12.9751892, -32.7208939, 16.4978409, -42.0886459, 45.6960831
4: -25.5741043, 15.7249880, -32.6812973, 20.0242710, -45.5983734, 48.4062843
5: -22.2725601, 14.1031237, -28.4210892, 17.9731121, -40.2456665, 42.5242119
6: -24.6856956, 14.4102125, -31.3498459, 18.4791641, -43.1648598, 45.7600594
7: -19.8520470, 18.8564796, -25.4646702, 24.0689812, -43.9210281, 44.3211517
8: -28.0505886, 14.9080362, -35.6862030, 18.9842796, -47.0348663, 50.5942307
9: -18.7780457, 18.9801521, -24.0359383, 24.3119488, -43.0899963, 43.0160904

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 88

## Relational analysis of IS_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2620276, upper bound: 81.2620245
time: 9.51 seconds

## Relational analysis of IS_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2620276, upper bound: 81.2620245
time: 8.34 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -28.8741894, 21.2814789, -25.3761444, 18.6902046, -47.5643883, 46.6576233
1: -21.9367828, 20.8015118, -19.1192036, 18.2771568, -40.2139359, 39.9207077
2: -30.1658401, 20.2008190, -26.3849792, 17.7452431, -47.9110756, 46.5858002
3: -34.1769638, 17.3006783, -29.9906082, 15.2067547, -49.3837204, 47.2912827
4: -33.8955688, 21.0238590, -29.7957726, 18.4313374, -52.3269043, 50.8196259
5: -29.5475769, 18.9164047, -25.9778214, 16.6047859, -46.1523628, 44.8942261
6: -32.2972260, 19.5802917, -28.5331841, 17.0550575, -49.3522835, 48.1134682
7: -26.6282444, 25.0745373, -23.2719498, 22.0314369, -48.6596680, 48.3464851
8: -37.2711411, 20.1110096, -32.7970963, 17.6029015, -54.8740425, 52.9081039
9: -25.0548458, 25.4105225, -21.9648037, 22.2574787, -47.3123245, 47.3753242

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 160
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2565697, upper bound: 81.2561910
time: 10.15 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2565697, upper bound: 81.2590231
time: 9.60 seconds

## BFS IS instance: IS_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -28.1219711, 20.5855503, -23.0220566, 16.8674545, -44.9894257, 43.6076050
1: -21.2073269, 20.2803650, -17.1750088, 16.6070290, -37.8143539, 37.4553719
2: -29.2897797, 19.6038780, -23.8524666, 16.0718765, -45.3616562, 43.4563446
3: -33.2376099, 16.7705593, -27.1387787, 13.7444553, -46.9820633, 43.9093399
4: -33.1855927, 20.2519283, -27.1344490, 16.5821648, -49.7677536, 47.3863754
5: -28.9220924, 18.2220364, -23.6552696, 14.9312458, -43.8533363, 41.8773041
6: -31.8463478, 18.5833607, -26.1899109, 15.1301613, -46.9765091, 44.7732697
7: -25.8139343, 24.4751892, -21.0138130, 20.0206699, -45.8345947, 45.4889984
8: -36.3900375, 19.2715244, -29.8371582, 15.7656002, -52.1556396, 49.1086807
9: -24.3303032, 24.6698246, -19.8665771, 20.1224709, -44.4527740, 44.5363998

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 193
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2325573, upper bound: 81.2332035
time: 11.06 seconds

## Relational analysis of IS_A2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2306019, upper bound: 81.2316076
time: 11.94 seconds

## BFS IS instance: IS_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -37.9015732, 28.4315834, -33.7188797, 25.2361507, -63.1377182, 62.1504631
1: -29.4614983, 27.1017761, -26.0017090, 24.1208992, -53.5823898, 53.1034851
2: -39.9078789, 26.6046562, -35.3497734, 23.6359482, -63.5438232, 61.9544296
3: -44.9355545, 22.8319759, -39.9512329, 20.2931004, -65.2286453, 62.7832031
4: -43.7524986, 28.4884052, -38.9951248, 25.1812801, -68.9337769, 67.4835205
5: -38.2921715, 25.5231705, -34.1235123, 22.6170502, -60.9092216, 59.6466751
6: -40.7272263, 27.6113529, -36.5061188, 24.2048016, -64.9320297, 64.1174622
7: -35.4439392, 32.6638756, -31.3407021, 29.0627670, -64.5066986, 64.0045776
8: -48.3502121, 27.5836506, -43.0865135, 24.3651276, -72.7153397, 70.6701660
9: -33.1223412, 33.6060677, -29.3755779, 29.8020229, -62.9243622, 62.9816437

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 56
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 168
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 193
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 56
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2632258, upper bound: 81.2627800
time: 10.67 seconds

## Relational analysis of IS_A2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2629961, upper bound: 81.2626198
time: 10.60 seconds

## BFS IS instance: IS_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -35.6749649, 26.5379601, -27.7494736, 20.6020393, -56.2769890, 54.2874336
1: -27.4251080, 25.4782562, -21.1274719, 19.8989601, -47.3240623, 46.6057281
2: -37.3611183, 24.8854923, -28.9124546, 19.3862000, -56.7473068, 53.7979393
3: -42.2543030, 21.3796005, -32.8138046, 16.6749020, -58.9292068, 54.1933975
4: -41.4671669, 26.4627094, -32.3281174, 20.4876728, -61.9548340, 58.7908249
5: -36.2014313, 23.7226028, -28.2140217, 18.3898869, -54.5913162, 51.9366226
6: -38.9351044, 25.3125229, -30.6222858, 19.3400631, -58.2751656, 55.9348068
7: -33.1676559, 30.8096695, -25.6106243, 23.9882317, -57.1558838, 56.4202957
8: -45.5519562, 25.3918457, -35.5850525, 19.6374702, -65.1894226, 60.9768982
9: -31.1266556, 31.5179863, -24.1268559, 24.4283218, -55.5549774, 55.6448441

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 160
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 193
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 193
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 56
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 168
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 56
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2622603, upper bound: 81.2621799
time: 9.87 seconds

## Relational analysis of IS_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -81.2620245, upper bound: 81.2620276
time: 61.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 72.12 seconds
IS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2305421, upper bound: 81.2311337
IS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2297990, upper bound: 81.2300685
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2310311, upper bound: 81.2309307
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2297990, upper bound: 81.2323822
IS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2299919, upper bound: 81.2303717
IS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2292359, upper bound: 81.2291979
IS_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2304197, upper bound: 81.2310606
IS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2292359, upper bound: 81.2306019
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2506460, upper bound: 81.2523931
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2506460, upper bound: 81.2753746
IS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2627800, upper bound: 81.2632258
IS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2626198, upper bound: 81.2629961
IS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2297988, upper bound: 81.2317385
IS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2297988, upper bound: 81.2622603
IS_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2620276, upper bound: 81.2620245
IS_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2620276, upper bound: 81.2620245
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2565697, upper bound: 81.2561910
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2565697, upper bound: 81.2590231
IS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2325573, upper bound: 81.2332035
IS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2306019, upper bound: 81.2316076
IS_A2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2632258, upper bound: 81.2627800
IS_A2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2629961, upper bound: 81.2626198
IS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2622603, upper bound: 81.2621799
IS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 72.12
Output dim: 6, lower bound: -81.2620245, upper bound: 81.2620276
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 72.12
Output dim: 6, lower bound: -81.2285443, upper bound: 81.2285443
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 72.12
Output dim: 6, lower bound: -81.2285443, upper bound: 81.2285443
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 72.12
Output dim: 6, lower bound: -81.2135603, upper bound: 81.2133984
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 72.12
Output dim: 6, lower bound: -81.2131937, upper bound: 81.2131940
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=90.90605926513672
rel_dist={6: [-81.29696620335748, 81.29696620339547]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1872.80 seconds
