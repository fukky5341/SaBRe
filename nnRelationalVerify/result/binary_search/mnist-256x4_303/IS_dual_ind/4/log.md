## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 10.8418399842
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673)
1: (-5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646)
2: (-7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834)
3: (-7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134)
4: (-8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527)
5: (-7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887)
6: (-6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578)
7: (-7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368)
8: (-9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864)
9: (-6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577)

## BASE Result
execution time: IAR + LP analysis = 1.22 + 4.15 = 5.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -10.8418977, upper bound: 10.8418970


# Binary Search by BASE starts (time budget: 2694.63 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=13.22586727142334
rel_dist={0: [-10.841897398404505, 10.841898463369382]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=13.22586727142334
rel_dist={0: [-10.84189901486047, 10.841897299947618]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=13.22586727142334
rel_dist={0: [-10.841896423745458, 10.841896847270611]}

## Binary Search Result
Binary search time: 18.90 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2675.73 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418812, upper bound: 10.8418722
time: 2.62 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418727, upper bound: 10.8418721
time: 2.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.50 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.50
Output dim: 0, lower bound: -10.8418812, upper bound: 10.8418722
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.50
Output dim: 0, lower bound: -10.8418727, upper bound: 10.8418721

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.2066984, 3.6700487, -8.7459450, 4.4799228, -11.6866198, 12.4159937
1: -4.7570004, 4.5818253, -5.7760954, 5.5418692, -10.2988701, 10.3579206
2: -6.1038303, 4.6374569, -7.5248165, 5.6160679, -11.7198982, 12.1622705
3: -6.4665151, 4.0927978, -7.9764252, 4.9407926, -11.4073076, 12.0692234
4: -7.0948610, 5.6892929, -8.6739941, 6.9140596, -14.0089207, 14.3632870
5: -5.8535242, 4.1872144, -7.1571879, 5.0683022, -10.9218264, 11.3444023
6: -5.4275541, 5.5503712, -6.6823978, 6.7787600, -12.2063141, 12.2327690
7: -5.9946299, 5.5622005, -7.3757725, 6.7552648, -12.7498941, 12.9379730
8: -7.6838732, 4.7240686, -9.3987417, 5.7281442, -13.4120169, 14.1228104
9: -5.3657575, 5.5053244, -6.5382051, 6.7157540, -12.0815115, 12.0435295

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418725, upper bound: 10.8418720
time: 2.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418725, upper bound: 10.8418727
time: 2.47 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.5377808, 3.8540711, -8.4148340, 4.3116217, -11.8494024, 12.2689056
1: -4.9751611, 4.7891555, -5.5554104, 5.3348246, -10.3099861, 10.3445663
2: -6.4123225, 4.8521576, -7.2192297, 5.4064736, -11.8187962, 12.0713873
3: -6.7883143, 4.2768574, -7.6492348, 4.7587304, -11.5470448, 11.9260921
4: -7.4346066, 5.9521360, -8.3313799, 6.6494684, -14.0840750, 14.2835150
5: -6.1374726, 4.3769474, -6.8775959, 4.8803320, -11.0178051, 11.2545433
6: -5.6966209, 5.8149390, -6.4136863, 6.5148754, -12.2114964, 12.2286253
7: -6.3026853, 5.8260241, -7.0813370, 6.4998589, -12.8025427, 12.9073610
8: -8.0521526, 4.9401875, -9.0292559, 5.5116916, -13.5638447, 13.9694433
9: -5.6227317, 5.7691216, -6.2864227, 6.4556870, -12.0784187, 12.0555439

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418680, upper bound: 10.8418729
time: 2.64 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418682, upper bound: 10.8418678
time: 2.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.58 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.58
Output dim: 0, lower bound: -10.8418725, upper bound: 10.8418720
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.58
Output dim: 0, lower bound: -10.8418725, upper bound: 10.8418727
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.58
Output dim: 0, lower bound: -10.8418680, upper bound: 10.8418729
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.58
Output dim: 0, lower bound: -10.8418682, upper bound: 10.8418678

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.2066984, 3.6700487, -7.2066984, 3.6700487, -10.8767471, 10.8767471
1: -4.7570004, 4.5818253, -4.7570004, 4.5818253, -9.3388252, 9.3388252
2: -6.1038303, 4.6374569, -6.1038303, 4.6374569, -10.7412872, 10.7412872
3: -6.4665151, 4.0927978, -6.4665151, 4.0927978, -10.5593128, 10.5593128
4: -7.0948610, 5.6892929, -7.0948610, 5.6892929, -12.7841539, 12.7841539
5: -5.8535242, 4.1872144, -5.8535242, 4.1872144, -10.0407391, 10.0407391
6: -5.4275541, 5.5503712, -5.4275541, 5.5503712, -10.9779253, 10.9779253
7: -5.9946299, 5.5622005, -5.9946299, 5.5622005, -11.5568304, 11.5568304
8: -7.6838732, 4.7240686, -7.6838732, 4.7240686, -12.4079418, 12.4079418
9: -5.3657575, 5.5053244, -5.3657575, 5.5053244, -10.8710823, 10.8710823

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418810, upper bound: 10.8418679
time: 2.12 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418762, upper bound: 10.8418671
time: 2.95 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.2066984, 3.6700487, -7.5377808, 3.8540711, -11.0607700, 11.2078295
1: -4.7570004, 4.5818253, -4.9751611, 4.7891555, -9.5461559, 9.5569859
2: -6.1038303, 4.6374569, -6.4123225, 4.8521576, -10.9559879, 11.0497799
3: -6.4665151, 4.0927978, -6.7883143, 4.2768574, -10.7433720, 10.8811121
4: -7.0948610, 5.6892929, -7.4346066, 5.9521360, -13.0469971, 13.1238995
5: -5.8535242, 4.1872144, -6.1374726, 4.3769474, -10.2304716, 10.3246870
6: -5.4275541, 5.5503712, -5.6966209, 5.8149390, -11.2424927, 11.2469921
7: -5.9946299, 5.5622005, -6.3026853, 5.8260241, -11.8206539, 11.8648853
8: -7.6838732, 4.7240686, -8.0521526, 4.9401875, -12.6240606, 12.7762213
9: -5.3657575, 5.5053244, -5.6227317, 5.7691216, -11.1348791, 11.1280556

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418810, upper bound: 10.8418682
time: 3.21 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418762, upper bound: 10.8418682
time: 3.03 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.4872518, 3.8283093, -7.0549593, 3.6517334, -11.1389847, 10.8832684
1: -4.9418020, 4.7576094, -4.6464891, 4.4737520, -9.4155540, 9.4040985
2: -6.3660021, 4.8203907, -5.9656377, 4.5548830, -10.9208851, 10.7860279
3: -6.7386794, 4.2490993, -6.2934737, 4.0089908, -10.7476702, 10.5425730
4: -7.3826947, 5.9118791, -6.9076209, 5.5545030, -12.9371977, 12.8195000
5: -6.0946121, 4.3483729, -5.7241497, 4.1193461, -10.2139587, 10.0725231
6: -5.6560106, 5.7752519, -5.3232594, 5.4470472, -11.1030579, 11.0985107
7: -6.2579765, 5.7872996, -5.8838434, 5.4584112, -11.7163877, 11.6711426
8: -7.9958887, 4.9075027, -7.5014896, 4.6308384, -12.6267271, 12.4089928
9: -5.5843239, 5.7295594, -5.2464671, 5.3854189, -10.9697428, 10.9760265

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418681, upper bound: 10.8418671
time: 2.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418653, upper bound: 10.8418675
time: 3.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.5023775, 3.8359356, -7.9166756, 4.0587020, -11.5610790, 11.7526112
1: -4.9517407, 4.7670236, -5.2235937, 5.0226860, -9.9744263, 9.9906158
2: -6.3797679, 4.8298278, -6.7597456, 5.0919313, -11.4716988, 11.5895729
3: -6.7534637, 4.2573829, -7.1557059, 4.4842439, -11.2377071, 11.4130859
4: -7.3981938, 5.9238601, -7.8151684, 6.2509513, -13.6491451, 13.7390261
5: -6.1073694, 4.3569102, -6.4540420, 4.5989947, -10.7063637, 10.8109512
6: -5.6681409, 5.7870216, -6.0127869, 6.1211014, -11.7892418, 11.7998085
7: -6.2712469, 5.7988214, -6.6389542, 6.1157436, -12.3869905, 12.4377756
8: -8.0126228, 4.9172592, -8.4716873, 5.1883664, -13.2009888, 13.3889465
9: -5.5957479, 5.7413273, -5.9063473, 6.0633631, -11.6591110, 11.6476746

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418683, upper bound: 10.8418681
time: 2.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418683, upper bound: 10.8418681
time: 2.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.11 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.11
Output dim: 0, lower bound: -10.8418810, upper bound: 10.8418679
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.11
Output dim: 0, lower bound: -10.8418762, upper bound: 10.8418671
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.11
Output dim: 0, lower bound: -10.8418810, upper bound: 10.8418682
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.11
Output dim: 0, lower bound: -10.8418762, upper bound: 10.8418682
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.11
Output dim: 0, lower bound: -10.8418681, upper bound: 10.8418671
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.11
Output dim: 0, lower bound: -10.8418653, upper bound: 10.8418675
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.11
Output dim: 0, lower bound: -10.8418683, upper bound: 10.8418681
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.11
Output dim: 0, lower bound: -10.8418683, upper bound: 10.8418681

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.9140306, 3.0436082, -7.1576452, 3.6449850, -9.5590153, 10.2012539
1: -3.8967600, 3.7642124, -4.7246408, 4.5512242, -8.4479847, 8.4888535
2: -4.9132462, 3.8297064, -6.0589170, 4.6066318, -9.5198784, 9.8886232
3: -5.2251310, 3.3809280, -6.4183478, 4.0658598, -9.2909908, 9.7992764
4: -5.7461271, 4.6486964, -7.0445094, 5.6502409, -11.3963680, 11.6932058
5: -4.7567749, 3.4694471, -5.8119140, 4.1594634, -8.9162388, 9.2813606
6: -4.3906856, 4.5409346, -5.3881407, 5.5118790, -9.9025650, 9.9290752
7: -4.8556237, 4.5812330, -5.9512606, 5.5246487, -10.3802719, 10.5324936
8: -6.2345629, 3.8940446, -7.6292787, 4.6923614, -10.9269238, 11.5233231
9: -4.3770018, 4.5038028, -5.3284826, 5.4669533, -9.8439550, 9.8322849

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418717, upper bound: 10.8418759
time: 2.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418720, upper bound: 10.8418680
time: 2.79 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.7393360, 3.4298229, -7.1720057, 3.6521847, -10.3915205, 10.6018286
1: -4.4478135, 4.2898297, -4.7340612, 4.5601482, -9.0079613, 9.0238914
2: -5.6740150, 4.3425436, -6.0719519, 4.6155653, -10.2895803, 10.4144955
3: -6.0082946, 3.8355651, -6.4323668, 4.0737128, -10.0820074, 10.2679319
4: -6.6142368, 5.3159237, -7.0592060, 5.6615982, -12.2758350, 12.3751297
5: -5.4556208, 3.9228878, -5.8240061, 4.1675591, -9.6231804, 9.7468939
6: -5.0513749, 5.1823626, -5.3996315, 5.5230141, -10.5743885, 10.5819941
7: -5.5792704, 5.2033987, -5.9638124, 5.5355520, -11.1148224, 11.1672115
8: -7.1623602, 4.4216914, -7.6451368, 4.7016096, -11.8639698, 12.0668278
9: -5.0092063, 5.1394053, -5.3393059, 5.4781008, -10.4873066, 10.4787111

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418757
time: 2.70 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418685
time: 2.91 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.9140306, 3.0436082, -7.4872518, 3.8283093, -9.7423401, 10.5308599
1: -3.8967600, 3.7642124, -4.9418020, 4.7576094, -8.6543694, 8.7060146
2: -4.9132462, 3.8297064, -6.3660021, 4.8203907, -9.7336369, 10.1957083
3: -5.2251310, 3.3809280, -6.7386794, 4.2490993, -9.4742298, 10.1196079
4: -5.7461271, 4.6486964, -7.3826947, 5.9118791, -11.6580067, 12.0313911
5: -4.7567749, 3.4694471, -6.0946121, 4.3483729, -9.1051483, 9.5640593
6: -4.3906856, 4.5409346, -5.6560106, 5.7752519, -10.1659374, 10.1969452
7: -4.8556237, 4.5812330, -6.2579765, 5.7872996, -10.6429234, 10.8392096
8: -6.2345629, 3.8940446, -7.9958887, 4.9075027, -11.1420650, 11.8899336
9: -4.3770018, 4.5038028, -5.5843239, 5.7295594, -10.1065617, 10.0881271

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418717, upper bound: 10.8418682
time: 2.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418722, upper bound: 10.8418647
time: 2.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.7393360, 3.4298229, -7.5023775, 3.8359356, -10.5752716, 10.9322004
1: -4.4478135, 4.2898297, -4.9517407, 4.7670236, -9.2148371, 9.2415705
2: -5.6740150, 4.3425436, -6.3797679, 4.8298278, -10.5038433, 10.7223110
3: -6.0082946, 3.8355651, -6.7534637, 4.2573829, -10.2656775, 10.5890293
4: -6.6142368, 5.3159237, -7.3981938, 5.9238601, -12.5380974, 12.7141171
5: -5.4556208, 3.9228878, -6.1073694, 4.3569102, -9.8125305, 10.0302572
6: -5.0513749, 5.1823626, -5.6681409, 5.7870216, -10.8383961, 10.8505039
7: -5.5792704, 5.2033987, -6.2712469, 5.7988214, -11.3780918, 11.4746456
8: -7.1623602, 4.4216914, -8.0126228, 4.9172592, -12.0796194, 12.4343147
9: -5.0092063, 5.1394053, -5.5957479, 5.7413273, -10.7505341, 10.7351532

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418750, upper bound: 10.8418672
time: 2.40 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418750, upper bound: 10.8418675
time: 2.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6248856, 3.4118989, -7.0549593, 3.6517334, -10.2766190, 10.4668579
1: -4.3688440, 4.2132726, -4.6464891, 4.4737520, -8.8425961, 8.8597622
2: -5.5747519, 4.2824383, -5.9656377, 4.5548830, -10.1296349, 10.2480755
3: -5.8933353, 3.7750039, -6.2934737, 4.0089908, -9.9023266, 10.0684776
4: -6.4846783, 5.2198944, -6.9076209, 5.5545030, -12.0391808, 12.1275158
5: -5.3658404, 3.8666933, -5.7241497, 4.1193461, -9.4851866, 9.5908432
6: -4.9644737, 5.1021767, -5.3232594, 5.4470472, -10.4115210, 10.4254360
7: -5.5017748, 5.1310849, -5.8838434, 5.4584112, -10.9601860, 11.0149288
8: -7.0322032, 4.3511925, -7.5014896, 4.6308384, -11.6630421, 11.8526821
9: -4.9273858, 5.0574002, -5.2464671, 5.3854189, -10.3128052, 10.3038673

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418267, upper bound: 10.8418640
time: 2.95 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418267, upper bound: 10.8418276
time: 2.31 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.3838596, 3.7681737, -6.9535813, 3.6028035, -10.9866629, 10.7217550
1: -4.8802261, 4.6993656, -4.5791860, 4.4098740, -9.2901001, 9.2785511
2: -6.2792187, 4.7581372, -5.8727446, 4.4917221, -10.7709408, 10.6308823
3: -6.6490340, 4.1951532, -6.1933117, 3.9533172, -10.6023512, 10.3884649
4: -7.2889719, 5.8385181, -6.8022504, 5.4732842, -12.7622566, 12.6407681
5: -6.0128870, 4.2866731, -5.6386099, 4.0625658, -10.0754528, 9.9252834
6: -5.5713139, 5.6978812, -5.2419605, 5.3679338, -10.9392471, 10.9398422
7: -6.1715102, 5.7110276, -5.7951107, 5.3811445, -11.5526543, 11.5061378
8: -7.8924174, 4.8418417, -7.3884506, 4.5652990, -12.4577160, 12.2302923
9: -5.5131154, 5.6553974, -5.1693912, 5.3062596, -10.8193750, 10.8247890

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418211, upper bound: 10.8418631
time: 2.75 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418286
time: 2.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.3053184, 3.2544355, -7.9166756, 4.0587020, -10.3640203, 11.1711111
1: -4.1552782, 4.0103369, -5.2235937, 5.0226860, -9.1779642, 9.2339296
2: -5.2785544, 4.0812855, -6.7597456, 5.0919313, -10.3704853, 10.8410311
3: -5.5891037, 3.5980761, -7.1557059, 4.4842439, -10.0733471, 10.7537804
4: -6.1496978, 4.9611111, -7.8151684, 6.2509513, -12.4006491, 12.7762794
5: -5.0921431, 3.6901307, -6.4540420, 4.5989947, -9.6911373, 10.1441727
6: -4.7078476, 4.8517275, -6.0127869, 6.1211014, -10.8289490, 10.8645144
7: -5.2179785, 4.8884001, -6.6389542, 6.1157436, -11.3337212, 11.5273542
8: -6.6713152, 4.1466579, -8.4716873, 5.1883664, -11.8596821, 12.6183453
9: -4.6810789, 4.8095245, -5.9063473, 6.0633631, -10.7444410, 10.7158718

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418633
time: 2.94 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418253
time: 2.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.0584607, 3.6089172, -7.9166756, 4.0587020, -11.1171627, 11.5255928
1: -4.6578755, 4.4893708, -5.2235937, 5.0226860, -9.6805611, 9.7129631
2: -5.9713945, 4.5498142, -6.7597456, 5.0919313, -11.0633259, 11.3095598
3: -6.3164043, 4.0131030, -7.1557059, 4.4842439, -10.8006477, 11.1688080
4: -6.9412670, 5.5690036, -7.8151684, 6.2509513, -13.1922188, 13.3841696
5: -5.7297173, 4.1058216, -6.4540420, 4.5989947, -10.3287125, 10.5598621
6: -5.3109918, 5.4372406, -6.0127869, 6.1211014, -11.4320927, 11.4500275
7: -5.8769274, 5.4576788, -6.6389542, 6.1157436, -11.9926682, 12.0966330
8: -7.5169945, 4.6297545, -8.4716873, 5.1883664, -12.7053604, 13.1014423
9: -5.2571592, 5.3929949, -5.9063473, 6.0633631, -11.3205214, 11.2993422

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418642
time: 2.59 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418271
time: 2.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418717, upper bound: 10.8418759
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418720, upper bound: 10.8418680
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418757
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418685
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418717, upper bound: 10.8418682
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418722, upper bound: 10.8418647
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418750, upper bound: 10.8418672
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418750, upper bound: 10.8418675
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418267, upper bound: 10.8418640
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418267, upper bound: 10.8418276
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418211, upper bound: 10.8418631
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418286
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418633
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418253
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418642
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.43
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418271

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9140306, 3.0436082, -6.3353863, 3.2485588, -9.1625891, 9.3789940
1: -3.8967600, 3.7642124, -4.1792064, 4.0327048, -7.9294648, 7.9434185
2: -4.9132462, 3.8297064, -5.3048878, 4.0940523, -9.0072985, 9.1345940
3: -5.2251310, 3.3809280, -5.6268563, 3.6139324, -8.8390636, 9.0077839
4: -5.7461271, 4.6486964, -6.1890211, 4.9908166, -10.7369442, 10.8377171
5: -4.7567749, 3.4694471, -5.1170745, 3.7021790, -8.4589539, 8.5865211
6: -4.3906856, 4.5409346, -4.7290378, 4.8720789, -9.2627640, 9.2699718
7: -4.8556237, 4.5812330, -5.2304592, 4.9018402, -9.7574635, 9.8116922
8: -6.2345629, 3.8940446, -6.7112384, 4.1642365, -10.3987999, 10.6052828
9: -4.3770018, 4.5038028, -4.7023783, 4.8306150, -9.2076168, 9.2061806

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418722, upper bound: 10.8418679
time: 2.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418722, upper bound: 10.8418688
time: 2.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.8116031, 2.9942837, -7.1982613, 3.6561396, -9.4677429, 10.1925449
1: -3.8288946, 3.6995342, -4.7593546, 4.5847540, -8.4136486, 8.4588890
2: -4.8191342, 3.7663870, -6.1062446, 4.6360502, -9.4551849, 9.8726311
3: -5.1289639, 3.3247123, -6.4718242, 4.0915508, -9.2205143, 9.7965364
4: -5.6396713, 4.5663939, -7.1015234, 5.6934166, -11.3330879, 11.6679173
5: -4.6701303, 3.4129658, -5.8529358, 4.1787171, -8.8488474, 9.2659016
6: -4.3085456, 4.4612160, -5.4195728, 5.5484815, -9.8570271, 9.8807888
7: -4.7657561, 4.5039239, -5.9945340, 5.5609932, -10.3267498, 10.4984579
8: -6.1200242, 3.8287840, -7.6877117, 4.7203941, -10.8404179, 11.5164957
9: -4.2990842, 4.4253054, -5.3687410, 5.5072198, -9.8063040, 9.7940464

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418709, upper bound: 10.8418324
time: 4.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418385, upper bound: 10.8418318
time: 2.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.7393360, 3.4298229, -6.3489361, 3.2553077, -9.9946442, 9.7787590
1: -4.4478135, 4.2898297, -4.1880636, 4.0411129, -8.4889259, 8.4778938
2: -5.6740150, 4.3425436, -5.3171782, 4.1024704, -9.7764854, 9.6597214
3: -6.0082946, 3.8355651, -5.6393843, 3.6213410, -9.6296358, 9.4749489
4: -6.6142368, 5.3159237, -6.2028656, 5.0015240, -11.6157608, 11.5187893
5: -5.4556208, 3.9228878, -5.1284804, 3.7097194, -9.1653404, 9.0513687
6: -5.0513749, 5.1823626, -4.7398849, 4.8825068, -9.9338818, 9.9222469
7: -5.5792704, 5.2033987, -5.2422991, 4.9119992, -10.4912701, 10.4456978
8: -7.1623602, 4.4216914, -6.7261724, 4.1728592, -11.3352194, 11.1478634
9: -5.0092063, 5.1394053, -4.7125816, 4.8409166, -9.8501225, 9.8519869

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418685, upper bound: 10.8418687
time: 2.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418685, upper bound: 10.8418684
time: 3.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.6348896, 3.3796997, -7.2114840, 3.6627753, -10.2976646, 10.5911837
1: -4.3786497, 4.2239819, -4.7680202, 4.5929632, -8.9716129, 8.9920025
2: -5.5782385, 4.2774725, -6.1182265, 4.6442709, -10.2225094, 10.3956985
3: -5.9102521, 3.7781830, -6.4847126, 4.0987763, -10.0090284, 10.2628956
4: -6.5056515, 5.2321835, -7.1150351, 5.7038603, -12.2095118, 12.3472185
5: -5.3674870, 3.8650787, -5.8640642, 4.1861744, -9.5536613, 9.7291431
6: -4.9675303, 5.1012621, -5.4301567, 5.5587230, -10.5262527, 10.5314188
7: -5.4877443, 5.1247644, -6.0060730, 5.5710282, -11.0587730, 11.1308374
8: -7.0458145, 4.3549380, -7.7022982, 4.7289062, -11.7747211, 12.0572357
9: -4.9297538, 5.0593185, -5.3786964, 5.5174665, -10.4472198, 10.4380150

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418659, upper bound: 10.8418334
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418321, upper bound: 10.8418325
time: 2.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.9140306, 3.0436082, -6.6248856, 3.4118989, -9.3259296, 9.6684933
1: -3.8967600, 3.7642124, -4.3688440, 4.2132726, -8.1100330, 8.1330566
2: -4.9132462, 3.8297064, -5.5747519, 4.2824383, -9.1956844, 9.4044580
3: -5.2251310, 3.3809280, -5.8933353, 3.7750039, -9.0001354, 9.2742634
4: -5.7461271, 4.6486964, -6.4846783, 5.2198944, -10.9660215, 11.1333752
5: -4.7567749, 3.4694471, -5.3658404, 3.8666933, -8.6234684, 8.8352871
6: -4.3906856, 4.5409346, -4.9644737, 5.1021767, -9.4928627, 9.5054083
7: -4.8556237, 4.5812330, -5.5017748, 5.1310849, -9.9867086, 10.0830078
8: -6.2345629, 3.8940446, -7.0322032, 4.3511925, -10.5857553, 10.9262476
9: -4.3770018, 4.5038028, -4.9273858, 5.0574002, -9.4344025, 9.4311886

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418721, upper bound: 10.8418648
time: 3.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418721, upper bound: 10.8418646
time: 3.21 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.8116031, 2.9942837, -7.3838596, 3.7681737, -9.5797768, 10.3781433
1: -3.8288946, 3.6995342, -4.8802261, 4.6993656, -8.5282602, 8.5797606
2: -4.8191342, 3.7663870, -6.2792187, 4.7581372, -9.5772715, 10.0456057
3: -5.1289639, 3.3247123, -6.6490340, 4.1951532, -9.3241177, 9.9737463
4: -5.6396713, 4.5663939, -7.2889719, 5.8385181, -11.4781895, 11.8553658
5: -4.6701303, 3.4129658, -6.0128870, 4.2866731, -8.9568033, 9.4258528
6: -4.3085456, 4.4612160, -5.5713139, 5.6978812, -10.0064268, 10.0325298
7: -4.7657561, 4.5039239, -6.1715102, 5.7110276, -10.4767838, 10.6754341
8: -6.1200242, 3.8287840, -7.8924174, 4.8418417, -10.9618664, 11.7212009
9: -4.2990842, 4.4253054, -5.5131154, 5.6553974, -9.9544811, 9.9384212

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418655, upper bound: 10.8418211
time: 3.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418385, upper bound: 10.8418211
time: 2.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.7393360, 3.4298229, -6.3053184, 3.2544355, -9.9937716, 9.7351418
1: -4.4478135, 4.2898297, -4.1552782, 4.0103369, -8.4581509, 8.4451084
2: -5.6740150, 4.3425436, -5.2785544, 4.0812855, -9.7553005, 9.6210976
3: -6.0082946, 3.8355651, -5.5891037, 3.5980761, -9.6063709, 9.4246693
4: -6.6142368, 5.3159237, -6.1496978, 4.9611111, -11.5753479, 11.4656219
5: -5.4556208, 3.9228878, -5.0921431, 3.6901307, -9.1457520, 9.0150309
6: -5.0513749, 5.1823626, -4.7078476, 4.8517275, -9.9031029, 9.8902102
7: -5.5792704, 5.2033987, -5.2179785, 4.8884001, -10.4676704, 10.4213772
8: -7.1623602, 4.4216914, -6.6713152, 4.1466579, -11.3090181, 11.0930061
9: -5.0092063, 5.1394053, -4.6810789, 4.8095245, -9.8187313, 9.8204842

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418752, upper bound: 10.8418648
time: 2.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418649
time: 2.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.7393360, 3.4298229, -7.0584607, 3.6089172, -10.3482533, 10.4882832
1: -4.4478135, 4.2898297, -4.6578755, 4.4893708, -8.9371843, 8.9477053
2: -5.6740150, 4.3425436, -5.9713945, 4.5498142, -10.2238293, 10.3139381
3: -6.0082946, 3.8355651, -6.3164043, 4.0131030, -10.0213976, 10.1519699
4: -6.6142368, 5.3159237, -6.9412670, 5.5690036, -12.1832409, 12.2571907
5: -5.4556208, 3.9228878, -5.7297173, 4.1058216, -9.5614424, 9.6526051
6: -5.0513749, 5.1823626, -5.3109918, 5.4372406, -10.4886150, 10.4933548
7: -5.5792704, 5.2033987, -5.8769274, 5.4576788, -11.0369492, 11.0803261
8: -7.1623602, 4.4216914, -7.5169945, 4.6297545, -11.7921143, 11.9386864
9: -5.0092063, 5.1394053, -5.2571592, 5.3929949, -10.4022007, 10.3965645

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418756, upper bound: 10.8418652
time: 2.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418643
time: 3.20 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.6248856, 3.4118989, -6.4354076, 3.3369887, -9.9618740, 9.8473063
1: -4.3688440, 4.2132726, -4.2396083, 4.0879951, -8.4568386, 8.4528809
2: -5.5747519, 4.2824383, -5.3993311, 4.1663837, -9.7411356, 9.6817694
3: -5.8933353, 3.7750039, -5.7057648, 3.6693094, -9.5626450, 9.4807682
4: -6.4846783, 5.2198944, -6.2736416, 5.0633287, -11.5480070, 11.4935360
5: -5.3658404, 3.8666933, -5.2017756, 3.7701623, -9.1360025, 9.0684690
6: -4.9644737, 5.1021767, -4.8227043, 4.9624438, -9.9269180, 9.9248810
7: -5.5017748, 5.1310849, -5.3366976, 4.9867616, -10.4885368, 10.4677830
8: -7.0322032, 4.3511925, -6.8160396, 4.2320890, -11.2642918, 11.1672325
9: -4.9273858, 5.0574002, -4.7773967, 4.9079146, -9.8353004, 9.8347969

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418262, upper bound: 10.8418283
time: 2.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418262, upper bound: 10.8418286
time: 2.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3838596, 3.7681737, -6.3361425, 3.2893867, -10.6732464, 10.1043167
1: -4.8802261, 4.6993656, -4.1739197, 4.0254574, -8.9056835, 8.8732853
2: -6.2792187, 4.7581372, -5.3084030, 4.1045804, -10.3837986, 10.0665398
3: -6.6490340, 4.1951532, -5.6126671, 3.6148014, -10.2638359, 9.8078203
4: -7.2889719, 5.8385181, -6.1705108, 4.9838142, -12.2727861, 12.0090294
5: -6.0128870, 4.2866731, -5.1180277, 3.7152758, -9.7281628, 9.4047012
6: -5.5713139, 5.6978812, -4.7430844, 4.8854232, -10.4567375, 10.4409657
7: -6.1715102, 5.7110276, -5.2497988, 4.9120317, -11.0835419, 10.9608269
8: -7.8924174, 4.8418417, -6.7053719, 4.1686897, -12.0611076, 11.5472136
9: -5.5131154, 5.6553974, -4.7019329, 4.8318510, -10.3449669, 10.3573303

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418282
time: 2.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418277
time: 2.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.3053184, 3.2544355, -7.2978354, 3.7428906, -10.0482092, 10.5522709
1: -4.1552782, 4.0103369, -4.8161936, 4.6370029, -8.7922812, 8.8265305
2: -5.2785544, 4.0812855, -6.1939793, 4.7036676, -9.9822216, 10.2752647
3: -5.5891037, 3.5980761, -6.5506630, 4.1448550, -9.7339592, 10.1487389
4: -6.1496978, 4.9611111, -7.1810155, 5.7602339, -11.9099312, 12.1421261
5: -5.0921431, 3.6901307, -5.9321175, 4.2477264, -9.3398695, 9.6222477
6: -4.7078476, 4.8517275, -5.5130339, 5.6347446, -10.3425922, 10.3647614
7: -5.2179785, 4.8884001, -6.0927730, 5.6414976, -10.8594761, 10.9811726
8: -6.6713152, 4.1466579, -7.7854891, 4.7875400, -11.4588547, 11.9321470
9: -4.6810789, 4.8095245, -5.4377251, 5.5806322, -10.2617111, 10.2472496

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418615
time: 3.09 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418281, upper bound: 10.8418613
time: 2.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.0584607, 3.6089172, -7.2978354, 3.7428906, -10.8013515, 10.9067526
1: -4.6578755, 4.4893708, -4.8161936, 4.6370029, -9.2948780, 9.3055649
2: -5.9713945, 4.5498142, -6.1939793, 4.7036676, -10.6750622, 10.7437935
3: -6.3164043, 4.0131030, -6.5506630, 4.1448550, -10.4612598, 10.5637665
4: -6.9412670, 5.5690036, -7.1810155, 5.7602339, -12.7015009, 12.7500191
5: -5.7297173, 4.1058216, -5.9321175, 4.2477264, -9.9774437, 10.0379391
6: -5.3109918, 5.4372406, -5.5130339, 5.6347446, -10.9457359, 10.9502745
7: -5.8769274, 5.4576788, -6.0927730, 5.6414976, -11.5184250, 11.5504513
8: -7.5169945, 4.6297545, -7.7854891, 4.7875400, -12.3045349, 12.4152431
9: -5.2571592, 5.3929949, -5.4377251, 5.5806322, -10.8377914, 10.8307199

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418610
time: 3.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418214, upper bound: 10.8418610
time: 2.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.00 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418722, upper bound: 10.8418679
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418722, upper bound: 10.8418688
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418709, upper bound: 10.8418324
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418385, upper bound: 10.8418318
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418685, upper bound: 10.8418687
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418685, upper bound: 10.8418684
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418659, upper bound: 10.8418334
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418321, upper bound: 10.8418325
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418721, upper bound: 10.8418648
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418721, upper bound: 10.8418646
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418655, upper bound: 10.8418211
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418385, upper bound: 10.8418211
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418752, upper bound: 10.8418648
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418649
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418756, upper bound: 10.8418652
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418643
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418262, upper bound: 10.8418283
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418262, upper bound: 10.8418286
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418282
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418277
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418615
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418281, upper bound: 10.8418613
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418610
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.00
Output dim: 0, lower bound: -10.8418214, upper bound: 10.8418610

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.1456804, 2.6726890, -6.3353863, 3.2485588, -8.3942394, 9.0080757
1: -3.3895102, 3.2789326, -4.1792064, 4.0327048, -7.4222150, 7.4581389
2: -4.2082262, 3.3543770, -5.3048878, 4.0940523, -8.3022785, 8.6592646
3: -4.5038347, 2.9588246, -5.6268563, 3.6139324, -8.1177673, 8.5856810
4: -4.9459782, 4.0353332, -6.1890211, 4.9908166, -9.9367943, 10.2243538
5: -4.1065888, 3.0467172, -5.1170745, 3.7021790, -7.8087678, 8.1637917
6: -3.7769420, 3.9443822, -4.7290378, 4.8720789, -8.6490211, 8.6734200
7: -4.1813045, 4.0000744, -5.2304592, 4.9018402, -9.0831451, 9.2305336
8: -5.3735533, 3.4055884, -6.7112384, 4.1642365, -9.5377903, 10.1168270
9: -3.7928834, 3.9135110, -4.7023783, 4.8306150, -8.6234989, 8.6158895

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418392, upper bound: 10.8418725
time: 3.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418383, upper bound: 10.8418417
time: 3.06 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0008326, 3.0761125, -6.3353863, 3.2485588, -9.2493916, 9.4114990
1: -3.9621720, 3.8269193, -4.1792064, 4.0327048, -7.9948769, 8.0061255
2: -5.0026145, 3.8867745, -5.3048878, 4.0940523, -9.0966663, 9.1916618
3: -5.3212023, 3.4317949, -5.6268563, 3.6139324, -8.9351349, 9.0586510
4: -5.8514624, 4.7289529, -6.1890211, 4.9908166, -10.8422794, 10.9179745
5: -4.8369226, 3.5136783, -5.1170745, 3.7021790, -8.5391016, 8.6307526
6: -4.4584570, 4.6130958, -4.7290378, 4.8720789, -9.3305359, 9.3421335
7: -4.9380789, 4.6513405, -5.2304592, 4.9018402, -9.8399191, 9.8817997
8: -6.3448548, 3.9510064, -6.7112384, 4.1642365, -10.5090914, 10.6622448
9: -4.4523010, 4.5783863, -4.7023783, 4.8306150, -9.2829161, 9.2807646

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418392, upper bound: 10.8418714
time: 3.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418383, upper bound: 10.8418421
time: 2.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.1995449, 2.6777778, -7.1982613, 3.6561396, -8.8556843, 9.8760395
1: -3.4298589, 3.3189757, -4.7593546, 4.5847540, -8.0146132, 8.0783300
2: -4.2601910, 3.3850479, -6.1062446, 4.6360502, -8.8962412, 9.4912930
3: -4.5644293, 2.9891331, -6.4718242, 4.0915508, -8.6559801, 9.4609575
4: -5.0145874, 4.0849466, -7.1015234, 5.6934166, -10.7080040, 11.1864700
5: -4.1532965, 3.0703590, -5.8529358, 4.1787171, -8.3320141, 8.9232950
6: -3.8144424, 3.9839282, -5.4195728, 5.5484815, -9.3629236, 9.4035015
7: -4.2239814, 4.0397096, -5.9945340, 5.5609932, -9.7849751, 10.0342436
8: -5.4410276, 3.4388592, -7.6877117, 4.7203941, -10.1614218, 11.1265707
9: -3.8373079, 3.9575059, -5.3687410, 5.5072198, -9.3445282, 9.3262472

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418396, upper bound: 10.8418331
time: 2.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418396, upper bound: 10.8418322
time: 2.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.9371543, 3.0434709, -6.3489361, 3.2553077, -9.1924620, 9.3924065
1: -3.9161406, 3.7839019, -4.1880636, 4.0411129, -7.9572535, 7.9719658
2: -4.9378753, 3.8432474, -5.3171782, 4.1024704, -9.0403461, 9.1604252
3: -5.2542038, 3.3949192, -5.6393843, 3.6213410, -8.8755445, 9.0343037
4: -5.7800894, 4.6722612, -6.2028656, 5.0015240, -10.7816133, 10.8751268
5: -4.7772279, 3.4792390, -5.1284804, 3.7097194, -8.4869471, 8.6077194
6: -4.4083147, 4.5593653, -4.7398849, 4.8825068, -9.2908211, 9.2992496
7: -4.8753810, 4.5987363, -5.2422991, 4.9119992, -9.7873802, 9.8410358
8: -6.2662902, 3.9093165, -6.7261724, 4.1728592, -10.4391499, 10.6354885
9: -4.3980331, 4.5241385, -4.7125816, 4.8409166, -9.2389498, 9.2367201

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418322, upper bound: 10.8418724
time: 2.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418426
time: 2.52 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.7957907, 3.4479465, -6.3489361, 3.2553077, -10.0510979, 9.7968826
1: -4.4929895, 4.3334131, -4.1880636, 4.0411129, -8.5341024, 8.5214767
2: -5.7356710, 4.3816500, -5.3171782, 4.1024704, -9.8381414, 9.6988277
3: -6.0753632, 3.8698959, -5.6393843, 3.6213410, -9.6967039, 9.5092802
4: -6.6879244, 5.3718209, -6.2028656, 5.0015240, -11.6894484, 11.5746861
5: -5.5099669, 3.9505434, -5.1284804, 3.7097194, -9.2196865, 9.0790234
6: -5.0953851, 5.2310023, -4.7398849, 4.8825068, -9.9778919, 9.9708872
7: -5.6359997, 5.2513366, -5.2422991, 4.9119992, -10.5479984, 10.4936352
8: -7.2384176, 4.4595127, -6.7261724, 4.1728592, -11.4112768, 11.1856852
9: -5.0615201, 5.1911583, -4.7125816, 4.8409166, -9.9024372, 9.9037399

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418322, upper bound: 10.8418725
time: 2.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418431
time: 2.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.0207863, 3.0634670, -7.2114840, 3.6627753, -9.6835613, 10.2749510
1: -3.9765973, 3.8427012, -4.7680202, 4.5929632, -8.5695610, 8.6107216
2: -5.0175257, 3.8928704, -6.1182265, 4.6442709, -9.6617966, 10.0110970
3: -5.3429594, 3.4415696, -6.4847126, 4.0987763, -9.4417362, 9.9262819
4: -5.8794065, 4.7463627, -7.1150351, 5.7038603, -11.5832672, 11.8613977
5: -4.8493214, 3.5197592, -5.8640642, 4.1861744, -9.0354958, 9.3838234
6: -4.4701662, 4.6224046, -5.4301567, 5.5587230, -10.0288887, 10.0525608
7: -4.9448180, 4.6609669, -6.0060730, 5.5710282, -10.5158463, 10.6670399
8: -6.3668699, 3.9619489, -7.7022982, 4.7289062, -11.0957756, 11.6642475
9: -4.4655128, 4.5909796, -5.3786964, 5.5174665, -9.9829788, 9.9696760

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418334, upper bound: 10.8418323
time: 3.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418334, upper bound: 10.8418322
time: 3.17 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.1456804, 2.6726890, -6.6248856, 3.4118989, -8.5575790, 9.2975750
1: -3.3895102, 3.2789326, -4.3688440, 4.2132726, -7.6027827, 7.6477766
2: -4.2082262, 3.3543770, -5.5747519, 4.2824383, -8.4906645, 8.9291286
3: -4.5038347, 2.9588246, -5.8933353, 3.7750039, -8.2788391, 8.8521595
4: -4.9459782, 4.0353332, -6.4846783, 5.2198944, -10.1658726, 10.5200119
5: -4.1065888, 3.0467172, -5.3658404, 3.8666933, -7.9732819, 8.4125576
6: -3.7769420, 3.9443822, -4.9644737, 5.1021767, -8.8791189, 8.9088554
7: -4.1813045, 4.0000744, -5.5017748, 5.1310849, -9.3123894, 9.5018492
8: -5.3735533, 3.4055884, -7.0322032, 4.3511925, -9.7247458, 10.4377918
9: -3.7928834, 3.9135110, -4.9273858, 5.0574002, -8.8502836, 8.8408966

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418625
time: 2.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418389, upper bound: 10.8418273
time: 3.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.0008326, 3.0761125, -6.6248856, 3.4118989, -9.4127312, 9.7009983
1: -3.9621720, 3.8269193, -4.3688440, 4.2132726, -8.1754446, 8.1957636
2: -5.0026145, 3.8867745, -5.5747519, 4.2824383, -9.2850533, 9.4615269
3: -5.3212023, 3.4317949, -5.8933353, 3.7750039, -9.0962067, 9.3251305
4: -5.8514624, 4.7289529, -6.4846783, 5.2198944, -11.0713568, 11.2136307
5: -4.8369226, 3.5136783, -5.3658404, 3.8666933, -8.7036161, 8.8795185
6: -4.4584570, 4.6130958, -4.9644737, 5.1021767, -9.5606337, 9.5775700
7: -4.9380789, 4.6513405, -5.5017748, 5.1310849, -10.0691643, 10.1531153
8: -6.3448548, 3.9510064, -7.0322032, 4.3511925, -10.6960468, 10.9832096
9: -4.4523010, 4.5783863, -4.9273858, 5.0574002, -9.5097008, 9.5057716

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418643
time: 3.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418389, upper bound: 10.8418266
time: 2.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.1995449, 2.6777778, -7.3838596, 3.7681737, -8.9677181, 10.0616379
1: -3.4298589, 3.3189757, -4.8802261, 4.6993656, -8.1292248, 8.1992016
2: -4.2601910, 3.3850479, -6.2792187, 4.7581372, -9.0183277, 9.6642666
3: -4.5644293, 2.9891331, -6.6490340, 4.1951532, -8.7595825, 9.6381674
4: -5.0145874, 4.0849466, -7.2889719, 5.8385181, -10.8531055, 11.3739185
5: -4.1532965, 3.0703590, -6.0128870, 4.2866731, -8.4399700, 9.0832462
6: -3.8144424, 3.9839282, -5.5713139, 5.6978812, -9.5123234, 9.5552425
7: -4.2239814, 4.0397096, -6.1715102, 5.7110276, -9.9350090, 10.2112198
8: -5.4410276, 3.4388592, -7.8924174, 4.8418417, -10.2828693, 11.3312769
9: -3.8373079, 3.9575059, -5.5131154, 5.6553974, -9.4927053, 9.4706211

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418384, upper bound: 10.8418217
time: 2.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418384, upper bound: 10.8418209
time: 3.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.9371543, 3.0434709, -6.3053184, 3.2544355, -9.1915894, 9.3487892
1: -3.9161406, 3.7839019, -4.1552782, 4.0103369, -7.9264774, 7.9391804
2: -4.9378753, 3.8432474, -5.2785544, 4.0812855, -9.0191612, 9.1218014
3: -5.2542038, 3.3949192, -5.5891037, 3.5980761, -8.8522797, 8.9840231
4: -5.7800894, 4.6722612, -6.1496978, 4.9611111, -10.7412004, 10.8219585
5: -4.7772279, 3.4792390, -5.0921431, 3.6901307, -8.4673586, 8.5713825
6: -4.4083147, 4.5593653, -4.7078476, 4.8517275, -9.2600422, 9.2672129
7: -4.8753810, 4.5987363, -5.2179785, 4.8884001, -9.7637806, 9.8167152
8: -6.2662902, 3.9093165, -6.6713152, 4.1466579, -10.4129486, 10.5806313
9: -4.3980331, 4.5241385, -4.6810789, 4.8095245, -9.2075577, 9.2052174

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418687, upper bound: 10.8418677
time: 2.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418687, upper bound: 10.8418675
time: 2.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.7957907, 3.4479465, -6.2008882, 3.2042334, -10.0000238, 9.6488342
1: -4.4929895, 4.3334131, -4.0861259, 3.9445333, -8.4375229, 8.4195385
2: -5.7356710, 4.3816500, -5.1828160, 4.0164285, -9.7521000, 9.5644665
3: -6.0753632, 3.8698959, -5.4911194, 3.5408072, -9.6161709, 9.3610153
4: -6.6879244, 5.3718209, -6.0413036, 4.8773999, -11.5653248, 11.4131241
5: -5.5099669, 3.9505434, -5.0040054, 3.6322920, -9.1422586, 8.9545488
6: -5.0953851, 5.2310023, -4.6240225, 4.7705998, -9.8659849, 9.8550243
7: -5.6359997, 5.2513366, -5.1264935, 4.8097401, -10.4457397, 10.3778305
8: -7.2384176, 4.4595127, -6.5547991, 4.0799513, -11.3183689, 11.0143118
9: -5.0615201, 5.1911583, -4.6016560, 4.7296047, -9.7911243, 9.7928143

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418328, upper bound: 10.8418636
time: 3.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418329, upper bound: 10.8418284
time: 3.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.9371543, 3.0434709, -7.0584607, 3.6089172, -9.5460720, 10.1019316
1: -3.9161406, 3.7839019, -4.6578755, 4.4893708, -8.4055119, 8.4417772
2: -4.9378753, 3.8432474, -5.9713945, 4.5498142, -9.4876900, 9.8146420
3: -5.2542038, 3.3949192, -6.3164043, 4.0131030, -9.2673073, 9.7113237
4: -5.7800894, 4.6722612, -6.9412670, 5.5690036, -11.3490925, 11.6135283
5: -4.7772279, 3.4792390, -5.7297173, 4.1058216, -8.8830490, 9.2089558
6: -4.4083147, 4.5593653, -5.3109918, 5.4372406, -9.8455553, 9.8703575
7: -4.8753810, 4.5987363, -5.8769274, 5.4576788, -10.3330593, 10.4756641
8: -6.2662902, 3.9093165, -7.5169945, 4.6297545, -10.8960447, 11.4263115
9: -4.3980331, 4.5241385, -5.2571592, 5.3929949, -9.7910280, 9.7812977

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418652
time: 3.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418653
time: 2.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.7957907, 3.4479465, -6.9492555, 3.5562177, -10.3520088, 10.3972015
1: -4.4929895, 4.3334131, -4.5853720, 4.4205446, -8.9135342, 8.9187851
2: -5.7356710, 4.3816500, -5.8713460, 4.4817681, -10.2174397, 10.2529964
3: -6.0753632, 3.8698959, -6.2084780, 3.9531331, -10.0284958, 10.0783739
4: -6.6879244, 5.3718209, -6.8277454, 5.4814944, -12.1694183, 12.1995659
5: -5.5099669, 3.9505434, -5.6376190, 4.0446033, -9.5545702, 9.5881624
6: -5.0953851, 5.2310023, -5.2233448, 5.3519945, -10.4473801, 10.4543476
7: -5.6359997, 5.2513366, -5.7813454, 5.3744936, -11.0104933, 11.0326824
8: -7.2384176, 4.4595127, -7.3952074, 4.5591307, -11.7975483, 11.8547201
9: -5.0615201, 5.1911583, -5.1741438, 5.3077502, -10.3692703, 10.3653021

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418333, upper bound: 10.8418616
time: 3.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418218
time: 3.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.4863024, 2.8597903, -7.2978354, 3.7428906, -9.2291927, 10.1576252
1: -3.6124439, 3.4929590, -4.8161936, 4.6370029, -8.2494469, 8.3091526
2: -4.5258231, 3.5734527, -6.1939793, 4.7036676, -9.2294903, 9.7674322
3: -4.8188763, 3.1482296, -6.5506630, 4.1448550, -8.9637318, 9.6988926
4: -5.2970591, 4.3041267, -7.1810155, 5.7602339, -11.0572929, 11.4851418
5: -4.3987713, 3.2382784, -5.9321175, 4.2477264, -8.6464977, 9.1703959
6: -4.0519695, 4.2145114, -5.5130339, 5.6347446, -9.6867142, 9.7275448
7: -4.4985981, 4.2694802, -6.0927730, 5.6414976, -10.1400957, 10.3622532
8: -5.7543092, 3.6243405, -7.7854891, 4.7875400, -10.5418491, 11.4098301
9: -4.0570917, 4.1808109, -5.4377251, 5.5806322, -9.6377239, 9.6185360

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418328, upper bound: 10.8418594
time: 3.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418328, upper bound: 10.8418601
time: 3.46 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1809220, 3.1845155, -7.1844974, 3.6882691, -9.8691912, 10.3690128
1: -4.0792761, 3.9383545, -4.7407961, 4.5653720, -8.6446476, 8.6791506
2: -5.1708407, 4.0050287, -6.0899415, 4.6329417, -9.8037825, 10.0949707
3: -5.4851189, 3.5320981, -6.4381733, 4.0825472, -9.5676661, 9.9702711
4: -6.0332732, 4.8699932, -7.0628767, 5.6692038, -11.7024765, 11.9328699
5: -4.9924374, 3.6173661, -5.8363509, 4.1842546, -9.1766920, 9.4537172
6: -4.6056709, 4.7573004, -5.4221506, 5.5460372, -10.1517086, 10.1794510
7: -5.1104822, 4.7960491, -5.9934464, 5.5550194, -10.6655016, 10.7894955
8: -6.5434933, 4.0675793, -7.6586161, 4.7142386, -11.2577324, 11.7261953
9: -4.5929127, 4.7196722, -5.3513689, 5.4918013, -10.0847139, 10.0710411

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418286, upper bound: 10.8418603
time: 2.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418283, upper bound: 10.8418602
time: 2.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.2038436, 3.1974018, -7.2978354, 3.7428906, -9.9467344, 10.4952374
1: -4.0909982, 3.9502230, -4.8161936, 4.6370029, -8.7280006, 8.7664165
2: -5.1873441, 4.0170794, -6.1939793, 4.7036676, -9.8910122, 10.2110586
3: -5.4998894, 3.5433230, -6.5506630, 4.1448550, -9.6447449, 10.0939865
4: -6.0518875, 4.8833451, -7.1810155, 5.7602339, -11.8121214, 12.0643606
5: -5.0075426, 3.6313958, -5.9321175, 4.2477264, -9.2552691, 9.5635128
6: -4.6255407, 4.7724142, -5.5130339, 5.6347446, -10.2602854, 10.2854481
7: -5.1273813, 4.8111987, -6.0927730, 5.6414976, -10.7688789, 10.9039717
8: -6.5625849, 4.0816412, -7.7854891, 4.7875400, -11.3501244, 11.8671303
9: -4.6062346, 4.7333307, -5.4377251, 5.5806322, -10.1868668, 10.1710558

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418605
time: 2.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418603
time: 2.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.9702139, 3.5557473, -7.1844974, 3.6882691, -10.6584835, 10.7402449
1: -4.6064243, 4.4408340, -4.7407961, 4.5653720, -9.1717968, 9.1816301
2: -5.8985720, 4.4969873, -6.0899415, 4.6329417, -10.5315132, 10.5869293
3: -6.2418804, 3.9674559, -6.4381733, 4.0825472, -10.3244276, 10.4056292
4: -6.8634853, 5.5078368, -7.0628767, 5.6692038, -12.5326891, 12.5707130
5: -5.6606913, 4.0525503, -5.8363509, 4.1842546, -9.8449459, 9.8889008
6: -5.2384191, 5.3718424, -5.4221506, 5.5460372, -10.7844563, 10.7939930
7: -5.8037043, 5.3930817, -5.9934464, 5.5550194, -11.3587236, 11.3865280
8: -7.4305086, 4.5739317, -7.6586161, 4.7142386, -12.1447468, 12.2325478
9: -5.1975718, 5.3307428, -5.3513689, 5.4918013, -10.6893730, 10.6821117

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418603
time: 3.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418605
time: 3.25 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.09 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418392, upper bound: 10.8418725
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418383, upper bound: 10.8418417
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418392, upper bound: 10.8418714
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418383, upper bound: 10.8418421
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418396, upper bound: 10.8418331
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418396, upper bound: 10.8418322
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418322, upper bound: 10.8418724
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418426
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418322, upper bound: 10.8418725
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418431
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418334, upper bound: 10.8418323
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418334, upper bound: 10.8418322
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418625
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418389, upper bound: 10.8418273
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418397, upper bound: 10.8418643
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418389, upper bound: 10.8418266
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418384, upper bound: 10.8418217
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418384, upper bound: 10.8418209
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418687, upper bound: 10.8418677
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418687, upper bound: 10.8418675
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418328, upper bound: 10.8418636
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418329, upper bound: 10.8418284
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418652
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418653
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418333, upper bound: 10.8418616
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418218
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418328, upper bound: 10.8418594
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418328, upper bound: 10.8418601
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418286, upper bound: 10.8418603
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418283, upper bound: 10.8418602
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418605
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418603
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418603
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.09
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418605

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.1456804, 2.6726890, -5.7167516, 2.9301786, -8.0758591, 8.3894405
1: -3.3895102, 3.2789326, -3.7737803, 3.6480432, -7.0375533, 7.0527129
2: -4.2082262, 3.3543770, -4.7392192, 3.7077065, -7.9159327, 8.0935965
3: -4.5038347, 2.9588246, -5.0547991, 3.2750421, -7.7788768, 8.0136242
4: -4.9459782, 4.0353332, -5.5577736, 4.5006485, -9.4466267, 9.5931072
5: -4.1065888, 3.0467172, -4.5944419, 3.3547754, -7.4613643, 7.6411591
6: -3.7769420, 3.9443822, -4.2282467, 4.3888531, -8.1657953, 8.1726284
7: -4.1813045, 4.0000744, -4.6830568, 4.4338446, -8.6151485, 8.6831312
8: -5.3735533, 3.4055884, -6.0262251, 3.7686982, -9.1422520, 9.4318132
9: -3.7928834, 3.9135110, -4.2342577, 4.3588066, -8.1516895, 8.1477690

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418523, upper bound: 10.8418575
time: 4.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418290, upper bound: 10.8418564
time: 3.13 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.9386725, 2.5670664, -8.2184830, 4.0730910, -9.0117636, 10.7855492
1: -3.2549624, 3.1499581, -5.4570708, 5.2541413, -8.5091038, 8.6070290
2: -4.0194125, 3.2258985, -7.0585070, 5.2668796, -9.2862921, 10.2844057
3: -4.3129034, 2.8453975, -7.4530101, 4.6577430, -8.9706459, 10.2984076
4: -4.7341208, 3.8732393, -8.2153492, 6.5421133, -11.2762337, 12.0885887
5: -3.9325888, 2.9309256, -6.7216654, 4.7207279, -8.6533165, 9.6525908
6: -3.6097758, 3.7835598, -6.2256303, 6.3375168, -9.9472923, 10.0091896
7: -3.9986043, 3.8434393, -6.8799734, 6.3237324, -10.3223362, 10.7234125
8: -5.1437283, 3.2739449, -8.8602180, 5.3695889, -10.5133171, 12.1341629
9: -3.6369002, 3.7554755, -6.1631775, 6.2981653, -9.9350653, 9.9186535

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418511, upper bound: 10.8418220
time: 5.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418298, upper bound: 10.8418218
time: 2.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.0008326, 3.0761125, -5.7167516, 2.9301786, -8.9310112, 8.7928638
1: -3.9621720, 3.8269193, -3.7737803, 3.6480432, -7.6102152, 7.6006994
2: -5.0026145, 3.8867745, -4.7392192, 3.7077065, -8.7103214, 8.6259937
3: -5.3212023, 3.4317949, -5.0547991, 3.2750421, -8.5962448, 8.4865942
4: -5.8514624, 4.7289529, -5.5577736, 4.5006485, -10.3521109, 10.2867260
5: -4.8369226, 3.5136783, -4.5944419, 3.3547754, -8.1916981, 8.1081200
6: -4.4584570, 4.6130958, -4.2282467, 4.3888531, -8.8473101, 8.8413429
7: -4.9380789, 4.6513405, -4.6830568, 4.4338446, -9.3719234, 9.3343973
8: -6.3448548, 3.9510064, -6.0262251, 3.7686982, -10.1135530, 9.9772320
9: -4.4523010, 4.5783863, -4.2342577, 4.3588066, -8.8111076, 8.8126440

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418398, upper bound: 10.8418428
time: 2.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418398, upper bound: 10.8418424
time: 2.96 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.7952414, 2.9710410, -8.2184830, 4.0730910, -9.8683319, 11.1895237
1: -3.8273544, 3.6989326, -5.4570708, 5.2541413, -9.0814953, 9.1560040
2: -4.8145595, 3.7586501, -7.0585070, 5.2668796, -10.0814390, 10.8171568
3: -5.1310892, 3.3192284, -7.4530101, 4.6577430, -9.7888317, 10.7722387
4: -5.6415353, 4.5658655, -8.2153492, 6.5421133, -12.1836491, 12.7812147
5: -4.6635051, 3.3982623, -6.7216654, 4.7207279, -9.3842335, 10.1199274
6: -4.2918272, 4.4524379, -6.2256303, 6.3375168, -10.6293440, 10.6780682
7: -4.7563772, 4.4959702, -6.8799734, 6.3237324, -11.0801096, 11.3759441
8: -6.1171217, 3.8195341, -8.8602180, 5.3695889, -11.4867105, 12.6797523
9: -4.2968655, 4.4216757, -6.1631775, 6.2981653, -10.5950308, 10.5848532

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418389, upper bound: 10.8418221
time: 2.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418208, upper bound: 10.8418222
time: 2.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9371543, 3.0434709, -5.7306690, 2.9371867, -8.8743410, 8.7741394
1: -3.9161406, 3.7839019, -3.7828979, 3.6566932, -7.5728340, 7.5668001
2: -4.9378753, 3.8432474, -4.7519073, 3.7163029, -8.6541786, 8.5951548
3: -5.2542038, 3.3949192, -5.0676908, 3.2826552, -8.5368595, 8.4626102
4: -5.7800894, 4.6722612, -5.5720148, 4.5116544, -10.2917442, 10.2442760
5: -4.7772279, 3.4792390, -4.6062193, 3.3624897, -8.1397171, 8.0854588
6: -4.4083147, 4.5593653, -4.2393541, 4.3996220, -8.8079367, 8.7987194
7: -4.8753810, 4.5987363, -4.6952648, 4.4443378, -9.3197193, 9.2940006
8: -6.2662902, 3.9093165, -6.0416327, 3.7775397, -10.0438299, 9.9509487
9: -4.3980331, 4.5241385, -4.2447462, 4.3694134, -8.7674465, 8.7688847

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418425, upper bound: 10.8418426
time: 2.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418425, upper bound: 10.8418427
time: 2.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.7280502, 2.9366791, -8.2318306, 4.0797620, -9.8078117, 11.1685095
1: -3.7790453, 3.6536474, -5.4658155, 5.2624440, -9.0414896, 9.1194630
2: -4.7465348, 3.7132132, -7.0706396, 5.2751913, -10.0217266, 10.7838526
3: -5.0608530, 3.2803850, -7.4653816, 4.6650515, -9.7259045, 10.7457666
4: -5.5664492, 4.5063019, -8.2290258, 6.5526838, -12.1191330, 12.7353277
5: -4.6007905, 3.3620973, -6.7329206, 4.7281590, -9.3289490, 10.0950184
6: -4.2389636, 4.3958955, -6.2363405, 6.3478093, -10.5867729, 10.6322365
7: -4.6906576, 4.4406719, -6.8916664, 6.3337545, -11.0244122, 11.3323383
8: -6.0345435, 3.7757549, -8.8749638, 5.3780918, -11.4126358, 12.6507187
9: -4.2400222, 4.3647075, -6.1732531, 6.3083324, -10.5483551, 10.5379601

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418408, upper bound: 10.8418221
time: 2.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418220, upper bound: 10.8418221
time: 2.57 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.7957907, 3.4479465, -5.7306690, 2.9371867, -9.7329769, 9.1786156
1: -4.4929895, 4.3334131, -3.7828979, 3.6566932, -8.1496830, 8.1163111
2: -5.7356710, 4.3816500, -4.7519073, 3.7163029, -9.4519739, 9.1335573
3: -6.0753632, 3.8698959, -5.0676908, 3.2826552, -9.3580189, 8.9375868
4: -6.6879244, 5.3718209, -5.5720148, 4.5116544, -11.1995792, 10.9438362
5: -5.5099669, 3.9505434, -4.6062193, 3.3624897, -8.8724566, 8.5567627
6: -5.0953851, 5.2310023, -4.2393541, 4.3996220, -9.4950066, 9.4703560
7: -5.6359997, 5.2513366, -4.6952648, 4.4443378, -10.0803375, 9.9466019
8: -7.2384176, 4.4595127, -6.0416327, 3.7775397, -11.0159569, 10.5011454
9: -5.0615201, 5.1911583, -4.2447462, 4.3694134, -9.4309330, 9.4359045

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418403
time: 3.48 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418428
time: 3.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.5876369, 3.3418067, -8.2318306, 4.0797620, -10.6673985, 11.5736370
1: -4.3566971, 4.2040653, -5.4658155, 5.2624440, -9.6191406, 9.6698809
2: -5.5457115, 4.2512989, -7.0706396, 5.2751913, -10.8209028, 11.3219385
3: -5.8831501, 3.7558107, -7.4653816, 4.6650515, -10.5482016, 11.2211924
4: -6.4754286, 5.2070999, -8.2290258, 6.5526838, -13.0281124, 13.4361258
5: -5.3347883, 3.8335452, -6.7329206, 4.7281590, -10.0629473, 10.5664654
6: -4.9266586, 5.0687890, -6.2363405, 6.3478093, -11.2744675, 11.3051300
7: -5.4523768, 5.0944114, -6.8916664, 6.3337545, -11.7861309, 11.9860783
8: -7.0084190, 4.3262224, -8.8749638, 5.3780918, -12.3865108, 13.2011862
9: -4.9043503, 5.0324574, -6.1732531, 6.3083324, -11.2126827, 11.2057104

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418219
time: 2.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418137, upper bound: 10.8418220
time: 2.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.1456804, 2.6726890, -5.9843440, 3.0858927, -8.2315731, 8.6570330
1: -3.3895102, 3.2789326, -3.9489391, 3.8147852, -7.2042952, 7.2278719
2: -4.2082262, 3.3543770, -4.9894814, 3.8815887, -8.0898151, 8.3438587
3: -4.5038347, 2.9588246, -5.3009396, 3.4240315, -7.9278660, 8.2597637
4: -4.9459782, 4.0353332, -5.8301935, 4.7122335, -9.6582117, 9.8655262
5: -4.1065888, 3.0467172, -4.8257642, 3.5072234, -7.6138124, 7.8724813
6: -3.7769420, 3.9443822, -4.4461956, 4.6027403, -8.3796825, 8.3905773
7: -4.1813045, 4.0000744, -4.9358459, 4.6469455, -8.8282499, 8.9359207
8: -5.3735533, 3.4055884, -6.3235755, 3.9412389, -9.3147926, 9.7291641
9: -3.7928834, 3.9135110, -4.4428444, 4.5686321, -8.3615150, 8.3563557

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418471, upper bound: 10.8418493
time: 3.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418300, upper bound: 10.8418493
time: 2.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.0008326, 3.0761125, -5.9843440, 3.0858927, -9.0867252, 9.0604563
1: -3.9621720, 3.8269193, -3.9489391, 3.8147852, -7.7769575, 7.7758584
2: -5.0026145, 3.8867745, -4.9894814, 3.8815887, -8.8842030, 8.8762560
3: -5.3212023, 3.4317949, -5.3009396, 3.4240315, -8.7452335, 8.7327347
4: -5.8514624, 4.7289529, -5.8301935, 4.7122335, -10.5636959, 10.5591469
5: -4.8369226, 3.5136783, -4.8257642, 3.5072234, -8.3441458, 8.3394423
6: -4.4584570, 4.6130958, -4.4461956, 4.6027403, -9.0611973, 9.0592918
7: -4.9380789, 4.6513405, -4.9358459, 4.6469455, -9.5850239, 9.5871868
8: -6.3448548, 3.9510064, -6.3235755, 3.9412389, -10.2860937, 10.2745819
9: -4.4523010, 4.5783863, -4.4428444, 4.5686321, -9.0209332, 9.0212307

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418386, upper bound: 10.8418260
time: 3.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418386, upper bound: 10.8418271
time: 3.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.9371543, 3.0434709, -5.4863024, 2.8597903, -8.7969446, 8.5297737
1: -3.9161406, 3.7839019, -3.6124439, 3.4929590, -7.4090996, 7.3963461
2: -4.9378753, 3.8432474, -4.5258231, 3.5734527, -8.5113277, 8.3690701
3: -5.2542038, 3.3949192, -4.8188763, 3.1482296, -8.4024334, 8.2137957
4: -5.7800894, 4.6722612, -5.2970591, 4.3041267, -10.0842161, 9.9693203
5: -4.7772279, 3.4792390, -4.3987713, 3.2382784, -8.0155067, 7.8780103
6: -4.4083147, 4.5593653, -4.0519695, 4.2145114, -8.6228256, 8.6113348
7: -4.8753810, 4.5987363, -4.4985981, 4.2694802, -9.1448612, 9.0973339
8: -6.2662902, 3.9093165, -5.7543092, 3.6243405, -9.8906307, 9.6636257
9: -4.3980331, 4.5241385, -4.0570917, 4.1808109, -8.5788441, 8.5812302

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418676, upper bound: 10.8418274
time: 3.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418426, upper bound: 10.8418284
time: 3.15 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.9371543, 3.0434709, -6.1809220, 3.1845155, -9.1216698, 9.2243929
1: -3.9161406, 3.7839019, -4.0792761, 3.9383545, -7.8544950, 7.8631783
2: -4.9378753, 3.8432474, -5.1708407, 4.0050287, -8.9429035, 9.0140877
3: -5.2542038, 3.3949192, -5.4851189, 3.5320981, -8.7863016, 8.8800383
4: -5.7800894, 4.6722612, -6.0332732, 4.8699932, -10.6500826, 10.7055340
5: -4.7772279, 3.4792390, -4.9924374, 3.6173661, -8.3945942, 8.4716759
6: -4.4083147, 4.5593653, -4.6056709, 4.7573004, -9.1656151, 9.1650362
7: -4.8753810, 4.5987363, -5.1104822, 4.7960491, -9.6714306, 9.7092190
8: -6.2662902, 3.9093165, -6.5434933, 4.0675793, -10.3338699, 10.4528103
9: -4.3980331, 4.5241385, -4.5929127, 4.7196722, -9.1177053, 9.1170511

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418676, upper bound: 10.8418278
time: 4.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418426, upper bound: 10.8418279
time: 3.18 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.7957907, 3.4479465, -5.5674343, 2.8801978, -9.6759882, 9.0153809
1: -4.4929895, 4.3334131, -3.6708441, 3.5502357, -8.0432253, 8.0042572
2: -5.7356710, 4.3816500, -4.6031914, 3.6214607, -9.3571320, 8.9848413
3: -6.0753632, 3.8698959, -4.9052415, 3.1937904, -9.2691536, 8.7751369
4: -6.6879244, 5.3718209, -5.3940477, 4.3753977, -11.0633221, 10.7658691
5: -5.5099669, 3.9505434, -4.4689746, 3.2773957, -8.7873631, 8.4195175
6: -5.0953851, 5.2310023, -4.1120558, 4.2757854, -9.3711700, 9.3430576
7: -5.6359997, 5.2513366, -4.5660567, 4.3299398, -9.9659395, 9.8173933
8: -7.2384176, 4.4595127, -5.8527985, 3.6751215, -10.9135389, 10.3123112
9: -5.0615201, 5.1911583, -4.1224999, 4.2460814, -9.3076019, 9.3136578

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418329, upper bound: 10.8418276
time: 2.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418329, upper bound: 10.8418288
time: 2.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.9371543, 3.0434709, -6.2038436, 3.1974018, -9.1345558, 9.2473145
1: -3.9161406, 3.7839019, -4.0909982, 3.9502230, -7.8663635, 7.8748999
2: -4.9378753, 3.8432474, -5.1873441, 4.0170794, -8.9549541, 9.0305920
3: -5.2542038, 3.3949192, -5.4998894, 3.5433230, -8.7975273, 8.8948088
4: -5.7800894, 4.6722612, -6.0518875, 4.8833451, -10.6634350, 10.7241488
5: -4.7772279, 3.4792390, -5.0075426, 3.6313958, -8.4086237, 8.4867821
6: -4.4083147, 4.5593653, -4.6255407, 4.7724142, -9.1807289, 9.1849060
7: -4.8753810, 4.5987363, -5.1273813, 4.8111987, -9.6865797, 9.7261181
8: -6.2662902, 3.9093165, -6.5625849, 4.0816412, -10.3479309, 10.4719009
9: -4.3980331, 4.5241385, -4.6062346, 4.7333307, -9.1313639, 9.1303730

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418669, upper bound: 10.8418205
time: 3.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418424, upper bound: 10.8418212
time: 3.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.9371543, 3.0434709, -6.9702139, 3.5557473, -9.4929018, 10.0136852
1: -3.9161406, 3.7839019, -4.6064243, 4.4408340, -8.3569746, 8.3903265
2: -4.9378753, 3.8432474, -5.8985720, 4.4969873, -9.4348621, 9.7418194
3: -5.2542038, 3.3949192, -6.2418804, 3.9674559, -9.2216597, 9.6367998
4: -5.7800894, 4.6722612, -6.8634853, 5.5078368, -11.2879257, 11.5357466
5: -4.7772279, 3.4792390, -5.6606913, 4.0525503, -8.8297787, 9.1399307
6: -4.4083147, 4.5593653, -5.2384191, 5.3718424, -9.7801571, 9.7977848
7: -4.8753810, 4.5987363, -5.8037043, 5.3930817, -10.2684631, 10.4024410
8: -6.2662902, 3.9093165, -7.4305086, 4.5739317, -10.8402214, 11.3398247
9: -4.3980331, 4.5241385, -5.1975718, 5.3307428, -9.7287760, 9.7217102

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418669, upper bound: 10.8418220
time: 3.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418424, upper bound: 10.8418206
time: 4.13 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.7957907, 3.4479465, -6.3192081, 3.2345004, -10.0302906, 9.7671547
1: -4.4929895, 4.3334131, -4.1724901, 4.0289717, -8.5219612, 8.5059032
2: -5.7356710, 4.3816500, -5.2962942, 4.0868263, -9.8224974, 9.6779442
3: -6.0753632, 3.8698959, -5.6185608, 3.6076765, -9.6830397, 9.4884567
4: -6.6879244, 5.3718209, -6.1842194, 4.9827271, -11.6706514, 11.5560398
5: -5.5099669, 3.9505434, -5.1067219, 3.6898530, -9.1998196, 9.0572653
6: -5.0953851, 5.2310023, -4.7134151, 4.8605013, -9.9558868, 9.9444180
7: -5.6359997, 5.2513366, -5.2250595, 4.8973064, -10.5333061, 10.4763966
8: -7.2384176, 4.4595127, -6.6988897, 4.1546659, -11.3930836, 11.1584024
9: -5.0615201, 5.1911583, -4.6980171, 4.8246908, -9.8862114, 9.8891754

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418330, upper bound: 10.8418221
time: 3.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418330, upper bound: 10.8418214
time: 2.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.4863024, 2.8597903, -6.1256390, 3.1139157, -8.6002178, 8.9854298
1: -3.6124439, 3.4929590, -4.0461578, 3.9088831, -7.5213270, 7.5391169
2: -4.5258231, 3.5734527, -5.1138372, 3.9579840, -8.4838066, 8.6872902
3: -4.8188763, 3.1482296, -5.4415450, 3.4990921, -8.3179684, 8.5897751
4: -5.2970591, 4.3041267, -5.9883862, 4.8305826, -10.1276417, 10.2925129
5: -4.3987713, 3.2382784, -4.9379535, 3.5778284, -7.9765997, 8.1762314
6: -4.0519695, 4.2145114, -4.5543671, 4.7039785, -8.7559481, 8.7688789
7: -4.4985981, 4.2694802, -5.0368814, 4.7400365, -9.2386341, 9.3063622
8: -5.7543092, 3.6243405, -6.4840741, 4.0289526, -9.7832623, 10.1084146
9: -4.0570917, 4.1808109, -4.5454125, 4.6713142, -8.7284060, 8.7262230

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418593
time: 4.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418595
time: 3.96 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.4863024, 2.8597903, -6.4239588, 3.2847714, -8.7710743, 9.2837486
1: -3.6124439, 3.4929590, -4.2418203, 4.0949788, -7.7074227, 7.7347794
2: -4.5258231, 3.5734527, -5.3922582, 4.1520615, -8.6778851, 8.9657106
3: -4.8188763, 3.1482296, -5.7168398, 3.6652050, -8.4840813, 8.8650694
4: -5.2970591, 4.3041267, -6.2930646, 5.0666671, -10.3637257, 10.5971909
5: -4.3987713, 3.2382784, -5.1950784, 3.7478294, -8.1466007, 8.4333572
6: -4.0519695, 4.2145114, -4.7975149, 4.9417963, -8.9937658, 9.0120258
7: -4.4985981, 4.2694802, -5.3167782, 4.9761252, -9.4747238, 9.5862579
8: -5.7543092, 3.6243405, -6.8157129, 4.2215929, -9.9759026, 10.4400539
9: -4.0570917, 4.1808109, -4.7776527, 4.9049506, -8.9620419, 8.9584637

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418602
time: 4.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418599
time: 3.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1809220, 3.1845155, -6.0207863, 3.0634670, -9.2443886, 9.2053013
1: -4.0792761, 3.9383545, -3.9765973, 3.8427012, -7.9219770, 7.9149518
2: -5.1708407, 4.0050287, -5.0175257, 3.8928704, -9.0637112, 9.0225544
3: -5.4851189, 3.5320981, -5.3429594, 3.4415696, -8.9266882, 8.8750572
4: -6.0332732, 4.8699932, -5.8794065, 4.7463627, -10.7796364, 10.7493992
5: -4.9924374, 3.6173661, -4.8493214, 3.5197592, -8.5121965, 8.4666872
6: -4.6056709, 4.7573004, -4.4701662, 4.6224046, -9.2280750, 9.2274666
7: -5.1104822, 4.7960491, -4.9448180, 4.6609669, -9.7714491, 9.7408676
8: -6.5434933, 4.0675793, -6.3668699, 3.9619489, -10.5054417, 10.4344492
9: -4.5929127, 4.7196722, -4.4655128, 4.5909796, -9.1838923, 9.1851845

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418603
time: 2.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418594
time: 2.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1809220, 3.1845155, -6.3192081, 3.2345004, -9.4154224, 9.5037231
1: -4.0792761, 3.9383545, -4.1724901, 4.0289717, -8.1082478, 8.1108446
2: -5.1708407, 4.0050287, -5.2962942, 4.0868263, -9.2576675, 9.3013229
3: -5.4851189, 3.5320981, -5.6185608, 3.6076765, -9.0927954, 9.1506586
4: -6.0332732, 4.8699932, -6.1842194, 4.9827271, -11.0160007, 11.0542126
5: -4.9924374, 3.6173661, -5.1067219, 3.6898530, -8.6822901, 8.7240877
6: -4.6056709, 4.7573004, -4.7134151, 4.8605013, -9.4661722, 9.4707155
7: -5.1104822, 4.7960491, -5.2250595, 4.8973064, -10.0077887, 10.0211086
8: -6.5434933, 4.0675793, -6.6988897, 4.1546659, -10.6981592, 10.7664690
9: -4.5929127, 4.7196722, -4.6980171, 4.8246908, -9.4176035, 9.4176893

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418601
time: 2.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418602
time: 2.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.2038436, 3.1974018, -6.1256390, 3.1139157, -9.3177595, 9.3230410
1: -4.0909982, 3.9502230, -4.0461578, 3.9088831, -7.9998813, 7.9963808
2: -5.1873441, 4.0170794, -5.1138372, 3.9579840, -9.1453285, 9.1309166
3: -5.4998894, 3.5433230, -5.4415450, 3.4990921, -8.9989815, 8.9848680
4: -6.0518875, 4.8833451, -5.9883862, 4.8305826, -10.8824701, 10.8717308
5: -5.0075426, 3.6313958, -4.9379535, 3.5778284, -8.5853710, 8.5693493
6: -4.6255407, 4.7724142, -4.5543671, 4.7039785, -9.3295193, 9.3267813
7: -5.1273813, 4.8111987, -5.0368814, 4.7400365, -9.8674183, 9.8480797
8: -6.5625849, 4.0816412, -6.4840741, 4.0289526, -10.5915375, 10.5657158
9: -4.6062346, 4.7333307, -4.5454125, 4.6713142, -9.2775488, 9.2787437

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418602
time: 3.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418590
time: 2.97 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.2038436, 3.1974018, -6.4239588, 3.2847714, -9.4886150, 9.6213608
1: -4.0909982, 3.9502230, -4.2418203, 4.0949788, -8.1859770, 8.1920433
2: -5.1873441, 4.0170794, -5.3922582, 4.1520615, -9.3394051, 9.4093380
3: -5.4998894, 3.5433230, -5.7168398, 3.6652050, -9.1650944, 9.2601624
4: -6.0518875, 4.8833451, -6.2930646, 5.0666671, -11.1185551, 11.1764097
5: -5.0075426, 3.6313958, -5.1950784, 3.7478294, -8.7553720, 8.8264742
6: -4.6255407, 4.7724142, -4.7975149, 4.9417963, -9.5673370, 9.5699291
7: -5.1273813, 4.8111987, -5.3167782, 4.9761252, -10.1035061, 10.1279774
8: -6.5625849, 4.0816412, -6.8157129, 4.2215929, -10.7841778, 10.8973541
9: -4.6062346, 4.7333307, -4.7776527, 4.9049506, -9.5111847, 9.5109835

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418604
time: 3.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418605
time: 3.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.9702139, 3.5557473, -6.0207863, 3.0634670, -10.0336809, 9.5765333
1: -4.6064243, 4.4408340, -3.9765973, 3.8427012, -8.4491253, 8.4174309
2: -5.8985720, 4.4969873, -5.0175257, 3.8928704, -9.7914429, 9.5145130
3: -6.2418804, 3.9674559, -5.3429594, 3.4415696, -9.6834497, 9.3104153
4: -6.8634853, 5.5078368, -5.8794065, 4.7463627, -11.6098480, 11.3872433
5: -5.6606913, 4.0525503, -4.8493214, 3.5197592, -9.1804504, 8.9018717
6: -5.2384191, 5.3718424, -4.4701662, 4.6224046, -9.8608236, 9.8420086
7: -5.8037043, 5.3930817, -4.9448180, 4.6609669, -10.4646711, 10.3379002
8: -7.4305086, 4.5739317, -6.3668699, 3.9619489, -11.3924580, 10.9408016
9: -5.1975718, 5.3307428, -4.4655128, 4.5909796, -9.7885513, 9.7962551

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418605
time: 3.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418604
time: 3.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.9702139, 3.5557473, -6.3192081, 3.2345004, -10.2047138, 9.8749552
1: -4.6064243, 4.4408340, -4.1724901, 4.0289717, -8.6353960, 8.6133242
2: -5.8985720, 4.4969873, -5.2962942, 4.0868263, -9.9853983, 9.7932816
3: -6.2418804, 3.9674559, -5.6185608, 3.6076765, -9.8495569, 9.5860167
4: -6.8634853, 5.5078368, -6.1842194, 4.9827271, -11.8462124, 11.6920567
5: -5.6606913, 4.0525503, -5.1067219, 3.6898530, -9.3505440, 9.1592722
6: -5.2384191, 5.3718424, -4.7134151, 4.8605013, -10.0989208, 10.0852575
7: -5.8037043, 5.3930817, -5.2250595, 4.8973064, -10.7010107, 10.6181412
8: -7.4305086, 4.5739317, -6.6988897, 4.1546659, -11.5851746, 11.2728214
9: -5.1975718, 5.3307428, -4.6980171, 4.8246908, -10.0222626, 10.0287600

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418211, upper bound: 10.8418602
time: 3.52 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418211, upper bound: 10.8418605
time: 3.43 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 8.37 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418523, upper bound: 10.8418575
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418290, upper bound: 10.8418564
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418511, upper bound: 10.8418220
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418298, upper bound: 10.8418218
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418398, upper bound: 10.8418428
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418398, upper bound: 10.8418424
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418389, upper bound: 10.8418221
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418208, upper bound: 10.8418222
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418425, upper bound: 10.8418426
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418425, upper bound: 10.8418427
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418408, upper bound: 10.8418221
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418220, upper bound: 10.8418221
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418403
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418428
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418219
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418137, upper bound: 10.8418220
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418471, upper bound: 10.8418493
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418300, upper bound: 10.8418493
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418386, upper bound: 10.8418260
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418386, upper bound: 10.8418271
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418676, upper bound: 10.8418274
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418426, upper bound: 10.8418284
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418676, upper bound: 10.8418278
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418426, upper bound: 10.8418279
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418329, upper bound: 10.8418276
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418329, upper bound: 10.8418288
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418669, upper bound: 10.8418205
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418424, upper bound: 10.8418212
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418669, upper bound: 10.8418220
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418424, upper bound: 10.8418206
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418330, upper bound: 10.8418221
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418330, upper bound: 10.8418214
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418593
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418595
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418602
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418599
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418603
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418594
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418601
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418602
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418602
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418590
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418604
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418605
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418605
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418604
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418211, upper bound: 10.8418602
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.37
Output dim: 0, lower bound: -10.8418211, upper bound: 10.8418605

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.8119898, 2.5056427, -5.7167516, 2.9301786, -7.7421684, 8.2223940
1: -3.1766400, 3.0753863, -3.7737803, 3.6480432, -6.8246832, 6.8491669
2: -3.9096470, 3.1503749, -4.7392192, 3.7077065, -7.6173534, 7.8895941
3: -4.2028399, 2.7788568, -5.0547991, 3.2750421, -7.4778819, 7.8336558
4: -4.6128912, 3.7783892, -5.5577736, 4.5006485, -9.1135397, 9.3361626
5: -3.8311107, 2.8588872, -4.5944419, 3.3547754, -7.1858864, 7.4533291
6: -3.5079789, 3.6881104, -4.2282467, 4.3888531, -7.8968320, 7.9163570
7: -3.8940656, 3.7525139, -4.6830568, 4.4338446, -8.3279104, 8.4355707
8: -5.0097694, 3.1947494, -6.0262251, 3.7686982, -8.7784672, 9.2209740
9: -3.5477219, 3.6644399, -4.2342577, 4.3588066, -7.9065285, 7.8986979

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418293, upper bound: 10.8418566
time: 3.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418293, upper bound: 10.8418564
time: 3.44 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1288214, 3.1725669, -5.5981703, 2.8709106, -8.9997320, 8.7707367
1: -4.0549479, 3.9193275, -3.6970248, 3.5757775, -7.6307254, 7.6163521
2: -5.1415930, 3.9826441, -4.6322160, 3.6348329, -8.7764263, 8.6148605
3: -5.4594541, 3.5130708, -4.9469690, 3.2111096, -8.6705637, 8.4600401
4: -6.0056953, 4.8388300, -5.4394784, 4.4075794, -10.4132748, 10.2783089
5: -4.9653387, 3.5846860, -4.4956245, 3.2880392, -8.2533779, 8.0803108
6: -4.5647469, 4.7264662, -4.1330147, 4.2965751, -8.8613224, 8.8594809
7: -5.0882754, 4.7717509, -4.5808077, 4.3457508, -9.4340267, 9.3525581
8: -6.5109921, 4.0362101, -5.8963890, 3.6932759, -10.2042675, 9.9325991
9: -4.5725589, 4.6985445, -4.1467609, 4.2699051, -8.8424644, 8.8453054

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418289, upper bound: 10.8418547
time: 3.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418289, upper bound: 10.8418565
time: 4.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.6055975, 2.4014173, -8.2184830, 4.0730910, -8.6786880, 10.6198997
1: -3.0437407, 2.9463153, -5.4570708, 5.2541413, -8.2978821, 8.4033861
2: -3.7217932, 3.0232837, -7.0585070, 5.2668796, -8.9886723, 10.0817909
3: -4.0128393, 2.6658244, -7.4530101, 4.6577430, -8.6705818, 10.1188345
4: -4.4018621, 3.6179276, -8.2153492, 6.5421133, -10.9439754, 11.8332767
5: -3.6578517, 2.7437932, -6.7216654, 4.7207279, -8.3785801, 9.4654589
6: -3.3412282, 3.5284278, -6.2256303, 6.3375168, -9.6787453, 9.7540579
7: -3.7122412, 3.5973830, -6.8799734, 6.3237324, -10.0359735, 10.4773560
8: -4.7816076, 3.0638921, -8.8602180, 5.3695889, -10.1511965, 11.9241104
9: -3.3922806, 3.5077710, -6.1631775, 6.2981653, -9.6904459, 9.6709480

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418505, upper bound: 10.8418222
time: 4.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418505, upper bound: 10.8418219
time: 4.11 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -5.3975844, 2.7638438, -5.7167516, 2.9301786, -8.3277626, 8.4805956
1: -3.5678692, 3.4520073, -3.7737803, 3.6480432, -7.2159123, 7.2257876
2: -4.4511304, 3.5107293, -4.7392192, 3.7077065, -8.1588364, 8.2499485
3: -4.7643251, 3.1011429, -5.0547991, 3.2750421, -8.0393677, 8.1559420
4: -5.2359729, 4.2527771, -5.5577736, 4.5006485, -9.7366219, 9.8105507
5: -4.3271589, 3.1754415, -4.5944419, 3.3547754, -7.6819344, 7.7698832
6: -3.9707050, 4.1420946, -4.2282467, 4.3888531, -8.3595581, 8.3703413
7: -4.4038935, 4.1943283, -4.6830568, 4.4338446, -8.8377380, 8.8773851
8: -5.6761880, 3.5660353, -6.0262251, 3.7686982, -9.4448862, 9.5922604
9: -3.9967065, 4.1177359, -4.2342577, 4.3588066, -8.3555126, 8.3519936

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418207, upper bound: 10.8418625
time: 4.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418210, upper bound: 10.8418547
time: 4.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.9234624, 3.9210975, -5.7167516, 2.9301786, -10.8536415, 9.6378489
1: -5.2687559, 5.0751228, -3.7737803, 3.6480432, -8.9167995, 8.8489037
2: -6.7962418, 5.0802250, -4.7392192, 3.7077065, -10.5039482, 9.8194447
3: -7.1881218, 4.4954014, -5.0547991, 3.2750421, -10.4631634, 9.5502005
4: -7.9187074, 6.3150415, -5.5577736, 4.5006485, -12.4193554, 11.8728151
5: -6.4781137, 4.5547523, -4.5944419, 3.3547754, -9.8328896, 9.1491947
6: -5.9864979, 6.1132488, -4.2282467, 4.3888531, -10.3753510, 10.3414955
7: -6.6252952, 6.1064835, -4.6830568, 4.4338446, -11.0591393, 10.7895403
8: -8.5423794, 5.1813574, -6.0262251, 3.7686982, -12.3110771, 11.2075825
9: -5.9462724, 6.0750666, -4.2342577, 4.3588066, -10.3050785, 10.3093243

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418207, upper bound: 10.8418635
time: 4.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418210, upper bound: 10.8418546
time: 4.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.3181658, 2.7239060, -5.7306690, 2.9371867, -8.2553520, 8.4545746
1: -3.5118306, 3.3989668, -3.7828979, 3.6566932, -7.1685238, 7.1818647
2: -4.3722482, 3.4579000, -4.7519073, 3.7163029, -8.0885506, 8.2098074
3: -4.6827130, 3.0555778, -5.0676908, 3.2826552, -7.9653683, 8.1232681
4: -5.1479559, 4.1841011, -5.5720148, 4.5116544, -9.6596107, 9.7561159
5: -4.2543335, 3.1327319, -4.6062193, 3.3624897, -7.6168232, 7.7389512
6: -3.9083495, 4.0764365, -4.2393541, 4.3996220, -8.3079720, 8.3157902
7: -4.3278456, 4.1298618, -4.6952648, 4.4443378, -8.7721834, 8.8251266
8: -5.5798359, 3.5146651, -6.0416327, 3.7775397, -9.3573761, 9.5562973
9: -3.9307966, 4.0514021, -4.2447462, 4.3694134, -8.3002100, 8.2961483

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418423, upper bound: 10.8418745
time: 3.47 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418419, upper bound: 10.8418745
time: 3.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.8202767, 3.8689373, -5.7306690, 2.9371867, -10.7574635, 9.5996065
1: -5.1967802, 5.0072808, -3.7828979, 3.6566932, -8.8534737, 8.7901783
2: -6.6958375, 5.0124760, -4.7519073, 3.7163029, -10.4121399, 9.7643833
3: -7.0841351, 4.4370327, -5.0676908, 3.2826552, -10.3667908, 9.5047235
4: -7.8065438, 6.2270393, -5.5720148, 4.5116544, -12.3181982, 11.7990541
5: -6.3851147, 4.4982843, -4.6062193, 3.3624897, -9.7476044, 9.1045036
6: -5.9042845, 6.0287142, -4.2393541, 4.3996220, -10.3039064, 10.2680683
7: -6.5287595, 6.0245438, -4.6952648, 4.4443378, -10.9730968, 10.7198086
8: -8.4198666, 5.1146483, -6.0416327, 3.7775397, -12.1974068, 11.1562805
9: -5.8622417, 5.9911718, -4.2447462, 4.3694134, -10.2316551, 10.2359180

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418423, upper bound: 10.8418745
time: 3.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418419, upper bound: 10.8418742
time: 4.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.3932414, 2.7683005, -8.2318306, 4.0797620, -9.4730034, 11.0001316
1: -3.5640817, 3.4494286, -5.4658155, 5.2624440, -8.8265257, 8.9152441
2: -4.4463410, 3.5079784, -7.0706396, 5.2751913, -9.7215328, 10.5786180
3: -4.7586560, 3.0997369, -7.4653816, 4.6650515, -9.4237080, 10.5651188
4: -5.2321830, 4.2465792, -8.2290258, 6.5526838, -11.7848663, 12.4756050
5: -4.3237114, 3.1733730, -6.7329206, 4.7281590, -9.0518703, 9.9062939
6: -3.9689631, 4.1374221, -6.2363405, 6.3478093, -10.3167725, 10.3737621
7: -4.4020429, 4.1918221, -6.8916664, 6.3337545, -10.7357979, 11.0834885
8: -5.6697035, 3.5632863, -8.8749638, 5.3780918, -11.0477953, 12.4382496
9: -3.9937782, 4.1143394, -6.1732531, 6.3083324, -10.3021107, 10.2875919

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418404, upper bound: 10.8418216
time: 4.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418404, upper bound: 10.8418225
time: 5.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1842360, 3.1328716, -5.7306690, 2.9371867, -9.1214228, 8.8635406
1: -4.0927343, 3.9537969, -3.7828979, 3.6566932, -7.7494278, 7.7366948
2: -5.1773438, 3.9980450, -4.7519073, 3.7163029, -8.8936462, 8.7499523
3: -5.5107250, 3.5344450, -5.0676908, 3.2826552, -8.7933807, 8.6021357
4: -6.0642614, 4.8881578, -5.5720148, 4.5116544, -10.5759163, 10.4601727
5: -4.9942179, 3.6065440, -4.6062193, 3.3624897, -8.3567076, 8.2127628
6: -4.5996704, 4.7544198, -4.2393541, 4.3996220, -8.9992924, 8.9937744
7: -5.0951371, 4.7896237, -4.6952648, 4.4443378, -9.5394745, 9.4848881
8: -6.5626469, 4.0679350, -6.0416327, 3.7775397, -10.3401871, 10.1095676
9: -4.5993681, 4.7244434, -4.2447462, 4.3694134, -8.9687815, 8.9691896

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418712
time: 3.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418726
time: 5.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.6198492, 4.2450581, -5.7306690, 2.9371867, -11.5570354, 9.9757271
1: -5.7310905, 5.5170021, -3.7828979, 3.6566932, -9.3877840, 9.2999001
2: -7.4351187, 5.5176396, -4.7519073, 3.7163029, -11.1514215, 10.2695465
3: -7.8446693, 4.8809452, -5.0676908, 3.2826552, -11.1273251, 9.9486361
4: -8.6513939, 6.8749657, -5.5720148, 4.5116544, -13.1630478, 12.4469805
5: -7.0641937, 4.9368281, -4.6062193, 3.3624897, -10.4266834, 9.5430470
6: -6.5445800, 6.6516228, -4.2393541, 4.3996220, -10.9442024, 10.8909769
7: -7.2338796, 6.6292601, -4.6952648, 4.4443378, -11.6782169, 11.3245249
8: -9.3205662, 5.6272001, -6.0416327, 3.7775397, -13.0981064, 11.6688328
9: -6.4769163, 6.6130018, -4.2447462, 4.3694134, -10.8463297, 10.8577480

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418724
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418719
time: 5.31 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 10.10 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418293, upper bound: 10.8418566
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418293, upper bound: 10.8418564
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418289, upper bound: 10.8418547
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418289, upper bound: 10.8418565
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418505, upper bound: 10.8418222
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418505, upper bound: 10.8418219
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418207, upper bound: 10.8418625
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418210, upper bound: 10.8418547
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418207, upper bound: 10.8418635
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418210, upper bound: 10.8418546
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418423, upper bound: 10.8418745
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418419, upper bound: 10.8418745
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418423, upper bound: 10.8418745
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418419, upper bound: 10.8418742
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418404, upper bound: 10.8418216
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418404, upper bound: 10.8418225
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418712
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418726
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418724
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 10.10
Output dim: 0, lower bound: -10.8418338, upper bound: 10.8418719
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418471, upper bound: 10.8418493
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418300, upper bound: 10.8418493
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418676, upper bound: 10.8418274
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418426, upper bound: 10.8418284
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418676, upper bound: 10.8418278
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418426, upper bound: 10.8418279
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418669, upper bound: 10.8418205
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418424, upper bound: 10.8418212
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418669, upper bound: 10.8418220
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418424, upper bound: 10.8418206
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418593
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418595
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418602
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418341, upper bound: 10.8418599
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418603
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418594
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418601
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418284, upper bound: 10.8418602
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418602
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418590
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418604
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418261, upper bound: 10.8418605
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418605
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418604
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418211, upper bound: 10.8418602
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 10.10
Output dim: 0, lower bound: -10.8418211, upper bound: 10.8418605
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=13.22586727142334
rel_dist={0: [-10.841897398404505, 10.841898463369382]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418804, upper bound: 10.8418719
time: 8.18 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418724, upper bound: 10.8418724
time: 2.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.71
Output dim: 0, lower bound: -10.8418804, upper bound: 10.8418719
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.71
Output dim: 0, lower bound: -10.8418724, upper bound: 10.8418724

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.2066984, 3.6700487, -8.2654743, 4.2291927, -11.4358912, 11.9355230
1: -4.7570004, 4.5818253, -5.4564252, 5.2416186, -9.9986191, 10.0382500
2: -6.1038303, 4.6374569, -7.0800390, 5.3101583, -11.4139881, 11.7174940
3: -6.4665151, 4.0927978, -7.5031548, 4.6759410, -11.1424561, 11.5959520
4: -7.0948610, 5.6892929, -8.1781969, 6.5308719, -13.6257324, 13.8674879
5: -5.8535242, 4.1872144, -6.7503586, 4.7941637, -10.6476879, 10.9375715
6: -5.4275541, 5.5503712, -6.2909393, 6.3943529, -11.8219070, 11.8413105
7: -5.9946299, 5.5622005, -6.9444499, 6.3817754, -12.3764048, 12.5066509
8: -7.6838732, 4.7240686, -8.8629284, 5.4135633, -13.0974369, 13.5869970
9: -5.3657575, 5.5053244, -6.1717129, 6.3372817, -11.7030382, 11.6770372

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418736, upper bound: 10.8418724
time: 18.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418671
time: 2.90 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.5377808, 3.8540711, -7.8917203, 4.0462055, -11.5839863, 11.7457914
1: -4.9751611, 4.7891555, -5.2075109, 5.0074821, -9.9826431, 9.9966660
2: -6.4123225, 4.8521576, -6.7368975, 5.0762606, -11.4885826, 11.5890551
3: -6.7883143, 4.2768574, -7.1325645, 4.4709806, -11.2592945, 11.4094219
4: -7.4346066, 5.9521360, -7.7911258, 6.2317262, -13.6663322, 13.7432613
5: -6.1374726, 4.3769474, -6.4355984, 4.5828524, -10.7203255, 10.8125458
6: -5.6966209, 5.8149390, -5.9891667, 6.0992041, -11.7958250, 11.8041058
7: -6.3026853, 5.8260241, -6.6173315, 6.0969286, -12.3996143, 12.4433556
8: -8.0521526, 4.9401875, -8.4449444, 5.1710291, -13.2231817, 13.3851318
9: -5.6227317, 5.7691216, -5.8890786, 6.0447054, -11.6674366, 11.6582003

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418722
time: 3.39 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418683, upper bound: 10.8418673
time: 2.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 7.46 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.46
Output dim: 0, lower bound: -10.8418736, upper bound: 10.8418724
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.46
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418671
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.46
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418722
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.46
Output dim: 0, lower bound: -10.8418683, upper bound: 10.8418673

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6.9777255, 3.5532782, -6.9011164, 3.5674465, -10.5451717, 10.4543943
1: -4.6058979, 4.4390020, -4.5451117, 4.3778467, -8.9837446, 8.9841137
2: -5.8941183, 4.4935989, -5.8228006, 4.4559479, -10.3500662, 10.3163996
3: -6.2418895, 3.9670393, -6.1439199, 3.9238918, -10.1657810, 10.1109591
4: -6.8597612, 5.5069399, -6.7507811, 5.4323735, -12.2921352, 12.2577209
5: -5.6592665, 4.0577669, -5.5932875, 4.0305457, -9.6898117, 9.6510544
6: -5.2435818, 5.3708844, -5.1968293, 5.3232474, -10.5668297, 10.5677137
7: -5.7921696, 5.3869224, -5.7431817, 5.3374381, -11.1296082, 11.1301041
8: -7.4292226, 4.5760522, -7.3304901, 4.5301681, -11.9593906, 11.9065418
9: -5.1917090, 5.3264284, -5.1284347, 5.2640247, -10.4557343, 10.4548626

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418732, upper bound: 10.8418678
time: 3.02 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418682, upper bound: 10.8418680
time: 2.81 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.0150061, 3.5714946, -7.7667141, 3.9758468, -10.9908524, 11.3382092
1: -4.6302238, 4.4620829, -5.1245556, 4.9291363, -9.5593605, 9.5866385
2: -5.9276495, 4.5165300, -6.6203303, 4.9954929, -10.9231424, 11.1368599
3: -6.2780204, 3.9873338, -7.0095873, 4.4011874, -10.6792078, 10.9969215
4: -6.8978109, 5.5362325, -7.6619654, 6.1319499, -13.0297604, 13.1981983
5: -5.6904049, 4.0786901, -6.3261957, 4.5124712, -10.2028761, 10.4048862
6: -5.2732897, 5.3993883, -5.8897223, 6.0004911, -11.2737808, 11.2891102
7: -5.8243456, 5.4149837, -6.5018015, 5.9975519, -11.8218975, 11.9167852
8: -7.4700179, 4.5999770, -8.3046551, 5.0902834, -12.5603008, 12.9046326
9: -5.2195716, 5.3550863, -5.7911863, 5.9447441, -11.1643162, 11.1462727

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418731, upper bound: 10.8418641
time: 3.23 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418687, upper bound: 10.8418648
time: 3.64 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.3020306, 3.7339296, -6.6188145, 3.4283571, -10.7303877, 10.3527441
1: -4.8194814, 4.6419296, -4.3592887, 4.2022748, -9.0217562, 9.0012188
2: -6.1961784, 4.7039423, -5.5652804, 4.2799916, -10.4761696, 10.2692223
3: -6.5566511, 4.1473341, -5.8751669, 3.7697380, -10.3263893, 10.0225010
4: -7.1923323, 5.7642679, -6.4622779, 5.2074680, -12.3998003, 12.2265453
5: -5.9374785, 4.2436442, -5.3562651, 3.8714547, -9.8089333, 9.5999088
6: -5.5071368, 5.6297684, -4.9686265, 5.1019530, -10.6090899, 10.5983944
7: -6.0940905, 5.6453443, -5.4970837, 5.1250710, -11.2191620, 11.1424274
8: -7.7895870, 4.7876863, -7.0172548, 4.3485746, -12.1381617, 11.8049412
9: -5.4434938, 5.5845194, -4.9161062, 5.0474024, -10.4908962, 10.5006256

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418670, upper bound: 10.8418681
time: 2.97 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418642, upper bound: 10.8418680
time: 2.72 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.3429375, 3.7542818, -7.4221644, 3.8068178, -11.1497555, 11.1764460
1: -4.8462296, 4.6673174, -4.8964891, 4.7135806, -9.5598106, 9.5638065
2: -6.2331257, 4.7292542, -6.3049049, 4.7801466, -11.0132723, 11.0341587
3: -6.5964317, 4.1696663, -6.6698198, 4.2126164, -10.8090477, 10.8394861
4: -7.2341413, 5.7964697, -7.3074422, 5.8562675, -13.0904083, 13.1039124
5: -5.9717579, 4.2666855, -6.0362186, 4.3174343, -10.2891922, 10.3029041
6: -5.5398817, 5.6612921, -5.6117687, 5.7287502, -11.2686319, 11.2730608
7: -6.1296492, 5.6763039, -6.2004519, 5.7358699, -11.8655186, 11.8767557
8: -7.8345456, 4.8140001, -7.9201341, 4.8670301, -12.7015762, 12.7341347
9: -5.4741855, 5.6161356, -5.5308046, 5.6758800, -11.1500654, 11.1469402

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418677, upper bound: 10.8418653
time: 2.83 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418653, upper bound: 10.8418652
time: 2.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.82 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 0, lower bound: -10.8418732, upper bound: 10.8418678
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 0, lower bound: -10.8418682, upper bound: 10.8418680
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 0, lower bound: -10.8418731, upper bound: 10.8418641
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 0, lower bound: -10.8418687, upper bound: 10.8418648
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 0, lower bound: -10.8418670, upper bound: 10.8418681
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 0, lower bound: -10.8418642, upper bound: 10.8418680
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 0, lower bound: -10.8418677, upper bound: 10.8418653
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.82
Output dim: 0, lower bound: -10.8418653, upper bound: 10.8418652

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1652794, 3.1619105, -6.6118560, 3.4279923, -9.5932713, 9.7737665
1: -4.0672197, 3.9266562, -4.3533034, 4.1955361, -8.2627563, 8.2799597
2: -5.1489072, 3.9873781, -5.5576477, 4.2757053, -9.4246120, 9.5450258
3: -5.4683018, 3.5205660, -5.8655190, 3.7649522, -9.2332535, 9.3860855
4: -6.0146246, 4.8552737, -6.4499712, 5.2005186, -11.2151432, 11.3052444
5: -4.9725623, 3.6071160, -5.3488464, 3.8697526, -8.8423147, 8.9559622
6: -4.5922856, 4.7393475, -4.9651403, 5.0982943, -9.6905804, 9.7044878
7: -5.0797729, 4.7729788, -5.4897180, 5.1182928, -10.1980658, 10.2626972
8: -6.5220156, 4.0555444, -7.0077209, 4.3444347, -10.8664503, 11.0632648
9: -4.5729876, 4.7002287, -4.9082651, 5.0402012, -9.6131887, 9.6084938

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418366, upper bound: 10.8418489
time: 3.73 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418344, upper bound: 10.8418286
time: 3.47 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.0283189, 3.5690274, -6.5568705, 3.4019430, -10.4302616, 10.1258984
1: -4.6472692, 4.4788322, -4.3169756, 4.1609144, -8.8081837, 8.7958078
2: -5.9505787, 4.5291824, -5.5073633, 4.2415447, -10.1921234, 10.0365458
3: -6.3051481, 3.9982028, -5.8141904, 3.7348387, -10.0399866, 9.8123932
4: -6.9272280, 5.5581594, -6.3929930, 5.1565371, -12.0837650, 11.9511528
5: -5.7087307, 4.0825124, -5.3027873, 3.8392267, -9.5479574, 9.3852997
6: -5.2829323, 5.4151697, -4.9207540, 5.0555282, -10.3384609, 10.3359241
7: -5.8441124, 5.4308600, -5.4418030, 5.0770006, -10.9211130, 10.8726635
8: -7.4987001, 4.6105032, -6.9465814, 4.3092504, -11.8079510, 11.5570850
9: -5.2396002, 5.3743391, -4.8666658, 4.9982367, -10.2378368, 10.2410049

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418321, upper bound: 10.8418491
time: 4.12 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418296, upper bound: 10.8418279
time: 3.71 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.2005606, 3.1790471, -7.4613810, 3.8280387, -10.0285988, 10.6404285
1: -4.0901971, 3.9485140, -4.9216037, 4.7363853, -8.8265820, 8.8701172
2: -5.1806660, 4.0089693, -6.3402653, 4.8049469, -9.9856129, 10.3492346
3: -5.5007687, 3.5397339, -6.7067795, 4.2333560, -9.7341251, 10.2465134
4: -6.0505972, 4.8830180, -7.3440022, 5.8869529, -11.9375496, 12.2270203
5: -5.0020514, 3.6266742, -6.0679975, 4.3415084, -9.3435593, 9.6946716
6: -4.6204200, 4.7661915, -5.6451769, 5.7617502, -10.3821697, 10.4113684
7: -5.1102347, 4.7992320, -6.2341337, 5.7646017, -10.8748360, 11.0333652
8: -6.5606589, 4.0779004, -7.9630475, 4.8929620, -11.4536209, 12.0409479
9: -4.5993633, 4.7268004, -5.5585656, 5.7055268, -10.3048897, 10.2853661

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418368, upper bound: 10.8418467
time: 3.21 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418347, upper bound: 10.8418213
time: 2.93 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.0613823, 3.5851438, -7.3761015, 3.7875609, -10.8489437, 10.9612455
1: -4.6687536, 4.4992332, -4.8646731, 4.6822567, -9.3510103, 9.3639069
2: -5.9801440, 4.5494270, -6.2617493, 4.7516861, -10.7318306, 10.8111763
3: -6.3370590, 4.0161514, -6.6218405, 4.1864524, -10.5235119, 10.6379919
4: -6.9608550, 5.5840082, -7.2547765, 5.8181906, -12.7790451, 12.8387852
5: -5.7362456, 4.1010609, -5.9961047, 4.2936978, -10.0299435, 10.0971661
6: -5.3092699, 5.4403067, -5.5764952, 5.6946783, -11.0039482, 11.0168018
7: -5.8724613, 5.4556513, -6.1594152, 5.6994820, -11.5719433, 11.6150665
8: -7.5347071, 4.6316719, -7.8673129, 4.8376517, -12.3723583, 12.4989853
9: -5.2641821, 5.3996315, -5.4935255, 5.6385741, -10.9027557, 10.8931570

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418315, upper bound: 10.8418471
time: 3.71 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418297, upper bound: 10.8418220
time: 3.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.4445133, 3.3205233, -6.3288841, 3.2889769, -9.7334900, 9.6494074
1: -4.2501707, 4.1008053, -4.1673012, 4.0195570, -8.2697277, 8.2681065
2: -5.4094567, 4.1691542, -5.2994776, 4.0993528, -9.5088100, 9.4686317
3: -5.7253699, 3.6759167, -5.6029453, 3.6104195, -9.3357897, 9.2788620
4: -6.2995872, 5.0762348, -6.1608515, 4.9750710, -11.2746582, 11.2370863
5: -5.2128611, 3.7659869, -5.1113024, 3.7112341, -8.9240952, 8.8772888
6: -4.8194323, 4.9615431, -4.7363129, 4.8769994, -9.6964321, 9.6978559
7: -5.3421268, 4.9945097, -5.2429266, 4.9066148, -10.2487411, 10.2374363
8: -6.8317528, 4.2358842, -6.6937766, 4.1633959, -10.9951487, 10.9296608
9: -4.7903123, 4.9190459, -4.6954179, 4.8249965, -9.6153088, 9.6144638

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418277, upper bound: 10.8418485
time: 4.45 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418263, upper bound: 10.8418278
time: 3.15 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2083406, 3.6783323, -6.2811089, 3.2663062, -10.4746466, 9.9594412
1: -4.7644510, 4.5899124, -4.1357970, 3.9895511, -8.7540016, 8.7257099
2: -6.1184316, 4.6477852, -5.2558565, 4.0697498, -10.1881809, 9.9036417
3: -6.4767394, 4.0987444, -5.5584908, 3.5843179, -10.0610571, 9.6572351
4: -7.1088781, 5.6987944, -6.1115179, 4.9369411, -12.0458193, 11.8103123
5: -5.8639712, 4.1873536, -5.0713763, 3.6846538, -9.5486250, 9.2587299
6: -5.4302273, 5.5600772, -4.6977186, 4.8398442, -10.2700710, 10.2577953
7: -6.0162325, 5.5765581, -5.2013655, 4.8707814, -10.8870144, 10.7779236
8: -7.6970716, 4.7283468, -6.6407728, 4.1328297, -11.8299007, 11.3691196
9: -5.3797808, 5.5180383, -4.6593752, 4.7886438, -10.1684246, 10.1774139

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418220, upper bound: 10.8418480
time: 3.15 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418217, upper bound: 10.8418285
time: 2.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.4821596, 3.3391209, -7.1146932, 3.6580644, -10.1402245, 10.4538136
1: -4.2746840, 4.1241264, -4.6921148, 4.5195217, -8.7942057, 8.8162413
2: -5.4434314, 4.1924028, -6.0228944, 4.5883226, -10.0317535, 10.2152977
3: -5.7599845, 3.6964612, -6.3650756, 4.0436096, -9.8035946, 10.0615368
4: -6.3379922, 5.1058311, -6.9872847, 5.6095686, -11.9475613, 12.0931158
5: -5.2443814, 3.7869134, -5.7762566, 4.1453085, -9.3896904, 9.5631695
6: -4.8495774, 4.9903264, -5.3654852, 5.4884791, -10.3380566, 10.3558121
7: -5.3748317, 5.0226421, -5.9309483, 5.5013161, -10.8761482, 10.9535904
8: -6.8730145, 4.2598090, -7.5763464, 4.6683130, -11.5413275, 11.8361549
9: -4.8185225, 4.9475245, -5.2965708, 5.4351549, -10.2536774, 10.2440948

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418269, upper bound: 10.8418455
time: 2.91 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418268, upper bound: 10.8418220
time: 3.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.2439957, 3.6961675, -7.0538054, 3.6291621, -10.8731575, 10.7499733
1: -4.7876387, 4.6119218, -4.6516252, 4.4810467, -9.2686853, 9.2635469
2: -6.1504531, 4.6697693, -5.9670463, 4.5504208, -10.7008743, 10.6368160
3: -6.5112033, 4.1181593, -6.3048487, 4.0102024, -10.5214062, 10.4230080
4: -7.1451097, 5.7266884, -6.9239039, 5.5606995, -12.7058086, 12.6505928
5: -5.8937550, 4.2074552, -5.7251711, 4.1110954, -10.0048504, 9.9326267
6: -5.4587474, 5.5874314, -5.3163362, 5.4407516, -10.8994989, 10.9037676
7: -6.0470786, 5.6034660, -5.8777981, 5.4549556, -11.5020342, 11.4812641
8: -7.7360382, 4.7512364, -7.5084162, 4.6288252, -12.3648634, 12.2596531
9: -5.4063911, 5.5454550, -5.2503719, 5.3876452, -10.7940369, 10.7958269

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418217, upper bound: 10.8418451
time: 2.83 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418207, upper bound: 10.8418223
time: 3.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418366, upper bound: 10.8418489
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418344, upper bound: 10.8418286
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418321, upper bound: 10.8418491
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418296, upper bound: 10.8418279
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418368, upper bound: 10.8418467
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418347, upper bound: 10.8418213
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418315, upper bound: 10.8418471
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418297, upper bound: 10.8418220
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418277, upper bound: 10.8418485
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418263, upper bound: 10.8418278
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418220, upper bound: 10.8418480
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418217, upper bound: 10.8418285
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418269, upper bound: 10.8418455
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418268, upper bound: 10.8418220
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418217, upper bound: 10.8418451
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 7.43
Output dim: 0, lower bound: -10.8418207, upper bound: 10.8418223

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.0989132, 3.1277769, -5.9895816, 3.1116281, -9.2105408, 9.1173582
1: -4.0237041, 3.8853922, -3.9449613, 3.8079915, -7.8316956, 7.8303537
2: -5.0882258, 3.9458899, -4.9884834, 3.8861873, -8.9744129, 8.9343739
3: -5.4068956, 3.4842343, -5.2893457, 3.4239907, -8.8308868, 8.7735806
4: -5.9469185, 4.8026886, -5.8135910, 4.7068911, -10.6538095, 10.6162796
5: -4.9165010, 3.5697932, -4.8236094, 3.5208354, -8.4373360, 8.3934021
6: -4.5385537, 4.6875086, -4.4621916, 4.6124725, -9.1510258, 9.1497002
7: -5.0210376, 4.7227640, -4.9395480, 4.6469874, -9.6680250, 9.6623116
8: -6.4485369, 4.0130792, -6.3186827, 3.9462533, -10.3947906, 10.3317623
9: -4.5227337, 4.6496305, -4.4367247, 4.5650373, -9.0877705, 9.0863552

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418256, upper bound: 10.8418314
time: 3.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418178, upper bound: 10.8418302
time: 3.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.9628439, 3.5351710, -5.9345942, 3.0855651, -10.0484085, 9.4697647
1: -4.6042871, 4.4381762, -3.9085627, 3.7733238, -8.3776112, 8.3467388
2: -5.8907838, 4.4880710, -4.9380908, 3.8521423, -9.7429256, 9.4261618
3: -6.2414293, 3.9622941, -5.2379327, 3.3939281, -9.6353569, 9.2002268
4: -6.8604145, 5.5063744, -5.7566228, 4.6628199, -11.5232344, 11.2629967
5: -5.6535277, 4.0452271, -4.7775025, 3.4902685, -9.1437960, 8.8227291
6: -5.2299109, 5.3638144, -4.4177756, 4.5696077, -9.7995186, 9.7815895
7: -5.7862120, 5.3807817, -4.8915563, 4.6056495, -10.3918610, 10.2723379
8: -7.4263458, 4.5680728, -6.2574282, 3.9110532, -11.3373985, 10.8255005
9: -5.1901035, 5.3234148, -4.3950682, 4.5231104, -9.7132139, 9.7184830

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418214, upper bound: 10.8418322
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418126, upper bound: 10.8418293
time: 3.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1344142, 3.1450257, -6.8407516, 3.5103772, -9.6447916, 9.9857769
1: -4.0468254, 3.9073887, -4.5134401, 4.3499351, -8.3967609, 8.4208288
2: -5.1201859, 3.9676206, -5.7731113, 4.4154134, -9.5355988, 9.7407322
3: -5.4395690, 3.5035217, -6.1008348, 3.8929915, -9.3325605, 9.6043568
4: -5.9831161, 4.8306108, -6.7088995, 5.3950539, -11.3781700, 11.5395107
5: -4.9461780, 3.5894766, -5.5445642, 3.9889359, -8.9351139, 9.1340408
6: -4.5668654, 4.7145238, -5.1434383, 5.2742972, -9.8411627, 9.8579617
7: -5.0516977, 4.7491837, -5.6859832, 5.2893014, -10.3409996, 10.4351673
8: -6.4874263, 4.0355783, -7.2756629, 4.4909687, -10.9783955, 11.3112411
9: -4.5492749, 4.6763735, -5.0887318, 5.2219353, -9.7712097, 9.7651052

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418255, upper bound: 10.8418296
time: 3.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418186, upper bound: 10.8418284
time: 3.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.9960203, 3.5513492, -6.7558722, 3.4702413, -10.4662619, 10.3072214
1: -4.6258435, 4.4586520, -4.4568119, 4.2961063, -8.9219494, 8.9154644
2: -5.9204626, 4.5083890, -5.6949749, 4.3624496, -10.2829123, 10.2033634
3: -6.2734509, 3.9803095, -6.0170932, 3.8463151, -10.1197662, 9.9974022
4: -6.8941641, 5.5323143, -6.6201491, 5.3266239, -12.2207880, 12.1524639
5: -5.6811409, 4.0638380, -5.4730649, 3.9414525, -9.6225929, 9.5369034
6: -5.2563334, 5.3890491, -5.0750380, 5.2077055, -10.4640388, 10.4640865
7: -5.8146648, 5.4056692, -5.6116276, 5.2246475, -11.0393124, 11.0172968
8: -7.4624805, 4.5893173, -7.1805696, 4.4359884, -11.8984690, 11.7698870
9: -5.2147765, 5.3487997, -5.0240321, 5.1556244, -10.3704014, 10.3728313

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418288
time: 3.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418115, upper bound: 10.8418274
time: 3.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.3740816, 3.2847626, -5.7174897, 2.9776816, -9.3517628, 9.0022526
1: -4.2040219, 4.0569916, -3.7660198, 3.6386306, -7.8426523, 7.8230114
2: -5.3451462, 4.1250248, -4.7398434, 3.7177429, -9.0628891, 8.8648682
3: -5.6602535, 3.6372981, -5.0368299, 3.2756283, -8.9358816, 8.6741276
4: -6.2275810, 5.0204401, -5.5357747, 4.4897361, -10.7173176, 10.5562153
5: -5.1535244, 3.7264853, -4.5948248, 3.3687246, -8.5222492, 8.3213100
6: -4.7624483, 4.9066620, -4.2423582, 4.3992071, -9.1616554, 9.1490202
7: -5.2799616, 4.9413166, -4.7020006, 4.4431953, -9.7231569, 9.6433172
8: -6.7538719, 4.1908064, -6.0162468, 3.7725351, -10.5264072, 10.2070532
9: -4.7370615, 4.8652844, -4.2321348, 4.3582897, -9.0953512, 9.0974197

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418191, upper bound: 10.8418304
time: 3.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418089, upper bound: 10.8418287
time: 4.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1416874, 3.6440878, -5.6696453, 2.9549327, -10.0966206, 9.3137331
1: -4.7206659, 4.5484724, -3.7343893, 3.6084995, -8.3291655, 8.2828617
2: -6.0575724, 4.6059685, -4.6959600, 3.6882648, -9.7458372, 9.3019285
3: -6.4117670, 4.0621977, -4.9921923, 3.2494740, -9.6612415, 9.0543900
4: -7.0407763, 5.6460314, -5.4862118, 4.4515009, -11.4922771, 11.1322432
5: -5.8077908, 4.1494493, -4.5546870, 3.3422322, -9.1500225, 8.7041359
6: -5.3762980, 5.5077896, -4.2038393, 4.3618364, -9.7381344, 9.7116289
7: -5.9573741, 5.5255880, -4.6603117, 4.4071875, -10.3645611, 10.1858997
8: -7.6233416, 4.6851683, -5.9629078, 3.7419760, -11.3653173, 10.6480761
9: -5.3294029, 5.4661589, -4.1960306, 4.3217897, -9.6511927, 9.6621895

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418165, upper bound: 10.8418301
time: 3.56 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418028, upper bound: 10.8418276
time: 2.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.4118280, 3.3034139, -6.4987297, 3.3443837, -9.7562122, 9.8021431
1: -4.2285986, 4.0803747, -4.2876053, 4.1361656, -8.3647642, 8.3679800
2: -5.3792129, 4.1483374, -5.4601512, 4.2021360, -9.5813484, 9.6084881
3: -5.6949635, 3.6578963, -5.7780628, 3.7058547, -9.4008179, 9.4359589
4: -6.2660894, 5.0501146, -6.3571577, 5.1214061, -11.3874950, 11.4072723
5: -5.1851273, 3.7474661, -5.2569494, 3.7975755, -8.9827023, 9.0044155
6: -4.7926774, 4.9355211, -4.8676310, 5.0067186, -9.7993965, 9.8031521
7: -5.3127542, 4.9695272, -5.3871670, 5.0324488, -10.3452034, 10.3566942
8: -6.7952452, 4.2147932, -6.8949347, 4.2714715, -11.0667171, 11.1097279
9: -4.7653456, 4.8938413, -4.8304005, 4.9597740, -9.7251196, 9.7242413

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418192, upper bound: 10.8418275
time: 3.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418084, upper bound: 10.8418266
time: 2.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1775575, 3.6620021, -6.4387774, 3.3161757, -10.4937334, 10.1007795
1: -4.7439966, 4.5706038, -4.2478628, 4.0982800, -8.8422766, 8.8184662
2: -6.0897913, 4.6280756, -5.4051590, 4.1648550, -10.2546463, 10.0332346
3: -6.4463844, 4.0817304, -5.7218781, 3.6729701, -10.1193542, 9.8036079
4: -7.0772209, 5.6740980, -6.2947655, 5.0732803, -12.1505013, 11.9688635
5: -5.8377619, 4.1696544, -5.2066860, 3.7643170, -9.6020794, 9.3763409
6: -5.4049869, 5.5352650, -4.8191948, 4.9599905, -10.3649769, 10.3544598
7: -5.9884014, 5.5526562, -5.3348465, 4.9874077, -10.9758091, 10.8875027
8: -7.6624937, 4.7081938, -6.8280501, 4.2330618, -11.8955555, 11.5362434
9: -5.3561754, 5.4936948, -4.7849360, 4.9139047, -10.2700806, 10.2786312

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418164, upper bound: 10.8418290
time: 4.20 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418033, upper bound: 10.8418260
time: 2.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 8.50 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418256, upper bound: 10.8418314
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418178, upper bound: 10.8418302
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418214, upper bound: 10.8418322
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418126, upper bound: 10.8418293
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418255, upper bound: 10.8418296
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418186, upper bound: 10.8418284
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418212, upper bound: 10.8418288
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418115, upper bound: 10.8418274
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418191, upper bound: 10.8418304
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418089, upper bound: 10.8418287
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418165, upper bound: 10.8418301
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418028, upper bound: 10.8418276
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418192, upper bound: 10.8418275
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418084, upper bound: 10.8418266
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418164, upper bound: 10.8418290
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.50
Output dim: 0, lower bound: -10.8418033, upper bound: 10.8418260
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=13.22586727142334
rel_dist={0: [-10.841896644575556, 10.841898312561103]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418804, upper bound: 10.8418731
time: 3.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418721, upper bound: 10.8418721
time: 3.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.83
Output dim: 0, lower bound: -10.8418804, upper bound: 10.8418731
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.83
Output dim: 0, lower bound: -10.8418721, upper bound: 10.8418721

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -7.2066984, 3.6700487, -8.5349045, 4.3701630, -11.5768614, 12.2049532
1: -4.7570004, 4.5818253, -5.6356192, 5.4101305, -10.1671314, 10.2174435
2: -6.1038303, 4.6374569, -7.3297429, 5.4817209, -11.5855513, 11.9671974
3: -6.4665151, 4.0927978, -7.7687054, 4.8246388, -11.2911539, 11.8615026
4: -7.0948610, 5.6892929, -8.4562721, 6.7459025, -13.8407631, 14.1455650
5: -5.8535242, 4.1872144, -6.9787960, 4.9480581, -10.8015823, 11.1660099
6: -5.4275541, 5.5503712, -6.5105915, 6.6101217, -12.0376759, 12.0609627
7: -5.9946299, 5.5622005, -7.1863894, 6.5914335, -12.5860634, 12.7485905
8: -7.6838732, 4.7240686, -9.1637259, 5.5900631, -13.2739363, 13.8877945
9: -5.3657575, 5.5053244, -6.3772860, 6.5497446, -11.9155025, 11.8826103

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418752, upper bound: 10.8418719
time: 3.02 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418743, upper bound: 10.8418678
time: 3.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -7.5377808, 3.8540711, -8.0954323, 4.1494641, -11.6872444, 11.9495029
1: -4.9751611, 4.7891555, -5.3426514, 5.1347938, -10.1099548, 10.1318054
2: -6.4123225, 4.8521576, -6.9242058, 5.2047687, -11.6170912, 11.7763634
3: -6.7883143, 4.2768574, -7.3331814, 4.5829711, -11.3712854, 11.6100378
4: -7.4346066, 5.9521360, -8.0007162, 6.3942552, -13.8288612, 13.9528522
5: -6.1374726, 4.3769474, -6.6075306, 4.6987925, -10.8362656, 10.9844780
6: -5.6966209, 5.8149390, -6.1544223, 6.2609434, -11.9575644, 11.9693613
7: -6.3026853, 5.8260241, -6.7979555, 6.2533035, -12.5559864, 12.6239796
8: -8.0521526, 4.9401875, -8.6723337, 5.3032675, -13.3554201, 13.6125212
9: -5.6227317, 5.7691216, -6.0436821, 6.2045507, -11.8272820, 11.8128033

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418680, upper bound: 10.8418720
time: 3.18 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418681, upper bound: 10.8418673
time: 3.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 7.72 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.72
Output dim: 0, lower bound: -10.8418752, upper bound: 10.8418719
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.72
Output dim: 0, lower bound: -10.8418743, upper bound: 10.8418678
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.72
Output dim: 0, lower bound: -10.8418680, upper bound: 10.8418720
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.72
Output dim: 0, lower bound: -10.8418681, upper bound: 10.8418673

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -7.0482078, 3.5891795, -7.1526136, 3.6993985, -10.7476063, 10.7417927
1: -4.6524177, 4.4829636, -4.7112131, 4.5347915, -9.1872091, 9.1941757
2: -5.9586797, 4.5378766, -6.0550566, 4.6157227, -10.5744019, 10.5929337
3: -6.3110027, 4.0057507, -6.3896971, 4.0623779, -10.3733807, 10.3954449
4: -6.9321423, 5.5630813, -7.0082483, 5.6326303, -12.5647726, 12.5713291
5: -5.7190671, 4.0975986, -5.8063555, 4.1744561, -9.8935232, 9.9039536
6: -5.3002081, 5.4261036, -5.4018378, 5.5238075, -10.8240156, 10.8279419
7: -5.8544893, 5.4408731, -5.9689431, 5.5325909, -11.3870783, 11.4098167
8: -7.5075788, 4.6216164, -7.6103020, 4.6941361, -12.2017145, 12.2319183
9: -5.2452936, 5.3814626, -5.3202333, 5.4614096, -10.7067032, 10.7016964

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418752, upper bound: 10.8418675
time: 2.96 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418680, upper bound: 10.8418674
time: 2.65 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -7.0763016, 3.6029696, -8.0358095, 4.1165428, -11.1928444, 11.6387787
1: -4.6707716, 4.5003705, -5.3030930, 5.0974007, -9.7681723, 9.8034620
2: -5.9839954, 4.5551939, -6.8692713, 5.1663122, -11.1503077, 11.4244652
3: -6.3382664, 4.0210600, -7.2739520, 4.5495768, -10.8878431, 11.2950115
4: -6.9608359, 5.5851874, -7.9389243, 6.3464718, -13.3073082, 13.5241089
5: -5.7425704, 4.1133747, -6.5544081, 4.6661735, -10.4087439, 10.6677828
6: -5.3226137, 5.4476357, -6.1088610, 6.2151551, -11.5377693, 11.5564966
7: -5.8787985, 5.4620533, -6.7429399, 6.2066107, -12.0854073, 12.2049932
8: -7.5383739, 4.6396580, -8.6051102, 5.2657857, -12.8041592, 13.2447681
9: -5.2663260, 5.4030943, -5.9964318, 6.1565762, -11.4229021, 11.3995266

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418750, upper bound: 10.8418654
time: 3.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418682, upper bound: 10.8418646
time: 3.09 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.3747506, 3.7709699, -6.7904673, 3.5161915, -10.8909416, 10.5614376
1: -4.8675137, 4.6873589, -4.4720297, 4.3089790, -9.1764927, 9.1593885
2: -6.2628570, 4.7496643, -5.7227211, 4.3880882, -10.6509457, 10.4723854
3: -6.6281357, 4.1872921, -6.0348735, 3.8638647, -10.4920006, 10.2221661
4: -7.2670908, 5.8222351, -6.6372471, 5.3438926, -12.6109829, 12.4594822
5: -5.9991770, 4.2847595, -5.5009618, 3.9683986, -9.9675751, 9.7857208
6: -5.5655894, 5.6868916, -5.1082702, 5.2373018, -10.8028908, 10.7951622
7: -6.1584368, 5.7010818, -5.6492634, 5.2552958, -11.4137325, 11.3503456
8: -7.8706007, 4.8347273, -7.2076182, 4.4589367, -12.3295374, 12.0423450
9: -5.4987974, 5.6414690, -5.0459771, 5.1789618, -10.6777592, 10.6874466

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418681, upper bound: 10.8418670
time: 2.92 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418644, upper bound: 10.8418673
time: 2.86 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.4052796, 3.7861943, -7.6193857, 3.9073603, -11.3126402, 11.4055805
1: -4.8874884, 4.7063112, -5.0268307, 4.8368073, -9.7242956, 9.7331419
2: -6.2904625, 4.7685795, -6.4862208, 4.9045377, -11.1949997, 11.2548008
3: -6.6578455, 4.2039661, -6.8634787, 4.3209209, -10.9787664, 11.0674448
4: -7.2982979, 5.8462877, -7.5096731, 6.0136423, -13.3119402, 13.3559608
5: -6.0247860, 4.3019595, -6.2027726, 4.4298282, -10.4546146, 10.5047321
6: -5.5900311, 5.7104497, -5.7718329, 5.8852506, -11.4752817, 11.4822826
7: -6.1850114, 5.7242079, -6.3754172, 5.8872871, -12.0722980, 12.0996246
8: -7.9041839, 4.8543730, -8.1399727, 4.9952102, -12.8993940, 12.9943457
9: -5.5217228, 5.6650887, -5.6805253, 5.8303633, -11.3520861, 11.3456135

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418652
time: 2.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418654, upper bound: 10.8418648
time: 2.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.77 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.77
Output dim: 0, lower bound: -10.8418752, upper bound: 10.8418675
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.77
Output dim: 0, lower bound: -10.8418680, upper bound: 10.8418674
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.77
Output dim: 0, lower bound: -10.8418750, upper bound: 10.8418654
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.77
Output dim: 0, lower bound: -10.8418682, upper bound: 10.8418646
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.77
Output dim: 0, lower bound: -10.8418681, upper bound: 10.8418670
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.77
Output dim: 0, lower bound: -10.8418644, upper bound: 10.8418673
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.77
Output dim: 0, lower bound: -10.8418678, upper bound: 10.8418652
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.77
Output dim: 0, lower bound: -10.8418654, upper bound: 10.8418648

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.2318430, 3.1958442, -6.9802260, 3.6158538, -9.8476963, 10.1760702
1: -4.1110868, 3.9681840, -4.5966554, 4.4261250, -8.5372124, 8.5648394
2: -5.2100096, 4.0290642, -5.8968992, 4.5082531, -9.7182627, 9.9259634
3: -5.5304122, 3.5570796, -6.2191057, 3.9676528, -9.4980650, 9.7761850
4: -6.0828762, 4.9083662, -6.8288155, 5.4944530, -11.5773296, 11.7371817
5: -5.0291634, 3.6443250, -5.6606302, 4.0779667, -9.1071301, 9.3049555
6: -4.6458025, 4.7913389, -5.2637272, 5.3892660, -10.0350685, 10.0550661
7: -5.1387906, 4.8234520, -5.8179140, 5.4009385, -10.5397291, 10.6413660
8: -6.5961366, 4.0980749, -7.4178572, 4.5826426, -11.1787796, 11.5159321
9: -4.6236601, 4.7512460, -5.1890244, 5.3266129, -9.9502735, 9.9402704

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418747, upper bound: 10.8418672
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418747, upper bound: 10.8418674
time: 2.93 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.0951748, 3.6032536, -6.9053636, 3.5798900, -10.6750650, 10.5086174
1: -4.6913695, 4.5204868, -4.5469885, 4.3789749, -9.0703449, 9.0674753
2: -6.0118208, 4.5712094, -5.8283482, 4.4616508, -10.4734716, 10.3995571
3: -6.3706446, 4.0349288, -6.1452303, 3.9265876, -10.2972317, 10.1801586
4: -6.9957952, 5.6113734, -6.7510996, 5.4345174, -12.4303131, 12.3624725
5: -5.7654653, 4.1203375, -5.5976515, 4.0359602, -9.8014259, 9.7179890
6: -5.3366876, 5.4675450, -5.2035379, 5.3307667, -10.6674538, 10.6710835
7: -5.9032860, 5.4820480, -5.7525158, 5.3439169, -11.2472029, 11.2345638
8: -7.5729795, 4.6537323, -7.3344760, 4.5341959, -12.1071758, 11.9882088
9: -5.2904038, 5.4265509, -5.1322250, 5.2682362, -10.5586395, 10.5587759

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418677
time: 3.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418671
time: 3.08 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.2584066, 3.2088075, -7.8542786, 4.0285683, -10.2869749, 11.0630856
1: -4.1284013, 3.9846478, -5.1820045, 4.9828978, -9.1112995, 9.1666517
2: -5.2339592, 4.0453591, -6.7025590, 5.0530529, -10.2870121, 10.7479172
3: -5.5548840, 3.5715253, -7.0937338, 4.4496541, -10.0045376, 10.6652594
4: -6.1099710, 4.9292769, -7.7493763, 6.2009358, -12.3109074, 12.6786528
5: -5.0514002, 3.6590633, -6.4009342, 4.5645719, -9.6159725, 10.0599976
6: -4.6669979, 4.8115931, -5.9634390, 6.0731020, -10.7400999, 10.7750320
7: -5.1617799, 4.8432493, -6.5838766, 6.0678349, -11.2296133, 11.4271259
8: -6.6252704, 4.1149201, -8.4018555, 5.1484108, -11.7736797, 12.5167751
9: -4.6435494, 4.7712860, -5.8582439, 6.0142202, -10.6577692, 10.6295300

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418411, upper bound: 10.8418532
time: 4.48 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418368, upper bound: 10.8418216
time: 3.26 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -7.1197405, 3.6152925, -7.7555695, 3.9812253, -11.1009655, 11.3708620
1: -4.7073483, 4.5356526, -5.1160793, 4.9203691, -9.6277180, 9.6517315
2: -6.0338335, 4.5862885, -6.6116071, 4.9914312, -11.0252647, 11.1978951
3: -6.3943777, 4.0482769, -6.9954882, 4.3953505, -10.7897282, 11.0437651
4: -7.0207887, 5.6306038, -7.6460381, 6.1214857, -13.1422749, 13.2766418
5: -5.7859416, 4.1341324, -6.3175445, 4.5092812, -10.2952232, 10.4516773
6: -5.3562698, 5.4862704, -5.8840771, 5.9956174, -11.3518867, 11.3703480
7: -5.9244084, 5.5005026, -6.4973826, 5.9922094, -11.9166183, 11.9978848
8: -7.5997748, 4.6694732, -8.2911348, 5.0843453, -12.6841183, 12.9606075
9: -5.3087034, 5.4453731, -5.7829800, 5.9367437, -11.2454462, 11.2283535

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418318, upper bound: 10.8418531
time: 2.90 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8418216
time: 3.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.5152731, 3.3563650, -6.6183257, 3.4333708, -9.9486437, 9.9746904
1: -4.2967315, 4.1449304, -4.3580050, 4.2004986, -8.4972305, 8.5029354
2: -5.4743061, 4.2135954, -5.5648890, 4.2808270, -9.7551327, 9.7784843
3: -5.7912712, 3.7147920, -5.8722744, 3.7692692, -9.5605402, 9.5870667
4: -6.3722048, 5.1325979, -6.4582906, 5.2059188, -11.5781231, 11.5908890
5: -5.2728782, 3.8054924, -5.3555231, 3.8731217, -9.1459999, 9.1610155
6: -4.8763342, 5.0167103, -4.9703264, 5.1036415, -9.9799757, 9.9870367
7: -5.4047556, 5.0480890, -5.4983635, 5.1254010, -10.5301571, 10.5464525
8: -6.9103956, 4.2811203, -7.0155592, 4.3488326, -11.2592278, 11.2966795
9: -4.8440905, 4.9733267, -4.9149618, 5.0466237, -9.8907146, 9.8882885

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418260, upper bound: 10.8418542
time: 2.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418287
time: 3.02 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -7.2771916, 3.7135780, -6.5465846, 3.3991213, -10.6763134, 10.2601624
1: -4.8098707, 4.6328521, -4.3105969, 4.1553526, -8.9652233, 8.9434490
2: -6.1815100, 4.6910777, -5.4992442, 4.2362175, -10.4177275, 10.1903219
3: -6.5443354, 4.1365666, -5.8052082, 3.7299364, -10.2742720, 9.9417744
4: -7.1795330, 5.7536092, -6.3839169, 5.1485338, -12.3280668, 12.1375256
5: -5.9223919, 4.2263145, -5.2952280, 3.8333569, -9.7557487, 9.5215425
6: -5.4855747, 5.6141372, -4.9126101, 5.0479145, -10.5334892, 10.5267467
7: -6.0771503, 5.6293106, -5.4357090, 5.0714579, -11.1486082, 11.0650196
8: -7.7737088, 4.7728701, -6.9357343, 4.3029675, -12.0766764, 11.7086048
9: -5.4320917, 5.5719271, -4.8605833, 4.9917831, -10.4238749, 10.4325104

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418219, upper bound: 10.8418537
time: 2.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418272
time: 2.97 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.5435724, 3.3703778, -7.4363918, 3.8187938, -10.3623657, 10.8067694
1: -4.3151798, 4.1624756, -4.9051919, 4.7212801, -9.0364599, 9.0676670
2: -5.4998894, 4.2311034, -6.3183680, 4.7903585, -10.2902479, 10.5494709
3: -5.8173294, 3.7302451, -6.6819878, 4.2203350, -10.0376644, 10.4122334
4: -6.4011002, 5.1548781, -7.3191042, 5.8668003, -12.2679005, 12.4739819
5: -5.2966022, 3.8212290, -6.0480299, 4.3273721, -9.6239738, 9.8692589
6: -4.8990045, 5.0383959, -5.6252851, 5.7421765, -10.6411810, 10.6636810
7: -5.4293938, 5.0692654, -6.2150187, 5.7476864, -11.1770802, 11.2842846
8: -6.9414635, 4.2991147, -7.9352427, 4.8769474, -11.8184109, 12.2343578
9: -4.8653288, 4.9947691, -5.5411119, 5.6870036, -10.5523319, 10.5358810

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418258, upper bound: 10.8418528
time: 2.91 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418260, upper bound: 10.8418219
time: 2.69 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.3041267, 3.7270756, -7.3409739, 3.7731638, -11.0772905, 11.0680494
1: -4.8274202, 4.6495028, -4.8416181, 4.6608639, -9.4882841, 9.4911213
2: -6.2057581, 4.7077208, -6.2306757, 4.7308159, -10.9365740, 10.9383965
3: -6.5704231, 4.1512480, -6.5871401, 4.1678801, -10.7383032, 10.7383881
4: -7.2069426, 5.7747278, -7.2194953, 5.7900305, -12.9969730, 12.9942226
5: -5.9449306, 4.2415042, -5.9675207, 4.2738981, -10.2188282, 10.2090244
6: -5.5071325, 5.6348572, -5.5486021, 5.6673288, -11.1744614, 11.1834593
7: -6.1005125, 5.6496725, -6.1314569, 5.6748838, -11.7753963, 11.7811298
8: -7.8032103, 4.7901759, -7.8283029, 4.8151526, -12.6183624, 12.6184788
9: -5.4522362, 5.5926809, -5.4684086, 5.6121874, -11.0644236, 11.0610895

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418205, upper bound: 10.8418523
time: 4.47 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418221
time: 2.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 8.74 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418747, upper bound: 10.8418672
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418747, upper bound: 10.8418674
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418677
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418689, upper bound: 10.8418671
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418411, upper bound: 10.8418532
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418368, upper bound: 10.8418216
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418318, upper bound: 10.8418531
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8418216
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418260, upper bound: 10.8418542
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418287
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418219, upper bound: 10.8418537
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418272
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418258, upper bound: 10.8418528
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418260, upper bound: 10.8418219
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418205, upper bound: 10.8418523
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 8.74
Output dim: 0, lower bound: -10.8418218, upper bound: 10.8418221

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.2318430, 3.1958442, -5.7442002, 2.9616141, -9.1934566, 8.9400444
1: -4.1110868, 3.9681840, -3.7841702, 3.6569154, -7.7680025, 7.7523541
2: -5.2100096, 4.0290642, -4.7570758, 3.7246771, -8.9346867, 8.7861404
3: -5.5304122, 3.5570796, -5.0654860, 3.2876368, -8.8180485, 8.6225653
4: -6.0828762, 4.9083662, -5.5694523, 4.5121393, -10.5950155, 10.4778185
5: -5.0291634, 3.6443250, -4.6128540, 3.3759198, -8.4050827, 8.2571793
6: -4.6458025, 4.7913389, -4.2546582, 4.4087515, -9.0545540, 9.0459976
7: -5.1387906, 4.8234520, -4.7064171, 4.4529171, -9.5917072, 9.5298691
8: -6.5961366, 4.0980749, -6.0444689, 3.7858834, -10.3820200, 10.1425438
9: -4.6236601, 4.7512460, -4.2476425, 4.3734765, -8.9971371, 8.9988880

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418558, upper bound: 10.8418287
time: 4.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418369, upper bound: 10.8418282
time: 3.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.2318430, 3.1958442, -6.1110640, 3.1611118, -9.3929548, 9.3069077
1: -4.1110868, 3.9681840, -4.0264215, 3.8877091, -7.9987960, 7.9946055
2: -5.2100096, 4.0290642, -5.1001973, 3.9606094, -9.1706190, 9.1292610
3: -5.5304122, 3.5570796, -5.4064250, 3.4914742, -9.0218868, 8.9635048
4: -6.0828762, 4.9083662, -5.9476700, 4.8051248, -10.8880005, 10.8560362
5: -5.0291634, 3.6443250, -4.9279480, 3.5826242, -8.6117878, 8.5722733
6: -4.6458025, 4.7913389, -4.5519633, 4.7006598, -9.3464622, 9.3433018
7: -5.1387906, 4.8234520, -5.0475521, 4.7419171, -9.8807077, 9.8710041
8: -6.5961366, 4.0980749, -6.4541855, 4.0225487, -10.6186848, 10.5522604
9: -4.6236601, 4.7512460, -4.5330410, 4.6606746, -9.2843342, 9.2842865

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418542, upper bound: 10.8418274
time: 4.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418369, upper bound: 10.8418276
time: 2.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.0951748, 3.6032536, -5.6652079, 2.9236984, -10.0188732, 9.2684612
1: -4.6913695, 4.5204868, -3.7318540, 3.6070948, -8.2984638, 8.2523403
2: -6.0118208, 4.5712094, -4.6844897, 3.6759100, -9.6877308, 9.2556992
3: -6.3706446, 4.0349288, -4.9914603, 3.2443433, -9.6149883, 9.0263891
4: -6.9957952, 5.6113734, -5.4874029, 4.4488525, -11.4446478, 11.0987759
5: -5.7654653, 4.1203375, -4.5461817, 3.3323314, -9.0977964, 8.6665192
6: -5.3366876, 5.4675450, -4.1913347, 4.3471665, -9.6838541, 9.6588802
7: -5.9032860, 5.4820480, -4.6372824, 4.3933172, -10.2966032, 10.1193304
8: -7.5729795, 4.6537323, -5.9561758, 3.7355061, -11.3084850, 10.6099081
9: -5.2904038, 5.4265509, -4.1877460, 4.3129930, -9.6033974, 9.6142969

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418510, upper bound: 10.8418287
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418281
time: 4.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.0951748, 3.6032536, -6.0392842, 3.1266046, -10.2217789, 9.6425381
1: -4.6913695, 4.5204868, -3.9790573, 3.8426585, -8.5340281, 8.4995441
2: -6.0118208, 4.5712094, -5.0346169, 3.9162111, -9.9280319, 9.6058264
3: -6.3706446, 4.0349288, -5.3394279, 3.4522491, -9.8228931, 9.3743572
4: -6.9957952, 5.6113734, -5.8735671, 4.7477951, -11.7435904, 11.4849405
5: -5.7654653, 4.1203375, -4.8676038, 3.5427675, -9.3082333, 8.9879417
6: -5.3366876, 5.4675450, -4.4942732, 4.6449690, -9.9816570, 9.9618187
7: -5.9032860, 5.4820480, -4.9849181, 4.6880112, -10.5912971, 10.4669666
8: -7.5729795, 4.6537323, -6.3744154, 3.9767265, -11.5497055, 11.0281477
9: -5.2904038, 5.4265509, -4.4787226, 4.6060085, -9.8964119, 9.9052734

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418510, upper bound: 10.8418284
time: 3.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418282
time: 3.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.2584066, 3.2088075, -7.2224121, 3.7060852, -9.9644918, 10.4312191
1: -4.1284013, 3.9846478, -4.7659016, 4.5888872, -8.7172890, 8.7505493
2: -5.2339592, 4.0453591, -6.1244941, 4.6564760, -9.8904352, 10.1698532
3: -5.5548840, 3.5715253, -6.4755287, 4.1030927, -9.6579762, 10.0470543
4: -6.1099710, 4.9292769, -7.1015100, 5.6996508, -11.8096218, 12.0307865
5: -5.0514002, 3.6590633, -5.8678079, 4.2059140, -9.2573147, 9.5268707
6: -4.6669979, 4.8115931, -5.4529543, 5.5763721, -10.2433701, 10.2645473
7: -5.1617799, 4.8432493, -6.0259762, 5.5832558, -10.7450352, 10.8692255
8: -6.6252704, 4.1149201, -7.7009802, 4.7389283, -11.3641987, 11.8159008
9: -4.6435494, 4.7712860, -5.3795900, 5.5211363, -10.1646862, 10.1508760

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418372, upper bound: 10.8418207
time: 2.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418372, upper bound: 10.8418210
time: 2.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.1197405, 3.6152925, -7.1237736, 3.6589534, -10.7786942, 10.7390661
1: -4.7073483, 4.5356526, -4.7001419, 4.5263891, -9.2337379, 9.2357941
2: -6.0338335, 4.5862885, -6.0337758, 4.5948853, -10.6287193, 10.6200638
3: -6.3943777, 4.0482769, -6.3774199, 4.0488453, -10.4432230, 10.4256973
4: -7.0207887, 5.6306038, -6.9984584, 5.6202407, -12.6410294, 12.6290627
5: -5.7859416, 4.1341324, -5.7845635, 4.1506338, -9.9365749, 9.9186954
6: -5.3562698, 5.4862704, -5.3736453, 5.4989357, -10.8552055, 10.8599157
7: -5.9244084, 5.5005026, -5.9395304, 5.5079513, -11.4323597, 11.4400330
8: -7.5997748, 4.6694732, -7.5903635, 4.6750240, -12.2747993, 12.2598362
9: -5.3087034, 5.4453731, -5.3043871, 5.4437389, -10.7524424, 10.7497597

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8418217
time: 4.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8418219
time: 3.17 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.5152731, 3.3563650, -6.0069714, 3.1226683, -9.6379414, 9.3633366
1: -4.2967315, 4.1449304, -3.9570017, 3.8199294, -8.1166611, 8.1019325
2: -5.4743061, 4.2135954, -5.0059090, 3.8981473, -9.3724537, 9.2195044
3: -5.7912712, 3.7147920, -5.3066087, 3.4343717, -9.2256432, 9.0214005
4: -6.3722048, 5.1325979, -5.8333917, 4.7211657, -11.0933704, 10.9659901
5: -5.2728782, 3.8054924, -4.8397722, 3.5302737, -8.8031521, 8.6452646
6: -4.8763342, 5.0167103, -4.4762106, 4.6264162, -9.5027504, 9.4929209
7: -5.4047556, 5.0480890, -4.9579034, 4.6624494, -10.0672054, 10.0059929
8: -6.9103956, 4.2811203, -6.3389874, 3.9576087, -10.8680038, 10.6201077
9: -4.8440905, 4.9733267, -4.4519043, 4.5799270, -9.4240170, 9.4252310

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418283
time: 2.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418285
time: 2.85 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.2771916, 3.7135780, -5.9348416, 3.0882034, -10.3653946, 9.6484194
1: -4.8098707, 4.6328521, -3.9092238, 3.7744541, -8.5843248, 8.5420761
2: -6.1815100, 4.6910777, -4.9397640, 3.8534341, -10.0349445, 9.6308422
3: -6.5443354, 4.1365666, -5.2390075, 3.3948803, -9.9392157, 9.3755741
4: -7.1795330, 5.7536092, -5.7585850, 4.6633310, -11.8428640, 11.5121937
5: -5.9223919, 4.2263145, -4.7790279, 3.4902759, -9.4126682, 9.0053425
6: -5.4855747, 5.6141372, -4.4181781, 4.5702786, -10.0558529, 10.0323153
7: -6.0771503, 5.6293106, -4.8947897, 4.6081238, -10.6852741, 10.5241003
8: -7.7737088, 4.7728701, -6.2585440, 3.9114962, -11.6852055, 11.0314140
9: -5.4320917, 5.5719271, -4.3971090, 4.5248146, -9.9569063, 9.9690361

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418225, upper bound: 10.8418282
time: 3.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418225, upper bound: 10.8418284
time: 4.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.5435724, 3.3703778, -6.8161583, 3.5021706, -10.0457430, 10.1865358
1: -4.3151798, 4.1624756, -4.4971986, 4.3349867, -8.6501665, 8.6596737
2: -5.4998894, 4.2311034, -5.7515926, 4.4012642, -9.9011536, 9.9826965
3: -5.8173294, 3.7302451, -6.0764275, 3.8802180, -9.6975479, 9.8066730
4: -6.4011002, 5.1548781, -6.6842012, 5.3750668, -11.7761669, 11.8390789
5: -5.2966022, 3.8212290, -5.5250196, 3.9752436, -9.2718458, 9.3462486
6: -4.8990045, 5.0383959, -5.1240873, 5.2554235, -10.1544285, 10.1624832
7: -5.4293938, 5.0692654, -5.6674976, 5.2728148, -10.7022085, 10.7367630
8: -6.9414635, 4.2991147, -7.2484260, 4.4752593, -11.4167233, 11.5475407
9: -4.8653288, 4.9947691, -5.0715542, 5.2038546, -10.0691833, 10.0663233

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418267, upper bound: 10.8418213
time: 2.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418267, upper bound: 10.8418209
time: 2.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.3041267, 3.7270756, -6.7208967, 3.4568958, -10.7610226, 10.4479723
1: -4.8274202, 4.6495028, -4.4338365, 4.2747145, -9.1021347, 9.0833397
2: -6.2057581, 4.7077208, -5.6640463, 4.3418760, -10.5476341, 10.3717670
3: -6.5704231, 4.1512480, -5.9844666, 3.8278537, -10.3982773, 10.1357145
4: -7.2069426, 5.7747278, -6.5848141, 5.2984209, -12.5053635, 12.3595419
5: -5.9449306, 4.2415042, -5.4446793, 3.9222474, -9.8671780, 9.6861839
6: -5.5071325, 5.6348572, -5.0474973, 5.1810555, -10.6881886, 10.6823540
7: -6.1005125, 5.6496725, -5.5840764, 5.2006311, -11.3011436, 11.2337494
8: -7.8032103, 4.7901759, -7.1418533, 4.4139490, -12.2171593, 11.9320297
9: -5.4522362, 5.5926809, -4.9989858, 5.1300502, -10.5822868, 10.5916672

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418225, upper bound: 10.8418219
time: 4.54 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418214, upper bound: 10.8418209
time: 3.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 9.27 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418558, upper bound: 10.8418287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418369, upper bound: 10.8418282
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418542, upper bound: 10.8418274
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418369, upper bound: 10.8418276
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418510, upper bound: 10.8418287
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418281
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418510, upper bound: 10.8418284
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418282
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418372, upper bound: 10.8418207
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418372, upper bound: 10.8418210
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8418217
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8418219
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418283
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418271, upper bound: 10.8418285
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418225, upper bound: 10.8418282
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418225, upper bound: 10.8418284
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418267, upper bound: 10.8418213
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418267, upper bound: 10.8418209
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418225, upper bound: 10.8418219
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.27
Output dim: 0, lower bound: -10.8418214, upper bound: 10.8418209

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.6122236, 2.8765712, -5.7442002, 2.9616141, -8.5738373, 8.6207714
1: -3.7049055, 3.5828412, -3.7841702, 3.6569154, -7.3618212, 7.3670111
2: -4.6430755, 3.6425214, -4.7570758, 3.7246771, -8.3677521, 8.3995972
3: -4.9573255, 3.2176249, -5.0654860, 3.2876368, -8.2449627, 8.2831106
4: -5.4504738, 4.4174089, -5.5694523, 4.5121393, -9.9626131, 9.9868612
5: -4.5053434, 3.2966452, -4.6128540, 3.3759198, -7.8812633, 7.9094992
6: -4.1445098, 4.3070698, -4.2546582, 4.4087515, -8.5532608, 8.5617275
7: -4.5903578, 4.3544321, -4.7064171, 4.4529171, -9.0432749, 9.0608492
8: -5.9096193, 3.7020433, -6.0444689, 3.7858834, -9.6955032, 9.7465124
9: -4.1548553, 4.2785234, -4.2476425, 4.3734765, -8.5283318, 8.5261660

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418413, upper bound: 10.8418398
time: 2.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418413, upper bound: 10.8418390
time: 2.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.6122236, 2.8765712, -6.1110640, 3.1611118, -8.7733355, 8.9876347
1: -3.7049055, 3.5828412, -4.0264215, 3.8877091, -7.5926147, 7.6092625
2: -4.6430755, 3.6425214, -5.1001973, 3.9606094, -8.6036854, 8.7427187
3: -4.9573255, 3.2176249, -5.4064250, 3.4914742, -8.4487991, 8.6240501
4: -5.4504738, 4.4174089, -5.9476700, 4.8051248, -10.2555981, 10.3650789
5: -4.5053434, 3.2966452, -4.9279480, 3.5826242, -8.0879679, 8.2245932
6: -4.1445098, 4.3070698, -4.5519633, 4.7006598, -8.8451691, 8.8590336
7: -4.5903578, 4.3544321, -5.0475521, 4.7419171, -9.3322754, 9.4019842
8: -5.9096193, 3.7020433, -6.4541855, 4.0225487, -9.9321680, 10.1562290
9: -4.1548553, 4.2785234, -4.5330410, 4.6606746, -8.8155298, 8.8115644

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418379, upper bound: 10.8418286
time: 3.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418379, upper bound: 10.8418287
time: 3.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.4843163, 3.2879820, -5.6652079, 2.9236984, -9.4080143, 8.9531898
1: -4.2909346, 4.1411982, -3.7318540, 3.6070948, -7.8980293, 7.8730521
2: -5.4539909, 4.1878099, -4.6844897, 3.6759100, -9.1299009, 8.8722992
3: -5.7915239, 3.6998591, -4.9914603, 3.2443433, -9.0358677, 8.6913195
4: -6.3725863, 5.1282148, -5.4874029, 4.4488525, -10.8214388, 10.6156178
5: -5.2503328, 3.7746532, -4.5461817, 3.3323314, -8.5826645, 8.3208351
6: -4.8417358, 4.9899430, -4.1913347, 4.3471665, -9.1889019, 9.1812782
7: -5.3630362, 5.0178428, -4.6372824, 4.3933172, -9.7563534, 9.6551247
8: -6.8979206, 4.2602444, -5.9561758, 3.7355061, -10.6334267, 10.2164202
9: -4.8286376, 4.9559016, -4.1877460, 4.3129930, -9.1416302, 9.1436481

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418400
time: 3.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418394
time: 3.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.4843163, 3.2879820, -6.0392842, 3.1266046, -9.6109209, 9.3272667
1: -4.2909346, 4.1411982, -3.9790573, 3.8426585, -8.1335926, 8.1202555
2: -5.4539909, 4.1878099, -5.0346169, 3.9162111, -9.3702021, 9.2224274
3: -5.7915239, 3.6998591, -5.3394279, 3.4522491, -9.2437725, 9.0392876
4: -6.3725863, 5.1282148, -5.8735671, 4.7477951, -11.1203814, 11.0017815
5: -5.2503328, 3.7746532, -4.8676038, 3.5427675, -8.7931004, 8.6422567
6: -4.8417358, 4.9899430, -4.4942732, 4.6449690, -9.4867048, 9.4842167
7: -5.3630362, 5.0178428, -4.9849181, 4.6880112, -10.0510473, 10.0027609
8: -6.8979206, 4.2602444, -6.3744154, 3.9767265, -10.8746471, 10.6346598
9: -4.8286376, 4.9559016, -4.4787226, 4.6060085, -9.4346466, 9.4346237

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418284
time: 4.24 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418280
time: 4.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 10.26 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.26
Output dim: 0, lower bound: -10.8418413, upper bound: 10.8418398
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.26
Output dim: 0, lower bound: -10.8418413, upper bound: 10.8418390
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 10.26
Output dim: 0, lower bound: -10.8418379, upper bound: 10.8418286
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 10.26
Output dim: 0, lower bound: -10.8418379, upper bound: 10.8418287
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.26
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418400
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 10.26
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418394
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 10.26
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418284
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 10.26
Output dim: 0, lower bound: -10.8418317, upper bound: 10.8418280

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.6122236, 2.8765712, -5.1321545, 2.6450915, -8.2573147, 8.0087261
1: -3.7049055, 3.5828412, -3.3854666, 3.2763965, -6.9813023, 6.9683075
2: -4.6430755, 3.6425214, -4.1984034, 3.3432989, -7.9863744, 7.8409247
3: -4.9573255, 3.2176249, -4.5011425, 2.9520674, -7.9093928, 7.7187672
4: -5.4504738, 4.4174089, -4.9443169, 4.0313325, -9.4818058, 9.3617258
5: -4.5053434, 3.2966452, -4.0962100, 3.0333257, -7.5386691, 7.3928552
6: -4.1445098, 4.3070698, -3.7607229, 3.9317055, -8.0762157, 8.0677929
7: -4.5903578, 4.3544321, -4.1647406, 3.9886436, -8.5790014, 8.5191727
8: -5.9096193, 3.7020433, -5.3654089, 3.3960791, -9.3056984, 9.0674524
9: -4.1548553, 4.2785234, -3.7860315, 3.9056301, -8.0604858, 8.0645552

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418206
time: 4.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418438, upper bound: 10.8418207
time: 5.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.6122236, 2.8765712, -7.7032261, 3.8237388, -9.4359627, 10.5797977
1: -3.7049055, 3.5828412, -5.1144595, 4.9302635, -8.6351690, 8.6973009
2: -4.6430755, 3.6425214, -6.5867109, 4.9339194, -9.5769949, 10.2292328
3: -4.9573255, 3.2176249, -6.9659839, 4.3728361, -9.3301620, 10.1836090
4: -5.4504738, 4.4174089, -7.6777821, 6.1302485, -11.5807228, 12.0951910
5: -4.5053434, 3.2966452, -6.2872696, 4.4304695, -8.9358130, 9.5839148
6: -4.1445098, 4.3070698, -5.8095798, 5.9385147, -10.0830250, 10.1166496
7: -4.5903578, 4.3544321, -6.4253373, 5.9369659, -10.5273237, 10.7797699
8: -5.9096193, 3.7020433, -8.2863445, 5.0354614, -10.9450808, 11.9883881
9: -4.1548553, 4.2785234, -5.7675667, 5.9002733, -10.0551281, 10.0460901

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418203
time: 3.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418438, upper bound: 10.8418209
time: 6.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.4843163, 3.2879820, -5.0526452, 2.6069617, -9.0912781, 8.3406277
1: -4.2909346, 4.1411982, -3.3332305, 3.2262535, -7.5171881, 7.4744287
2: -5.4539909, 4.1878099, -4.1257119, 3.2942119, -8.7482033, 8.3135223
3: -5.7915239, 3.6998591, -4.4268894, 2.9084842, -8.7000084, 8.1267490
4: -6.3725863, 5.1282148, -4.8617239, 3.9682374, -10.3408241, 9.9899387
5: -5.2503328, 3.7746532, -4.0293875, 2.9894333, -8.2397661, 7.8040409
6: -4.8417358, 4.9899430, -3.6969788, 3.8700693, -8.7118053, 8.6869221
7: -5.3630362, 5.0178428, -4.0952520, 3.9286737, -9.2917099, 9.1130943
8: -6.8979206, 4.2602444, -5.2765217, 3.3455398, -10.2434607, 9.5367661
9: -4.8286376, 4.9559016, -3.7259219, 3.8447423, -8.6733799, 8.6818237

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418203
time: 4.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418402, upper bound: 10.8418206
time: 6.59 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 12.66 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.66
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418206
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.66
Output dim: 0, lower bound: -10.8418438, upper bound: 10.8418207
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.66
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418203
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.66
Output dim: 0, lower bound: -10.8418438, upper bound: 10.8418209
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.66
Output dim: 0, lower bound: -10.8418470, upper bound: 10.8418203
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.66
Output dim: 0, lower bound: -10.8418402, upper bound: 10.8418206

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2745924, 2.7063580, -5.1289635, 2.6434779, -7.9180703, 7.8353214
1: -3.4887996, 3.3774612, -3.3834176, 3.2744541, -6.7632537, 6.7608786
2: -4.3407760, 3.4352801, -4.1955366, 3.3413320, -7.6821079, 7.6308165
3: -4.6529016, 3.0356750, -4.4982510, 2.9503450, -7.6032467, 7.5339260
4: -5.1144304, 4.1560440, -4.9411373, 4.0288510, -9.1432819, 9.0971813
5: -4.2259188, 3.1061149, -4.0935555, 3.0315261, -7.2574449, 7.1996703
6: -3.8726864, 4.0464787, -3.7581582, 3.9292343, -7.8019209, 7.8046370
7: -4.2995119, 4.1036067, -4.1619835, 3.9862692, -8.2857809, 8.2655907
8: -5.5415092, 3.4880748, -5.3619123, 3.3940568, -8.9355659, 8.8499870
9: -3.9069099, 4.0260301, -3.7836804, 3.9032340, -7.8101439, 7.8097105

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418593, upper bound: 10.8418570
time: 3.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418593, upper bound: 10.8418555
time: 4.30 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.5856481, 3.3743370, -4.9080753, 2.5335331, -9.1191807, 8.2824125
1: -4.3695850, 4.2173033, -3.2418461, 3.1395426, -7.5091276, 7.4591494
2: -5.5723734, 4.2651196, -3.9973507, 3.2060831, -8.7784567, 8.2624702
3: -5.9072285, 3.7669845, -4.2980957, 2.8311539, -8.7383823, 8.0650806
4: -6.5006208, 5.2203312, -4.7202387, 3.8577235, -10.3583441, 9.9405699
5: -5.3597059, 3.8295255, -3.9107785, 2.9072099, -8.2669163, 7.7403040
6: -4.9256983, 5.0861354, -3.5804610, 3.7591591, -8.6848574, 8.6665964
7: -5.4909263, 5.1188593, -3.9718161, 3.8223958, -9.3133221, 9.0906754
8: -7.0362930, 4.3290234, -5.1202049, 3.2542481, -10.2905407, 9.4492283
9: -4.9294987, 5.0561848, -3.6210818, 3.7377992, -8.6672974, 8.6772671

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418589, upper bound: 10.8418561
time: 3.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418589, upper bound: 10.8418560
time: 4.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2745924, 2.7063580, -7.7001333, 3.8221645, -9.0967569, 10.4064913
1: -3.4887996, 3.3774612, -5.1124630, 4.9283829, -8.4171829, 8.4899244
2: -4.3407760, 3.4352801, -6.5839224, 4.9320126, -9.2727890, 10.0192022
3: -4.6529016, 3.0356750, -6.9631748, 4.3711667, -9.0240688, 9.9988499
4: -5.1144304, 4.1560440, -7.6747074, 6.1278181, -11.2422485, 11.8307514
5: -4.2259188, 3.1061149, -6.2846870, 4.4287229, -8.6546421, 9.3908024
6: -3.8726864, 4.0464787, -5.8070841, 5.9361129, -9.8087997, 9.8535633
7: -4.2995119, 4.1036067, -6.4226589, 5.9346681, -10.2341805, 10.5262661
8: -5.5415092, 3.4880748, -8.2829599, 5.0334973, -10.5750065, 11.7710342
9: -3.9069099, 4.0260301, -5.7652822, 5.8979540, -9.8048639, 9.7913122

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418437, upper bound: 10.8418207
time: 4.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418437, upper bound: 10.8418191
time: 4.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.5856481, 3.3743370, -7.4867663, 3.7148283, -10.3004761, 10.8611031
1: -4.3695850, 4.2173033, -4.9745464, 4.7982664, -9.1678514, 9.1918497
2: -5.5723734, 4.2651196, -6.3916602, 4.8008065, -10.3731804, 10.6567802
3: -5.9072285, 3.7669845, -6.7692704, 4.2560558, -10.1632843, 10.5362549
4: -6.5006208, 5.2203312, -7.4620128, 5.9600325, -12.4606533, 12.6823444
5: -5.3597059, 3.8295255, -6.1068268, 4.3084373, -9.6681433, 9.9363518
6: -4.9256983, 5.0861354, -5.6350894, 5.7705746, -10.6962729, 10.7212248
7: -5.4909263, 5.1188593, -6.2382917, 5.7763643, -11.2672901, 11.3571510
8: -7.0362930, 4.3290234, -8.0495520, 4.8979487, -11.9342422, 12.3785753
9: -4.9294987, 5.0561848, -5.6076479, 5.7380495, -10.6675482, 10.6638327

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418270, upper bound: 10.8417940
time: 4.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418201
time: 4.31 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1324825, 3.1106546, -5.0494528, 2.6053462, -8.7378292, 8.1601076
1: -4.0638747, 3.9275215, -3.3311818, 3.2243114, -7.2881861, 7.2587032
2: -5.1376500, 3.9696767, -4.1228447, 3.2922463, -8.4298964, 8.0925217
3: -5.4723787, 3.5101275, -4.4239979, 2.9067616, -8.3791409, 7.9341254
4: -6.0230141, 4.8524408, -4.8585463, 3.9657545, -9.9887686, 9.7109871
5: -4.9577351, 3.5752568, -4.0267324, 2.9876330, -7.9453678, 7.6019893
6: -4.5579519, 4.7171111, -3.6944122, 3.8675985, -8.4255505, 8.4115238
7: -5.0591755, 4.7570286, -4.0924954, 3.9262996, -8.9854755, 8.8495235
8: -6.5143123, 4.0360355, -5.2730255, 3.3435163, -9.8578281, 9.3090611
9: -4.5688062, 4.6923742, -3.7235713, 3.8423471, -8.4111538, 8.4159451

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418527, upper bound: 10.8418570
time: 2.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418527, upper bound: 10.8418562
time: 4.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -7.4807014, 3.7927194, -4.8283877, 2.4957347, -9.9764366, 8.6211071
1: -4.9691696, 4.7895594, -3.1898851, 3.0892694, -8.0584393, 7.9794445
2: -6.3993268, 4.8280506, -3.9246955, 3.1571956, -9.5565224, 8.7527466
3: -6.7604184, 4.2623396, -4.2238016, 2.7875524, -9.5479708, 8.4861412
4: -7.4462657, 5.9476886, -4.6374416, 3.7950754, -11.2413406, 10.5851307
5: -6.1187134, 4.3227158, -3.8441391, 2.8633244, -8.9820375, 8.1668549
6: -5.6456861, 5.7819877, -3.5165367, 3.6978371, -9.3435230, 9.2985249
7: -6.2807198, 5.7953553, -3.9023938, 3.7625985, -10.0433178, 9.6977491
8: -8.0443039, 4.9034781, -5.0313268, 3.2038651, -11.2481689, 9.9348049
9: -5.6199727, 5.7490721, -3.5609832, 3.6771605, -9.2971334, 9.3100548

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418537, upper bound: 10.8418568
time: 2.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418531, upper bound: 10.8418558
time: 2.65 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.90 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418593, upper bound: 10.8418570
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418593, upper bound: 10.8418555
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418589, upper bound: 10.8418561
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418589, upper bound: 10.8418560
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418437, upper bound: 10.8418207
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418437, upper bound: 10.8418191
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418270, upper bound: 10.8417940
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418201
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418527, upper bound: 10.8418570
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418527, upper bound: 10.8418562
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418537, upper bound: 10.8418568
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.90
Output dim: 0, lower bound: -10.8418531, upper bound: 10.8418558

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -5.2745924, 2.7063580, -4.7862034, 2.4717019, -7.7462940, 7.4925613
1: -3.4887996, 3.3774612, -3.1648619, 3.0653086, -6.5541081, 6.5423231
2: -4.3407760, 3.4352801, -3.8889902, 3.1318808, -7.4726567, 7.3242702
3: -4.6529016, 3.0356750, -4.1891918, 2.7654545, -7.4183559, 7.2248669
4: -5.1144304, 4.1560440, -4.5988717, 3.7651281, -8.8795586, 8.7549152
5: -4.2259188, 3.1061149, -3.8108020, 2.8384566, -7.0643754, 6.9169168
6: -3.8726864, 4.0464787, -3.4816730, 3.6663513, -7.5390377, 7.5281515
7: -4.2995119, 4.1036067, -3.8669081, 3.7321873, -8.0316992, 7.9705148
8: -5.5415092, 3.4880748, -4.9884443, 3.1774559, -8.7189655, 8.4765186
9: -3.9069099, 4.0260301, -3.5319252, 3.6474938, -7.5544038, 7.5579553

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418706, upper bound: 10.8418563
time: 4.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418685, upper bound: 10.8418554
time: 3.46 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -5.2745924, 2.7063580, -6.0943389, 3.1378570, -8.4124489, 8.8006973
1: -3.4887996, 3.3774612, -4.0370364, 3.9030671, -7.3918667, 7.4144974
2: -4.3407760, 3.4352801, -5.1125093, 3.9590747, -8.2998505, 8.5477896
3: -4.6529016, 3.0356750, -5.4370642, 3.4951227, -8.1480246, 8.4727392
4: -5.1144304, 4.1560440, -5.9814911, 4.8178892, -9.9323196, 10.1375351
5: -4.2259188, 3.1061149, -4.9380894, 3.5600979, -7.7860165, 8.0442047
6: -3.8726864, 4.0464787, -4.5324469, 4.6978145, -8.5705013, 8.5789261
7: -4.2995119, 4.1036067, -5.0539818, 4.7444067, -9.0439186, 9.1575890
8: -5.5415092, 3.4880748, -6.4795713, 4.0133958, -9.5549049, 9.9676456
9: -3.9069099, 4.0260301, -4.5497007, 4.6747322, -8.5816422, 8.5757313

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418706, upper bound: 10.8418565
time: 4.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418685, upper bound: 10.8418566
time: 3.20 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.4133925, 3.2900054, -4.0984864, 2.1364689, -8.5498619, 7.3884916
1: -4.2555513, 4.1087170, -2.7222576, 2.6349325, -6.8904839, 6.8309746
2: -5.4139004, 4.1583347, -3.2653389, 2.7105818, -8.1244822, 7.4236736
3: -5.7456970, 3.6722934, -3.5542996, 2.3900843, -8.1357813, 7.2265930
4: -6.3215537, 5.0825701, -3.8976185, 3.2306137, -9.5521679, 8.9801884
5: -5.2134805, 3.7346647, -3.2355239, 2.4604266, -7.6739073, 6.9701886
6: -4.7881746, 4.9520898, -2.9296153, 3.1374884, -7.9256630, 7.8817053
7: -5.3389988, 4.9880819, -3.2639117, 3.2177010, -8.5566998, 8.2519932
8: -6.8437252, 4.2192688, -4.2286425, 2.7429824, -9.5867081, 8.4479113
9: -4.7982874, 4.9237537, -3.0137095, 3.1279736, -7.9262609, 7.9374633

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418332, upper bound: 10.8418509
time: 2.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418590, upper bound: 10.8418569
time: 5.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.5191374, 3.3418639, -4.5737600, 2.3722558, -8.8913937, 7.9156237
1: -4.3254857, 4.1753035, -3.0239103, 2.9278069, -7.2532926, 7.1992140
2: -5.5110970, 4.2238708, -3.6911952, 3.0002885, -8.5113850, 7.9150658
3: -5.8447385, 3.7303953, -3.9849739, 2.6473815, -8.4921198, 7.7153692
4: -6.4313459, 5.1670523, -4.3721471, 3.5944028, -10.0257492, 9.5391998
5: -5.3032055, 3.7929227, -3.6283181, 2.7238834, -8.0270891, 7.4212408
6: -4.8725948, 5.0343280, -3.3134761, 3.5012732, -8.3738680, 8.3478041
7: -5.4322290, 5.0683336, -3.6775608, 3.5702150, -9.0024443, 8.7458944
8: -6.9618230, 4.2866287, -4.7464890, 3.0428677, -10.0046902, 9.0331173
9: -4.8787680, 5.0049887, -3.3666611, 3.4818964, -8.3606644, 8.3716497

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418333, upper bound: 10.8418499
time: 3.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418590, upper bound: 10.8418562
time: 3.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.2745924, 2.7063580, -7.3610902, 3.6496980, -8.9242907, 10.0674477
1: -3.4887996, 3.3774612, -4.8934560, 4.7220035, -8.2108030, 8.2709169
2: -4.3407760, 3.4352801, -6.2782779, 4.7230940, -9.0638695, 9.7135582
3: -4.6529016, 3.0356750, -6.6552553, 4.1881943, -8.8410959, 9.6909304
4: -5.1144304, 4.1560440, -7.3374386, 5.8613663, -10.9757967, 11.4934826
5: -4.2259188, 3.1061149, -6.0016165, 4.2372351, -8.4631538, 9.1077309
6: -3.8726864, 4.0464787, -5.5336881, 5.6727734, -9.5454597, 9.5801668
7: -4.2995119, 4.1036067, -6.1289997, 5.6827979, -9.9823093, 10.2326069
8: -5.5415092, 3.4880748, -7.9120979, 4.8180642, -10.3595734, 11.4001732
9: -3.9069099, 4.0260301, -5.5146856, 5.6436896, -9.5506001, 9.5407162

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418204
time: 5.47 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418210
time: 6.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -5.2745924, 2.7063580, -8.5811882, 4.2727308, -9.5473232, 11.2875462
1: -3.4887996, 3.3774612, -5.7130919, 5.5039620, -8.9927616, 9.0905533
2: -4.3407760, 3.4352801, -7.4224372, 5.4929485, -9.8337250, 10.8577175
3: -4.6529016, 3.0356750, -7.8213339, 4.8693604, -9.5222626, 10.8570089
4: -5.1144304, 4.1560440, -8.6263294, 6.8541613, -11.9685917, 12.7823734
5: -4.2259188, 3.1061149, -7.0557289, 4.9106350, -9.1365538, 10.1618443
6: -3.8726864, 4.0464787, -6.5185928, 6.6386819, -10.5113678, 10.5650711
7: -4.2995119, 4.1036067, -7.2362137, 6.6245427, -10.9240551, 11.3398209
8: -5.5415092, 3.4880748, -9.3005190, 5.6000676, -11.1415768, 12.7885933
9: -3.9069099, 4.0260301, -6.4657459, 6.5999022, -10.5068121, 10.4917755

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418210
time: 4.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418213
time: 4.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.5856481, 3.3743370, -7.2522144, 3.5959570, -10.1816053, 10.6265516
1: -4.3695850, 4.2173033, -4.8239565, 4.6567068, -9.0262918, 9.0412598
2: -5.5723734, 4.2651196, -6.1816907, 4.6570139, -10.2293873, 10.4468098
3: -5.9072285, 3.7669845, -6.5578289, 4.1302471, -10.0374756, 10.3248138
4: -6.5006208, 5.2203312, -7.2310185, 5.7766309, -12.2772522, 12.4513493
5: -5.3597059, 3.8295255, -5.9121103, 4.1755366, -9.5352421, 9.7416363
6: -4.9256983, 5.0861354, -5.4460974, 5.5890961, -10.5147943, 10.5322323
7: -5.4909263, 5.1188593, -6.0371184, 5.6035075, -11.0944338, 11.1559772
8: -7.0362930, 4.3290234, -7.7945910, 4.7492008, -11.7854939, 12.1236143
9: -4.9294987, 5.0561848, -5.4359241, 5.5635467, -10.4930458, 10.4921093

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418202
time: 3.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418196
time: 3.48 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1324825, 3.1106546, -4.7077522, 2.4345841, -8.5670662, 7.8184071
1: -4.0638747, 3.9275215, -3.1138613, 3.0156887, -7.0795631, 7.0413828
2: -5.1376500, 3.9696767, -3.8173900, 3.0839779, -8.2216282, 7.7870665
3: -5.4723787, 3.5101275, -4.1161180, 2.7224622, -8.1948414, 7.6262455
4: -6.0230141, 4.8524408, -4.5175314, 3.7033844, -9.7263985, 9.3699722
5: -4.9577351, 3.5752568, -3.7449927, 2.7953055, -7.7530403, 7.3202496
6: -4.5579519, 4.7171111, -3.4188099, 3.6058614, -8.1638136, 8.1359215
7: -5.0591755, 4.7570286, -3.7985063, 3.6734595, -8.7326355, 8.5555344
8: -6.5143123, 4.0360355, -4.9010406, 3.1278219, -9.6421337, 8.9370766
9: -4.5688062, 4.6923742, -3.4726772, 3.5878329, -8.1566391, 8.1650515

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418656, upper bound: 10.8418565
time: 3.56 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418660, upper bound: 10.8418568
time: 5.07 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1324825, 3.1106546, -6.0138817, 3.0994489, -9.2319317, 9.1245365
1: -4.0638747, 3.9275215, -3.9842410, 3.8523123, -7.9161873, 7.9117622
2: -5.1376500, 3.9696767, -5.0390229, 3.9094877, -9.0471382, 9.0086994
3: -5.4723787, 3.5101275, -5.3620520, 3.4510400, -8.9234190, 8.8721790
4: -6.0230141, 4.8524408, -5.8978796, 4.7541523, -10.7771664, 10.7503204
5: -4.9577351, 3.5752568, -4.8706055, 3.5157151, -8.4734497, 8.4458618
6: -4.5579519, 4.7171111, -4.4679813, 4.6355467, -9.1934986, 9.1850929
7: -5.0591755, 4.7570286, -4.9837327, 4.6837072, -9.7428827, 9.7407608
8: -6.5143123, 4.0360355, -6.3897676, 3.9622788, -10.4765911, 10.4258032
9: -4.5688062, 4.6923742, -4.4889221, 4.6131649, -9.1819706, 9.1812963

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418656, upper bound: 10.8418562
time: 3.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418660, upper bound: 10.8418557
time: 5.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -7.3082271, 3.7084472, -4.0239067, 2.1025541, -9.4107809, 7.7323542
1: -4.8553095, 4.6811337, -2.6744356, 2.5884960, -7.4438057, 7.3555694
2: -6.2413855, 4.7203383, -3.1979005, 2.6660607, -8.9074459, 7.9182386
3: -6.5991359, 4.1674933, -3.4853764, 2.3495696, -8.9487057, 7.6528697
4: -7.2674589, 5.8099504, -3.8264375, 3.1726797, -10.4401388, 9.6363878
5: -5.9729404, 4.2272120, -3.1732469, 2.4213786, -8.3943195, 7.4004588
6: -5.5073609, 5.6483831, -2.8703074, 3.0810449, -8.5884056, 8.5186901
7: -6.1288528, 5.6649618, -3.1991262, 3.1623223, -9.2911749, 8.8640881
8: -7.8524294, 4.7933292, -4.1461449, 2.6963081, -10.5487375, 8.9394741
9: -5.4886827, 5.6167269, -2.9574580, 3.0731263, -8.5618095, 8.5741844

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418315, upper bound: 10.8418510
time: 3.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418536, upper bound: 10.8418568
time: 4.28 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -7.4156365, 3.7610645, -4.4950905, 2.3350968, -9.7507334, 8.2561550
1: -4.9261169, 4.7485466, -2.9729497, 2.8781362, -7.8042531, 7.7214966
2: -6.3396215, 4.7873907, -3.6197183, 2.9521837, -9.2918053, 8.4071093
3: -6.6994057, 4.2265229, -3.9115911, 2.6043959, -9.3038015, 8.1381140
4: -7.3786197, 5.8955984, -4.2904758, 3.5328639, -10.9114838, 10.1860743
5: -6.0636778, 4.2867107, -3.5624371, 2.6809785, -8.7446566, 7.8491478
6: -5.5934796, 5.7314992, -3.2505474, 3.4407148, -9.0341949, 8.9820461
7: -6.2233734, 5.7461329, -3.6089540, 3.5115004, -9.7348738, 9.3550873
8: -7.9717569, 4.8618927, -4.6590304, 2.9932194, -10.9649763, 9.5209236
9: -5.5703454, 5.6990633, -3.3071353, 3.4222367, -8.9925823, 9.0061989

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 93

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418325, upper bound: 10.8418505
time: 2.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418536, upper bound: 10.8418563
time: 2.92 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 7.16 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418706, upper bound: 10.8418563
IS_A1_B1_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418685, upper bound: 10.8418554
IS_A1_B1_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418706, upper bound: 10.8418565
IS_A1_B1_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418685, upper bound: 10.8418566
IS_A1_B1_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418332, upper bound: 10.8418509
IS_A1_B1_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418590, upper bound: 10.8418569
IS_A1_B1_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418333, upper bound: 10.8418499
IS_A1_B1_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418590, upper bound: 10.8418562
IS_A1_B1_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418204
IS_A1_B1_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418210
IS_A1_B1_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418210
IS_A1_B1_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418213
IS_A1_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418202
IS_A1_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418196
IS_A1_B1_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418656, upper bound: 10.8418565
IS_A1_B1_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418660, upper bound: 10.8418568
IS_A1_B1_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418656, upper bound: 10.8418562
IS_A1_B1_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418660, upper bound: 10.8418557
IS_A1_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418315, upper bound: 10.8418510
IS_A1_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418536, upper bound: 10.8418568
IS_A1_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418325, upper bound: 10.8418505
IS_A1_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.16
Output dim: 0, lower bound: -10.8418536, upper bound: 10.8418563

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.4363551, 2.2941785, -4.6070666, 2.3851354, -6.8214903, 6.9012451
1: -2.9473343, 2.8543468, -3.0483532, 2.9519598, -5.8992939, 5.9026999
2: -3.5812991, 2.9206049, -3.7251120, 3.0218730, -6.6031723, 6.6457167
3: -3.8817768, 2.5782537, -4.0218101, 2.6670156, -6.5487924, 6.6000638
4: -4.2560139, 3.5028915, -4.4128580, 3.6242099, -7.8802238, 7.9157495
5: -3.5264707, 2.6405244, -3.6594591, 2.7401261, -6.2665968, 6.2999835
6: -3.1982424, 3.4014854, -3.3386812, 3.5282915, -6.7265339, 6.7401667
7: -3.5673370, 3.4754684, -3.7093506, 3.5972595, -7.1645966, 7.1848192
8: -4.6158886, 2.9573741, -4.7886314, 3.0641787, -7.6800671, 7.7460055
9: -3.2785830, 3.3911195, -3.3957543, 3.5105476, -6.7891307, 6.7868738

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418696, upper bound: 10.8418487
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418754, upper bound: 10.8418701
time: 3.19 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.9391317, 2.5429313, -4.7204552, 2.4399724, -7.3791041, 7.2633867
1: -3.2681527, 3.1653872, -3.1220012, 3.0236464, -6.2917991, 6.2873883
2: -4.0330329, 3.2272940, -3.8287172, 3.0914745, -7.1245074, 7.0560112
3: -4.3385234, 2.8510425, -4.1276560, 2.7292714, -7.0677948, 6.9786987
4: -4.7646532, 3.8898122, -4.5304942, 3.7132468, -8.4778996, 8.4203062
5: -3.9422717, 2.9213891, -3.7551656, 2.8023853, -6.7446570, 6.6765547
6: -3.6045351, 3.7865639, -3.4291875, 3.6155751, -7.2201099, 7.2157516
7: -4.0036316, 3.8488936, -3.8090248, 3.6825838, -7.6862154, 7.6579185
8: -5.1654863, 3.2749743, -4.9149151, 3.1358676, -8.3013535, 8.1898899
9: -3.6515489, 3.7677307, -3.4818828, 3.5971601, -7.2487087, 7.2496138

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418690, upper bound: 10.8418479
time: 3.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418706
time: 2.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.4363551, 2.2941785, -5.9195623, 3.0525429, -7.4888983, 8.2137413
1: -2.9473343, 2.8543468, -3.9222250, 3.7928188, -6.7401533, 6.7765718
2: -3.5812991, 2.9206049, -4.9524813, 3.8506937, -7.4319925, 7.8730860
3: -3.8817768, 2.5782537, -5.2736807, 3.3990247, -7.2808018, 7.8519344
4: -4.2560139, 3.5028915, -5.7996922, 4.6793060, -8.9353199, 9.3025837
5: -3.5264707, 2.6405244, -4.7903528, 3.4638455, -6.9903164, 7.4308772
6: -3.1982424, 3.4014854, -4.3928566, 4.5625253, -7.7607679, 7.7943420
7: -3.5673370, 3.4754684, -4.9000015, 4.6116729, -8.1790104, 8.3754702
8: -4.6158886, 2.9573741, -6.2840915, 3.9024029, -8.5182915, 9.2414656
9: -3.2785830, 3.3911195, -4.4168768, 4.5403056, -7.8188887, 7.8079963

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418581, upper bound: 10.8418309
time: 4.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418696, upper bound: 10.8418556
time: 4.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.9391317, 2.5429313, -6.0296450, 3.1062884, -8.0454197, 8.5725765
1: -3.2681527, 3.1653872, -3.9945164, 3.8622351, -7.1303878, 7.1599035
2: -4.0330329, 3.2272940, -5.0532355, 3.9189448, -7.9519777, 8.2805290
3: -4.3385234, 2.8510425, -5.3765469, 3.4595404, -7.7980638, 8.2275896
4: -4.7646532, 3.8898122, -5.9141536, 4.7665634, -9.5312166, 9.8039656
5: -3.9422717, 2.9213891, -4.8833828, 3.5244741, -7.4667459, 7.8047719
6: -3.6045351, 3.7865639, -4.4807787, 4.6477165, -8.2522516, 8.2673426
7: -4.0036316, 3.8488936, -4.9969597, 4.6952496, -8.6988811, 8.8458538
8: -5.1654863, 3.2749743, -6.4071741, 3.9723063, -9.1377926, 9.6821480
9: -3.6515489, 3.7677307, -4.5005054, 4.6249447, -8.2764931, 8.2682362

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418581, upper bound: 10.8418317
time: 4.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418694, upper bound: 10.8418565
time: 4.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.9456105, 3.5269728, -3.8422961, 2.0168171, -8.9624271, 7.3692689
1: -4.6275392, 4.4645863, -2.5621142, 2.4807317, -7.1082706, 7.0267005
2: -5.9260097, 4.4944296, -3.0398657, 2.5592918, -8.4853020, 7.5342951
3: -6.2815990, 3.9732242, -3.3275292, 2.2529383, -8.5345373, 7.3007536
4: -6.9143515, 5.5337524, -3.6610909, 3.0371552, -9.9515066, 9.1948433
5: -5.6812592, 4.0180020, -3.0265346, 2.3233972, -8.0046558, 7.0445366
6: -5.2092094, 5.3761740, -2.7246699, 2.9464180, -8.1556273, 8.1008434
7: -5.8217926, 5.4005766, -3.0454881, 3.0311773, -8.8529701, 8.4460649
8: -7.4719858, 4.5616760, -3.9530311, 2.5834084, -10.0553942, 8.5147076
9: -5.2256670, 5.3517342, -2.8264813, 2.9446120, -8.1702785, 8.1782150

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418332, upper bound: 10.8418502
time: 4.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418332, upper bound: 10.8418512
time: 4.18 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.1981816, 3.1834860, -4.0984864, 2.1364689, -8.3346500, 7.2819724
1: -4.1178560, 3.9777346, -2.7222576, 2.6349325, -6.7527885, 6.6999922
2: -5.2220955, 4.0278621, -3.2653389, 2.7105818, -7.9326773, 7.2932010
3: -5.5534630, 3.5566719, -3.5542996, 2.3900843, -7.9435472, 7.1109715
4: -6.1073847, 4.9166131, -3.8976185, 3.2306137, -9.3379984, 8.8142319
5: -5.0372090, 3.6134515, -3.2355239, 2.4604266, -7.4976358, 6.8489752
6: -4.6140518, 4.7868385, -2.9296153, 3.1374884, -7.7515402, 7.7164536
7: -5.1555352, 4.8287797, -3.2639117, 3.2177010, -8.3732357, 8.0926914
8: -6.6117382, 4.0826793, -4.2286425, 2.7429824, -9.3547211, 8.3113213
9: -4.6413403, 4.7643595, -3.0137095, 3.1279736, -7.7693138, 7.7780690

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418522, upper bound: 10.8418324
time: 4.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418522, upper bound: 10.8418572
time: 3.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -7.0536499, 3.5796938, -4.3232837, 2.2495770, -9.3032265, 7.9029775
1: -4.6994185, 4.5330520, -2.8660905, 2.7744200, -7.4738388, 7.3991423
2: -6.0258284, 4.5615811, -3.4693112, 2.8490620, -8.8748903, 8.0308924
3: -6.3836412, 4.0328093, -3.7600482, 2.5127006, -8.8963413, 7.7928572
4: -7.0273933, 5.6206145, -4.1214771, 3.4042454, -10.4316387, 9.7420921
5: -5.7734571, 4.0772533, -3.4232435, 2.5844877, -8.3579445, 7.5004969
6: -5.2952971, 5.4603429, -3.1113131, 3.3109927, -8.6062899, 8.5716562
7: -5.9174185, 5.4829254, -3.4634194, 3.3865869, -9.3040056, 8.9463444
8: -7.5933938, 4.6305227, -4.4759722, 2.8853757, -10.4787693, 9.1064949
9: -5.3084650, 5.4352283, -3.1830230, 3.2970574, -8.6055222, 8.6182518

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418499
time: 2.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418504
time: 2.59 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.3035374, 3.2350001, -4.5737600, 2.3722558, -8.6757927, 7.8087602
1: -4.1876497, 4.0442019, -3.0239103, 2.9278069, -7.1154566, 7.0681124
2: -5.3190699, 4.0931892, -3.6911952, 3.0002885, -8.3193588, 7.7843847
3: -5.6523509, 3.6146121, -3.9849739, 2.6473815, -8.2997322, 7.5995860
4: -6.2170110, 5.0009379, -4.3721471, 3.5944028, -9.8114138, 9.3730850
5: -5.1266870, 3.6714411, -3.6283181, 2.7238834, -7.8505707, 7.2997589
6: -4.6981716, 4.8688412, -3.3134761, 3.5012732, -8.1994448, 8.1823177
7: -5.2484760, 4.9088030, -3.6775608, 3.5702150, -8.8186913, 8.5863638
8: -6.7296057, 4.1498117, -4.7464890, 3.0428677, -9.7724733, 8.8963013
9: -4.7216444, 4.8454070, -3.3666611, 3.4818964, -8.2035408, 8.2120686

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418524, upper bound: 10.8418328
time: 3.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418524, upper bound: 10.8418566
time: 3.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.4363551, 2.2941785, -7.1831694, 3.5618410, -7.9981961, 9.4773483
1: -2.9473343, 2.8543468, -4.7757864, 4.6100812, -7.5574155, 7.6301332
2: -3.5812991, 2.9206049, -6.1145463, 4.6126161, -8.1939154, 9.0351515
3: -3.8817768, 2.5782537, -6.4885087, 4.0904360, -7.9722128, 9.0667629
4: -4.2560139, 3.5028915, -7.1532965, 5.7185388, -9.9745522, 10.6561880
5: -3.5264707, 2.6405244, -5.8504333, 4.1389132, -7.6653838, 8.4909573
6: -3.1982424, 3.4014854, -5.3911300, 5.5340014, -8.7322435, 8.7926159
7: -3.5673370, 3.4754684, -5.9718032, 5.5479827, -9.1153202, 9.4472713
8: -4.6158886, 2.9573741, -7.7131758, 4.7047024, -9.3205910, 10.6705494
9: -3.2785830, 3.3911195, -5.3791475, 5.5069861, -8.7855692, 8.7702675

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418481, upper bound: 10.8418156
time: 3.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418632, upper bound: 10.8418390
time: 3.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.9391317, 2.5429313, -7.2944937, 3.6169162, -8.5560474, 9.8374252
1: -3.2681527, 3.1653872, -4.8493166, 4.6799974, -7.9481502, 8.0147038
2: -4.0330329, 3.2272940, -6.2168751, 4.6817045, -8.7147369, 9.4441690
3: -4.3385234, 2.8510425, -6.5926771, 4.1515579, -8.4900818, 9.4437199
4: -4.7646532, 3.8898122, -7.2683096, 5.8077912, -10.5724449, 11.1581221
5: -3.9422717, 2.9213891, -5.9449763, 4.2004623, -8.1427345, 8.8663654
6: -3.6045351, 3.7865639, -5.4802957, 5.6207547, -9.2252903, 9.2668591
7: -4.0036316, 3.8488936, -6.0700836, 5.6322737, -9.6359053, 9.9189777
8: -5.1654863, 3.2749743, -7.8374815, 4.7756057, -9.9410915, 11.1124554
9: -3.6515489, 3.7677307, -5.4638486, 5.5924292, -9.2439785, 9.2315788

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418473, upper bound: 10.8418177
time: 3.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418630, upper bound: 10.8418383
time: 3.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.4363551, 2.2941785, -8.4063034, 4.1867757, -8.6231308, 10.7004814
1: -2.9473343, 2.8543468, -5.5974698, 5.3936982, -8.3410320, 8.4518166
2: -3.5812991, 2.9206049, -7.2616711, 5.3845429, -8.9658422, 10.1822758
3: -3.8817768, 2.5782537, -7.6575980, 4.7731867, -8.6549635, 10.2358513
4: -4.2560139, 3.5028915, -8.4448643, 6.7139983, -10.9700127, 11.9477558
5: -3.5264707, 2.6405244, -6.9074397, 4.8141298, -8.3406010, 9.5479641
6: -3.1982424, 3.4014854, -6.3783121, 6.5026217, -9.7008638, 9.7797976
7: -3.5673370, 3.4754684, -7.0817451, 6.4919119, -10.0592489, 10.5572138
8: -4.6158886, 2.9573741, -9.1052437, 5.4886684, -10.1045570, 12.0626183
9: -3.2785830, 3.3911195, -6.3324981, 6.4655943, -9.7441769, 9.7236176

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8417927
time: 4.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418197
time: 4.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.9391317, 2.5429313, -8.5158081, 4.2406883, -9.1798201, 11.0587397
1: -3.2681527, 3.1653872, -5.6697941, 5.4626598, -8.7308121, 8.8351812
2: -4.0330329, 3.2272940, -7.3622427, 5.4524031, -9.4854355, 10.5895367
3: -4.3385234, 2.8510425, -7.7599845, 4.8333769, -9.1718998, 10.6110268
4: -4.7646532, 3.8898122, -8.5583420, 6.8016653, -11.5663185, 12.4481544
5: -3.9422717, 2.9213891, -7.0002441, 4.8745852, -8.8168564, 9.9216328
6: -3.6045351, 3.7865639, -6.4661469, 6.5877600, -10.1922951, 10.2527103
7: -4.0036316, 3.8488936, -7.1784267, 6.5749259, -10.5785580, 11.0273209
8: -5.1654863, 3.2749743, -9.2273874, 5.5584068, -10.7238932, 12.5023613
9: -3.6515489, 3.7677307, -6.4158554, 6.5496240, -10.2011728, 10.1835861

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8417947
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418487, upper bound: 10.8418205
time: 5.23 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.8017673, 2.9883249, -7.0759978, 3.5092070, -9.3109741, 10.0643225
1: -3.8553381, 3.7276177, -4.7073975, 4.5455618, -8.4008999, 8.4350147
2: -4.8563447, 3.7813106, -6.0196123, 4.5477648, -9.4041100, 9.8009224
3: -5.1822128, 3.3387408, -6.3927507, 4.0333118, -9.2155247, 9.7314911
4: -5.6956167, 4.5993204, -7.0481076, 5.6353431, -11.3309593, 11.6474285
5: -4.7013950, 3.3939152, -5.7626781, 4.0782213, -8.7796164, 9.1565933
6: -4.2957773, 4.4764175, -5.3046069, 5.4519091, -9.7476864, 9.7810249
7: -4.8043327, 4.5272179, -5.8814240, 5.4699521, -10.2742844, 10.4086418
8: -6.1682587, 3.8293390, -7.5977139, 4.6369023, -10.8051605, 11.4270535
9: -4.3392000, 4.4593315, -5.3016343, 5.4281964, -9.7673969, 9.7609653

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418428, upper bound: 10.8418200
time: 6.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418428, upper bound: 10.8418196
time: 5.60 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.2515249, 3.2114227, -7.1860299, 3.5634263, -9.8149509, 10.3974524
1: -4.1479044, 4.0061531, -4.7801180, 4.6148968, -8.7628012, 8.7862711
2: -5.2643766, 4.0578709, -6.1207371, 4.6159511, -9.8803272, 10.1786079
3: -5.5930719, 3.5831327, -6.4957218, 4.0938120, -9.6868839, 10.0788546
4: -6.1523428, 4.9524851, -7.1622038, 5.7234879, -11.8758307, 12.1146889
5: -5.0758376, 3.6456614, -5.8559375, 4.1390057, -9.2148438, 9.5015984
6: -4.6588507, 4.8257275, -5.3929491, 5.5375166, -10.1963673, 10.2186766
7: -5.1959877, 4.8649998, -5.9785848, 5.5532975, -10.7492847, 10.8435841
8: -6.6619458, 4.1159916, -7.7205405, 4.7070088, -11.3689547, 11.8365326
9: -4.6745152, 4.7989016, -5.3854151, 5.5126467, -10.1871624, 10.1843166

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418434, upper bound: 10.8418204
time: 4.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418434, upper bound: 10.8418206
time: 5.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -5.2429228, 2.6708193, -4.5281148, 2.3478050, -7.5907278, 7.1989341
1: -3.4818058, 3.3723457, -2.9972601, 2.9021146, -6.3839207, 6.3696060
2: -4.3270836, 3.4193678, -3.6534090, 2.9736288, -7.3007126, 7.0727768
3: -4.6503496, 3.0240979, -3.9482307, 2.6238804, -7.2742300, 6.9723287
4: -5.1110449, 4.1490097, -4.3309603, 3.5625110, -8.6735554, 8.4799700
5: -4.2119427, 3.0790157, -3.5933514, 2.6969986, -6.9089413, 6.6723671
6: -3.8407640, 4.0271907, -3.2755480, 3.4675922, -7.3083563, 7.3027387
7: -4.2794285, 4.0874143, -3.6405516, 3.5383482, -7.8177767, 7.7279658
8: -5.5296926, 3.4690039, -4.7009468, 3.0143611, -8.5440540, 8.1699505
9: -3.8993592, 4.0155458, -3.3360610, 3.4506650, -7.3500242, 7.3516068

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418649, upper bound: 10.8418494
time: 3.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418683, upper bound: 10.8418705
time: 2.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -5.7970800, 2.9468598, -4.6422119, 2.4029562, -8.2000360, 7.5890718
1: -3.8412936, 3.7158566, -3.0711801, 2.9741533, -6.8154469, 6.7870369
2: -4.8287902, 3.7605157, -3.7573543, 3.0436966, -7.8724871, 7.5178699
3: -5.1567841, 3.3257680, -4.0547671, 2.6864259, -7.8432102, 7.3805351
4: -5.6744604, 4.5828371, -4.4493599, 3.6517587, -9.3262196, 9.0321970
5: -4.6730113, 3.3894963, -3.6895833, 2.7593439, -7.4323549, 7.0790796
6: -4.2889624, 4.4556608, -3.3664873, 3.5553055, -7.8442678, 7.8221483
7: -4.7626987, 4.5027723, -3.7408018, 3.6240497, -8.3867483, 8.2435741
8: -6.1389070, 3.8216777, -4.8278394, 3.0863614, -9.2252684, 8.6495171
9: -4.3123202, 4.4344792, -3.4227862, 3.5376637, -7.8499842, 7.8572655

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 181

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418637, upper bound: 10.8418483
time: 3.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418679, upper bound: 10.8418705
time: 3.16 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -5.2429228, 2.6708193, -5.8381538, 3.0136678, -8.2565908, 8.5089731
1: -3.4818058, 3.3723457, -3.8688030, 3.7414684, -7.2232742, 7.2411489
2: -4.3270836, 3.4193678, -4.8781281, 3.8005202, -8.1276035, 8.2974958
3: -4.6503496, 3.0240979, -5.1977782, 3.3544185, -8.0047684, 8.2218761
4: -5.1110449, 4.1490097, -5.7150898, 4.6148157, -9.7258606, 9.8640995
5: -4.2119427, 3.0790157, -4.7220564, 3.4189494, -7.6308918, 7.8010721
6: -3.8407640, 4.0271907, -4.3276453, 4.4995317, -8.3402958, 8.3548355
7: -4.2794285, 4.0874143, -4.8289175, 4.5502510, -8.8296795, 8.9163322
8: -5.5296926, 3.4690039, -6.1932259, 3.8506880, -9.3803806, 9.6622295
9: -3.8993592, 4.0155458, -4.3553724, 4.4780045, -8.3773632, 8.3709183

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418553, upper bound: 10.8418316
time: 4.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418654, upper bound: 10.8418567
time: 3.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -5.7970800, 2.9468598, -5.9493542, 3.0679665, -8.8650465, 8.8962135
1: -3.8412936, 3.7158566, -3.9418244, 3.8115768, -7.6528702, 7.6576810
2: -4.8287902, 3.7605157, -4.9798942, 3.8694570, -8.6982470, 8.7404099
3: -5.1567841, 3.3257680, -5.3016806, 3.4155476, -8.5723314, 8.6274490
4: -5.6744604, 4.5828371, -5.8307056, 4.7029495, -10.3774099, 10.4135427
5: -4.6730113, 3.3894963, -4.8160362, 3.4801855, -8.1531963, 8.2055321
6: -4.2889624, 4.4556608, -4.4164448, 4.5855703, -8.8745327, 8.8721056
7: -4.7626987, 4.5027723, -4.9268508, 4.6346731, -9.3973713, 9.4296227
8: -6.1389070, 3.8216777, -6.3175459, 3.9212923, -10.0601997, 10.1392231
9: -4.3123202, 4.4344792, -4.4398451, 4.5634990, -8.8758192, 8.8743248

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418557, upper bound: 10.8418319
time: 4.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418649, upper bound: 10.8418563
time: 3.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.8745928, 3.9599338, -3.7683260, 1.9843509, -9.8589439, 7.7282600
1: -5.2483463, 5.0572023, -2.5147884, 2.4352677, -7.6836138, 7.5719910
2: -6.7817445, 5.0801725, -2.9737923, 2.5151589, -9.2969036, 8.0539646
3: -7.1643810, 4.4868784, -3.2602506, 2.2127728, -9.3771534, 7.7471290
4: -7.8932819, 6.2872767, -3.5907946, 2.9800816, -10.8733635, 9.8780708
5: -6.4662533, 4.5314755, -2.9650843, 2.2847357, -8.7509890, 7.4965601
6: -5.9580126, 6.0964594, -2.6661525, 2.8911219, -8.8491344, 8.7626114
7: -6.6401615, 6.1011310, -2.9814601, 2.9765532, -9.6167145, 9.0825911
8: -8.5145302, 5.1584072, -3.8714283, 2.5376873, -11.0522175, 9.0298357
9: -5.9409900, 6.0694809, -2.7709219, 2.8908579, -8.8318481, 8.8404026

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 135
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 181

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418215, upper bound: 10.8418271
time: 3.76 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418234, upper bound: 10.8418423
time: 3.32 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 8.44 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418696, upper bound: 10.8418487
IS_A1_B1_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418754, upper bound: 10.8418701
IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418690, upper bound: 10.8418479
IS_A1_B1_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418744, upper bound: 10.8418706
IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418581, upper bound: 10.8418309
IS_A1_B1_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418696, upper bound: 10.8418556
IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418581, upper bound: 10.8418317
IS_A1_B1_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418694, upper bound: 10.8418565
IS_A1_B1_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418332, upper bound: 10.8418502
IS_A1_B1_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418332, upper bound: 10.8418512
IS_A1_B1_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418522, upper bound: 10.8418324
IS_A1_B1_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418522, upper bound: 10.8418572
IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418499
IS_A1_B1_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418331, upper bound: 10.8418504
IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418524, upper bound: 10.8418328
IS_A1_B1_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418524, upper bound: 10.8418566
IS_A1_B1_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418481, upper bound: 10.8418156
IS_A1_B1_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418632, upper bound: 10.8418390
IS_A1_B1_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418473, upper bound: 10.8418177
IS_A1_B1_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418630, upper bound: 10.8418383
IS_A1_B1_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8417927
IS_A1_B1_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418197
IS_A1_B1_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418311, upper bound: 10.8417947
IS_A1_B1_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418487, upper bound: 10.8418205
IS_A1_B1_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418428, upper bound: 10.8418200
IS_A1_B1_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418428, upper bound: 10.8418196
IS_A1_B1_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418434, upper bound: 10.8418204
IS_A1_B1_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418434, upper bound: 10.8418206
IS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418649, upper bound: 10.8418494
IS_A1_B1_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418683, upper bound: 10.8418705
IS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418637, upper bound: 10.8418483
IS_A1_B1_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418679, upper bound: 10.8418705
IS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418553, upper bound: 10.8418316
IS_A1_B1_A2_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418654, upper bound: 10.8418567
IS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418557, upper bound: 10.8418319
IS_A1_B1_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418649, upper bound: 10.8418563
IS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418215, upper bound: 10.8418271
IS_A1_B1_A2_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 8.44
Output dim: 0, lower bound: -10.8418234, upper bound: 10.8418423
IS_A1_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 8.44
Output dim: 0, lower bound: -10.8418536, upper bound: 10.8418568
IS_A1_B1_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 8.44
Output dim: 0, lower bound: -10.8418325, upper bound: 10.8418505
IS_A1_B1_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 8.44
Output dim: 0, lower bound: -10.8418536, upper bound: 10.8418563
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=13.22586727142334
rel_dist={0: [-10.84189645044881, 10.84189726697796]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 1421.03 seconds
