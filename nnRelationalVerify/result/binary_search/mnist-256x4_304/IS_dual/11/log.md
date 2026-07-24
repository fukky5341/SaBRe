## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 7.6259627037
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305)
1: (-4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668)
2: (-6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820)
3: (-5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251)
4: (-6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038)
5: (-4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198)
6: (-4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974)
7: (-5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517)
8: (-6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594)
9: (-4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873)

## BASE Result
execution time: IAR + LP analysis = 1.45 + 5.70 = 7.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -7.6336112, upper bound: 7.6336109


# Binary Search by BASE starts (time budget: 1992.85 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=9.388381958007812
rel_dist={2: [-7.633606580437654, 7.633606672931759]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=9.388381958007812
rel_dist={2: [-7.633595790761577, 7.633596173769259]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=9.388381958007812
rel_dist={2: [-7.63358222082411, 7.633582568333054]}

## Binary Search Result
Binary search time: 18.92 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1973.93 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315509, upper bound: 7.6327101
time: 2.39 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314567, upper bound: 7.6314568
time: 1.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 4.51 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 4.51
Output dim: 2, lower bound: -7.6315509, upper bound: 7.6327101
IS_B2, status: Status.UNKNOWN, split count: 1, time: 4.51
Output dim: 2, lower bound: -7.6314567, upper bound: 7.6314568

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -5.1286879, 3.9331427, -4.6870031, 3.6019335, -8.7306213, 8.6201458
1: -4.0091529, 3.6131139, -3.6542621, 3.3308623, -7.3400154, 7.2673759
2: -6.6639881, 2.7243943, -6.0363026, 2.4904585, -9.1544466, 8.7606964
3: -5.8328662, 2.9040592, -5.3217468, 2.6795204, -8.5123863, 8.2258062
4: -6.1130657, 3.8349376, -5.6128726, 3.5257287, -9.6387939, 9.4478102
5: -4.6064701, 4.0478497, -4.2223544, 3.6964202, -8.3028908, 8.2702045
6: -4.8427486, 4.1230493, -4.4340792, 3.7822859, -8.6250343, 8.5571289
7: -5.5768681, 4.1297841, -5.0997634, 3.7987463, -9.3756142, 9.2295475
8: -6.3826962, 3.8029628, -5.8376360, 3.4970667, -9.8797626, 9.6405983
9: -4.5053129, 5.1798744, -4.1510506, 4.7564611, -9.2617741, 9.3309250

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303598, upper bound: 7.6322762
time: 3.50 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303199, upper bound: 7.6318296
time: 2.72 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -4.8434224, 3.7171459, -6.2712379, 4.7431469, -9.5865688, 9.9883842
1: -3.7794468, 3.4320366, -4.9379225, 4.3718748, -8.1513214, 8.3699589
2: -6.2518530, 2.5630457, -8.1674137, 3.0112796, -9.2631321, 10.7304592
3: -5.5055265, 2.7580702, -7.2472944, 3.4771147, -8.9826412, 10.0053644
4: -5.7941098, 3.6333003, -7.5289741, 4.6118517, -10.4059620, 11.1622744
5: -4.3576050, 3.8170950, -5.5662413, 4.8327031, -9.1903076, 9.3833361
6: -4.5793586, 3.9034712, -5.9290524, 5.0279045, -9.6072636, 9.8325233
7: -5.2679534, 3.9155214, -6.7920446, 4.9908943, -10.2588482, 10.7075663
8: -6.0301533, 3.6022162, -7.8048959, 4.5699730, -10.6001263, 11.4071121
9: -4.2779164, 4.9090796, -5.4692111, 6.3742647, -10.6521816, 10.3782902

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311912, upper bound: 7.6302674
time: 3.27 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302139
time: 1.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.61 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 6.61
Output dim: 2, lower bound: -7.6303598, upper bound: 7.6322762
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 6.61
Output dim: 2, lower bound: -7.6303199, upper bound: 7.6318296
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 6.61
Output dim: 2, lower bound: -7.6311912, upper bound: 7.6302674
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 6.61
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302139

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -5.1286879, 3.9331427, -3.8411446, 2.9820642, -8.1107521, 7.7742872
1: -4.0091529, 3.6131139, -2.9738793, 2.7744110, -6.7835636, 6.5869932
2: -6.6639881, 2.7243943, -4.8838549, 2.1416857, -8.8056736, 7.6082492
3: -5.8328662, 2.9040592, -4.3142233, 2.2556181, -8.0884838, 7.2182827
4: -6.1130657, 3.8349376, -4.6136365, 2.9400029, -9.0530682, 8.4485741
5: -4.6064701, 4.0478497, -3.4920440, 3.0690916, -7.6755619, 7.5398936
6: -4.8427486, 4.1230493, -3.6465309, 3.1185267, -7.9612751, 7.7695799
7: -5.5768681, 4.1297841, -4.1928883, 3.1591718, -8.7360401, 8.3226719
8: -6.3826962, 3.8029628, -4.7903576, 2.9237399, -9.3064365, 8.5933208
9: -4.5053129, 5.1798744, -3.4538531, 3.9136477, -8.4189606, 8.6337280

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303205, upper bound: 7.6318296
time: 4.13 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303205, upper bound: 7.6318297
time: 4.46 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -5.0093522, 3.8459010, -7.8474908, 5.9147978, -10.9241505, 11.6933918
1: -3.9128380, 3.5350213, -6.2018151, 5.3870506, -9.2998886, 9.7368364
2: -6.5023088, 2.6747284, -10.2793226, 3.8671761, -10.3694849, 12.9540510
3: -5.6912117, 2.8437872, -9.0614338, 4.2798848, -9.9710960, 11.9052210
4: -5.9729309, 3.7523787, -9.3044252, 5.7151203, -11.6880512, 13.0568037
5: -4.5041556, 3.9591372, -6.9447851, 6.0614977, -10.5656528, 10.9039221
6: -4.7312536, 4.0299139, -7.3762808, 6.2456083, -10.9768620, 11.4061947
7: -5.4488769, 4.0398254, -8.4882631, 6.1739264, -11.6228027, 12.5280886
8: -6.2352085, 3.7215319, -9.7457047, 5.6260433, -11.8612518, 13.4672365
9: -4.4072256, 5.0615902, -6.7400947, 7.8452983, -12.2525234, 11.8016853

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281604, upper bound: 7.6308942
time: 2.19 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303189, upper bound: 7.6318072
time: 2.54 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -3.9922912, 3.0941501, -6.2712379, 4.7431469, -8.7354383, 9.3653879
1: -3.0935822, 2.8727627, -4.9379225, 4.3718748, -7.4654570, 7.8106852
2: -5.0954485, 2.2125888, -8.1674137, 3.0112796, -8.1067276, 10.3800030
3: -4.4921103, 2.3302932, -7.2472944, 3.4771147, -7.9692249, 9.5775871
4: -4.7896214, 3.0441232, -7.5289741, 4.6118517, -9.4014730, 10.5730972
5: -3.6236882, 3.1859820, -5.5662413, 4.8327031, -8.4563913, 8.7522230
6: -3.7858045, 3.2366152, -5.9290524, 5.0279045, -8.8137093, 9.1656675
7: -4.3554506, 3.2723999, -6.7920446, 4.9908943, -9.3463449, 10.0644445
8: -4.9768515, 3.0256767, -7.8048959, 4.5699730, -9.5468245, 10.8305721
9: -3.5770454, 4.0620775, -5.4692111, 6.3742647, -9.9513102, 9.5312881

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 173

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302144, upper bound: 7.6302139
time: 2.03 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302144, upper bound: 7.6302144
time: 2.10 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -8.0170670, 6.0385313, -6.1540556, 4.6580434, -12.6751099, 12.1925869
1: -6.3375893, 5.4964390, -4.8436136, 4.2948666, -10.6324558, 10.3400526
2: -10.5062456, 3.9429576, -8.0082445, 2.9648843, -13.4711304, 11.9512024
3: -9.2603989, 4.3651304, -7.1074286, 3.4181731, -12.6785717, 11.4725590
4: -9.5007601, 5.8316431, -7.3904686, 4.5310082, -14.0317688, 13.2221117
5: -7.0902209, 6.1901631, -5.4659543, 4.7457986, -11.8360195, 11.6561174
6: -7.5332799, 6.3772149, -5.8194675, 4.9362507, -12.4695301, 12.1966820
7: -8.6691980, 6.3001738, -6.6665635, 4.9027033, -13.5719013, 12.9667377
8: -9.9540081, 5.7397442, -7.6600680, 4.4903469, -14.4443550, 13.3998127
9: -6.8775125, 8.0104961, -5.3725309, 6.2574883, -13.1350002, 13.3830271

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302144
time: 2.28 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302144
time: 2.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 6.44 seconds
IS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 2, lower bound: -7.6303205, upper bound: 7.6318296
IS_B1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 2, lower bound: -7.6303205, upper bound: 7.6318297
IS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 2, lower bound: -7.6281604, upper bound: 7.6308942
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 2, lower bound: -7.6303189, upper bound: 7.6318072
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 2, lower bound: -7.6302144, upper bound: 7.6302139
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 2, lower bound: -7.6302144, upper bound: 7.6302144
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302144
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 6.44
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302144

## BFS IS instance: IS_B1_B1_A1

### Backsubstitution after applying IS history:
0: -4.2686481, 3.3034801, -3.8411446, 2.9820642, -7.2507124, 7.1446247
1: -3.3155420, 3.0498059, -2.9738793, 2.7744110, -6.0899529, 6.0236855
2: -5.4949217, 2.3689001, -4.8838549, 2.1416857, -7.6366072, 7.2527552
3: -4.8110600, 2.4704089, -4.3142233, 2.2556181, -7.0666780, 6.7846322
4: -5.1011324, 3.2392945, -4.6136365, 2.9400029, -8.0411358, 7.8529310
5: -3.8673596, 3.4099646, -3.4920440, 3.0690916, -6.9364510, 6.9020085
6: -4.0397010, 3.4505873, -3.6465309, 3.1185267, -7.1582279, 7.0971184
7: -4.6541395, 3.4808147, -4.1928883, 3.1591718, -7.8133116, 7.6737032
8: -5.3186646, 3.2162716, -4.7903576, 2.9237399, -8.2424049, 8.0066290
9: -3.7986977, 4.3246951, -3.4538531, 3.9136477, -7.7123451, 7.7785482

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5646487, upper bound: 7.6295464
time: 3.44 seconds

## Relational analysis of IS_B1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303598, upper bound: 7.6322763
time: 4.10 seconds

## BFS IS instance: IS_B1_B1_A2

### Backsubstitution after applying IS history:
0: -8.3132782, 6.2759089, -3.8411446, 2.9820642, -11.2953424, 10.1170540
1: -6.5844011, 5.6741643, -2.9738793, 2.7744110, -9.3588123, 8.6480436
2: -10.9843225, 4.1436543, -4.8838549, 2.1416857, -13.1260080, 9.0275097
3: -9.5884886, 4.5238266, -4.3142233, 2.2556181, -11.8441067, 8.8380499
4: -9.8131104, 6.0509901, -4.6136365, 2.9400029, -12.7531128, 10.6646271
5: -7.3376122, 6.4407911, -3.4920440, 3.0690916, -10.4067039, 9.9328346
6: -7.8093801, 6.6083603, -3.6465309, 3.1185267, -10.9279070, 10.2548914
7: -8.9933872, 6.5240011, -4.1928883, 3.1591718, -12.1525593, 10.7168894
8: -10.3325996, 5.9963574, -4.7903576, 2.9237399, -13.2563400, 10.7867146
9: -7.1123838, 8.2964840, -3.4538531, 3.9136477, -11.0260315, 11.7503376

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_A2_A1

### Relational analysis result of IS_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4463669, upper bound: 7.6309187
time: 5.18 seconds

## Relational analysis of IS_B1_B1_A2_A2

### Relational analysis result of IS_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295074, upper bound: 7.6317537
time: 2.73 seconds

## BFS IS instance: IS_B1_B2_A1

### Backsubstitution after applying IS history:
0: -5.5322533, 4.2217755, -7.7243218, 5.8250055, -11.3572588, 11.9460974
1: -4.3243022, 3.8728077, -6.1014423, 5.3060460, -9.6303482, 9.9742498
2: -7.2015219, 2.8409548, -10.1169844, 3.8134727, -11.0149946, 12.9579391
3: -6.3346186, 3.0989356, -8.9165468, 4.2170033, -10.5516224, 12.0154819
4: -6.6107616, 4.1033335, -9.1610870, 5.6294408, -12.2402020, 13.2644205
5: -4.9494290, 4.3341074, -6.8386002, 5.9702997, -10.9197292, 11.1727076
6: -5.2255716, 4.4442544, -7.2617373, 6.1499901, -11.3755617, 11.7059917
7: -6.0116749, 4.4208307, -8.3567677, 6.0803251, -12.0920000, 12.7775984
8: -6.8852110, 4.0637231, -9.5940552, 5.5424399, -12.4276505, 13.6577778
9: -4.8413191, 5.5929785, -6.6392221, 7.7243199, -12.5656395, 12.2322006

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_B2_A1_A1

### Relational analysis result of IS_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281605, upper bound: 7.6308940
time: 2.39 seconds

## Relational analysis of IS_B1_B2_A1_A2

### Relational analysis result of IS_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281605, upper bound: 7.6308943
time: 2.76 seconds

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.8379469, 3.7174051, -7.8474908, 5.9147978, -10.7527447, 11.5648956
1: -3.7729442, 3.4230950, -6.2018151, 5.3870506, -9.1599951, 9.6249104
2: -6.2596750, 2.5884390, -10.2793226, 3.8671761, -10.1268511, 12.8677616
3: -5.4938083, 2.7551901, -9.0614338, 4.2798848, -9.7736931, 11.8166237
4: -5.7777557, 3.6306758, -9.3044252, 5.7151203, -11.4928761, 12.9351006
5: -4.3559961, 3.8271627, -6.9447851, 6.0614977, -10.4174938, 10.7719479
6: -4.5725079, 3.8970954, -7.3762808, 6.2456083, -10.8181162, 11.2733765
7: -5.2640791, 3.9076703, -8.4882631, 6.1739264, -11.4380054, 12.3959332
8: -6.0227752, 3.5968170, -9.7457047, 5.6260433, -11.6488190, 13.3425217
9: -4.2680917, 4.8936806, -6.7400947, 7.8452983, -12.1133900, 11.6337757

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303195, upper bound: 7.6318070
time: 2.29 seconds

## Relational analysis of IS_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303195, upper bound: 7.6318072
time: 4.03 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.9922912, 3.0941501, -5.3819876, 4.0995798, -8.0918713, 8.4761372
1: -3.0935822, 2.8727627, -4.2226143, 3.7880433, -6.8816252, 7.0953770
2: -5.0954485, 2.2125888, -6.9588943, 2.6669736, -7.7624221, 9.1714830
3: -4.4921103, 2.3302932, -6.1860161, 3.0297389, -7.5218492, 8.5163097
4: -4.7896214, 3.0441232, -6.4774103, 3.9987886, -8.7884102, 9.5215340
5: -3.6236882, 3.1859820, -4.8070345, 4.1748304, -7.7985187, 7.9930162
6: -3.7858045, 3.2366152, -5.0985279, 4.3322272, -8.1180315, 8.3351431
7: -4.3554506, 3.2723999, -5.8421459, 4.3220015, -8.6774521, 9.1145458
8: -4.9768515, 3.0256767, -6.7065644, 3.9671144, -8.9439659, 9.7322407
9: -3.5770454, 4.0620775, -4.7365842, 5.4877405, -9.0647860, 8.7986622

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6297276, upper bound: 7.4462588
time: 2.48 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294102
time: 2.45 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.9922912, 3.0941501, -9.1081705, 6.7988100, -10.7911015, 12.2023201
1: -3.0935822, 2.8727627, -7.2136149, 6.2370625, -9.3306446, 10.0863781
2: -5.0954485, 2.2125888, -12.0102386, 4.1338797, -9.2293282, 14.2228279
3: -4.4921103, 2.3302932, -10.6376057, 4.9016495, -9.3937597, 12.9678993
4: -4.7896214, 3.0441232, -10.8863506, 6.5671210, -11.3567429, 13.9304733
5: -3.6236882, 3.1859820, -7.9977584, 6.9330654, -10.5567532, 11.1837406
6: -3.7858045, 3.2366152, -8.5807638, 7.2531233, -11.0389280, 11.8173790
7: -4.3554506, 3.2723999, -9.8344851, 7.1246367, -11.4800873, 13.1068850
8: -4.9768515, 3.0256767, -11.3155050, 6.4960618, -11.4729137, 14.3411818
9: -3.5770454, 4.0620775, -7.8206162, 9.1847649, -12.7618103, 11.8826942

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6297276, upper bound: 7.4462585
time: 4.18 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294102
time: 2.69 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.0170670, 6.0385313, -5.3819876, 4.0995798, -12.1166468, 11.4205189
1: -6.3375893, 5.4964390, -4.2226143, 3.7880433, -10.1256323, 9.7190533
2: -10.5062456, 3.9429576, -6.9588943, 2.6669736, -13.1732197, 10.9018517
3: -9.2603989, 4.3651304, -6.1860161, 3.0297389, -12.2901382, 10.5511465
4: -9.5007601, 5.8316431, -6.4774103, 3.9987886, -13.4995489, 12.3090534
5: -7.0902209, 6.1901631, -4.8070345, 4.1748304, -11.2650509, 10.9971981
6: -7.5332799, 6.3772149, -5.0985279, 4.3322272, -11.8655071, 11.4757423
7: -8.6691980, 6.3001738, -5.8421459, 4.3220015, -12.9911995, 12.1423197
8: -9.9540081, 5.7397442, -6.7065644, 3.9671144, -13.9211226, 12.4463081
9: -6.8775125, 8.0104961, -4.7365842, 5.4877405, -12.3652534, 12.7470798

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4461945, upper bound: 7.6284031
time: 3.50 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6293501, upper bound: 7.6293504
time: 2.61 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.0170670, 6.0385313, -9.1081705, 6.7988100, -14.8158770, 15.1467018
1: -6.3375893, 5.4964390, -7.2136149, 6.2370625, -12.5746517, 12.7100544
2: -10.5062456, 3.9429576, -12.0102386, 4.1338797, -14.6401253, 15.9531965
3: -9.2603989, 4.3651304, -10.6376057, 4.9016495, -14.1620483, 15.0027361
4: -9.5007601, 5.8316431, -10.8863506, 6.5671210, -16.0678806, 16.7179947
5: -7.0902209, 6.1901631, -7.9977584, 6.9330654, -14.0232868, 14.1879215
6: -7.5332799, 6.3772149, -8.5807638, 7.2531233, -14.7864037, 14.9579792
7: -8.6691980, 6.3001738, -9.8344851, 7.1246367, -15.7938347, 16.1346588
8: -9.9540081, 5.7397442, -11.3155050, 6.4960618, -16.4500694, 17.0552483
9: -6.8775125, 8.0104961, -7.8206162, 9.1847649, -16.0622768, 15.8311119

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302144, upper bound: 7.6302144
time: 1.95 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302144, upper bound: 7.6302139
time: 2.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.85 seconds
IS_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.5646487, upper bound: 7.6295464
IS_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6303598, upper bound: 7.6322763
IS_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.4463669, upper bound: 7.6309187
IS_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6295074, upper bound: 7.6317537
IS_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6281605, upper bound: 7.6308940
IS_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6281605, upper bound: 7.6308943
IS_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6303195, upper bound: 7.6318070
IS_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6303195, upper bound: 7.6318072
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6297276, upper bound: 7.4462588
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294102
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6297276, upper bound: 7.4462585
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294102
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.4461945, upper bound: 7.6284031
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6293501, upper bound: 7.6293504
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6302144, upper bound: 7.6302144
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.85
Output dim: 2, lower bound: -7.6302144, upper bound: 7.6302139

## BFS IS instance: IS_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.1318207, 3.2026668, -5.8788328, 4.4753709, -8.6071911, 9.0814991
1: -3.2052956, 2.9599786, -4.5774941, 4.1051755, -7.3104711, 7.5374727
2: -5.3070688, 2.3115497, -7.6721725, 3.0676310, -8.3746996, 9.9837227
3: -4.6482587, 2.4018841, -6.7199149, 3.2587385, -7.9069972, 9.1217995
4: -4.9395218, 3.1442196, -6.9897199, 4.3451872, -9.2847090, 10.1339397
5: -3.7489400, 3.3082638, -5.2565708, 4.6184645, -8.3674049, 8.5648346
6: -3.9124284, 3.3429971, -5.5115995, 4.7038031, -8.6162319, 8.8545971
7: -4.5073729, 3.3770695, -6.3731327, 4.6870232, -9.1943960, 9.7502022
8: -5.1489520, 3.1224020, -7.3193836, 4.2829409, -9.4318924, 10.4417858
9: -3.6859000, 4.1878457, -5.1280084, 5.9031305, -9.5890303, 9.3158541

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_B1_A1_B1_A1

### Relational analysis result of IS_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6301716, upper bound: 7.6306046
time: 4.59 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2

### Relational analysis result of IS_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6301705, upper bound: 7.6306482
time: 6.80 seconds

## BFS IS instance: IS_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.2686481, 3.3034801, -3.7711341, 2.9309092, -7.1995573, 7.0746145
1: -3.3155420, 3.0498059, -2.9179156, 2.7284474, -6.0439892, 5.9677215
2: -5.4949217, 2.3689001, -4.7880740, 2.1122229, -7.6071444, 7.1569738
3: -4.8110600, 2.4704089, -4.2308588, 2.2210655, -7.0321255, 6.7012677
4: -5.1011324, 3.2392945, -4.5309696, 2.8919618, -7.9930944, 7.7702641
5: -3.8673596, 3.4099646, -3.4313211, 3.0169790, -6.8843384, 6.8412857
6: -4.0397010, 3.4505873, -3.5815358, 3.0640309, -7.1037321, 7.0321231
7: -4.6541395, 3.4808147, -4.1177950, 3.1064944, -7.7606339, 7.5986099
8: -5.3186646, 3.2162716, -4.7037334, 2.8767529, -8.1954174, 7.9200048
9: -3.7986977, 4.3246951, -3.3961568, 3.8438315, -7.6425295, 7.7208519

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5963078, upper bound: 7.6314723
time: 3.51 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309131, upper bound: 7.6322266
time: 3.37 seconds

## BFS IS instance: IS_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -3.4289303, 2.6780100, -2.8485129, 2.2557290, -5.6846590, 5.5265226
1: -2.6615841, 2.4935343, -2.2336926, 2.1126685, -4.7742529, 4.7272272
2: -4.3062325, 1.9688954, -3.4863071, 1.7488687, -6.0551014, 5.4552026
3: -3.8253829, 2.0555108, -3.1182642, 1.7620014, -5.5873842, 5.1737747
4: -4.1241121, 2.6504107, -3.4284332, 2.2502775, -6.3743896, 6.0788441
5: -3.1364799, 2.7556326, -2.6185107, 2.3044384, -5.4409180, 5.3741436
6: -3.2842293, 2.7937129, -2.7222340, 2.3430896, -5.6273189, 5.5159469
7: -3.7546012, 2.8442411, -3.1170912, 2.4061971, -6.1607981, 5.9613323
8: -4.2720189, 2.6389854, -3.5578825, 2.2495763, -6.5215950, 6.1968679
9: -3.1114426, 3.4878597, -2.6262088, 2.9283700, -6.0398126, 6.1140685

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_B1_B1_A2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.2769116, upper bound: 7.6299035
time: 2.37 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.2769237, upper bound: 7.6299895
time: 3.55 seconds

## BFS IS instance: IS_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -7.0665984, 5.3569503, -3.8411446, 2.9820642, -10.0486622, 9.1980953
1: -5.5791984, 4.8653116, -2.9738793, 2.7744110, -8.3536091, 7.8391910
2: -9.2917576, 3.5834966, -4.8838549, 2.1416857, -11.4334431, 8.4673519
3: -8.1203499, 3.8885853, -4.3142233, 2.2556181, -10.3759680, 8.2028084
4: -8.3703003, 5.1766109, -4.6136365, 2.9400029, -11.3103027, 9.7902470
5: -6.2708530, 5.5036459, -3.4920440, 3.0690916, -9.3399448, 8.9956894
6: -6.6540017, 5.6346269, -3.6465309, 3.1185267, -9.7725286, 9.2811575
7: -7.6584935, 5.5861216, -4.1928883, 3.1591718, -10.8176651, 9.7790098
8: -8.7819738, 5.1313066, -4.7903576, 2.9237399, -11.7057133, 9.9216642
9: -6.0929327, 7.0746951, -3.4538531, 3.9136477, -10.0065804, 10.5285482

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_B1_A2_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4200133, upper bound: 7.6279760
time: 2.90 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295074, upper bound: 7.6317533
time: 3.07 seconds

## BFS IS instance: IS_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -4.7967854, 3.6851165, -7.7243218, 5.8250055, -10.6217909, 11.4094381
1: -3.7323868, 3.3906033, -6.1014423, 5.3060460, -9.0384331, 9.4920454
2: -6.2061720, 2.5392091, -10.1169844, 3.8134727, -10.0196447, 12.6561937
3: -5.4602466, 2.7283242, -8.9165468, 4.2170033, -9.6772499, 11.6448708
4: -5.7448153, 3.5955670, -9.1610870, 5.6294408, -11.3742561, 12.7566538
5: -4.3175664, 3.7879877, -6.8386002, 5.9702997, -10.2878666, 10.6265879
6: -4.5391769, 3.8701496, -7.2617373, 6.1499901, -10.6891670, 11.1318874
7: -5.2229223, 3.8674383, -8.3567677, 6.0803251, -11.3032475, 12.2242060
8: -5.9766102, 3.5662158, -9.5940552, 5.5424399, -11.5190506, 13.1602707
9: -4.2377043, 4.8621831, -6.6392221, 7.7243199, -11.9620247, 11.5014057

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_B2_A1_A1_B1

### Relational analysis result of IS_B1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5379163, upper bound: 7.5421711
time: 3.90 seconds

## Relational analysis of IS_B1_B2_A1_A1_B2

### Relational analysis result of IS_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5942751, upper bound: 7.6303577
time: 4.13 seconds

## BFS IS instance: IS_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.5155888, 6.4035997, -7.7243218, 5.8250055, -14.3405943, 14.1279221
1: -6.7298136, 5.8137283, -6.1014423, 5.3060460, -12.0358601, 11.9151707
2: -11.1836510, 4.1420307, -10.1169844, 3.8134727, -14.9971237, 14.2590151
3: -9.8592854, 4.6113749, -8.9165468, 4.2170033, -14.0762882, 13.5279217
4: -10.0908918, 6.1718817, -9.1610870, 5.6294408, -15.7203331, 15.3329687
5: -7.5160408, 6.5621881, -6.8386002, 5.9702997, -13.4863405, 13.4007883
6: -7.9988790, 6.7687163, -7.2617373, 6.1499901, -14.1488686, 14.0304537
7: -9.2054605, 6.6633658, -8.3567677, 6.0803251, -15.2857857, 15.0201340
8: -10.5730801, 6.0725203, -9.5940552, 5.5424399, -16.1155205, 15.6665754
9: -7.2833834, 8.5054226, -6.6392221, 7.7243199, -15.0077038, 15.1446447

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_B2_A1_A2_A1

### Relational analysis result of IS_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4681698, upper bound: 7.5510238
time: 2.96 seconds

## Relational analysis of IS_B1_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_B2_A1_A2_B1

### Relational analysis result of IS_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281487, upper bound: 7.6308943
time: 5.34 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2

### Relational analysis result of IS_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308804
time: 3.21 seconds

## BFS IS instance: IS_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -4.1041327, 3.1805727, -7.8474908, 5.9147978, -10.0189304, 11.0280638
1: -3.1813231, 2.9409537, -6.2018151, 5.3870506, -8.5683737, 9.1427689
2: -5.2651973, 2.2880259, -10.2793226, 3.8671761, -9.1323738, 12.5673485
3: -4.6198821, 2.3863373, -9.0614338, 4.2798848, -8.8997669, 11.4477711
4: -4.9113965, 3.1226285, -9.3044252, 5.7151203, -10.6265163, 12.4270535
5: -3.7235584, 3.2842174, -6.9447851, 6.0614977, -9.7850561, 10.2290020
6: -3.8881786, 3.3221297, -7.3762808, 6.2456083, -10.1337872, 10.6984100
7: -4.4776583, 3.3532567, -8.4882631, 6.1739264, -10.6515846, 11.8415203
8: -5.1148095, 3.0997121, -9.7457047, 5.6260433, -10.7408524, 12.8454170
9: -3.6638427, 4.1628437, -6.7400947, 7.8452983, -11.5091410, 10.9029388

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_B2_A2_A1_B1

### Relational analysis result of IS_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286865, upper bound: 7.6291468
time: 2.58 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2

### Relational analysis result of IS_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6294674, upper bound: 7.6312762
time: 3.24 seconds

## BFS IS instance: IS_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.1198177, 6.1140661, -7.8474908, 5.9147978, -14.0346155, 13.9615574
1: -6.4181318, 5.5581045, -6.2018151, 5.3870506, -11.8051825, 11.7599201
2: -10.6458311, 4.0024557, -10.2793226, 3.8671761, -14.5130072, 14.2817783
3: -9.3769054, 4.4163599, -9.0614338, 4.2798848, -13.6567898, 13.4777937
4: -9.6118574, 5.9058447, -9.3044252, 5.7151203, -15.3269777, 15.2102699
5: -7.1813607, 6.2736654, -6.9447851, 6.0614977, -13.2428589, 13.2184505
6: -7.6258817, 6.4554477, -7.3762808, 6.2456083, -13.8714905, 13.8317280
7: -8.7770195, 6.3729239, -8.4882631, 6.1739264, -14.9509459, 14.8611870
8: -10.0784760, 5.8079462, -9.7457047, 5.6260433, -15.7045193, 15.5536509
9: -6.9563274, 8.1032085, -6.7400947, 7.8452983, -14.8016262, 14.8433037

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_B2_A2_A2_B1

### Relational analysis result of IS_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303190, upper bound: 7.6318071
time: 2.22 seconds

## Relational analysis of IS_B1_B2_A2_A2_B2

### Relational analysis result of IS_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303175, upper bound: 7.6317927
time: 2.68 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2.9935913, 2.3598413, -1.3239425, 1.1216354, -4.1152267, 3.6837840
1: -2.3331151, 2.2095194, -1.1004875, 1.0334187, -3.3665338, 3.3100069
2: -3.6986365, 1.7966759, -1.1730111, 1.2664483, -4.9650850, 2.9696870
3: -3.2945924, 1.8339106, -1.2896901, 0.9636180, -4.2582102, 3.1236007
4: -3.6035929, 2.3506122, -1.5453866, 1.1412911, -4.7448840, 3.8959987
5: -2.7452724, 2.4215212, -1.2164173, 1.1772044, -3.9224768, 3.6379385
6: -2.8578942, 2.4563713, -1.2136378, 1.1434219, -4.0013161, 3.6700091
7: -3.2740285, 2.5155191, -1.4023838, 1.2176877, -4.4917164, 3.9179029
8: -3.7364671, 2.3488247, -1.5090184, 1.2115004, -4.9479675, 3.8578432
9: -2.7475948, 3.0727167, -1.2450711, 1.3870066, -4.1346016, 4.3177876

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279284, upper bound: 7.3382475
time: 4.41 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6282155, upper bound: 7.3382496
time: 2.30 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -3.9922912, 3.0941501, -4.2960958, 3.3149490, -7.3072405, 7.3902459
1: -3.0935822, 2.8727627, -3.3490167, 3.0689065, -6.1624889, 6.2217793
2: -5.0954485, 2.2125888, -5.4857354, 2.2493892, -7.3448377, 7.6983242
3: -4.4921103, 2.3302932, -4.8915582, 2.4820070, -6.9741173, 7.2218513
4: -4.7896214, 3.0441232, -5.1937985, 3.2445681, -8.0341892, 8.2379217
5: -3.6236882, 3.1859820, -3.8790288, 3.3747022, -6.9983902, 7.0650110
6: -3.7858045, 3.2366152, -4.0881305, 3.4787266, -7.2645311, 7.3247457
7: -4.3554506, 3.2723999, -4.6843715, 3.4987607, -7.8542113, 7.9567714
8: -4.9768515, 3.0256767, -5.3621178, 3.2304080, -8.2072592, 8.3877945
9: -3.5770454, 4.0620775, -3.8389866, 4.4067969, -7.9838424, 7.9010639

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5962090, upper bound: 7.6301940
time: 2.66 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5962090, upper bound: 7.6308224
time: 2.88 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2.9935913, 2.3598413, -4.6163473, 3.5446234, -6.5382147, 6.9761887
1: -2.3331151, 2.2095194, -3.6038563, 3.2782011, -5.6113162, 5.8133755
2: -3.6986365, 1.7966759, -5.9224572, 2.3654823, -6.0641189, 7.7191334
3: -3.2945924, 1.8339106, -5.2754774, 2.6393905, -5.9339828, 7.1093879
4: -3.6035929, 2.3506122, -5.5727034, 3.4641471, -7.0677400, 7.9233155
5: -2.7452724, 2.4215212, -4.1525769, 3.6067333, -6.3520060, 6.5740981
6: -2.8578942, 2.4563713, -4.3899145, 3.7351730, -6.5930672, 6.8462858
7: -3.2740285, 2.5155191, -5.0274491, 3.7361324, -7.0101609, 7.5429683
8: -3.7364671, 2.3488247, -5.7576594, 3.4452345, -7.1817017, 8.1064844
9: -2.7475948, 3.0727167, -4.1038008, 4.7199178, -7.4675126, 7.1765175

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275757, upper bound: 7.3134002
time: 3.02 seconds

## Relational analysis of IS_B2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5809121, upper bound: 7.1278790
time: 3.21 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -3.9922912, 3.0941501, -7.9550819, 5.9623775, -9.9546690, 11.0492325
1: -3.0935822, 2.8727627, -6.2860422, 5.4770222, -8.5706043, 9.1588049
2: -5.0954485, 2.2125888, -10.4491272, 3.6730886, -8.7685375, 12.6617165
3: -4.4921103, 2.3302932, -9.2615461, 4.3208528, -8.8129635, 11.5918388
4: -4.7896214, 3.0441232, -9.5239658, 5.7698388, -10.5594597, 12.5680885
5: -3.6236882, 3.1859820, -7.0092583, 6.0786748, -9.7023630, 10.1952400
6: -3.7858045, 3.2366152, -7.5039740, 6.3504782, -10.1362829, 10.7405891
7: -4.3554506, 3.2723999, -8.5988626, 6.2546387, -10.6100893, 11.8712626
8: -4.9768515, 3.0256767, -9.8890581, 5.7106099, -10.6874619, 12.9147348
9: -3.5770454, 4.0620775, -6.8655496, 8.0400314, -11.6170769, 10.9276276

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6291212, upper bound: 7.6278923
time: 3.33 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294103
time: 4.00 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.1775239, 2.4899387, -4.4019928, 3.3915286, -6.5690527, 6.8919315
1: -2.4668922, 2.3265679, -3.4350991, 3.1409805, -5.6078730, 5.7616673
2: -3.9342504, 1.8591143, -5.6281967, 2.2916877, -6.2259378, 7.4873109
3: -3.5261378, 1.9277678, -5.0157800, 2.5357170, -6.0618548, 6.9435477
4: -3.8273141, 2.4755931, -5.3172464, 3.3202555, -7.1475697, 7.7928395
5: -2.9095769, 2.5432673, -3.9698849, 3.4510772, -6.3606539, 6.5131521
6: -3.0442059, 2.5991576, -4.1853099, 3.5626757, -6.6068816, 6.7844677
7: -3.4786110, 2.6502266, -4.7962933, 3.5808213, -7.0594320, 7.4465199
8: -3.9653938, 2.4670744, -5.4931092, 3.3027983, -7.2681923, 7.9601836
9: -2.9040656, 3.2417586, -3.9265342, 4.5115414, -7.4156070, 7.1682930

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A1_B1

### Relational analysis result of IS_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1278775, upper bound: 7.5586323
time: 2.00 seconds

## Relational analysis of IS_B2_A2_B1_A1_B2

### Relational analysis result of IS_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1278788, upper bound: 7.5809121
time: 2.30 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.7807870, 5.1339335, -5.3819876, 4.0995798, -10.8803673, 10.5159206
1: -5.3446326, 4.6897540, -4.2226143, 3.7880433, -9.1326761, 8.9123688
2: -8.8531914, 3.4052317, -6.9588943, 2.6669736, -11.5201645, 10.3641262
3: -7.7995238, 3.7382524, -6.1860161, 3.0297389, -10.8292627, 9.9242687
4: -8.0608282, 4.9686856, -6.4774103, 3.9987886, -12.0596170, 11.4460964
5: -6.0278864, 5.2681341, -4.8070345, 4.1748304, -10.2027168, 10.0751686
6: -6.3887343, 5.4127569, -5.0985279, 4.3322272, -10.7209616, 10.5112848
7: -7.3475165, 5.3703842, -5.8421459, 4.3220015, -11.6695175, 11.2125301
8: -8.4225311, 4.9033704, -6.7065644, 3.9671144, -12.3896456, 11.6099348
9: -5.8661280, 6.8001566, -4.7365842, 5.4877405, -11.3538685, 11.5367413

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_A2_B1

### Relational analysis result of IS_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284907, upper bound: 7.5954779
time: 4.87 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2

### Relational analysis result of IS_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284907, upper bound: 7.6305242
time: 2.90 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -7.9243875, 5.9709558, -7.7002954, 5.7780294, -13.7024174, 13.6712513
1: -6.2627816, 5.4358072, -6.0808873, 5.3083715, -11.5711536, 11.5166950
2: -10.3827000, 3.9045658, -10.1060753, 3.5695834, -13.9522839, 14.0106411
3: -9.1503620, 4.3182521, -8.9559231, 4.1919689, -13.3423309, 13.2741756
4: -9.3917770, 5.7676334, -9.2217960, 5.5931292, -14.9849062, 14.9894295
5: -7.0105772, 6.1217976, -6.7881498, 5.8907986, -12.9013758, 12.9099474
6: -7.4468040, 6.3048959, -7.2653570, 6.1505637, -13.5973682, 13.5702534
7: -8.5698967, 6.2303624, -8.3246021, 6.0611353, -14.6310320, 14.5549641
8: -9.8395109, 5.6770978, -9.5737057, 5.5354171, -15.3749275, 15.2508030
9: -6.8014011, 7.9188404, -6.6524782, 7.7876930, -14.5890942, 14.5713186

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 211

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5322507, upper bound: 7.2169345
time: 1.86 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2684674, upper bound: 7.2167074
time: 2.12 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.0113945, 6.0343947, -8.6445274, 6.4619808, -14.4733753, 14.6789227
1: -6.3330016, 5.4927273, -6.8404965, 5.9325838, -12.2655849, 12.3332233
2: -10.4986935, 3.9406223, -11.3802662, 3.9501324, -14.4488258, 15.3208885
3: -9.2536545, 4.3622589, -10.0832367, 4.6682968, -13.9219513, 14.4454956
4: -9.4940805, 5.8277254, -10.3373842, 6.2474213, -15.7415018, 16.1651096
5: -7.0853424, 6.1859879, -7.6002817, 6.5891843, -13.6745262, 13.7862701
6: -7.5279794, 6.3727884, -8.1469727, 6.8905358, -14.4185152, 14.5197611
7: -8.6631193, 6.2959013, -9.3373289, 6.7758870, -15.4390068, 15.6332302
8: -9.9469986, 5.7359076, -10.7421255, 6.1808410, -16.1278400, 16.4780331
9: -6.8728533, 8.0048733, -7.4376712, 8.7222271, -15.5950804, 15.4425449

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302144
time: 1.87 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302144
time: 2.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 6.27 seconds
IS_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6301716, upper bound: 7.6306046
IS_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6301705, upper bound: 7.6306482
IS_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.5963078, upper bound: 7.6314723
IS_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6309131, upper bound: 7.6322266
IS_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.2769116, upper bound: 7.6299035
IS_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.2769237, upper bound: 7.6299895
IS_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.4200133, upper bound: 7.6279760
IS_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6295074, upper bound: 7.6317533
IS_B1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.5379163, upper bound: 7.5421711
IS_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.5942751, upper bound: 7.6303577
IS_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6281487, upper bound: 7.6308943
IS_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308804
IS_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6286865, upper bound: 7.6291468
IS_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6294674, upper bound: 7.6312762
IS_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6303190, upper bound: 7.6318071
IS_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6303175, upper bound: 7.6317927
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6279284, upper bound: 7.3382475
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6282155, upper bound: 7.3382496
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.5962090, upper bound: 7.6301940
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.5962090, upper bound: 7.6308224
IS_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6275757, upper bound: 7.3134002
IS_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.5809121, upper bound: 7.1278790
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6291212, upper bound: 7.6278923
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294103
IS_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.1278775, upper bound: 7.5586323
IS_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.1278788, upper bound: 7.5809121
IS_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6284907, upper bound: 7.5954779
IS_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6284907, upper bound: 7.6305242
IS_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.5322507, upper bound: 7.2169345
IS_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.2684674, upper bound: 7.2167074
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302144
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.27
Output dim: 2, lower bound: -7.6302139, upper bound: 7.6302144

## BFS IS instance: IS_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2.7583365, 2.1975303, -5.7984819, 4.4168892, -7.1752257, 7.9960122
1: -2.1684287, 2.0470228, -4.5134516, 4.0524149, -6.2208433, 6.5604744
2: -3.4166212, 1.7335253, -7.5639963, 3.0343127, -6.4509339, 9.2975216
3: -2.9997208, 1.7162313, -6.6243467, 3.2191024, -6.2188234, 8.3405781
4: -3.3102117, 2.1841488, -6.8948364, 4.2900257, -7.6002374, 9.0789852
5: -2.5391273, 2.2846255, -5.1871696, 4.5592098, -7.0983372, 7.4717951
6: -2.6378453, 2.2694678, -5.4372053, 4.6411347, -7.2789803, 7.7066731
7: -3.0206823, 2.3356647, -6.2871952, 4.6265440, -7.6472263, 8.6228600
8: -3.4395552, 2.1875536, -7.2198906, 4.2290735, -7.6686287, 9.4074440
9: -2.5412538, 2.8365419, -5.0616484, 5.8233361, -8.3645897, 7.8981905

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5503515, upper bound: 7.4734779
time: 5.53 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289966, upper bound: 7.6296709
time: 6.25 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.7221415, 2.9006858, -5.8745856, 4.4722805, -8.1944218, 8.7752714
1: -2.8738658, 2.6911838, -4.5741048, 4.1023870, -6.9762526, 7.2652884
2: -4.7453251, 2.1364012, -7.6664677, 3.0658698, -7.8111949, 9.8028688
3: -4.1614532, 2.1960385, -6.7148666, 3.2566428, -7.4180961, 8.9109049
4: -4.4567099, 2.8590803, -6.9847074, 4.3422685, -8.7989788, 9.8437881
5: -3.3935823, 3.0039182, -5.2529006, 4.6153369, -8.0089188, 8.2568188
6: -3.5308197, 3.0215454, -5.5076656, 4.7004910, -8.2313108, 8.5292110
7: -4.0677638, 3.0656476, -6.3685932, 4.6838245, -8.7515888, 9.4342403
8: -4.6412039, 2.8414750, -7.3141294, 4.2800918, -8.9212952, 10.1556044
9: -3.3480804, 3.7803302, -5.1245022, 5.8989096, -9.2469902, 8.9048328

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5503511, upper bound: 7.4896764
time: 4.07 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289966, upper bound: 7.6297385
time: 3.18 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4509927, 0.5204877, -2.7806711, 2.2070007, -2.6579933, 3.3011589
1: -0.4899695, 0.5177615, -2.1870410, 2.0672841, -2.5572536, 2.7048025
2: 0.1305590, 1.0913000, -3.3901360, 1.7263676, -1.5958086, 4.4814358
3: -0.3232668, 0.5498731, -3.0351186, 1.7283721, -2.0516388, 3.5849917
4: -0.5242356, 0.5465826, -3.3460612, 2.2031314, -2.7273669, 3.8926439
5: -0.4814742, 0.5579521, -2.5588822, 2.2525856, -2.7340598, 3.1168344
6: -0.4239199, 0.5575832, -2.6586695, 2.2901330, -2.7140529, 3.2162528
7: -0.4820627, 0.6032573, -3.0433204, 2.3552217, -2.8372843, 3.6465778
8: -0.5474737, 0.6434382, -3.4741127, 2.2032363, -2.7507100, 4.1175508
9: -0.5339578, 0.5699098, -2.5690913, 2.8603649, -3.3943226, 3.1390011

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.3383577, upper bound: 7.6300796
time: 4.04 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.3383750, upper bound: 7.6303069
time: 13.39 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -3.7711341, 2.9309092, -6.0604033, 6.2328043
1: -2.4251304, 2.2983003, -2.9179156, 2.7284474, -5.1535778, 5.2162161
2: -3.9117460, 1.8518873, -4.7880740, 2.1122229, -6.0239687, 6.6399612
3: -3.4586265, 1.9007586, -4.2308588, 2.2210655, -5.6796923, 6.1316175
4: -3.7661660, 2.4433300, -4.5309696, 2.8919618, -6.6581278, 6.9742994
5: -2.8676689, 2.5432758, -3.4313211, 3.0169790, -5.8846478, 5.9745970
6: -2.9870868, 2.5610185, -3.5815358, 3.0640309, -6.0511179, 6.1425543
7: -3.4249156, 2.6162057, -4.1177950, 3.1064944, -6.5314102, 6.7340007
8: -3.9026318, 2.4418151, -4.7037334, 2.8767529, -6.7793846, 7.1455488
9: -2.8588891, 3.2085273, -3.3961568, 3.8438315, -6.7027206, 6.6046839

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303982, upper bound: 7.6306935
time: 3.09 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303982, upper bound: 7.6322263
time: 5.28 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -3.2991657, 2.5814331, -1.4550622, 1.2157210, -4.5148869, 4.0364952
1: -2.5556443, 2.4069495, -1.2017486, 1.1310760, -3.6867204, 3.6086981
2: -4.1274829, 1.9167413, -1.3723750, 1.3106838, -5.4381666, 3.2891164
3: -3.6686006, 1.9904231, -1.4480087, 1.0328410, -4.7014418, 3.4384317
4: -3.9681654, 2.5605903, -1.7059625, 1.2396755, -5.2078409, 4.2665529
5: -3.0218735, 2.6575723, -1.3356619, 1.2692595, -4.2911329, 3.9932342
6: -3.1618674, 2.6919823, -1.3326101, 1.2457312, -4.4075985, 4.0245924
7: -3.6135166, 2.7440152, -1.5479141, 1.3293593, -4.9428759, 4.2919292
8: -4.1112270, 2.5508080, -1.6916983, 1.3026962, -5.4139233, 4.2425060
9: -3.0030138, 3.3574250, -1.3691740, 1.5105356, -4.5135493, 4.7265987

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.1342644, upper bound: 7.6276368
time: 2.55 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9621034, upper bound: 7.5744705
time: 2.12 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -3.2686677, 2.5584579, -1.7978070, 1.4735185, -4.7421861, 4.3562651
1: -2.5304127, 2.3863688, -1.4632750, 1.3699645, -3.9003773, 3.8496437
2: -4.0851040, 1.9038808, -1.9118865, 1.4092780, -5.4943819, 3.8157673
3: -3.6318660, 1.9748540, -1.8640885, 1.2134869, -4.8453531, 3.8389425
4: -3.9316375, 2.5391803, -2.1200476, 1.4885551, -5.4201927, 4.6592278
5: -2.9945536, 2.6340556, -1.6572421, 1.5170685, -4.5116220, 4.2912979
6: -3.1329427, 2.6680713, -1.6723914, 1.5112429, -4.6441855, 4.3404627
7: -3.5800676, 2.7200844, -1.9373448, 1.5956358, -5.1757035, 4.6574292
8: -4.0734086, 2.5300012, -2.1641603, 1.5359149, -5.6093235, 4.6941614
9: -2.9774530, 3.3269577, -1.6822411, 1.8637688, -4.8412218, 5.0091987

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.1342903, upper bound: 7.6282883
time: 2.26 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9621538, upper bound: 7.6171505
time: 2.26 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -6.9336033, 5.2597418, -5.8788328, 4.4753709, -11.4089737, 11.1385746
1: -5.4720449, 4.7785091, -4.5774941, 4.1051755, -9.5772209, 9.3560028
2: -9.1123562, 3.5281138, -7.6721725, 3.0676310, -12.1799870, 11.2002869
3: -7.9623547, 3.8215346, -6.7199149, 3.2587385, -11.2210932, 10.5414495
4: -8.2140923, 5.0847821, -6.9897199, 4.3451872, -12.5592794, 12.0745020
5: -6.1568098, 5.4046974, -5.2565708, 4.6184645, -10.7752743, 10.6612682
6: -6.5298166, 5.5309153, -5.5115995, 4.7038031, -11.2336197, 11.0425148
7: -7.5158625, 5.4862204, -6.3731327, 4.6870232, -12.2028856, 11.8593531
8: -8.6176672, 5.0404816, -7.3193836, 4.2829409, -12.9006081, 12.3598652
9: -5.9837914, 6.9425769, -5.1280084, 5.9031305, -11.8869219, 12.0705853

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3816083, upper bound: 7.5446719
time: 2.74 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4200133, upper bound: 7.6279760
time: 3.82 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -7.0665984, 5.3569503, -3.7711341, 2.9309092, -9.9975071, 9.1280842
1: -5.5791984, 4.8653116, -2.9179156, 2.7284474, -8.3076458, 7.7832270
2: -9.2917576, 3.5834966, -4.7880740, 2.1122229, -11.4039803, 8.3715706
3: -8.1203499, 3.8885853, -4.2308588, 2.2210655, -10.3414154, 8.1194439
4: -8.3703003, 5.1766109, -4.5309696, 2.8919618, -11.2622623, 9.7075806
5: -6.2708530, 5.5036459, -3.4313211, 3.0169790, -9.2878323, 8.9349670
6: -6.6540017, 5.6346269, -3.5815358, 3.0640309, -9.7180328, 9.2161627
7: -7.6584935, 5.5861216, -4.1177950, 3.1064944, -10.7649879, 9.7039165
8: -8.7819738, 5.1313066, -4.7037334, 2.8767529, -11.6587267, 9.8350401
9: -6.0929327, 7.0746951, -3.3961568, 3.8438315, -9.9367638, 10.4708519

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6282264, upper bound: 7.6301483
time: 4.77 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295073, upper bound: 7.6317536
time: 2.41 seconds

## BFS IS instance: IS_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -4.7967854, 3.6851165, -6.4884582, 4.9201121, -9.7168980, 10.1735744
1: -3.7323868, 3.3906033, -5.1089339, 4.4995127, -8.2318993, 8.4995375
2: -6.2061720, 2.5392091, -8.4605274, 3.2737737, -9.4799461, 10.9997368
3: -5.4602466, 2.7283242, -7.4562993, 3.5904226, -9.0506687, 10.1846237
4: -5.7448153, 3.5955670, -7.7218442, 4.7666931, -10.5115089, 11.3174114
5: -4.3175664, 3.7879877, -5.7759690, 5.0469036, -9.3644695, 9.5639572
6: -4.5391769, 3.8701496, -6.1175137, 5.1862001, -9.7253771, 9.9876633
7: -5.2229223, 3.8674383, -7.0348496, 5.1506877, -10.3736095, 10.9022884
8: -5.9766102, 3.5662158, -8.0630178, 4.7061872, -10.6827974, 11.6292334
9: -4.2377043, 4.8621831, -5.6282506, 6.5144768, -10.7521811, 10.4904337

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_B2_A1_A1_B2_A1

### Relational analysis result of IS_B1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.3082287, upper bound: 7.6282643
time: 2.05 seconds

## Relational analysis of IS_B1_B2_A1_A1_B2_A2

### Relational analysis result of IS_B1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.3082287, upper bound: 7.6306110
time: 3.97 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4224510, 6.3357053, -6.2819428, 4.7715302, -13.1939812, 12.6176481
1: -6.6546073, 5.7527900, -4.9321213, 4.3596702, -11.0142775, 10.6849117
2: -11.0596981, 4.1035709, -8.1964598, 3.2028580, -14.2625561, 12.3000307
3: -9.7486916, 4.5642633, -7.2030063, 3.4814873, -13.2301788, 11.7672691
4: -9.9813604, 6.1075382, -7.4703984, 4.6212096, -14.6025696, 13.5779362
5: -7.4360290, 6.4935946, -5.5936394, 4.9101338, -12.3461628, 12.0872345
6: -7.9119449, 6.6960268, -5.9159803, 5.0220213, -12.9339657, 12.6120071
7: -9.1056738, 6.5932193, -6.8101983, 4.9880953, -14.0937691, 13.4034176
8: -10.4579992, 6.0095530, -7.8058262, 4.5631461, -15.0211449, 13.8153791
9: -7.2068748, 8.4132833, -5.4507275, 6.3085823, -13.5154572, 13.8640108

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_B2_A1_A2_B1_A1

### Relational analysis result of IS_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308784
time: 4.50 seconds

## Relational analysis of IS_B1_B2_A1_A2_B1_A2

### Relational analysis result of IS_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308802
time: 2.73 seconds

## BFS IS instance: IS_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.5100555, 6.3995667, -7.2914085, 5.5092411, -14.0192966, 13.6909752
1: -6.7253404, 5.8101072, -5.7514229, 5.0229692, -11.7483101, 11.5615301
2: -11.1762981, 4.1397581, -9.5395489, 3.6345134, -14.8108120, 13.6793070
3: -9.8527060, 4.6085749, -8.4021521, 3.9979467, -13.8506527, 13.0107269
4: -10.0843811, 6.1680603, -8.6517696, 5.3304434, -15.4148245, 14.8198299
5: -7.5112858, 6.5581236, -6.4663787, 5.6510348, -13.1623211, 13.0245018
6: -7.9937096, 6.7643991, -6.8572760, 5.8123622, -13.8060722, 13.6216755
7: -9.1995316, 6.6591992, -7.8927817, 5.7543335, -14.9538651, 14.5519810
8: -10.5662451, 6.0687776, -9.0592556, 5.2496920, -15.8159370, 15.1280327
9: -7.2788386, 8.4999390, -6.2838120, 7.2954574, -14.5742960, 14.7837505

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308784
time: 4.66 seconds

## Relational analysis of IS_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308801
time: 2.91 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -3.0996242, 2.4413133, -3.0149438, 2.3743861, -5.4740105, 5.4562569
1: -2.4050732, 2.2783751, -2.3557260, 2.2184715, -4.6235447, 4.6341009
2: -3.8758757, 1.8463492, -3.6948304, 1.8049128, -5.6807885, 5.5411797
3: -3.4224510, 1.8861226, -3.3309722, 1.8459786, -5.2684298, 5.2170949
4: -3.7279999, 2.4242122, -3.6329050, 2.3632812, -6.0912809, 6.0571175
5: -2.8431208, 2.5239224, -2.7676797, 2.4093752, -5.2524958, 5.2916021
6: -2.9585686, 2.5391905, -2.8899817, 2.4737902, -5.4323587, 5.4291725
7: -3.3944073, 2.5941417, -3.3043084, 2.5273905, -5.9217978, 5.8984499
8: -3.8664663, 2.4223671, -3.7671199, 2.3567216, -6.2231879, 6.1894870
9: -2.8340626, 3.1751676, -2.7691569, 3.0824423, -5.9165049, 5.9443245

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_B1_B2_A2_A1_B1_A1

### Relational analysis result of IS_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283507, upper bound: 7.5645506
time: 10.77 seconds

## Relational analysis of IS_B1_B2_A2_A1_B1_A2

### Relational analysis result of IS_B1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6285789, upper bound: 7.5645870
time: 3.07 seconds

## BFS IS instance: IS_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -4.1041327, 3.1805727, -6.6115294, 5.0100403, -9.1141729, 9.7921019
1: -3.1813231, 2.9409537, -5.2091675, 4.5804949, -7.7618179, 8.1501217
2: -5.2651973, 2.2880259, -8.6242132, 3.3285465, -8.5937443, 10.9122391
3: -4.6198821, 2.3863373, -7.6008682, 3.6532593, -8.2731419, 9.9872055
4: -4.9113965, 3.1226285, -7.8647785, 4.8523750, -9.7637711, 10.9874067
5: -3.7235584, 3.2842174, -5.8823118, 5.1387844, -8.8623428, 9.1665287
6: -3.8881786, 3.3221297, -6.2318931, 5.2815685, -9.1697474, 9.5540228
7: -4.4776583, 3.3532567, -7.1664228, 5.2442799, -9.7219381, 10.5196800
8: -5.1148095, 3.0997121, -8.2144985, 4.7897620, -9.9045715, 11.3142109
9: -3.6638427, 4.1628437, -5.7289343, 6.6351810, -10.2990236, 9.8917780

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_B1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1775400, upper bound: 7.2651954
time: 5.41 seconds

## Relational analysis of IS_B1_B2_A2_A1_B2_B2

### Relational analysis result of IS_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307318, upper bound: 7.6313911
time: 2.49 seconds

## BFS IS instance: IS_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.0269976, 6.0464077, -6.4073286, 4.8629994, -12.8899975, 12.4537363
1: -6.3432007, 5.4973879, -5.0342441, 4.4421697, -10.7853699, 10.5316315
2: -10.5223007, 3.9640536, -8.3628836, 3.2582178, -13.7805185, 12.3269367
3: -9.2667036, 4.3694119, -7.3503299, 3.5454721, -12.8121758, 11.7197418
4: -9.5027208, 5.8416839, -7.6160703, 4.7084866, -14.2112074, 13.4577541
5: -7.1015444, 6.2052794, -5.7018614, 5.0035405, -12.1050854, 11.9071407
6: -7.5392785, 6.3830242, -6.0324841, 5.1192350, -12.6585140, 12.4155083
7: -8.6775942, 6.3030138, -6.9440937, 5.0833960, -13.7609901, 13.2471075
8: -9.9638100, 5.7451820, -7.9600902, 4.6482973, -14.6121073, 13.7052727
9: -6.8801022, 8.0114155, -5.5532951, 6.4316568, -13.3117590, 13.5647106

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_B2_A2_A2_B1_A1

### Relational analysis result of IS_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279900, upper bound: 7.6290181
time: 3.02 seconds

## Relational analysis of IS_B1_B2_A2_A2_B1_A2

### Relational analysis result of IS_B1_B2_A2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1779004, upper bound: 7.5707023
time: 2.30 seconds

## BFS IS instance: IS_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.1141319, 6.1099234, -7.4134459, 5.5982227, -13.7123547, 13.5233688
1: -6.4135356, 5.5543852, -5.8508787, 5.1032228, -11.5167580, 11.4052639
2: -10.6382790, 4.0001192, -9.7006741, 3.6879392, -14.3262177, 13.7007933
3: -9.3701496, 4.4134831, -8.5456333, 4.0602455, -13.4303951, 12.9591160
4: -9.6051655, 5.9019160, -8.7936869, 5.4153433, -15.0205088, 14.6956024
5: -7.1764708, 6.2694879, -6.5716166, 5.7415514, -12.9180222, 12.8411045
6: -7.6205730, 6.4510121, -6.9707365, 5.9070468, -13.5276203, 13.4217491
7: -8.7709284, 6.3686428, -8.0230675, 5.8470612, -14.6179895, 14.3917103
8: -10.0714550, 5.8041019, -9.2094574, 5.3325076, -15.4039631, 15.0135593
9: -6.9516611, 8.0975714, -6.3837132, 7.4152765, -14.3669376, 14.4812851

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_B2_A2_A2_B2_A1

### Relational analysis result of IS_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303175, upper bound: 7.6317834
time: 3.67 seconds

## Relational analysis of IS_B1_B2_A2_A2_B2_A2

### Relational analysis result of IS_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303175, upper bound: 7.6317927
time: 2.36 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.9215061, 0.8422726, -1.2254752, 1.0500937, -1.9715998, 2.0677478
1: -0.8078814, 0.7900633, -1.0271922, 0.9688413, -1.7767227, 1.8172555
2: -0.5502269, 1.1804223, -1.0177077, 1.2402557, -1.7904826, 2.1981301
3: -0.8031018, 0.7683184, -1.1706777, 0.9131690, -1.7162709, 1.9389961
4: -1.0632244, 0.8707467, -1.4266045, 1.0730712, -2.1362958, 2.2973514
5: -0.8489783, 0.9130602, -1.1261477, 1.1097583, -1.9587365, 2.0392079
6: -0.8298422, 0.8621604, -1.1188492, 1.0712469, -1.9010891, 1.9810095
7: -0.9535363, 0.9284251, -1.2929826, 1.1428101, -2.0963464, 2.2214077
8: -1.0165153, 0.9494683, -1.3744918, 1.1473403, -2.1638556, 2.3239601
9: -0.9063756, 1.0019305, -1.1584077, 1.2880969, -2.1944726, 2.1603382

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279054, upper bound: 7.3382471
time: 4.12 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279054, upper bound: 7.3382465
time: 5.14 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1.7551173, 1.4429920, -1.1345656, 0.9858438, -2.7409611, 2.5775576
1: -1.4307915, 1.3387165, -0.9614096, 0.9097232, -2.3405147, 2.3001261
2: -1.8493936, 1.3958887, -0.8758149, 1.2169657, -3.0663593, 2.2717037
3: -1.8130950, 1.1904706, -1.0614752, 0.8678237, -2.6809187, 2.2519457
4: -2.0690022, 1.4568048, -1.3166511, 1.0111474, -3.0801497, 2.7734559
5: -1.6156819, 1.4877362, -1.0441494, 1.0485871, -2.6642690, 2.5318856
6: -1.6295102, 1.4765315, -1.0338272, 1.0054855, -2.6349957, 2.5103588
7: -1.8906827, 1.5602169, -1.1919224, 1.0744700, -2.9651527, 2.7521393
8: -2.1060436, 1.5054897, -1.2529733, 1.0882094, -3.1942530, 2.7584629
9: -1.6416720, 1.8233292, -1.0801580, 1.1986178, -2.8402898, 2.9034872

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4920922, upper bound: 7.1861185
time: 2.53 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5356744, upper bound: 7.1861698
time: 3.92 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4124850, 0.4788342, -4.2960958, 3.3149490, -3.7274342, 4.7749300
1: -0.4517946, 0.4817170, -3.3490167, 3.0689065, -3.5207012, 3.8307338
2: 0.1904097, 1.0817281, -5.4857354, 2.2493892, -2.0589795, 6.5674634
3: -0.2937512, 0.5212060, -4.8915582, 2.4820070, -2.7757583, 5.4127641
4: -0.4884118, 0.5025003, -5.1937985, 3.2445681, -3.7329798, 5.6962986
5: -0.4423507, 0.5151439, -3.8790288, 3.3747022, -3.8170528, 4.3941727
6: -0.3886202, 0.5214964, -4.0881305, 3.4787266, -3.8673468, 4.6096268
7: -0.4450938, 0.5586728, -4.6843715, 3.4987607, -3.9438546, 5.2430444
8: -0.5108202, 0.6002739, -5.3621178, 3.2304080, -3.7412281, 5.9623919
9: -0.4996807, 0.5173190, -3.8389866, 4.4067969, -4.9064775, 4.3563056

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.3381811, upper bound: 7.6279284
time: 3.69 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.3381433, upper bound: 7.6282155
time: 2.84 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -2.8854003, 2.2831175, -4.2960958, 3.3149490, -6.2003493, 6.5792131
1: -2.2588069, 2.1350756, -3.3490167, 3.0689065, -5.3277135, 5.4840922
2: -3.5347953, 1.7592452, -5.4857354, 2.2493892, -5.7841845, 7.2449808
3: -3.1660383, 1.7792099, -4.8915582, 2.4820070, -5.6480455, 6.6707678
4: -3.4757137, 2.2737236, -5.1937985, 3.2445681, -6.7202816, 7.4675221
5: -2.6499650, 2.3284945, -3.8790288, 3.3747022, -6.0246673, 6.2075233
6: -2.7588854, 2.3706551, -4.0881305, 3.4787266, -6.2376118, 6.4587855
7: -3.1579261, 2.4309921, -4.6843715, 3.4987607, -6.6566868, 7.1153636
8: -3.6045885, 2.2730327, -5.3621178, 3.2304080, -6.8349962, 7.6351504
9: -2.6563036, 2.9686413, -3.8389866, 4.4067969, -7.0631008, 6.8076277

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.3381811, upper bound: 7.6296911
time: 1.92 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.3381433, upper bound: 7.6298457
time: 2.13 seconds

## BFS IS instance: IS_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -2.8519695, 2.2580185, -2.6849976, 2.1410673, -4.9930367, 4.9430161
1: -2.2359099, 2.1143737, -2.1190431, 1.9881384, -4.2240486, 4.2334166
2: -3.4963076, 1.7493064, -3.2604780, 1.6815801, -5.1778879, 5.0097847
3: -3.1217809, 1.7635781, -2.9343529, 1.6745905, -4.7963715, 4.6979308
4: -3.4317422, 2.2523177, -3.2395947, 2.1258936, -5.5576358, 5.4919124
5: -2.6211424, 2.3108377, -2.4744313, 2.1753957, -4.7965384, 4.7852688
6: -2.7259490, 2.3454952, -2.5864949, 2.2116771, -4.9376259, 4.9319901
7: -3.1204257, 2.4082694, -2.9435201, 2.2648487, -5.3852744, 5.3517895
8: -3.5610211, 2.2521849, -3.3542600, 2.1342087, -5.6952295, 5.6064448
9: -2.6281118, 2.9318202, -2.4813373, 2.7796640, -5.4077759, 5.4131575

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A1_B2_B1_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5578383, upper bound: 7.1278763
time: 3.83 seconds

## Relational analysis of IS_B2_A1_B2_B1_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5578383, upper bound: 7.1278784
time: 2.83 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -3.7660370, 2.9260561, -7.7145057, 5.7875309, -9.5535679, 10.6405621
1: -2.9212441, 2.7291427, -6.0924363, 5.3192892, -8.2405338, 8.8215790
2: -4.7665234, 2.1052055, -10.1218643, 3.5781035, -8.3446274, 12.2270699
3: -4.2271256, 2.2236676, -8.9737110, 4.1998453, -8.4269714, 11.1973782
4: -4.5252490, 2.8952808, -9.2389355, 5.6041679, -10.1294174, 12.1342163
5: -3.4271171, 3.0011704, -6.8030586, 5.9001760, -9.3272934, 9.8042288
6: -3.5790677, 3.0619555, -7.2787614, 6.1623836, -9.7414513, 10.3407173
7: -4.1115742, 3.1088505, -8.3408775, 6.0739355, -10.1855097, 11.4497280
8: -4.7011738, 2.8742797, -9.5915966, 5.5472345, -10.2484083, 12.4658766
9: -3.3959942, 3.8341162, -6.6670771, 7.7996817, -11.1956758, 10.5011930

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289308, upper bound: 7.6251432
time: 2.16 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286523, upper bound: 7.6195855
time: 2.28 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -3.6922517, 2.8743668, -7.9550819, 5.9623775, -9.6546288, 10.8294487
1: -2.8533370, 2.6758895, -6.2860422, 5.4770222, -8.3303595, 8.9619312
2: -4.6852260, 2.0885468, -10.4491272, 3.6730886, -8.3583145, 12.5376740
3: -4.1336260, 2.1815400, -9.2615461, 4.3208528, -8.4544792, 11.4430866
4: -4.4342575, 2.8372912, -9.5239658, 5.7698388, -10.2040958, 12.3612576
5: -3.3636880, 2.9637656, -7.0092583, 6.0786748, -9.4423628, 9.9730244
6: -3.5067868, 3.0017023, -7.5039740, 6.3504782, -9.8572655, 10.5056763
7: -4.0335407, 3.0463476, -8.5988626, 6.2546387, -10.2881794, 11.6452103
8: -4.6050034, 2.8231895, -9.8890581, 5.7106099, -10.3156128, 12.7122478
9: -3.3293736, 3.7623889, -6.8655496, 8.0400314, -11.3694048, 10.6279383

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5689777, upper bound: 7.6197797
time: 2.25 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294102
time: 3.17 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.7807870, 5.1339335, -1.3239425, 1.1216354, -7.9024224, 6.4578762
1: -5.3446326, 4.6897540, -1.1004875, 1.0334187, -6.3780513, 5.7902412
2: -8.8531914, 3.4052317, -1.1730111, 1.2664483, -10.1196394, 4.5782428
3: -7.7995238, 3.7382524, -1.2896901, 0.9636180, -8.7631416, 5.0279427
4: -8.0608282, 4.9686856, -1.5453866, 1.1412911, -9.2021198, 6.5140724
5: -6.0278864, 5.2681341, -1.2164173, 1.1772044, -7.2050905, 6.4845514
6: -6.3887343, 5.4127569, -1.2136378, 1.1434219, -7.5321560, 6.6263947
7: -7.3475165, 5.3703842, -1.4023838, 1.2176877, -8.5652046, 6.7727680
8: -8.4225311, 4.9033704, -1.5090184, 1.2115004, -9.6340313, 6.4123888
9: -5.8661280, 6.8001566, -1.2450711, 1.3870066, -7.2531347, 8.0452280

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4745981, upper bound: 7.3333025
time: 2.98 seconds

## Relational analysis of IS_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4837392, upper bound: 7.3324538
time: 3.12 seconds

## BFS IS instance: IS_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.7807870, 5.1339335, -4.2960958, 3.3149490, -10.0957355, 9.4300289
1: -5.3446326, 4.6897540, -3.3490167, 3.0689065, -8.4135389, 8.0387707
2: -8.8531914, 3.4052317, -5.4857354, 2.2493892, -11.1025810, 8.8909674
3: -7.7995238, 3.7382524, -4.8915582, 2.4820070, -10.2815304, 8.6298103
4: -8.0608282, 4.9686856, -5.1937985, 3.2445681, -11.3053961, 10.1624842
5: -6.0278864, 5.2681341, -3.8790288, 3.3747022, -9.4025888, 9.1471634
6: -6.3887343, 5.4127569, -4.0881305, 3.4787266, -9.8674612, 9.5008869
7: -7.3475165, 5.3703842, -4.6843715, 3.4987607, -10.8462772, 10.0547562
8: -8.4225311, 4.9033704, -5.3621178, 3.2304080, -11.6529388, 10.2654877
9: -5.8661280, 6.8001566, -3.8389866, 4.4067969, -10.2729244, 10.6391430

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4745981, upper bound: 7.6294480
time: 3.78 seconds

## Relational analysis of IS_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4837393, upper bound: 7.6294668
time: 4.73 seconds

## BFS IS instance: IS_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -6.5766506, 4.9869409, -8.6445274, 6.4619808, -13.0386314, 13.6314678
1: -5.1696811, 4.5512891, -6.8404965, 5.9325838, -11.1022644, 11.3917856
2: -8.5922661, 3.3357744, -11.3802662, 3.9501324, -12.5423985, 14.7160406
3: -7.5487266, 3.6306424, -10.0832367, 4.6682968, -12.2170238, 13.7138786
4: -7.8115149, 4.8252296, -10.3373842, 6.2474213, -14.0589361, 15.1626139
5: -5.8473496, 5.1334963, -7.6002817, 6.5891843, -12.4365339, 12.7337780
6: -6.1890645, 5.2505584, -8.1469727, 6.8905358, -13.0796003, 13.3975315
7: -7.1249576, 5.2094851, -9.3373289, 6.7758870, -13.9008446, 14.5468140
8: -8.1682968, 4.7618990, -10.7421255, 6.1808410, -14.3491383, 15.5040245
9: -5.6903148, 6.5960035, -7.4376712, 8.7222271, -14.4125423, 14.0336742

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4605820, upper bound: 7.1779141
time: 3.12 seconds

## Relational analysis of IS_B2_A2_B2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1030228, upper bound: 7.1576924
time: 1.85 seconds

## BFS IS instance: IS_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -7.5821204, 5.7213197, -8.6445274, 6.4619808, -14.0441017, 14.3658466
1: -5.9858732, 5.2119942, -6.8404965, 5.9325838, -11.9184570, 12.0524902
2: -9.9269371, 3.7636333, -11.3802662, 3.9501324, -13.8770695, 15.1438999
3: -8.7434378, 4.1450176, -10.0832367, 4.6682968, -13.4117346, 14.2282543
4: -8.9888868, 5.5312209, -10.3373842, 6.2474213, -15.2363081, 15.8686047
5: -6.7162724, 5.8698053, -7.6002817, 6.5891843, -13.3054562, 13.4700871
6: -7.1268291, 6.0379157, -8.1469727, 6.8905358, -14.0173645, 14.1848888
7: -8.2030220, 5.9726176, -9.3373289, 6.7758870, -14.9789085, 15.3099461
8: -9.4166441, 5.4455733, -10.7421255, 6.1808410, -15.5974846, 16.1876984
9: -6.5203571, 7.5794892, -7.4376712, 8.7222271, -15.2425842, 15.0171604

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4605820, upper bound: 7.1778897
time: 3.11 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1030228, upper bound: 7.1776555
time: 2.28 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.98 seconds
IS_B1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.5503515, upper bound: 7.4734779
IS_B1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6289966, upper bound: 7.6296709
IS_B1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.5503511, upper bound: 7.4896764
IS_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6289966, upper bound: 7.6297385
IS_B1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3383577, upper bound: 7.6300796
IS_B1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3383750, upper bound: 7.6303069
IS_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6303982, upper bound: 7.6306935
IS_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6303982, upper bound: 7.6322263
IS_B1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.1342644, upper bound: 7.6276368
IS_B1_B1_A2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -6.9621034, upper bound: 7.5744705
IS_B1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.1342903, upper bound: 7.6282883
IS_B1_B1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -6.9621538, upper bound: 7.6171505
IS_B1_B1_A2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3816083, upper bound: 7.5446719
IS_B1_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.4200133, upper bound: 7.6279760
IS_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6282264, upper bound: 7.6301483
IS_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6295073, upper bound: 7.6317536
IS_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3082287, upper bound: 7.6282643
IS_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3082287, upper bound: 7.6306110
IS_B1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308784
IS_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308802
IS_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308784
IS_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308801
IS_B1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6283507, upper bound: 7.5645506
IS_B1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6285789, upper bound: 7.5645870
IS_B1_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.1775400, upper bound: 7.2651954
IS_B1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6307318, upper bound: 7.6313911
IS_B1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6279900, upper bound: 7.6290181
IS_B1_B2_A2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.1779004, upper bound: 7.5707023
IS_B1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6303175, upper bound: 7.6317834
IS_B1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6303175, upper bound: 7.6317927
IS_B2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6279054, upper bound: 7.3382471
IS_B2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6279054, upper bound: 7.3382465
IS_B2_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.4920922, upper bound: 7.1861185
IS_B2_A1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.5356744, upper bound: 7.1861698
IS_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3381811, upper bound: 7.6279284
IS_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3381433, upper bound: 7.6282155
IS_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3381811, upper bound: 7.6296911
IS_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.3381433, upper bound: 7.6298457
IS_B2_A1_B2_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.5578383, upper bound: 7.1278763
IS_B2_A1_B2_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.5578383, upper bound: 7.1278784
IS_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6289308, upper bound: 7.6251432
IS_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6286523, upper bound: 7.6195855
IS_B2_A1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.5689777, upper bound: 7.6197797
IS_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294102
IS_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.4745981, upper bound: 7.3333025
IS_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.4837392, upper bound: 7.3324538
IS_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.4745981, upper bound: 7.6294480
IS_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.4837393, upper bound: 7.6294668
IS_B2_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.4605820, upper bound: 7.1779141
IS_B2_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.1030228, upper bound: 7.1576924
IS_B2_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.4605820, upper bound: 7.1778897
IS_B2_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 6.98
Output dim: 2, lower bound: -7.1030228, upper bound: 7.1776555

## BFS IS instance: IS_B1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.7583365, 2.1975303, -4.7577033, 3.6570287, -6.4153652, 6.9552336
1: -2.1684287, 2.0470228, -3.6879022, 3.3686776, -5.5371065, 5.7349253
2: -3.4166212, 1.7335253, -6.1541176, 2.5774944, -5.9941158, 7.8876429
3: -2.9997208, 1.7162313, -5.3921947, 2.7042320, -5.7039528, 7.1084261
4: -3.3102117, 2.1841488, -5.6790109, 3.5670347, -6.8772464, 7.8631597
5: -2.5391273, 2.2846255, -4.2858133, 3.7828324, -6.3219595, 6.5704389
6: -2.6378453, 2.2694678, -4.4809308, 3.8292837, -6.4671288, 6.7503986
7: -3.0206823, 2.3356647, -5.1755810, 3.8428583, -6.8635406, 7.5112457
8: -3.4395552, 2.1875536, -5.9278159, 3.5300746, -6.9696298, 8.1153698
9: -2.5412538, 2.8365419, -4.2046809, 4.8020205, -7.3432741, 7.0412226

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5583486, upper bound: 7.6131681
time: 5.22 seconds

## Relational analysis of IS_B1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5563494, upper bound: 7.6054897
time: 3.37 seconds

## BFS IS instance: IS_B1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.7221415, 2.9006858, -4.8338804, 3.7125416, -7.4346828, 7.7345662
1: -2.8738658, 2.6911838, -3.7486165, 3.4186985, -6.2925644, 6.4398003
2: -4.7453251, 2.1364012, -6.2573571, 2.6093826, -7.3547077, 8.3937588
3: -4.1614532, 2.1960385, -5.4827766, 2.7418098, -6.9032631, 7.6788149
4: -4.4567099, 2.8590803, -5.7688847, 3.6193135, -8.0760231, 8.6279650
5: -3.3935823, 3.0039182, -4.3517017, 3.8393259, -7.2329082, 7.3556199
6: -3.5308197, 3.0215454, -4.5514741, 3.8886440, -7.4194636, 7.5730195
7: -4.0677638, 3.0656476, -5.2571268, 3.9001970, -7.9679608, 8.3227749
8: -4.6412039, 2.8414750, -6.0220671, 3.5811415, -8.2223454, 8.8635426
9: -3.3480804, 3.7803302, -4.2675686, 4.8776536, -8.2257338, 8.0478992

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5587639, upper bound: 7.6279446
time: 4.98 seconds

## Relational analysis of IS_B1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_B1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5568421, upper bound: 7.6277935
time: 2.99 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4271084, 0.4942832, -0.7540269, 0.7343497, -1.1614581, 1.2483101
1: -0.4654395, 0.4951848, -0.6913316, 0.6943027, -1.1597422, 1.1865164
2: 0.1681221, 1.0850841, -0.2998856, 1.1493220, -0.9811999, 1.3849697
3: -0.3050309, 0.5317209, -0.6065565, 0.6927909, -0.9978217, 1.1382773
4: -0.5014338, 0.5190079, -0.8618459, 0.7638992, -1.2653329, 1.3808537
5: -0.4564503, 0.5308417, -0.7035303, 0.8065343, -1.2629846, 1.2343720
6: -0.4015422, 0.5350480, -0.6818031, 0.7495090, -1.1510512, 1.2168511
7: -0.4590012, 0.5750179, -0.7700191, 0.8143637, -1.2733649, 1.3450370
8: -0.5243088, 0.6157832, -0.8376355, 0.8428617, -1.3671705, 1.4534187
9: -0.5123287, 0.5364254, -0.7650511, 0.8581845, -1.3705132, 1.3014765

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.1862527, upper bound: 7.6280945
time: 2.15 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.1862834, upper bound: 7.6284915
time: 2.56 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4032527, 0.4682322, -1.5733756, 1.3070676, -1.7103204, 2.0416079
1: -0.4429742, 0.4721122, -1.2924001, 1.2113595, -1.6543337, 1.7645123
2: 0.2054854, 1.0785766, -1.5638127, 1.3436865, -1.1382010, 2.6423893
3: -0.2864389, 0.5135676, -1.5932244, 1.0944699, -1.3809088, 2.1067920
4: -0.4788212, 0.4909663, -1.8510408, 1.3244344, -1.8032557, 2.3420072
5: -0.4336670, 0.5031477, -1.4452169, 1.3570000, -1.7906669, 1.9483646
6: -0.3800945, 0.5120108, -1.4479425, 1.3360484, -1.7161429, 1.9599533
7: -0.4360116, 0.5466110, -1.6846129, 1.4190888, -1.8551004, 2.2312238
8: -0.5004768, 0.5907781, -1.8562331, 1.3815845, -1.8820614, 2.4470112
9: -0.4901814, 0.5045211, -1.4765712, 1.6363499, -2.1265314, 1.9810922

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.1862607, upper bound: 7.6287975
time: 1.95 seconds

## Relational analysis of IS_B1_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_B1_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.1863060, upper bound: 7.6292109
time: 2.14 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -0.3757299, 0.4385934, -3.5680876, 2.8374002
1: -2.4251304, 2.2983003, -0.4187502, 0.4465809, -2.8717113, 2.7170506
2: -3.9117460, 1.8518873, 0.2481663, 1.0724428, -4.9841890, 1.6037211
3: -3.4586265, 1.9007586, -0.2651372, 0.4933855, -3.9520121, 2.1658959
4: -3.7661660, 2.4433300, -0.4544440, 0.4593208, -4.2254868, 2.8977740
5: -2.8676689, 2.5432758, -0.4080147, 0.4731213, -3.3407900, 2.9512906
6: -2.9870868, 2.5610185, -0.3559362, 0.4863845, -3.4734712, 2.9169545
7: -3.4249156, 2.6162057, -0.4101494, 0.5149477, -3.9398632, 3.0263550
8: -3.9026318, 2.4418151, -0.4751519, 0.5632613, -4.4658928, 2.9169669
9: -2.8588891, 3.2085273, -0.4661730, 0.4688219, -3.3277109, 3.6747003

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287406, upper bound: 7.6287567
time: 2.59 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6288356, upper bound: 7.6286888
time: 3.86 seconds

## BFS IS instance: IS_B1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -2.6794171, 2.1358154, -5.2653093, 5.1410875
1: -2.4251304, 2.2983003, -2.1174948, 1.9977398, -4.4228702, 4.4157953
2: -3.9117460, 1.8518873, -3.2404983, 1.6914064, -5.6031523, 5.0923858
3: -3.4586265, 1.9007586, -2.9226379, 1.6773391, -5.1359653, 4.8233967
4: -3.7661660, 2.4433300, -3.2260008, 2.1315095, -5.8976755, 5.6693306
5: -2.8676689, 2.5432758, -2.4691849, 2.1705532, -5.0382223, 5.0124607
6: -2.9870868, 2.5610185, -2.5658884, 2.2108388, -5.1979256, 5.1269069
7: -3.4249156, 2.6162057, -2.9340572, 2.2778964, -5.7028122, 5.5502629
8: -3.9026318, 2.4418151, -3.3505292, 2.1337020, -6.0363340, 5.7923441
9: -2.8588891, 3.2085273, -2.4830496, 2.7655108, -5.6244001, 5.6915770

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287406, upper bound: 7.6315860
time: 2.89 seconds

## Relational analysis of IS_B1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_B1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6288356, upper bound: 7.6316200
time: 4.04 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.4493155, 1.2098210, -1.3434418, 1.1330652, -2.5823808, 2.5532627
1: -1.1976373, 1.1195531, -1.1160163, 1.0562403, -2.2538776, 2.2355695
2: -1.3650365, 1.3035675, -1.1966918, 1.2793392, -2.6443758, 2.5002594
3: -1.4389732, 1.0272485, -1.3116214, 0.9752318, -2.4142051, 2.3388700
4: -1.6928862, 1.2317009, -1.5705938, 1.1606678, -2.8535540, 2.8022947
5: -1.3268120, 1.2617052, -1.2318771, 1.1914653, -2.5182772, 2.4935822
6: -1.3342030, 1.2407057, -1.2250969, 1.1611276, -2.4953306, 2.4658027
7: -1.5441769, 1.3150357, -1.4232943, 1.2432532, -2.7874303, 2.7383299
8: -1.6806755, 1.2937479, -1.5390041, 1.2268938, -2.9075694, 2.8327520
9: -1.3562841, 1.5030695, -1.2688261, 1.3978652, -2.7541494, 2.7718956

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9620937, upper bound: 7.5339414
time: 2.63 seconds

## Relational analysis of IS_B1_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9620937, upper bound: 7.5744705
time: 2.49 seconds

## BFS IS instance: IS_B1_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.4093380, 1.1798502, -1.6738324, 1.3810493, -2.7903872, 2.8536825
1: -1.1668918, 1.0925207, -1.3692747, 1.2819340, -2.4488258, 2.4617953
2: -1.3024813, 1.2918483, -1.7186816, 1.3732573, -2.6757386, 3.0105300
3: -1.3898871, 1.0063906, -1.7145017, 1.1467748, -2.5366619, 2.7208924
4: -1.6441749, 1.2032182, -1.9711111, 1.3972197, -3.0413947, 3.1743293
5: -1.2896481, 1.2335820, -1.5410731, 1.4271635, -2.7168117, 2.7746551
6: -1.2953608, 1.2104204, -1.5439175, 1.4138288, -2.7091897, 2.7543378
7: -1.4993726, 1.2839645, -1.7979063, 1.4968593, -2.9962320, 3.0818708
8: -1.6259675, 1.2667897, -1.9933050, 1.4511932, -3.0771608, 3.2600946
9: -1.3200788, 1.4627522, -1.5686765, 1.7368355, -3.0569143, 3.0314288

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9621289, upper bound: 7.5542447
time: 2.22 seconds

## Relational analysis of IS_B1_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_B1_B1_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9621289, upper bound: 7.6171505
time: 4.01 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.5128613, 4.9518657, -5.8745856, 4.4722805, -10.9851418, 10.8264513
1: -5.1312685, 4.5041256, -4.5741048, 4.1023870, -9.2336559, 9.0782299
2: -8.5463114, 3.3501475, -7.6664677, 3.0658698, -11.6121807, 11.0166149
3: -7.4630375, 3.6083336, -6.7148666, 3.2566428, -10.7196808, 10.3232002
4: -7.7207422, 4.7936544, -6.9847074, 4.3422685, -12.0630112, 11.7783623
5: -5.7954903, 5.0919800, -5.2529006, 4.6153369, -10.4108276, 10.3448811
6: -6.1363516, 5.2029133, -5.5076656, 4.7004910, -10.8368425, 10.7105789
7: -7.0644946, 5.1694031, -6.3685932, 4.6838245, -11.7483196, 11.5379963
8: -8.0975590, 4.7538056, -7.3141294, 4.2800918, -12.3776512, 12.0679350
9: -5.6382923, 6.5261049, -5.1245022, 5.8989096, -11.5372019, 11.6506071

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0913304, upper bound: 7.4527742
time: 2.96 seconds

## Relational analysis of IS_B1_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.0913304, upper bound: 7.6279761
time: 2.69 seconds

## BFS IS instance: IS_B1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -6.9096441, 5.2423658, -5.1763005, 3.9672503, -10.8768940, 10.4186668
1: -5.4521885, 4.7628059, -4.0301423, 3.6396792, -9.0918674, 8.7929478
2: -9.0808659, 3.5176759, -6.7342319, 2.7398500, -11.8207159, 10.2519073
3: -7.9340968, 3.8091946, -5.8981314, 2.9127958, -10.8468924, 9.7073259
4: -8.1865387, 5.0678654, -6.1821489, 3.8531108, -12.0396500, 11.2500143
5: -6.1360221, 5.3879976, -4.6458259, 4.0937705, -10.2297926, 10.0338230
6: -6.5069914, 5.5117245, -4.8742437, 4.1486177, -10.6556091, 10.3859682
7: -7.4898524, 5.4680958, -5.6234097, 4.1566892, -11.6465416, 11.0915051
8: -8.5878477, 5.0231576, -6.4463711, 3.8126554, -12.4005032, 11.4695282
9: -5.9637508, 6.9198046, -4.5446272, 5.2423530, -11.2061043, 11.4644318

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 182

## Relational analysis of IS_B1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 182

## Relational analysis of IS_B1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1_B1_A2_A2_B2_B1_B1

### Relational analysis result of IS_B1_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279827, upper bound: 7.6301476
time: 11.78 seconds

## Relational analysis of IS_B1_B1_A2_A2_B2_B1_B2

### Relational analysis result of IS_B1_B1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6282263, upper bound: 7.6301483
time: 4.19 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 21.99 seconds
IS_B1_B1_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.5583486, upper bound: 7.6131681
IS_B1_B1_A1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.5563494, upper bound: 7.6054897
IS_B1_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.5587639, upper bound: 7.6279446
IS_B1_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.5568421, upper bound: 7.6277935
IS_B1_B1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.1862527, upper bound: 7.6280945
IS_B1_B1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.1862834, upper bound: 7.6284915
IS_B1_B1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.1862607, upper bound: 7.6287975
IS_B1_B1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.1863060, upper bound: 7.6292109
IS_B1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.6287406, upper bound: 7.6287567
IS_B1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.6288356, upper bound: 7.6286888
IS_B1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.6287406, upper bound: 7.6315860
IS_B1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.6288356, upper bound: 7.6316200
IS_B1_B1_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 21.99
Output dim: 2, lower bound: -6.9620937, upper bound: 7.5339414
IS_B1_B1_A2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 21.99
Output dim: 2, lower bound: -6.9620937, upper bound: 7.5744705
IS_B1_B1_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 21.99
Output dim: 2, lower bound: -6.9621289, upper bound: 7.5542447
IS_B1_B1_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 21.99
Output dim: 2, lower bound: -6.9621289, upper bound: 7.6171505
IS_B1_B1_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.0913304, upper bound: 7.4527742
IS_B1_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.0913304, upper bound: 7.6279761
IS_B1_B1_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.6279827, upper bound: 7.6301476
IS_B1_B1_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 21.99
Output dim: 2, lower bound: -7.6282263, upper bound: 7.6301483
IS_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6295073, upper bound: 7.6317536
IS_B1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.3082287, upper bound: 7.6282643
IS_B1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.3082287, upper bound: 7.6306110
IS_B1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308784
IS_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308802
IS_B1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308784
IS_B1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6281479, upper bound: 7.6308801
IS_B1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6283507, upper bound: 7.5645506
IS_B1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6285789, upper bound: 7.5645870
IS_B1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6307318, upper bound: 7.6313911
IS_B1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6279900, upper bound: 7.6290181
IS_B1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6303175, upper bound: 7.6317834
IS_B1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6303175, upper bound: 7.6317927
IS_B2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6279054, upper bound: 7.3382471
IS_B2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6279054, upper bound: 7.3382465
IS_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.3381811, upper bound: 7.6279284
IS_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.3381433, upper bound: 7.6282155
IS_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.3381811, upper bound: 7.6296911
IS_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.3381433, upper bound: 7.6298457
IS_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6289308, upper bound: 7.6251432
IS_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6286523, upper bound: 7.6195855
IS_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.6305242, upper bound: 7.6294102
IS_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.4745981, upper bound: 7.6294480
IS_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 21.99
Output dim: 2, lower bound: -7.4837393, upper bound: 7.6294668
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=9.388381958007812
rel_dist={2: [-7.633606580437654, 7.633606672931759]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335479, upper bound: 7.6335761
time: 3.73 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335463, upper bound: 7.6335463
time: 2.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.55 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 6.55
Output dim: 2, lower bound: -7.6335479, upper bound: 7.6335761
IS_B2, status: Status.UNKNOWN, split count: 1, time: 6.55
Output dim: 2, lower bound: -7.6335463, upper bound: 7.6335463

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -4.7909756, 3.6853030, -3.7251990, 2.9015779, -7.6925535, 7.4105020
1: -3.7366765, 3.3927815, -2.8778124, 2.6893115, -6.4259882, 6.2705936
2: -6.2026405, 2.5792634, -4.7487893, 2.1088452, -8.3114853, 7.3280525
3: -5.4332085, 2.7331824, -4.1680532, 2.1933427, -7.6265512, 6.9012356
4: -5.7178774, 3.6008029, -4.4700050, 2.8505054, -8.5683823, 8.0708084
5: -4.3159161, 3.7948947, -3.3880157, 3.0042415, -7.3201575, 7.1829104
6: -4.5275164, 3.8596551, -3.5393488, 3.0220010, -7.5495176, 7.3990040
7: -5.2140265, 3.8749912, -4.0690737, 3.0618806, -8.2759075, 7.9440651
8: -5.9650707, 3.5711460, -4.6395154, 2.8396196, -8.8046904, 8.2106609
9: -4.2284045, 4.8461180, -3.3468199, 3.8004518, -8.0288563, 8.1929379

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330454, upper bound: 7.6330971
time: 2.57 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330765, upper bound: 7.6331121
time: 2.66 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -4.9511919, 3.8029757, -4.6887450, 3.6103084, -8.5615005, 8.4917202
1: -3.8656716, 3.4973142, -3.6534214, 3.3259871, -7.1916590, 7.1507359
2: -6.4220614, 2.6485746, -6.0639324, 2.5362964, -8.9583578, 8.7125072
3: -5.6225247, 2.8142078, -5.3117399, 2.6811905, -8.3037148, 8.1259480
4: -5.9051828, 3.7118700, -5.5978603, 3.5297210, -9.4349041, 9.3097305
5: -4.4537635, 3.9152968, -4.2277665, 3.7193160, -8.1730795, 8.1430635
6: -4.6768012, 3.9845791, -4.4315343, 3.7799363, -8.4567375, 8.4161129
7: -5.3861384, 3.9958732, -5.1041079, 3.7976844, -9.1838226, 9.0999813
8: -6.1632180, 3.6809516, -5.8385901, 3.5008440, -9.6640625, 9.5195417
9: -4.3596549, 5.0042391, -4.1445389, 4.7445335, -9.1041889, 9.1487780

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335464, upper bound: 7.6335463
time: 4.75 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335464, upper bound: 7.6335465
time: 2.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 8.37 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 8.37
Output dim: 2, lower bound: -7.6330454, upper bound: 7.6330971
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 8.37
Output dim: 2, lower bound: -7.6330765, upper bound: 7.6331121
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 8.37
Output dim: 2, lower bound: -7.6335464, upper bound: 7.6335463
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 8.37
Output dim: 2, lower bound: -7.6335464, upper bound: 7.6335465

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -2.9899712, 2.3628576, -2.9973176, 2.3697085, -5.3596797, 5.3601751
1: -2.3294446, 2.2065747, -2.3335009, 2.2069466, -4.5363913, 4.5400753
2: -3.7335980, 1.8199666, -3.7462654, 1.8130248, -5.5466228, 5.5662317
3: -3.2849290, 1.8319349, -3.2937617, 1.8342454, -5.1191745, 5.1256967
4: -3.5904775, 2.3491380, -3.6018212, 2.3500030, -5.9404802, 5.9509592
5: -2.7549639, 2.4502921, -2.7513871, 2.4577379, -5.2127018, 5.2016792
6: -2.8579345, 2.4540000, -2.8649039, 2.4561791, -5.3141136, 5.3189039
7: -3.2786298, 2.5131192, -3.2827492, 2.5135856, -5.7922153, 5.7958684
8: -3.7249141, 2.3525136, -3.7358642, 2.3512032, -6.0761175, 6.0883780
9: -2.7413840, 3.0623896, -2.7425733, 3.0774326, -5.8188167, 5.8049631

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6327716, upper bound: 7.6324390
time: 3.02 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323612, upper bound: 7.6324192
time: 4.22 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -3.4656997, 2.7083182, -3.1214218, 2.4605050, -5.9262047, 5.8297400
1: -2.6719747, 2.5218318, -2.4187989, 2.2900453, -4.9620199, 4.9406309
2: -4.3868742, 2.0041418, -3.9193032, 1.8571093, -6.2439833, 5.9234447
3: -3.8597226, 2.0659814, -3.4465160, 1.8956268, -5.7553492, 5.5124974
4: -4.1618671, 2.6753273, -3.7534587, 2.4359410, -6.5978079, 6.4287863
5: -3.1715245, 2.8048944, -2.8613255, 2.5510793, -5.7226038, 5.6662197
6: -3.3018813, 2.8223674, -2.9815106, 2.5530663, -5.8549476, 5.8038778
7: -3.7930665, 2.8692207, -3.4191084, 2.6067133, -6.3997798, 6.2883291
8: -4.3153620, 2.6711991, -3.8909020, 2.4353621, -6.7507238, 6.5621014
9: -3.1373191, 3.5353332, -2.8471899, 3.2023582, -6.3396773, 6.3825231

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6328218, upper bound: 7.6324557
time: 4.15 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324287, upper bound: 7.6324396
time: 2.70 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -3.7251990, 2.9015779, -4.6887450, 3.6103084, -7.3355074, 7.5903230
1: -2.8778124, 2.6893115, -3.6534214, 3.3259871, -6.2037992, 6.3427329
2: -4.7487893, 2.1088452, -6.0639324, 2.5362964, -7.2850857, 8.1727772
3: -4.1680532, 2.1933427, -5.3117399, 2.6811905, -6.8492436, 7.5050826
4: -4.4700050, 2.8505054, -5.5978603, 3.5297210, -7.9997263, 8.4483662
5: -3.3880157, 3.0042415, -4.2277665, 3.7193160, -7.1073318, 7.2320080
6: -3.5393488, 3.0220010, -4.4315343, 3.7799363, -7.3192854, 7.4535351
7: -4.0690737, 3.0618806, -5.1041079, 3.7976844, -7.8667583, 8.1659889
8: -4.6395154, 2.8396196, -5.8385901, 3.5008440, -8.1403599, 8.6782093
9: -3.3468199, 3.8004518, -4.1445389, 4.7445335, -8.0913534, 7.9449906

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330610, upper bound: 7.6330446
time: 8.11 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330747, upper bound: 7.6330748
time: 2.72 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -4.6887450, 3.6103084, -4.6887450, 3.6103084, -8.2990532, 8.2990532
1: -3.6534214, 3.3259871, -3.6534214, 3.3259871, -6.9794083, 6.9794083
2: -6.0639324, 2.5362964, -6.0639324, 2.5362964, -8.6002293, 8.6002293
3: -5.3117399, 2.6811905, -5.3117399, 2.6811905, -7.9929304, 7.9929304
4: -5.5978603, 3.5297210, -5.5978603, 3.5297210, -9.1275816, 9.1275816
5: -4.2277665, 3.7193160, -4.2277665, 3.7193160, -7.9470825, 7.9470825
6: -4.4315343, 3.7799363, -4.4315343, 3.7799363, -8.2114706, 8.2114706
7: -5.1041079, 3.7976844, -5.1041079, 3.7976844, -8.9017925, 8.9017925
8: -5.8385901, 3.5008440, -5.8385901, 3.5008440, -9.3394337, 9.3394337
9: -4.1445389, 4.7445335, -4.1445389, 4.7445335, -8.8890724, 8.8890724

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330610, upper bound: 7.6330441
time: 3.24 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330747, upper bound: 7.6330753
time: 3.16 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 8.00 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 8.00
Output dim: 2, lower bound: -7.6327716, upper bound: 7.6324390
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 8.00
Output dim: 2, lower bound: -7.6323612, upper bound: 7.6324192
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 8.00
Output dim: 2, lower bound: -7.6328218, upper bound: 7.6324557
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 8.00
Output dim: 2, lower bound: -7.6324287, upper bound: 7.6324396
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 8.00
Output dim: 2, lower bound: -7.6330610, upper bound: 7.6330446
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 8.00
Output dim: 2, lower bound: -7.6330747, upper bound: 7.6330748
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 8.00
Output dim: 2, lower bound: -7.6330610, upper bound: 7.6330441
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 8.00
Output dim: 2, lower bound: -7.6330747, upper bound: 7.6330753

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -2.3151975, 1.8625641, -2.7276950, 2.1729178, -4.4881153, 4.5902591
1: -1.8558949, 1.7448099, -2.1480026, 2.0260634, -3.8819585, 3.8928125
2: -2.7631500, 1.5726111, -3.3686826, 1.7159586, -4.4791088, 4.9412937
3: -2.4932914, 1.4910572, -2.9669166, 1.7006347, -4.1939259, 4.4579735
4: -2.7511826, 1.8712901, -3.2738788, 2.1627779, -4.9139605, 5.1451688
5: -2.1427438, 1.9427977, -2.5115216, 2.2552068, -4.3979506, 4.4543190
6: -2.2092307, 1.9236181, -2.6118550, 2.2460961, -4.4553270, 4.5354729
7: -2.5191453, 2.0075314, -2.9860487, 2.3115568, -4.8307018, 4.9935799
8: -2.8667698, 1.8952205, -3.4003899, 2.1680882, -5.0348577, 5.2956104
9: -2.1585655, 2.3879640, -2.5151827, 2.8083096, -4.9668751, 4.9031467

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323605, upper bound: 7.6324198
time: 4.78 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323605, upper bound: 7.6324198
time: 2.82 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -5.7190075, 4.3689775, -2.6799695, 2.1383300, -7.8573375, 7.0489473
1: -4.2014623, 4.0458031, -2.1153226, 1.9939835, -6.1954460, 6.1611257
2: -7.5626879, 2.8519831, -3.3015814, 1.6984549, -9.2611427, 6.1535645
3: -6.6175451, 3.1929245, -2.9139428, 1.6769474, -8.2944927, 6.1068673
4: -6.8958654, 4.2538137, -3.2160358, 2.1296773, -9.0255432, 7.4698496
5: -5.2023110, 4.5596952, -2.4689925, 2.2197416, -7.4220524, 7.0286875
6: -5.4277568, 4.5908957, -2.5672934, 2.2093468, -7.6371036, 7.1581888
7: -6.2923150, 4.5697994, -2.9334710, 2.2764809, -8.5687962, 7.5032701
8: -7.1313362, 4.2114410, -3.3410714, 2.1363740, -9.2677097, 7.5525122
9: -5.0440254, 5.7778521, -2.4748447, 2.7623358, -7.8063612, 8.2526970

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324197
time: 3.86 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324192
time: 2.88 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -2.7200260, 2.1661086, -2.8512540, 2.2627337, -4.9827595, 5.0173626
1: -2.1433887, 2.0247507, -2.2328126, 2.1082034, -4.2515922, 4.2575636
2: -3.3543868, 1.7220030, -3.5412455, 1.7594935, -5.1138802, 5.2632484
3: -2.9604073, 1.6977860, -3.1148162, 1.7615002, -4.7219076, 4.8126020
4: -3.2632613, 2.1607199, -3.4245656, 2.2476377, -5.5108991, 5.5852852
5: -2.5136003, 2.2481799, -2.6210511, 2.3473935, -4.8609939, 4.8692312
6: -2.6038549, 2.2434061, -2.7279384, 2.3417883, -4.9456434, 4.9713445
7: -2.9809713, 2.3109436, -3.1219184, 2.4031744, -5.3841457, 5.4328623
8: -3.3895664, 2.1683950, -3.5538847, 2.2512808, -5.6408472, 5.7222795
9: -2.5137594, 2.7942827, -2.6194108, 2.9310386, -5.4447980, 5.4136934

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324280, upper bound: 7.6324401
time: 2.86 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324280, upper bound: 7.6324401
time: 2.57 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -6.3205810, 4.8004994, -2.8037958, 2.2279718, -8.5485525, 7.6042953
1: -4.9374208, 4.3939800, -2.2000954, 2.0762708, -7.0136919, 6.5940752
2: -8.2714624, 3.2594941, -3.4745617, 1.7419813, -10.0134439, 6.7340555
3: -7.2546554, 3.4685085, -3.0567567, 1.7379328, -8.9925880, 6.5252652
4: -7.5177426, 4.6388021, -3.3670454, 2.2146854, -9.7324276, 8.0058479
5: -5.6542635, 4.9533744, -2.5787539, 2.3113594, -7.9656229, 7.5321283
6: -5.9270120, 5.0531297, -2.6836472, 2.3047366, -8.2317486, 7.7367768
7: -6.8490715, 5.0186992, -3.0696311, 2.3672628, -9.2163343, 8.0883303
8: -7.8645062, 4.5837641, -3.4948890, 2.2189007, -10.0834064, 8.0786533
9: -5.4922948, 6.3318939, -2.5792959, 2.8837731, -8.3760681, 8.9111900

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
time: 2.89 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
time: 2.28 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2.9973176, 2.3697085, -2.8952935, 2.2936373, -5.2909546, 5.2650023
1: -2.3335009, 2.2069466, -2.2642684, 2.1427622, -4.4762630, 4.4712152
2: -3.7462654, 1.8130248, -3.6013370, 1.7856565, -5.5319219, 5.4143620
3: -3.2937617, 1.8342454, -3.1685982, 1.7848133, -5.0785751, 5.0028439
4: -3.6018212, 2.3500030, -3.4752483, 2.2829573, -5.8847785, 5.8252516
5: -2.7513871, 2.4577379, -2.6703582, 2.3790715, -5.1304588, 5.1280961
6: -2.8649039, 2.4561791, -2.7688076, 2.3800302, -5.2449341, 5.2249870
7: -3.2827492, 2.5135856, -3.1743832, 2.4417624, -5.7245116, 5.6879687
8: -3.7358642, 2.3512032, -3.6069160, 2.2877617, -6.0236259, 5.9581194
9: -2.7425733, 3.0774326, -2.6614912, 2.9672120, -5.7097855, 5.7389240

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324383, upper bound: 7.6327710
time: 3.74 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323612
time: 3.45 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.1214218, 2.4605050, -3.3594167, 2.6305923, -5.7520142, 5.8199215
1: -2.4187989, 2.2900453, -2.5864820, 2.4517391, -4.8705378, 4.8765273
2: -3.9193032, 1.8571093, -4.2425985, 1.9599543, -5.8792572, 6.0997076
3: -3.4465160, 1.8956268, -3.7323713, 2.0136664, -5.4601822, 5.6279984
4: -3.7534587, 2.4359410, -4.0356216, 2.6024401, -6.3558989, 6.4715624
5: -2.8613255, 2.5510793, -3.0788145, 2.7263789, -5.5877047, 5.6298938
6: -2.9815106, 2.5530663, -3.2031105, 2.7394447, -5.7209554, 5.7561769
7: -3.4191084, 2.6067133, -3.6790948, 2.7887216, -6.2078300, 6.2858081
8: -3.8909020, 2.4353621, -4.1837339, 2.5995026, -6.4904046, 6.6190958
9: -2.8471899, 3.2023582, -3.0494063, 3.4293561, -6.2765460, 6.2517643

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324556, upper bound: 7.6328217
time: 15.84 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324394, upper bound: 7.6324286
time: 2.98 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.9076688, 3.0339515, -2.8952935, 2.2936373, -6.2013063, 5.9292450
1: -3.0225821, 2.8135800, -2.2642684, 2.1427622, -5.1653442, 5.0778484
2: -4.9962831, 2.2021904, -3.6013370, 1.7856565, -6.7819395, 5.8035274
3: -4.3835006, 2.2860026, -3.1685982, 1.7848133, -6.1683140, 5.4546008
4: -4.6791110, 2.9837708, -3.4752483, 2.2829573, -6.9620686, 6.4590192
5: -3.5542026, 3.1377485, -2.6703582, 2.3790715, -5.9332743, 5.8081064
6: -3.7059753, 3.1675823, -2.7688076, 2.3800302, -6.0860052, 5.9363899
7: -4.2665973, 3.2041330, -3.1743832, 2.4417624, -6.7083597, 6.3785162
8: -4.8673449, 2.9676681, -3.6069160, 2.2877617, -7.1551065, 6.5745840
9: -3.5012918, 3.9700670, -2.6614912, 2.9672120, -6.4685040, 6.6315584

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324311, upper bound: 7.6327734
time: 3.45 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324095, upper bound: 7.6323609
time: 3.08 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0467625, 3.1360638, -3.3594167, 2.6305923, -6.6773548, 6.4954805
1: -3.1339869, 2.9049907, -2.5864820, 2.4517391, -5.5857258, 5.4914727
2: -5.1853676, 2.2577319, -4.2425985, 1.9599543, -7.1453218, 6.5003304
3: -4.5500417, 2.3552206, -3.7323713, 2.0136664, -6.5637083, 6.0875921
4: -4.8448181, 3.0797195, -4.0356216, 2.6024401, -7.4472580, 7.1153412
5: -3.6740952, 3.2401381, -3.0788145, 2.7263789, -6.4004741, 6.3189526
6: -3.8351860, 3.2770946, -3.2031105, 2.7394447, -6.5746307, 6.4802051
7: -4.4155798, 3.3093321, -3.6790948, 2.7887216, -7.2043014, 6.9884272
8: -5.0399632, 3.0617819, -4.1837339, 2.5995026, -7.6394658, 7.2455158
9: -3.6164696, 4.1096478, -3.0494063, 3.4293561, -7.0458260, 7.1590538

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324473, upper bound: 7.6328235
time: 2.89 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324284, upper bound: 7.6324278
time: 2.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.42 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6323605, upper bound: 7.6324198
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6323605, upper bound: 7.6324198
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324197
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324192
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324280, upper bound: 7.6324401
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324280, upper bound: 7.6324401
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324383, upper bound: 7.6327710
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323612
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324556, upper bound: 7.6328217
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324394, upper bound: 7.6324286
IS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324311, upper bound: 7.6327734
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324095, upper bound: 7.6323609
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324473, upper bound: 7.6328235
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.42
Output dim: 2, lower bound: -7.6324284, upper bound: 7.6324278

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -2.3151975, 1.8625641, -2.2380745, 1.8050854, -4.1202831, 4.1006384
1: -1.8558949, 1.7448099, -1.7980882, 1.6848613, -3.5407562, 3.5428982
2: -2.7631500, 1.5726111, -2.6492724, 1.5403659, -4.3035159, 4.2218838
3: -2.4932914, 1.4910572, -2.3990428, 1.4490991, -3.9423904, 3.8901000
4: -2.7511826, 1.8712901, -2.6521759, 1.8108498, -4.5620322, 4.5234661
5: -2.1427438, 1.9427977, -2.0623641, 1.8858886, -4.0286322, 4.0051618
6: -2.2092307, 1.9236181, -2.1315165, 1.8590215, -4.0682521, 4.0551348
7: -2.5191453, 2.0075314, -2.4316661, 1.9428017, -4.4619470, 4.4391975
8: -2.8667698, 1.8952205, -2.7659321, 1.8354522, -4.7022219, 4.6611528
9: -2.1585655, 2.3879640, -2.0822058, 2.3160677, -4.4746332, 4.4701700

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307164, upper bound: 7.6281447
time: 4.98 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323442, upper bound: 7.6320199
time: 5.84 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -2.3151975, 1.8625641, -5.6185951, 4.2716193, -6.5868168, 7.4811592
1: -1.8558949, 1.7448099, -4.1213198, 3.9566376, -5.8125324, 5.8661299
2: -2.7631500, 1.5726111, -7.4328771, 2.7999535, -5.5631037, 9.0054884
3: -2.4932914, 1.4910572, -6.1725292, 3.1276188, -5.6209102, 7.6635866
4: -2.7511826, 1.8712901, -6.7716045, 4.1562223, -6.9074049, 8.6428947
5: -2.1427438, 1.9427977, -5.0997190, 4.4419203, -6.5846643, 7.0425167
6: -2.2092307, 1.9236181, -5.3159370, 4.4713960, -6.6806269, 7.2395554
7: -2.5191453, 2.0075314, -6.1811018, 4.4241567, -6.9433022, 8.1886330
8: -2.8667698, 1.8952205, -6.9815183, 4.0893936, -6.9561634, 8.8767385
9: -2.1585655, 2.3879640, -4.9549112, 5.5824480, -7.7410135, 7.3428755

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307164, upper bound: 7.6281441
time: 6.69 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323442, upper bound: 7.6320198
time: 5.86 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -5.7190075, 4.3689775, -2.2380745, 1.8050854, -7.5240927, 6.6070518
1: -4.2014623, 4.0458031, -1.7980882, 1.6848613, -5.8863235, 5.8438911
2: -7.5626879, 2.8519831, -2.6492724, 1.5403659, -9.1030540, 5.5012555
3: -6.6175451, 3.1929245, -2.3990428, 1.4490991, -8.0666447, 5.5919676
4: -6.8958654, 4.2538137, -2.6521759, 1.8108498, -8.7067156, 6.9059896
5: -5.2023110, 4.5596952, -2.0623641, 1.8858886, -7.0881996, 6.6220593
6: -5.4277568, 4.5908957, -2.1315165, 1.8590215, -7.2867785, 6.7224121
7: -6.2923150, 4.5697994, -2.4316661, 1.9428017, -8.2351170, 7.0014658
8: -7.1313362, 4.2114410, -2.7659321, 1.8354522, -8.9667883, 6.9773731
9: -5.0440254, 5.7778521, -2.0822058, 2.3160677, -7.3600931, 7.8600578

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283727, upper bound: 7.6304619
time: 3.16 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319270, upper bound: 7.6320006
time: 3.65 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -5.7190075, 4.3689775, -5.6185951, 4.2716193, -9.9906273, 9.9875727
1: -4.2014623, 4.0458031, -4.1213198, 3.9566376, -8.1581001, 8.1671228
2: -7.5626879, 2.8519831, -7.4328771, 2.7999535, -10.3626413, 10.2848606
3: -6.6175451, 3.1929245, -6.1725292, 3.1276188, -9.7451639, 9.3654537
4: -6.8958654, 4.2538137, -6.7716045, 4.1562223, -11.0520878, 11.0254183
5: -5.2023110, 4.5596952, -5.0997190, 4.4419203, -9.6442318, 9.6594143
6: -5.4277568, 4.5908957, -5.3159370, 4.4713960, -9.8991528, 9.9068327
7: -6.2923150, 4.5697994, -6.1811018, 4.4241567, -10.7164717, 10.7509012
8: -7.1313362, 4.2114410, -6.9815183, 4.0893936, -11.2207298, 11.1929588
9: -5.0440254, 5.7778521, -4.9549112, 5.5824480, -10.6264734, 10.7327633

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324198
time: 2.62 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324191
time: 3.28 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -2.7200260, 2.1661086, -2.3338535, 1.8806324, -4.6006584, 4.4999619
1: -2.1433887, 2.0247507, -1.8705406, 1.7552400, -3.8986287, 3.8952913
2: -3.3543868, 1.7220030, -2.7982168, 1.5723163, -4.9267030, 4.5202198
3: -2.9604073, 1.6977860, -2.5174484, 1.5006043, -4.4610114, 4.2152343
4: -3.2632613, 2.1607199, -2.7822888, 1.8823733, -5.1456347, 4.9430084
5: -2.5136003, 2.2481799, -2.1522994, 1.9604743, -4.4740744, 4.4004793
6: -2.6038549, 2.2434061, -2.2313132, 1.9377041, -4.5415592, 4.4747190
7: -2.9809713, 2.3109436, -2.5428867, 2.0178661, -4.9988375, 4.8538303
8: -3.3895664, 2.1683950, -2.8974190, 1.9023521, -5.2919188, 5.0658140
9: -2.5137594, 2.7942827, -2.1729269, 2.4178085, -4.9315681, 4.9672098

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307375, upper bound: 7.6281819
time: 2.94 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324225, upper bound: 7.6320435
time: 4.92 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -2.7200260, 2.1661086, -5.7596278, 4.3945508, -7.1145768, 7.9257364
1: -2.1433887, 2.0247507, -4.2299485, 4.0584455, -6.2018342, 6.2546992
2: -3.3543868, 1.7220030, -7.6227016, 2.8566599, -6.2110467, 9.3447046
3: -2.9604073, 1.6977860, -6.6704755, 3.2000456, -6.1604528, 8.3682613
4: -3.2632613, 2.1607199, -6.9428921, 4.2715836, -7.5348449, 9.1036119
5: -2.5136003, 2.2481799, -5.2225118, 4.5900369, -7.1036372, 7.4706917
6: -2.6038549, 2.2434061, -5.4462862, 4.6206875, -7.2245426, 7.6896925
7: -2.9809713, 2.3109436, -6.3334379, 4.5902991, -7.5712705, 8.6443815
8: -3.3895664, 2.1683950, -7.1581602, 4.2342615, -7.6238279, 9.3265553
9: -2.5137594, 2.7942827, -5.0722699, 5.8229094, -8.3366690, 7.8665524

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307375, upper bound: 7.6281815
time: 3.27 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324225, upper bound: 7.6320434
time: 5.29 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -6.3205810, 4.8004994, -2.3338535, 1.8806324, -8.2012138, 7.1343527
1: -4.9374208, 4.3939800, -1.8705406, 1.7552400, -6.6926608, 6.2645206
2: -8.2714624, 3.2594941, -2.7982168, 1.5723163, -9.8437786, 6.0577106
3: -7.2546554, 3.4685085, -2.5174484, 1.5006043, -8.7552595, 5.9859571
4: -7.5177426, 4.6388021, -2.7822888, 1.8823733, -9.4001160, 7.4210911
5: -5.6542635, 4.9533744, -2.1522994, 1.9604743, -7.6147375, 7.1056738
6: -5.9270120, 5.0531297, -2.2313132, 1.9377041, -7.8647161, 7.2844429
7: -6.8490715, 5.0186992, -2.5428867, 2.0178661, -8.8669376, 7.5615859
8: -7.8645062, 4.5837641, -2.8974190, 1.9023521, -9.7668581, 7.4811831
9: -5.4922948, 6.3318939, -2.1729269, 2.4178085, -7.9101033, 8.5048208

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261535, upper bound: 7.6304678
time: 5.24 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320165, upper bound: 7.6320278
time: 3.71 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -6.3205810, 4.8004994, -5.7575884, 4.3945508, -10.7151318, 10.5580883
1: -4.9374208, 4.3939800, -4.2299485, 4.0489388, -8.9863596, 8.6239281
2: -8.2714624, 3.2594941, -7.6227016, 2.8517237, -11.1231861, 10.8821955
3: -7.2546554, 3.4685085, -6.6704755, 3.1960638, -10.4507189, 10.1389837
4: -7.5177426, 4.6388021, -6.9428921, 4.2515326, -11.7692757, 11.5816936
5: -5.6542635, 4.9533744, -5.2222462, 4.5900369, -10.2443008, 10.1756210
6: -5.9270120, 5.0531297, -5.4462862, 4.6091986, -10.5362110, 10.4994164
7: -6.8490715, 5.0186992, -6.3334379, 4.5891066, -11.4381781, 11.3521366
8: -7.8645062, 4.5837641, -7.1542482, 4.2342615, -12.0987682, 11.7380123
9: -5.4922948, 6.3318939, -5.0722699, 5.8165703, -11.3088646, 11.4041634

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
time: 3.13 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324396
time: 2.17 seconds

## BFS IS instance: IS_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -2.7276950, 2.1729178, -2.2355657, 1.7990253, -4.5267200, 4.4084835
1: -2.1480026, 2.0260634, -1.7956465, 1.6860437, -3.8340464, 3.8217101
2: -3.3686826, 1.7159586, -2.6386285, 1.5432963, -4.9119787, 4.3545871
3: -2.9669166, 1.7006347, -2.3953915, 1.4480960, -4.4150124, 4.0960264
4: -3.2738788, 2.1627779, -2.6439750, 1.8106759, -5.0845547, 4.8067532
5: -2.5115216, 2.2552068, -2.0660846, 1.8807189, -4.3922405, 4.3212914
6: -2.6118550, 2.2460961, -2.1259298, 1.8584037, -4.4702587, 4.3720260
7: -2.9860487, 2.3115568, -2.4252632, 1.9449842, -4.9310331, 4.7368202
8: -3.4003899, 2.1680882, -2.7569976, 1.8393042, -5.2396941, 4.9250860
9: -2.5151827, 2.8083096, -2.0834792, 2.3034928, -4.8186755, 4.8917885

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A1_B1_B1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324197, upper bound: 7.6323606
time: 11.87 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324197, upper bound: 7.6323611
time: 4.43 seconds

## BFS IS instance: IS_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2.6799695, 2.1383300, -5.6204772, 4.2910242, -6.9709940, 7.7588072
1: -2.1153226, 1.9939835, -4.1353722, 3.9699244, -6.0852470, 6.1293554
2: -3.3015814, 1.6984549, -7.4287090, 2.8142042, -6.1157856, 9.1271639
3: -2.9139428, 1.6769474, -6.4994969, 3.1324167, -6.0463595, 8.1764441
4: -3.2160358, 2.1296773, -6.7710991, 4.1788130, -7.3948488, 8.9007759
5: -2.4689925, 2.2197416, -5.1077037, 4.4831533, -6.9521456, 7.3274450
6: -2.5672934, 2.2093468, -5.3151417, 4.5160017, -7.0832949, 7.5244884
7: -2.9334710, 2.2764809, -6.1839952, 4.4906549, -7.4241257, 8.4604759
8: -3.3410714, 2.1363740, -6.9841442, 4.1458650, -7.4869366, 9.1205177
9: -2.4748447, 2.7623358, -4.9601774, 5.6757460, -8.1505909, 7.7225132

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323612
time: 2.88 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323605
time: 4.24 seconds

## BFS IS instance: IS_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2.8512540, 2.2627337, -2.6196442, 2.0933833, -4.9446373, 4.8823776
1: -2.2328126, 2.1082034, -2.0746675, 1.9575135, -4.1903262, 4.1828709
2: -3.5412455, 1.7594935, -3.2131815, 1.6855036, -5.2267489, 4.9726748
3: -3.1148162, 1.7615002, -2.8486047, 1.6480401, -4.7628565, 4.6101050
4: -3.4245656, 2.2476377, -3.1411972, 2.0911605, -5.5157261, 5.3888350
5: -2.6210511, 2.3473935, -2.4237320, 2.1737525, -4.7948036, 4.7711258
6: -2.7279384, 2.3417883, -2.5094218, 2.1663046, -4.8942432, 4.8512101
7: -3.1219184, 2.4031744, -2.8701410, 2.2375689, -5.3594875, 5.2733154
8: -3.5538847, 2.2512808, -3.2649107, 2.1014581, -5.6553431, 5.5161915
9: -2.6194108, 2.9310386, -2.4290111, 2.6969142, -5.3163252, 5.3600497

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324401, upper bound: 7.6324286
time: 2.78 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324401, upper bound: 7.6324287
time: 13.31 seconds

## BFS IS instance: IS_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2.8037958, 2.2279718, -6.1708174, 4.6784267, -7.4822226, 8.3987894
1: -2.2000954, 2.0762708, -4.8206434, 4.2905436, -6.4906387, 6.8969145
2: -3.4745617, 1.7419813, -8.0471668, 3.1853733, -6.6599350, 9.7891483
3: -3.0567567, 1.7379328, -7.0704432, 3.3784189, -6.4351759, 8.8083763
4: -3.3670454, 2.2146854, -7.3469925, 4.5044041, -7.8714495, 9.5616779
5: -2.5787539, 2.3113594, -5.5247989, 4.8391924, -7.4179463, 7.8361583
6: -2.6836472, 2.3047366, -5.7866602, 4.9247112, -7.6083584, 8.0913963
7: -3.0696311, 2.3672628, -6.6892972, 4.9057126, -7.9753437, 9.0565605
8: -3.4948890, 2.2189007, -7.6365604, 4.4801340, -7.9750233, 9.8554611
9: -2.5792959, 2.8837731, -5.3680449, 6.1882858, -8.7675819, 8.2518177

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324395, upper bound: 7.6324281
time: 3.61 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324395, upper bound: 7.6324286
time: 3.08 seconds

## BFS IS instance: IS_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -3.6245940, 2.8274207, -2.2355657, 1.7990253, -5.4236193, 5.0629864
1: -2.7959232, 2.6270945, -1.7956465, 1.6860437, -4.4819670, 4.4227409
2: -4.6116285, 2.0867095, -2.6386285, 1.5432963, -6.1549249, 4.7253380
3: -4.0458851, 2.1457882, -2.3953915, 1.4480960, -5.4939814, 4.5411797
4: -4.3439732, 2.7887988, -2.6439750, 1.8106759, -6.1546488, 5.4327736
5: -3.3095040, 2.9292896, -2.0660846, 1.8807189, -5.1902227, 4.9953742
6: -3.4430983, 2.9461675, -2.1259298, 1.8584037, -5.3015022, 5.0720973
7: -3.9634364, 2.9904122, -2.4252632, 1.9449842, -5.9084206, 5.4156752
8: -4.5167484, 2.7771845, -2.7569976, 1.8393042, -6.3560524, 5.5341821
9: -3.2673078, 3.6875274, -2.0834792, 2.3034928, -5.5708008, 5.7710066

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324089, upper bound: 7.6323609
time: 3.18 seconds

## Relational analysis of IS_B2_A2_B1_B1_A2

### Relational analysis result of IS_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324089, upper bound: 7.6323609
time: 4.67 seconds

## BFS IS instance: IS_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -3.5749564, 2.7911797, -5.6204772, 4.2910242, -7.8659806, 8.4116573
1: -2.7562363, 2.5942686, -4.1353722, 3.9699244, -6.7261610, 6.7296410
2: -4.5440865, 2.0658956, -7.4287090, 2.8142042, -7.3582907, 9.4946041
3: -3.9869032, 2.1212063, -6.4994969, 3.1324167, -7.1193199, 8.6207027
4: -4.2853813, 2.7546802, -6.7710991, 4.1788130, -8.4641943, 9.5257797
5: -3.2665367, 2.8925681, -5.1077037, 4.4831533, -7.7496901, 8.0002718
6: -3.3974385, 2.9073250, -5.3151417, 4.5160017, -7.9134402, 8.2224665
7: -3.9103365, 2.9527512, -6.1839952, 4.4906549, -8.4009914, 9.1367464
8: -4.4551463, 2.7438636, -6.9841442, 4.1458650, -8.6010113, 9.7280083
9: -3.2262805, 3.6382346, -4.9601774, 5.6757460, -8.9020262, 8.5984116

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6323609
time: 3.60 seconds

## Relational analysis of IS_B2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6323609
time: 4.65 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -3.7643478, 2.9293232, -2.6196442, 2.0933833, -5.8577309, 5.5489674
1: -2.9076014, 2.7188129, -2.0746675, 1.9575135, -4.8651147, 4.7934804
2: -4.8008823, 2.1425965, -3.2131815, 1.6855036, -6.4863858, 5.3557777
3: -4.2132230, 2.2147810, -2.8486047, 1.6480401, -5.8612633, 5.0633860
4: -4.5102758, 2.8844206, -3.1411972, 2.0911605, -6.6014366, 6.0256176
5: -3.4300013, 3.0320659, -2.4237320, 2.1737525, -5.6037540, 5.4557981
6: -3.5729051, 3.0552044, -2.5094218, 2.1663046, -5.7392097, 5.5646262
7: -4.1131449, 3.0956268, -2.8701410, 2.2375689, -6.3507137, 5.9657679
8: -4.6902037, 2.8706493, -3.2649107, 2.1014581, -6.7916617, 6.1355600
9: -3.3830318, 3.8275721, -2.4290111, 2.6969142, -6.0799460, 6.2565832

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324279, upper bound: 7.6324284
time: 4.40 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324279, upper bound: 7.6324278
time: 7.68 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -3.7136314, 2.8923538, -6.1708174, 4.6784267, -8.3920584, 9.0631714
1: -2.8671050, 2.6852334, -4.8206434, 4.2905436, -7.1576486, 7.5058765
2: -4.7320290, 2.1213841, -8.0471668, 3.1853733, -7.9174023, 10.1685505
3: -4.1530290, 2.1896510, -7.0704432, 3.3784189, -7.5314479, 9.2600937
4: -4.4504952, 2.8494086, -7.3469925, 4.5044041, -8.9548988, 10.1964016
5: -3.3861663, 2.9945903, -5.5247989, 4.8391924, -8.2253590, 8.5193892
6: -3.5260615, 3.0155828, -5.7866602, 4.9247112, -8.4507732, 8.8022432
7: -4.0588980, 3.0571866, -6.6892972, 4.9057126, -8.9646111, 9.7464838
8: -4.6273704, 2.8366013, -7.6365604, 4.4801340, -9.1075039, 10.4731617
9: -3.3410745, 3.7772696, -5.3680449, 6.1882858, -9.5293598, 9.1453142

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324277
time: 3.31 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324284
time: 2.94 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.78 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6307164, upper bound: 7.6281447
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6323442, upper bound: 7.6320199
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6307164, upper bound: 7.6281441
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6323442, upper bound: 7.6320198
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6283727, upper bound: 7.6304619
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6319270, upper bound: 7.6320006
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324198
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324191
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6307375, upper bound: 7.6281819
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324225, upper bound: 7.6320435
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6307375, upper bound: 7.6281815
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324225, upper bound: 7.6320434
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6261535, upper bound: 7.6304678
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6320165, upper bound: 7.6320278
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324396
IS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324197, upper bound: 7.6323606
IS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324197, upper bound: 7.6323611
IS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323612
IS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323605
IS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324401, upper bound: 7.6324286
IS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324401, upper bound: 7.6324287
IS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324395, upper bound: 7.6324281
IS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324395, upper bound: 7.6324286
IS_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324089, upper bound: 7.6323609
IS_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324089, upper bound: 7.6323609
IS_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6323609
IS_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6323609
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324279, upper bound: 7.6324284
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324279, upper bound: 7.6324278
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324277
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.78
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324284

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.5742193, 0.6161574, -0.1558935, 0.1683881, -0.7426074, 0.7720509
1: -0.5799464, 0.5992749, -0.2020659, 0.2123953, -0.7923417, 0.8013408
2: -0.0456628, 1.1171672, 0.6011345, 1.0350590, -1.0807219, 0.5160328
3: -0.4268622, 0.6182083, -0.0762342, 0.2830463, -0.7099085, 0.6944425
4: -0.6577558, 0.6442816, -0.2180401, 0.2160968, -0.8738526, 0.8623216
5: -0.5705507, 0.6804613, -0.1963118, 0.2151469, -0.7856975, 0.8767731
6: -0.5334853, 0.6385880, -0.1595097, 0.2242475, -0.7577328, 0.7980976
7: -0.6026897, 0.7060025, -0.1881549, 0.2411661, -0.8438557, 0.8941574
8: -0.6733687, 0.7379457, -0.2117272, 0.3074552, -0.9808239, 0.9496729
9: -0.6323479, 0.6944107, -0.2238538, 0.1974250, -0.8297729, 0.9182645

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6300020, upper bound: 7.5860490
time: 3.90 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302489, upper bound: 7.5860504
time: 2.70 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2.0870566, 1.6858394, -1.3644453, 1.1501882, -3.2372448, 3.0502849
1: -1.6829654, 1.5765797, -1.1318805, 1.0669785, -2.7499437, 2.7084603
2: -2.3958759, 1.4946616, -1.2343626, 1.2828192, -3.6786952, 2.7290242
3: -2.2124774, 1.3681638, -1.3378156, 0.9844181, -3.1968956, 2.7059793
4: -2.4623585, 1.6999377, -1.5969075, 1.1730683, -3.6354268, 3.2968452
5: -1.9270306, 1.7584631, -1.2479331, 1.2082707, -3.1353011, 3.0063963
6: -1.9723170, 1.7388371, -1.2479024, 1.1776177, -3.1499348, 2.9867396
7: -2.2560096, 1.8258598, -1.4493136, 1.2553008, -3.5113103, 3.2751734
8: -2.5543363, 1.7349540, -1.5698447, 1.2372620, -3.7915983, 3.3047986
9: -1.9438657, 2.1525607, -1.2857995, 1.4212538, -3.3651195, 3.4383602

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305727, upper bound: 7.6315649
time: 3.11 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305728, upper bound: 7.6326769
time: 3.83 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.5742193, 0.6161574, -1.4339643, 1.1984482, -1.7726674, 2.0501218
1: -0.5799464, 0.5992749, -1.1851937, 1.1114624, -1.6914088, 1.7844685
2: -0.0456628, 1.1171672, -1.3429151, 1.2996322, -1.3452950, 2.4600823
3: -0.4268622, 0.6182083, -1.4198550, 1.0189645, -1.4458268, 2.0380633
4: -0.6577558, 0.6442816, -1.6753489, 1.2212707, -1.8790264, 2.3196304
5: -0.5705507, 0.6804613, -1.3111669, 1.2524818, -1.8230325, 1.9916282
6: -0.5334853, 0.6385880, -1.3173723, 1.2309831, -1.7644684, 1.9559603
7: -0.6026897, 0.7060025, -1.5275031, 1.3041041, -1.9067938, 2.2335057
8: -0.6733687, 0.7379457, -1.6616356, 1.2835367, -1.9569054, 2.3995814
9: -0.6323479, 0.6944107, -1.3437171, 1.4875968, -2.1199446, 2.0381279

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290859, upper bound: 7.4743615
time: 3.11 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_B2

### Relational analysis result of IS_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289733, upper bound: 7.4419433
time: 3.17 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2.0870566, 1.6858394, -4.5606079, 3.5044854, -5.5915422, 6.2464476
1: -1.6829654, 1.5765797, -3.3987250, 3.2476947, -4.9306602, 4.9753046
2: -2.3958759, 1.4946616, -5.9579086, 2.4050877, -4.8009634, 7.4525700
3: -2.2124774, 1.3681638, -5.0008011, 2.6033912, -4.8158684, 6.3689651
4: -2.4623585, 1.6999377, -5.4930792, 3.4236238, -5.8859825, 7.1930170
5: -1.9270306, 1.7584631, -4.1529512, 3.6440611, -5.5710917, 5.9114141
6: -1.9723170, 1.7388371, -4.3277369, 3.6579456, -5.6302624, 6.0665741
7: -2.2560096, 1.8258598, -5.0142360, 3.6484683, -5.9044781, 6.8400955
8: -2.5543363, 1.7349540, -5.6715260, 3.3842456, -5.9385819, 7.4064798
9: -1.9438657, 2.1525607, -4.0622034, 4.5646563, -6.5085220, 6.2147641

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316419, upper bound: 7.6309815
time: 4.90 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317932, upper bound: 7.6314472
time: 3.05 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.5729212, 1.3003097, -0.5290306, 0.5901527, -2.1630738, 1.8293402
1: -1.2916257, 1.2064134, -0.5534776, 0.5771359, -1.8687615, 1.7598910
2: -1.5531341, 1.3401685, 0.0126180, 1.1089032, -2.6620374, 1.3275505
3: -1.5894464, 1.0895388, -0.3844105, 0.6012738, -2.1907203, 1.4739493
4: -1.8438501, 1.3189608, -0.6121894, 0.6145369, -2.4583871, 1.9311502
5: -1.4452966, 1.3476498, -0.5431017, 0.6472400, -2.0925367, 1.8907515
6: -1.4485787, 1.3399221, -0.4972519, 0.6147041, -2.0632830, 1.8371739
7: -1.6846026, 1.4115483, -0.5641073, 0.6789837, -2.3635864, 1.9756556
8: -1.8512034, 1.3800359, -0.6342140, 0.7126825, -2.5638859, 2.0142498
9: -1.4756049, 1.6203052, -0.6002016, 0.6578673, -2.1334722, 2.2205067

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4878332, upper bound: 7.6293744
time: 2.24 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4224551, upper bound: 7.6292976
time: 3.64 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.6485190, 3.5828099, -2.0160203, 1.6369640, -6.2854829, 5.5988302
1: -3.4665895, 3.3214092, -1.6295309, 1.5221844, -4.9887738, 4.9509401
2: -6.0722737, 2.4485002, -2.2899611, 1.4721330, -7.5444069, 4.7384615
3: -5.3124390, 2.6569118, -2.1254168, 1.3296511, -6.6420898, 4.7823286
4: -5.6009140, 3.5029910, -2.3793466, 1.6467444, -7.2476583, 5.8823376
5: -4.2426496, 3.7354851, -1.8554683, 1.7073951, -5.9500446, 5.5909534
6: -4.4214711, 3.7535670, -1.9012587, 1.6823790, -6.1038504, 5.6548257
7: -5.1125979, 3.7600589, -2.1823444, 1.7655002, -6.8780980, 5.9424033
8: -5.7955399, 3.4800866, -2.4629154, 1.6820664, -7.4776063, 5.9430017
9: -4.1408706, 4.7101393, -1.8773210, 2.0877132, -6.2285838, 6.5874605

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304189, upper bound: 7.6299327
time: 5.58 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304189, upper bound: 7.6324261
time: 13.26 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.7120776, 3.6348763, -5.6185951, 4.2716193, -8.9836969, 9.2534714
1: -3.5086536, 3.3622956, -4.1213198, 3.9566376, -7.4652910, 7.4836154
2: -6.1824889, 2.4841852, -7.4328771, 2.7999535, -8.9824429, 9.9170628
3: -5.3841496, 2.6883509, -6.1725292, 3.1276188, -8.5117683, 8.8608799
4: -5.6728554, 3.5463605, -6.7716045, 4.1562223, -9.8290777, 10.3179646
5: -4.2994499, 3.8076627, -5.0997190, 4.4419203, -8.7413702, 8.9073820
6: -4.4795990, 3.8009748, -5.3159370, 4.4713960, -8.9509945, 9.1169119
7: -5.1841855, 3.8064499, -6.1811018, 4.4241567, -9.6083422, 9.9875517
8: -5.8723760, 3.5217280, -6.9815183, 4.0893936, -9.9617691, 10.5032463
9: -4.1889925, 4.7741551, -4.9549112, 5.5824480, -9.7714405, 9.7290668

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_A1_A2_B2_A1_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302952, upper bound: 7.6290127
time: 3.48 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_A2

### Relational analysis result of IS_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6288282, upper bound: 7.6289546
time: 3.31 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.6229591, 4.2989349, -5.6185951, 4.2716193, -9.8945789, 9.9175301
1: -4.1353722, 3.9811015, -4.1213198, 3.9566376, -8.0920095, 8.1024208
2: -7.4313059, 2.8179891, -7.4328771, 2.7999535, -10.2312593, 10.2508659
3: -6.4994969, 3.1449418, -6.1725292, 3.1276188, -9.6271152, 9.3174706
4: -6.7787857, 4.1866455, -6.7716045, 4.1562223, -10.9350080, 10.9582500
5: -5.1166859, 4.4884729, -5.0997190, 4.4419203, -9.5586061, 9.5881920
6: -5.3368411, 4.5160017, -5.3159370, 4.4713960, -9.8082371, 9.8319387
7: -6.1868453, 4.4975691, -6.1811018, 4.4241567, -10.6110020, 10.6786709
8: -7.0113182, 4.1458650, -6.9815183, 4.0893936, -11.1007118, 11.1273832
9: -4.9629936, 5.6809969, -4.9549112, 5.5824480, -10.5454416, 10.6359081

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of IS_B1_A1_A2_B2_A2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323388, upper bound: 7.6323961
time: 4.23 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323366, upper bound: 7.6323965
time: 3.96 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.8380679, 0.7874784, -0.1623101, 0.1752629, -1.0133308, 0.9497885
1: -0.7495179, 0.7397859, -0.2093137, 0.2204718, -0.9699897, 0.9490996
2: -0.4249045, 1.1632850, 0.5897677, 1.0350578, -1.4599622, 0.5735173
3: -0.7033481, 0.7291363, -0.0815222, 0.2916855, -0.9950336, 0.8106585
4: -0.9612570, 0.8164982, -0.2258524, 0.2216960, -1.1829530, 1.0423506
5: -0.7759328, 0.8592340, -0.2038466, 0.2235304, -0.9994631, 1.0630807
6: -0.7561188, 0.8049411, -0.1647633, 0.2340025, -0.9901213, 0.9697043
7: -0.8623356, 0.8683124, -0.1936130, 0.2502241, -1.1125597, 1.0619254
8: -0.9263675, 0.8941428, -0.2202273, 0.3167633, -1.2431308, 1.1143701
9: -0.8332343, 0.9297156, -0.2322594, 0.2051974, -1.0384316, 1.1619750

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6300528, upper bound: 7.5861040
time: 3.87 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303786, upper bound: 7.5861222
time: 3.90 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -2.4637976, 1.9779339, -1.4592549, 1.2207263, -3.6845238, 3.4371886
1: -1.9675179, 1.8516169, -1.2046081, 1.1302004, -3.0977182, 3.0562248
2: -2.9840589, 1.6205074, -1.3846933, 1.3090674, -4.2931261, 3.0052006
3: -2.6755011, 1.5698578, -1.4540393, 1.0333321, -3.7088332, 3.0238972
4: -2.9522014, 1.9818386, -1.7119064, 1.2399557, -4.1921568, 3.6937451
5: -2.2806354, 2.0506394, -1.3359098, 1.2751590, -3.5557942, 3.3865492
6: -2.3630037, 2.0457332, -1.3397732, 1.2494270, -3.6124306, 3.3855064
7: -2.6945055, 2.1217723, -1.5558772, 1.3278602, -4.0223656, 3.6776495
8: -3.0702014, 1.9966767, -1.6996015, 1.3014858, -4.3716869, 3.6962781
9: -2.2962828, 2.5471468, -1.3710129, 1.5176690, -3.8139517, 3.9181597

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303598, upper bound: 7.6315650
time: 2.48 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303594, upper bound: 7.6326976
time: 3.69 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.8380679, 0.7874784, -1.5092630, 1.2556083, -2.0936761, 2.2967415
1: -0.7495179, 0.7397859, -1.2431486, 1.1624177, -1.9119356, 1.9829345
2: -0.4249045, 1.1632850, -1.4624932, 1.3207645, -1.7456690, 2.6257782
3: -0.7033481, 0.7291363, -1.5121390, 1.0581765, -1.7615247, 2.2412753
4: -0.9612570, 0.8164982, -1.7669803, 1.2748854, -2.2361424, 2.5834785
5: -0.7759328, 0.8592340, -1.3823025, 1.3060142, -2.0819468, 2.2415366
6: -0.7561188, 0.8049411, -1.3903584, 1.2886757, -2.0447946, 2.1952996
7: -0.8623356, 0.8683124, -1.6135566, 1.3615885, -2.2239242, 2.4818690
8: -0.9263675, 0.8941428, -1.7648863, 1.3349650, -2.2613325, 2.6590290
9: -0.8332343, 0.9297156, -1.4125023, 1.5650142, -2.3982487, 2.3422179

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6293385, upper bound: 7.4749466
time: 3.59 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292978, upper bound: 7.4420164
time: 7.27 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2.4637976, 1.9779339, -4.6796112, 3.6039925, -6.0677900, 6.6575451
1: -1.9675179, 1.8516169, -3.4879055, 3.3318214, -5.2993393, 5.3395224
2: -2.9840589, 1.6205074, -6.1202345, 2.4515746, -5.4356337, 7.7407417
3: -2.6755011, 1.5698578, -5.3519697, 2.6638720, -5.3393731, 6.9218273
4: -2.9522014, 1.9818386, -5.6376638, 3.5168540, -6.4690552, 7.6195025
5: -2.2806354, 2.0506394, -4.2570877, 3.7608597, -6.0414953, 6.3077269
6: -2.3630037, 2.0457332, -4.4381380, 3.7753828, -6.1383867, 6.4838715
7: -2.6945055, 2.1217723, -5.1433854, 3.7758074, -6.4703131, 7.2651577
8: -3.0702014, 1.9966767, -5.8202367, 3.4960437, -6.5662451, 7.8169136
9: -2.2962828, 2.5471468, -4.1615582, 4.7464414, -7.0427241, 6.7087049

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317450, upper bound: 7.6309989
time: 2.90 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319177, upper bound: 7.6314989
time: 2.86 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.8952280, 1.5408925, -0.5749281, 0.6165418, -2.5117698, 2.1158206
1: -1.5365030, 1.4320529, -0.5802902, 0.5992781, -2.1357810, 2.0123429
2: -2.0620646, 1.4333646, -0.0465086, 1.1170228, -3.1790874, 1.4798732
3: -1.9795961, 1.2601513, -0.4274951, 0.6183562, -2.5979524, 1.6876464
4: -2.2320299, 1.5530980, -0.6582962, 0.6446346, -2.8766646, 2.2113941
5: -1.7470505, 1.5851779, -0.5711213, 0.6807762, -2.4278266, 2.1562991
6: -1.7761716, 1.5907991, -0.5341682, 0.6387562, -2.4149277, 2.1249673
7: -2.0472651, 1.6634418, -0.6032372, 0.7061540, -2.7534189, 2.2666790
8: -2.2936382, 1.5987157, -0.6736836, 0.7382634, -3.0319016, 2.2723992
9: -1.7685250, 1.9517260, -0.6324757, 0.6950939, -2.4636188, 2.5842018

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4716106, upper bound: 7.6294428
time: 3.25 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4404536, upper bound: 7.6294079
time: 2.38 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.1909037, 3.9723678, -2.1138093, 1.7098024, -6.9007063, 6.0861769
1: -4.0388975, 3.6511269, -1.7035588, 1.5923096, -5.6312070, 5.3546858
2: -6.7433052, 2.7617259, -2.4423940, 1.5007048, -8.2440100, 5.2041197
3: -5.9124675, 2.9107666, -2.2464468, 1.3815364, -7.2940040, 5.1572132
4: -6.1932940, 3.8548326, -2.4962273, 1.7174634, -7.9107575, 6.3510599
5: -4.6728020, 4.1081657, -1.9467959, 1.7799761, -6.4527779, 6.0549617
6: -4.8909769, 4.1707125, -2.0029566, 1.7588335, -6.6498103, 6.1736689
7: -5.6431022, 4.1655025, -2.2908437, 1.8423295, -7.4854317, 6.4563465
8: -6.4577360, 3.8242424, -2.5968261, 1.7474377, -8.2051735, 6.4210682
9: -4.5612731, 5.2246933, -1.9660535, 2.1876166, -6.7488899, 7.1907468

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304622, upper bound: 7.6299404
time: 2.76 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304622, upper bound: 7.6324578
time: 5.56 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.2258110, 4.0003090, -5.7575884, 4.3945508, -9.6203613, 9.7578974
1: -4.0617042, 3.6687789, -4.2299485, 4.0489388, -8.1106434, 7.8987274
2: -6.8062730, 2.7964144, -7.6227016, 2.8517237, -9.6579971, 10.4191160
3: -5.9453349, 2.9258800, -6.6704755, 3.1960638, -9.1413984, 9.5963554
4: -6.2242832, 3.8764381, -6.9428921, 4.2515326, -10.4758158, 10.8193302
5: -4.6994996, 4.1530495, -5.2222462, 4.5900369, -9.2895365, 9.3752956
6: -4.9172645, 4.1935554, -5.4462862, 4.6091986, -9.5264626, 9.6398411
7: -5.6777139, 4.1865282, -6.3334379, 4.5891066, -10.2668209, 10.5199661
8: -6.4980412, 3.8435464, -7.1542482, 4.2342615, -10.7323027, 10.9977951
9: -4.5820532, 5.2544990, -5.0722699, 5.8165703, -10.3986235, 10.3267689

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of IS_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B1_A2_A2_B2_A1_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303956, upper bound: 7.6290450
time: 3.81 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_A2

### Relational analysis result of IS_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289787, upper bound: 7.6289850
time: 2.92 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.2124267, 4.7214203, -5.7575884, 4.3945508, -10.6069775, 10.4790087
1: -4.8502493, 4.3227358, -4.2299485, 4.0489388, -8.8991880, 8.5526848
2: -8.1273041, 3.2157297, -7.6227016, 2.8517237, -10.9790277, 10.8384314
3: -7.1248002, 3.4151611, -6.6704755, 3.1960638, -10.3208637, 10.0856361
4: -7.3889132, 4.5644493, -6.9428921, 4.2515326, -11.6404457, 11.5073414
5: -5.5602236, 4.8745117, -5.2222462, 4.5900369, -10.1502609, 10.0967579
6: -5.8263245, 4.9686246, -5.4462862, 4.6091986, -10.4355230, 10.4149113
7: -6.7332931, 4.9369349, -6.3334379, 4.5891066, -11.3223991, 11.2703724
8: -7.7301259, 4.5107884, -7.1542482, 4.2342615, -11.9643879, 11.6650372
9: -5.4027495, 6.2235818, -5.0722699, 5.8165703, -11.2193203, 11.2958517

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of IS_B1_A2_A2_B2_A2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324069, upper bound: 7.6324195
time: 3.34 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6324208
time: 3.09 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2.2380745, 1.8050854, -2.2355657, 1.7990253, -4.0370998, 4.0406513
1: -1.7980882, 1.6848613, -1.7956465, 1.6860437, -3.4841318, 3.4805079
2: -2.6492724, 1.5403659, -2.6386285, 1.5432963, -4.1925688, 4.1789942
3: -2.3990428, 1.4490991, -2.3953915, 1.4480960, -3.8471389, 3.8444905
4: -2.6521759, 1.8108498, -2.6439750, 1.8106759, -4.4628515, 4.4548249
5: -2.0623641, 1.8858886, -2.0660846, 1.8807189, -3.9430830, 3.9519732
6: -2.1315165, 1.8590215, -2.1259298, 1.8584037, -3.9899201, 3.9849515
7: -2.4316661, 1.9428017, -2.4252632, 1.9449842, -4.3766503, 4.3680649
8: -2.7659321, 1.8354522, -2.7569976, 1.8393042, -4.6052361, 4.5924497
9: -2.0822058, 2.3160677, -2.0834792, 2.3034928, -4.3856983, 4.3995466

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281443, upper bound: 7.6307164
time: 2.49 seconds

## Relational analysis of IS_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320192, upper bound: 7.6323443
time: 2.61 seconds

## BFS IS instance: IS_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -5.6185951, 4.2716193, -2.2355657, 1.7990253, -7.4176207, 6.5071850
1: -4.1213198, 3.9566376, -1.7956465, 1.6860437, -5.8073635, 5.7522840
2: -7.4328771, 2.7999535, -2.6386285, 1.5432963, -8.9761734, 5.4385819
3: -6.1725292, 3.1276188, -2.3953915, 1.4480960, -7.6206255, 5.5230103
4: -6.7716045, 4.1562223, -2.6439750, 1.8106759, -8.5822802, 6.8001976
5: -5.0997190, 4.4419203, -2.0660846, 1.8807189, -6.9804382, 6.5080051
6: -5.3159370, 4.4713960, -2.1259298, 1.8584037, -7.1743407, 6.5973258
7: -6.1811018, 4.4241567, -2.4252632, 1.9449842, -8.1260862, 6.8494196
8: -6.9815183, 4.0893936, -2.7569976, 1.8393042, -8.8208227, 6.8463912
9: -4.9549112, 5.5824480, -2.0834792, 2.3034928, -7.2584038, 7.6659269

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281443, upper bound: 7.6307164
time: 2.63 seconds

## Relational analysis of IS_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320192, upper bound: 7.6323443
time: 3.91 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -2.2380745, 1.8050854, -5.6204772, 4.2910242, -6.5290985, 7.4255629
1: -1.7980882, 1.6848613, -4.1353722, 3.9699244, -5.7680125, 5.8202333
2: -2.6492724, 1.5403659, -7.4287090, 2.8142042, -5.4634767, 8.9690752
3: -2.3990428, 1.4490991, -6.4994969, 3.1324167, -5.5314598, 7.9485960
4: -2.6521759, 1.8108498, -6.7710991, 4.1788130, -6.8309889, 8.5819492
5: -2.0623641, 1.8858886, -5.1077037, 4.4831533, -6.5455174, 6.9935923
6: -2.1315165, 1.8590215, -5.3151417, 4.5160017, -6.6475182, 7.1741633
7: -2.4316661, 1.9428017, -6.1839952, 4.4906549, -6.9223213, 8.1267967
8: -2.7659321, 1.8354522, -6.9841442, 4.1458650, -6.9117970, 8.8195963
9: -2.0822058, 2.3160677, -4.9601774, 5.6757460, -7.7579517, 7.2762451

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304620, upper bound: 7.6283727
time: 3.58 seconds

## Relational analysis of IS_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320005, upper bound: 7.6319276
time: 3.66 seconds

## BFS IS instance: IS_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -5.6185951, 4.2716193, -5.6204772, 4.2910242, -9.9096193, 9.8920965
1: -4.1213198, 3.9566376, -4.1353722, 3.9699244, -8.0912437, 8.0920095
2: -7.4328771, 2.7999535, -7.4287090, 2.8142042, -10.2470818, 10.2286625
3: -6.1725292, 3.1276188, -6.4994969, 3.1324167, -9.3049459, 9.6271152
4: -6.7716045, 4.1562223, -6.7710991, 4.1788130, -10.9504175, 10.9273214
5: -5.0997190, 4.4419203, -5.1077037, 4.4831533, -9.5828724, 9.5496235
6: -5.3159370, 4.4713960, -5.3151417, 4.5160017, -9.8319387, 9.7865372
7: -6.1811018, 4.4241567, -6.1839952, 4.4906549, -10.6717567, 10.6081524
8: -6.9815183, 4.0893936, -6.9841442, 4.1458650, -11.1273832, 11.0735378
9: -4.9549112, 5.5824480, -4.9601774, 5.6757460, -10.6306572, 10.5426254

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of IS_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323954, upper bound: 7.6323387
time: 3.39 seconds

## Relational analysis of IS_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323964, upper bound: 7.6323371
time: 13.96 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.3338535, 1.8806324, -2.6196442, 2.0933833, -4.4272366, 4.5002766
1: -1.8705406, 1.7552400, -2.0746675, 1.9575135, -3.8280540, 3.8299074
2: -2.7982168, 1.5723163, -3.2131815, 1.6855036, -4.4837203, 4.7854977
3: -2.5174484, 1.5006043, -2.8486047, 1.6480401, -4.1654882, 4.3492088
4: -2.7822888, 1.8823733, -3.1411972, 2.0911605, -4.8734493, 5.0235705
5: -2.1522994, 1.9604743, -2.4237320, 2.1737525, -4.3260517, 4.3842063
6: -2.2313132, 1.9377041, -2.5094218, 2.1663046, -4.3976178, 4.4471259
7: -2.5428867, 2.0178661, -2.8701410, 2.2375689, -4.7804556, 4.8880072
8: -2.8974190, 1.9023521, -3.2649107, 2.1014581, -4.9988770, 5.1672630
9: -2.1729269, 2.4178085, -2.4290111, 2.6969142, -4.8698411, 4.8468199

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281815, upper bound: 7.6307377
time: 3.18 seconds

## Relational analysis of IS_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320428, upper bound: 7.6324226
time: 3.26 seconds

## BFS IS instance: IS_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -5.7596278, 4.3945508, -2.6196442, 2.0933833, -7.8530111, 7.0141950
1: -4.2299485, 4.0584455, -2.0746675, 1.9575135, -6.1874619, 6.1331129
2: -7.6227016, 2.8566599, -3.2131815, 1.6855036, -9.3082056, 6.0698414
3: -6.6704755, 3.2000456, -2.8486047, 1.6480401, -8.3185158, 6.0486503
4: -6.9428921, 4.2715836, -3.1411972, 2.0911605, -9.0340528, 7.4127808
5: -5.2225118, 4.5900369, -2.4237320, 2.1737525, -7.3962641, 7.0137691
6: -5.4462862, 4.6206875, -2.5094218, 2.1663046, -7.6125908, 7.1301093
7: -6.3334379, 4.5902991, -2.8701410, 2.2375689, -8.5710068, 7.4604402
8: -7.1581602, 4.2342615, -3.2649107, 2.1014581, -9.2596188, 7.4991722
9: -5.0722699, 5.8229094, -2.4290111, 2.6969142, -7.7691841, 8.2519207

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281815, upper bound: 7.6307377
time: 2.13 seconds

## Relational analysis of IS_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320428, upper bound: 7.6324226
time: 2.78 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2.3338535, 1.8806324, -6.1708174, 4.6784267, -7.0122805, 8.0514498
1: -1.8705406, 1.7552400, -4.8206434, 4.2905436, -6.1610842, 6.5758834
2: -2.7982168, 1.5723163, -8.0471668, 3.1853733, -5.9835901, 9.6194830
3: -2.5174484, 1.5006043, -7.0704432, 3.3784189, -5.8958673, 8.5710478
4: -2.7822888, 1.8823733, -7.3469925, 4.5044041, -7.2866926, 9.2293663
5: -2.1522994, 1.9604743, -5.5247989, 4.8391924, -6.9914918, 7.4852734
6: -2.2313132, 1.9377041, -5.7866602, 4.9247112, -7.1560245, 7.7243643
7: -2.5428867, 2.0178661, -6.6892972, 4.9057126, -7.4485993, 8.7071629
8: -2.8974190, 1.9023521, -7.6365604, 4.4801340, -7.3775530, 9.5389128
9: -2.1729269, 2.4178085, -5.3680449, 6.1882858, -8.3612127, 7.7858534

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A1_B2_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304678, upper bound: 7.6261537
time: 4.01 seconds

## Relational analysis of IS_B2_A1_B2_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320278, upper bound: 7.6320166
time: 3.32 seconds

## BFS IS instance: IS_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -5.7575884, 4.3945508, -6.1708174, 4.6784267, -10.4360151, 10.5653687
1: -4.2299485, 4.0489388, -4.8206434, 4.2905436, -8.5204926, 8.8695822
2: -7.6227016, 2.8517237, -8.0471668, 3.1853733, -10.8080750, 10.8988905
3: -6.6704755, 3.1960638, -7.0704432, 3.3784189, -10.0488949, 10.2665071
4: -6.9428921, 4.2515326, -7.3469925, 4.5044041, -11.4472961, 11.5985250
5: -5.2222462, 4.5900369, -5.5247989, 4.8391924, -10.0614386, 10.1148357
6: -5.4462862, 4.6091986, -5.7866602, 4.9247112, -10.3709974, 10.3958588
7: -6.3334379, 4.5891066, -6.6892972, 4.9057126, -11.2391510, 11.2784042
8: -7.1542482, 4.2342615, -7.6365604, 4.4801340, -11.6343822, 11.8708220
9: -5.0722699, 5.8165703, -5.3680449, 6.1882858, -11.2605553, 11.1846151

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of IS_B2_A1_B2_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324195, upper bound: 7.6324074
time: 3.81 seconds

## Relational analysis of IS_B2_A1_B2_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324201, upper bound: 7.6324095
time: 3.04 seconds

## BFS IS instance: IS_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -3.1187835, 2.4610908, -2.2355657, 1.7990253, -4.9178085, 4.6966562
1: -2.4174776, 2.2936308, -1.7956465, 1.6860437, -4.1035213, 4.0892773
2: -3.9226675, 1.8811945, -2.6386285, 1.5432963, -5.4659638, 4.5198231
3: -3.4405944, 1.8972802, -2.3953915, 1.4480960, -4.8886905, 4.2926717
4: -3.7432592, 2.4423981, -2.6439750, 1.8106759, -5.5539351, 5.0863733
5: -2.8696542, 2.5558004, -2.0660846, 1.8807189, -4.7503729, 4.6218853
6: -2.9738081, 2.5542490, -2.1259298, 1.8584037, -4.8322115, 4.6801786
7: -3.4211364, 2.6120365, -2.4252632, 1.9449842, -5.3661203, 5.0372996
8: -3.8898296, 2.4390495, -2.7569976, 1.8393042, -5.7291336, 5.1960468
9: -2.8492255, 3.1873357, -2.0834792, 2.3034928, -5.1527185, 5.2708149

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261239, upper bound: 7.6307169
time: 3.10 seconds

## Relational analysis of IS_B2_A2_B1_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320108, upper bound: 7.6323470
time: 2.74 seconds

## BFS IS instance: IS_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -6.7113256, 5.0800357, -2.2355657, 1.7990253, -8.5103512, 7.3156013
1: -5.2438636, 4.6465702, -1.7956465, 1.6860437, -6.9299073, 6.4422169
2: -8.8025398, 3.4226661, -2.6386285, 1.5432963, -10.3458366, 6.0612946
3: -7.7096291, 3.6651051, -2.3953915, 1.4480960, -9.1577253, 6.0604963
4: -7.9684415, 4.9040861, -2.6439750, 1.8106759, -9.7791176, 7.5480614
5: -5.9831090, 5.2374344, -2.0660846, 1.8807189, -7.8638277, 7.3035192
6: -6.2747841, 5.3573856, -2.1259298, 1.8584037, -8.1331882, 7.4833155
7: -7.2645969, 5.3097229, -2.4252632, 1.9449842, -9.2095814, 7.7349863
8: -8.3465309, 4.8438973, -2.7569976, 1.8393042, -10.1858349, 7.6008949
9: -5.8098416, 6.7254162, -2.0834792, 2.3034928, -8.1133347, 8.8088951

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261239, upper bound: 7.6307169
time: 4.20 seconds

## Relational analysis of IS_B2_A2_B1_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320108, upper bound: 7.6323470
time: 2.86 seconds

## BFS IS instance: IS_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -3.1187835, 2.4610908, -5.6204772, 4.2910242, -7.4098077, 8.0815678
1: -2.4174776, 2.2936308, -4.1353722, 3.9699244, -6.3874021, 6.4290028
2: -3.9226675, 1.8811945, -7.4287090, 2.8142042, -6.7368717, 9.3099031
3: -3.4405944, 1.8972802, -6.4994969, 3.1324167, -6.5730114, 8.3967772
4: -3.7432592, 2.4423981, -6.7710991, 4.1788130, -7.9220724, 9.2134972
5: -2.8696542, 2.5558004, -5.1077037, 4.4831533, -7.3528075, 7.6635041
6: -2.9738081, 2.5542490, -5.3151417, 4.5160017, -7.4898100, 7.8693905
7: -3.4211364, 2.6120365, -6.1839952, 4.4906549, -7.9117913, 8.7960320
8: -3.8898296, 2.4390495, -6.9841442, 4.1458650, -8.0356941, 9.4231939
9: -2.8492255, 3.1873357, -4.9601774, 5.6757460, -8.5249710, 8.1475134

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B1_B2_A1_B1

### Relational analysis result of IS_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304162, upper bound: 7.6283728
time: 6.27 seconds

## Relational analysis of IS_B2_A2_B1_B2_A1_B2

### Relational analysis result of IS_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319897, upper bound: 7.6319274
time: 2.82 seconds

## BFS IS instance: IS_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -6.7113256, 5.0795760, -5.6204772, 4.2910242, -11.0023499, 10.7000532
1: -5.2438636, 4.6446295, -4.1353722, 3.9699244, -9.2137880, 8.7800016
2: -8.8019905, 3.4226661, -7.4287090, 2.8142042, -11.6161947, 10.8513756
3: -7.7068291, 3.6651051, -6.4994969, 3.1324167, -10.8392458, 10.1646023
4: -7.9671807, 4.9040861, -6.7710991, 4.1788130, -12.1459942, 11.6751852
5: -5.9821901, 5.2374344, -5.1077037, 4.4831533, -10.4653435, 10.3451385
6: -6.2747841, 5.3565526, -5.3151417, 4.5160017, -10.7907858, 10.6716938
7: -7.2645969, 5.3065777, -6.1839952, 4.4906549, -11.7552519, 11.4905729
8: -8.3465309, 4.8385329, -6.9841442, 4.1458650, -12.4923954, 11.8226776
9: -5.8058958, 6.7254162, -4.9601774, 5.6757460, -11.4816418, 11.6855936

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_B1_B2_A2_A1

### Relational analysis result of IS_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308999, upper bound: 7.6297265
time: 4.89 seconds

## Relational analysis of IS_B2_A2_B1_B2_A2_A2

### Relational analysis result of IS_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6297871, upper bound: 7.6296657
time: 3.24 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -3.2403135, 2.5479503, -2.6196442, 2.0933833, -5.3336968, 5.1675944
1: -2.4995601, 2.3735836, -2.0746675, 1.9575135, -4.4570737, 4.4482508
2: -4.0877514, 1.9280030, -3.2131815, 1.6855036, -5.7732549, 5.1411843
3: -3.5869420, 1.9565104, -2.8486047, 1.6480401, -5.2349820, 4.8051152
4: -3.8888640, 2.5249925, -3.1411972, 2.0911605, -5.9800243, 5.6661897
5: -2.9752712, 2.6450162, -2.4237320, 2.1737525, -5.1490240, 5.0687485
6: -3.0867250, 2.6473107, -2.5094218, 2.1663046, -5.2530298, 5.1567326
7: -3.5516436, 2.7015889, -2.8701410, 2.2375689, -5.7892122, 5.5717297
8: -4.0402813, 2.5194533, -3.2649107, 2.1014581, -6.1417394, 5.7843637
9: -2.9498816, 3.3069129, -2.4290111, 2.6969142, -5.6467957, 5.7359238

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261728, upper bound: 7.6307397
time: 2.80 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320342, upper bound: 7.6324249
time: 5.85 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -6.8564267, 5.1819391, -2.6196442, 2.0933833, -8.9498100, 7.8015833
1: -5.3554454, 4.7429519, -2.0746675, 1.9575135, -7.3129587, 6.8176193
2: -8.9886723, 3.4785094, -3.2131815, 1.6855036, -10.6741762, 6.6916909
3: -7.8770094, 3.7373860, -2.8486047, 1.6480401, -9.5250492, 6.5859909
4: -8.1349955, 5.0116653, -3.1411972, 2.0911605, -10.2261562, 8.1528625
5: -6.1032343, 5.3428826, -2.4237320, 2.1737525, -8.2769871, 7.7666149
6: -6.4196472, 5.4665518, -2.5094218, 2.1663046, -8.5859518, 7.9759736
7: -7.4192195, 5.4149461, -2.8701410, 2.2375689, -9.6567879, 8.2850876
8: -8.5203619, 4.9379287, -3.2649107, 2.1014581, -10.6218204, 8.2028389
9: -5.9301400, 6.8654613, -2.4290111, 2.6969142, -8.6270542, 9.2944727

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_A2_B2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261728, upper bound: 7.6307398
time: 5.18 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320342, upper bound: 7.6324248
time: 3.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 10.56 seconds
IS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6300020, upper bound: 7.5860490
IS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6302489, upper bound: 7.5860504
IS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6305727, upper bound: 7.6315649
IS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6305728, upper bound: 7.6326769
IS_B1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6290859, upper bound: 7.4743615
IS_B1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6289733, upper bound: 7.4419433
IS_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6316419, upper bound: 7.6309815
IS_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6317932, upper bound: 7.6314472
IS_B1_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.4878332, upper bound: 7.6293744
IS_B1_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.4224551, upper bound: 7.6292976
IS_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6304189, upper bound: 7.6299327
IS_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6304189, upper bound: 7.6324261
IS_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6302952, upper bound: 7.6290127
IS_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6288282, upper bound: 7.6289546
IS_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6323388, upper bound: 7.6323961
IS_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6323366, upper bound: 7.6323965
IS_B1_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6300528, upper bound: 7.5861040
IS_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6303786, upper bound: 7.5861222
IS_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6303598, upper bound: 7.6315650
IS_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6303594, upper bound: 7.6326976
IS_B1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6293385, upper bound: 7.4749466
IS_B1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6292978, upper bound: 7.4420164
IS_B1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6317450, upper bound: 7.6309989
IS_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6319177, upper bound: 7.6314989
IS_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.4716106, upper bound: 7.6294428
IS_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.4404536, upper bound: 7.6294079
IS_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6304622, upper bound: 7.6299404
IS_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6304622, upper bound: 7.6324578
IS_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6303956, upper bound: 7.6290450
IS_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6289787, upper bound: 7.6289850
IS_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6324069, upper bound: 7.6324195
IS_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6324208
IS_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6281443, upper bound: 7.6307164
IS_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320192, upper bound: 7.6323443
IS_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6281443, upper bound: 7.6307164
IS_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320192, upper bound: 7.6323443
IS_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6304620, upper bound: 7.6283727
IS_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320005, upper bound: 7.6319276
IS_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6323954, upper bound: 7.6323387
IS_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6323964, upper bound: 7.6323371
IS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6281815, upper bound: 7.6307377
IS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320428, upper bound: 7.6324226
IS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6281815, upper bound: 7.6307377
IS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320428, upper bound: 7.6324226
IS_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6304678, upper bound: 7.6261537
IS_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320278, upper bound: 7.6320166
IS_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6324195, upper bound: 7.6324074
IS_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6324201, upper bound: 7.6324095
IS_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6261239, upper bound: 7.6307169
IS_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320108, upper bound: 7.6323470
IS_B2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6261239, upper bound: 7.6307169
IS_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320108, upper bound: 7.6323470
IS_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6304162, upper bound: 7.6283728
IS_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6319897, upper bound: 7.6319274
IS_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6308999, upper bound: 7.6297265
IS_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6297871, upper bound: 7.6296657
IS_B2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6261728, upper bound: 7.6307397
IS_B2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320342, upper bound: 7.6324249
IS_B2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6261728, upper bound: 7.6307398
IS_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.56
Output dim: 2, lower bound: -7.6320342, upper bound: 7.6324248
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 10.56
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324277
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 10.56
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324284
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=9.388381958007812
rel_dist={2: [-7.633595790761577, 7.633596173769259]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335335, upper bound: 7.6335487
time: 3.56 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335331, upper bound: 7.6335334
time: 4.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.55 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 8.55
Output dim: 2, lower bound: -7.6335335, upper bound: 7.6335487
IS_B2, status: Status.UNKNOWN, split count: 1, time: 8.55
Output dim: 2, lower bound: -7.6335331, upper bound: 7.6335334

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -4.5237212, 3.4892483, -3.7251990, 2.9015779, -7.4252992, 7.2144470
1: -3.5211136, 3.2183642, -2.8778124, 2.6893115, -6.2104254, 6.0961766
2: -5.8374209, 2.4647236, -4.7487893, 2.1088452, -7.9462662, 7.2135129
3: -5.1169090, 2.5979800, -4.1680532, 2.1933427, -7.3102517, 6.7660332
4: -5.4049444, 3.4156179, -4.4700050, 2.8505054, -8.2554493, 7.8856230
5: -4.0858850, 3.5947173, -3.3880157, 3.0042415, -7.0901265, 6.9827328
6: -4.2781229, 3.6513577, -3.5393488, 3.0220010, -7.3001242, 7.1907063
7: -4.9269443, 3.6733992, -4.0690737, 3.0618806, -7.9888248, 7.7424726
8: -5.6346431, 3.3881350, -4.6395154, 2.8396196, -8.4742622, 8.0276508
9: -4.0094662, 4.5817900, -3.3468199, 3.8004518, -7.8099179, 7.9286098

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330325, upper bound: 7.6330594
time: 4.97 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330637, upper bound: 7.6330843
time: 3.65 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -4.7834167, 3.6798470, -4.6887450, 3.6103084, -8.3937254, 8.3685923
1: -3.7300522, 3.3877926, -3.6534214, 3.3259871, -7.0560393, 7.0412140
2: -6.1931820, 2.5770607, -6.0639324, 2.5362964, -8.7294788, 8.6409931
3: -5.4237604, 2.7292209, -5.3117399, 2.6811905, -8.1049509, 8.0409603
4: -5.7086158, 3.5954719, -5.5978603, 3.5297210, -9.2383366, 9.1933327
5: -4.3093128, 3.7901092, -4.2277665, 3.7193160, -8.0286293, 8.0178757
6: -4.5199976, 3.8536978, -4.4315343, 3.7799363, -8.2999344, 8.2852325
7: -5.2058315, 3.8692214, -5.1041079, 3.7976844, -9.0035162, 8.9733295
8: -5.9556684, 3.5658152, -5.8385901, 3.5008440, -9.4565125, 9.4044056
9: -4.2220955, 4.8381348, -4.1445389, 4.7445335, -8.9666290, 8.9826736

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330411, upper bound: 7.6330326
time: 4.12 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330630, upper bound: 7.6330636
time: 3.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.44 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 9.44
Output dim: 2, lower bound: -7.6330325, upper bound: 7.6330594
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 9.44
Output dim: 2, lower bound: -7.6330637, upper bound: 7.6330843
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 9.44
Output dim: 2, lower bound: -7.6330411, upper bound: 7.6330326
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 9.44
Output dim: 2, lower bound: -7.6330630, upper bound: 7.6330636

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -2.7381444, 2.1783710, -2.4202323, 1.9473032, -4.6854477, 4.5986032
1: -2.1564004, 2.0376217, -1.9358550, 1.8178914, -3.9742918, 3.9734769
2: -3.3780308, 1.7265962, -2.9258666, 1.6001980, -4.9782286, 4.6524630
3: -2.9807174, 1.7070343, -2.6232903, 1.5467935, -4.5275106, 4.3303246
4: -3.2845793, 2.1742134, -2.8971772, 1.9480476, -5.2326269, 5.0713906
5: -2.5296078, 2.2585366, -2.2353895, 2.0197203, -4.5493279, 4.4939260
6: -2.6214032, 2.2579334, -2.3223388, 2.0083542, -4.6297574, 4.5802722
7: -3.0005431, 2.3243308, -2.6431932, 2.0840619, -5.0846052, 4.9675241
8: -3.4118843, 2.1811774, -3.0144904, 1.9631433, -5.3750277, 5.1956677
9: -2.5289559, 2.8116975, -2.2527685, 2.5091908, -5.0381470, 5.0644660

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307904, upper bound: 7.6304446
time: 5.26 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325851, upper bound: 7.6326296
time: 3.44 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -3.2039921, 2.5187201, -2.6783552, 2.1347322, -5.3387241, 5.1970754
1: -2.4758227, 2.3497462, -2.1145093, 1.9918587, -4.4676814, 4.4642553
2: -4.0294271, 1.8949492, -3.2930102, 1.6906044, -5.7200317, 5.1879597
3: -3.5470028, 1.9381409, -2.9137521, 1.6754668, -5.2224693, 4.8518929
4: -3.8515916, 2.4970443, -3.2156665, 2.1277153, -5.9793072, 5.7127109
5: -2.9432049, 2.6101747, -2.4659753, 2.2116313, -5.1548362, 5.0761499
6: -3.0595021, 2.6201391, -2.5675845, 2.2081749, -5.2676773, 5.1877236
7: -3.5121913, 2.6734858, -2.9303653, 2.2733600, -5.7855511, 5.6038513
8: -3.9923794, 2.4962444, -3.3387947, 2.1350727, -6.1274519, 5.8350391
9: -2.9212067, 3.2774603, -2.4735689, 2.7618692, -5.6830759, 5.7510290

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307865, upper bound: 7.6304465
time: 5.37 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6326463, upper bound: 7.6326713
time: 3.84 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -3.3877015, 2.6521649, -2.8952935, 2.2936373, -5.6813388, 5.5474586
1: -2.6082523, 2.4710717, -2.2642684, 2.1427622, -4.7510147, 4.7353401
2: -4.2848082, 1.9803195, -3.6013370, 1.7856565, -6.0704646, 5.5816565
3: -3.7638547, 2.0281560, -3.1685982, 1.7848133, -5.5486679, 5.1967545
4: -4.0654545, 2.6239765, -3.4752483, 2.2829573, -6.3484116, 6.0992250
5: -3.1043308, 2.7500200, -2.6703582, 2.3790715, -5.4834023, 5.4203782
6: -3.2269964, 2.7613752, -2.7688076, 2.3800302, -5.6070266, 5.5301828
7: -3.7095153, 2.8108661, -3.1743832, 2.4417624, -6.1512775, 5.9852495
8: -4.2194753, 2.6188879, -3.6069160, 2.2877617, -6.5072370, 6.2258039
9: -3.0719180, 3.4542468, -2.6614912, 2.9672120, -6.0391302, 6.1157379

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325288, upper bound: 7.6323509
time: 5.05 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323780, upper bound: 7.6323446
time: 3.11 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -3.6775258, 2.8642888, -3.3594167, 2.6305923, -6.3081179, 6.2237053
1: -2.8400819, 2.6614575, -2.5864820, 2.4517391, -5.2918210, 5.2479396
2: -4.6786795, 2.0984056, -4.2425985, 1.9599543, -6.6386337, 6.3410044
3: -4.1111426, 2.1711473, -3.7323713, 2.0136664, -6.1248093, 5.9035187
4: -4.4104013, 2.8223696, -4.0356216, 2.6024401, -7.0128412, 6.8579912
5: -3.3550296, 2.9643435, -3.0788145, 2.7263789, -6.0814085, 6.0431581
6: -3.4955194, 2.9877150, -3.2031105, 2.7394447, -6.2349644, 6.1908255
7: -4.0201173, 3.0295544, -3.6790948, 2.7887216, -6.8088388, 6.7086492
8: -4.5800591, 2.8127925, -4.1837339, 2.5995026, -7.1795616, 6.9965267
9: -3.3117580, 3.7439997, -3.0494063, 3.4293561, -6.7411141, 6.7934060

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325713, upper bound: 7.6324207
time: 5.62 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324132, upper bound: 7.6324137
time: 3.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 10.46 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 2, lower bound: -7.6307904, upper bound: 7.6304446
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 2, lower bound: -7.6325851, upper bound: 7.6326296
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 2, lower bound: -7.6307865, upper bound: 7.6304465
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 2, lower bound: -7.6326463, upper bound: 7.6326713
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 2, lower bound: -7.6325288, upper bound: 7.6323509
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 2, lower bound: -7.6323780, upper bound: 7.6323446
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 2, lower bound: -7.6325713, upper bound: 7.6324207
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 10.46
Output dim: 2, lower bound: -7.6324132, upper bound: 7.6324137

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3215799, 0.3782694, -0.1717716, 0.1858835, -0.5074633, 0.5500410
1: -0.3690244, 0.3933338, -0.2199834, 0.2320807, -0.6011051, 0.6133172
2: 0.3344536, 1.0572246, 0.5732257, 1.0417510, -0.7072974, 0.4839990
3: -0.2227339, 0.4510955, -0.0895332, 0.3041013, -0.5268353, 0.5406287
4: -0.4023683, 0.3943540, -0.2373023, 0.2304131, -0.6327814, 0.6316564
5: -0.3571987, 0.4084581, -0.2144945, 0.2358484, -0.5930470, 0.6229526
6: -0.3071234, 0.4333916, -0.1721875, 0.2490436, -0.5561670, 0.6055791
7: -0.3580728, 0.4483745, -0.2020960, 0.2631232, -0.6211960, 0.6504706
8: -0.4200011, 0.5083054, -0.2332413, 0.3302424, -0.7502435, 0.7415468
9: -0.4147376, 0.3961046, -0.2442621, 0.2170286, -0.6317662, 0.6403667

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6291916, upper bound: 7.6289111
time: 3.99 seconds

## Relational analysis of IS_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6291748, upper bound: 7.6261458
time: 4.68 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.0330646, 1.6462981, -1.5153086, 1.2624767, -3.2955413, 3.1616068
1: -1.6417727, 1.5353556, -1.2474169, 1.1667470, -2.8085198, 2.7827725
2: -2.2886469, 1.4776545, -1.4736154, 1.3234075, -3.6120543, 2.9512699
3: -2.1470318, 1.3386359, -1.5220289, 1.0622299, -3.2092617, 2.8606648
4: -2.3978996, 1.6594166, -1.7782695, 1.2793334, -3.6772330, 3.4376860
5: -1.8771505, 1.6978521, -1.3893075, 1.3136376, -3.1907883, 3.0871596
6: -1.9175863, 1.6948583, -1.3952235, 1.2918832, -3.2094696, 3.0900817
7: -2.1977644, 1.7792150, -1.6199672, 1.3686359, -3.5664003, 3.3991823
8: -2.4828439, 1.6953671, -1.7748883, 1.3395880, -3.8224320, 3.4702554
9: -1.8931782, 2.1012187, -1.4205166, 1.5754106, -3.4685888, 3.5217352

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A1_B2_B1

### Relational analysis result of IS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319163, upper bound: 7.6321226
time: 4.64 seconds

## Relational analysis of IS_B1_A1_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319099, upper bound: 7.6319628
time: 8.10 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3915448, 0.4554980, -0.1900081, 0.2061674, -0.5977122, 0.6455061
1: -0.4323283, 0.4604924, -0.2397705, 0.2527692, -0.6850976, 0.7002629
2: 0.2237388, 1.0750108, 0.5437586, 1.0417507, -0.8180118, 0.5312521
3: -0.2772732, 0.5045166, -0.1063891, 0.3253229, -0.6025961, 0.6109056
4: -0.4674502, 0.4770141, -0.2585489, 0.2484625, -0.7159127, 0.7355630
5: -0.4231265, 0.4890333, -0.2326923, 0.2588077, -0.6819343, 0.7217255
6: -0.3697763, 0.5004441, -0.1856890, 0.2769534, -0.6467297, 0.6861330
7: -0.4247372, 0.5323674, -0.2197958, 0.2857786, -0.7105158, 0.7521632
8: -0.4881888, 0.5790800, -0.2589378, 0.3539795, -0.8421682, 0.8380178
9: -0.4788909, 0.4892352, -0.2665470, 0.2393169, -0.7182078, 0.7557822

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6294232, upper bound: 7.6280912
time: 5.26 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292638, upper bound: 7.6280880
time: 6.32 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -2.4467812, 1.9657372, -1.7452478, 1.4350554, -3.8818364, 3.7109852
1: -1.9551378, 1.8378396, -1.4227135, 1.3271093, -3.2822471, 3.2605531
2: -2.9378023, 1.6115057, -1.8359181, 1.3894708, -4.3272734, 3.4474239
3: -2.6575363, 1.5603653, -1.8010373, 1.1824899, -3.8400261, 3.3614025
4: -2.9331379, 1.9680252, -2.0556378, 1.4455732, -4.3787112, 4.0236630
5: -2.2643728, 2.0185962, -1.6048076, 1.4799536, -3.7443266, 3.6234038
6: -2.3478417, 2.0305796, -1.6221548, 1.4693173, -3.8171592, 3.6527343
7: -2.6751394, 2.1051531, -1.8815010, 1.5462532, -4.2213926, 3.9866540
8: -3.0504863, 1.9820280, -2.0916247, 1.4957134, -4.5461998, 4.0736527
9: -2.2800512, 2.5339248, -1.6300431, 1.8123301, -4.0923815, 4.1639681

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320095, upper bound: 7.6321788
time: 2.72 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320023, upper bound: 7.6320099
time: 3.28 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6585221, 2.1234524, -2.3722861, 1.9086498, -4.5671721, 4.4957385
1: -2.1017075, 1.9847608, -1.8994061, 1.7867826, -3.8884902, 3.8841667
2: -3.2734067, 1.7051687, -2.8540211, 1.5944785, -4.8678851, 4.5591898
3: -2.8920360, 1.6681871, -2.5636072, 1.5216662, -4.4137020, 4.2317944
4: -3.1866710, 2.1202855, -2.8282669, 1.9148810, -5.1015520, 4.9485521
5: -2.4603374, 2.2064281, -2.1972859, 1.9886073, -4.4489446, 4.4037142
6: -2.5450001, 2.1968358, -2.2689888, 1.9712191, -4.5162191, 4.4658246
7: -2.9146369, 2.2676373, -2.5874658, 2.0523348, -4.9669714, 4.8551030
8: -3.3141284, 2.1285615, -2.9458766, 1.9349499, -5.2490783, 5.0744381
9: -2.4618449, 2.7333882, -2.2123365, 2.4486344, -4.9104795, 4.9457245

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287596, upper bound: 7.6291903
time: 6.50 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321042, upper bound: 7.6319157
time: 4.63 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -6.2224550, 4.7288728, -2.4216352, 1.9477571, -8.1702118, 7.1505079
1: -4.8579550, 4.3297663, -1.9367254, 1.8227786, -6.6807337, 6.2664919
2: -8.1410723, 3.2246819, -2.9303212, 1.6131845, -9.7542572, 6.1550031
3: -7.1350336, 3.4204943, -2.6243370, 1.5479134, -8.6829472, 6.0448313
4: -7.3976192, 4.5732870, -2.8946478, 1.9521781, -9.3497972, 7.4679346
5: -5.5694642, 4.8826632, -2.2443824, 2.0264642, -7.5959282, 7.1270456
6: -5.8341093, 4.9765673, -2.3206730, 2.0124230, -7.8465323, 7.2972403
7: -6.7434812, 4.9448009, -2.6476912, 2.0905519, -8.8340330, 7.5924921
8: -7.7428970, 4.5181847, -3.0134678, 1.9694839, -9.7123814, 7.5316525
9: -5.4103546, 6.2310481, -2.2581058, 2.5011902, -7.9115448, 8.4891539

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6236163, upper bound: 7.6291746
time: 4.81 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319556, upper bound: 7.6319098
time: 10.75 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2.9093149, 2.3060150, -2.7834067, 2.2121871, -5.1215019, 5.0894217
1: -2.2736385, 2.1521797, -2.1866779, 2.0668497, -4.3404884, 4.3388577
2: -3.6247780, 1.7953339, -3.4442549, 1.7450286, -5.3698068, 5.2395887
3: -3.1857662, 1.7922595, -3.0314610, 1.7290161, -4.9147825, 4.8237205
4: -3.4921508, 2.2937145, -3.3402719, 2.2041543, -5.6963053, 5.6339865
5: -2.6828909, 2.3940825, -2.5696876, 2.2959130, -4.9788036, 4.9637699
6: -2.7797716, 2.3908379, -2.6631813, 2.2921765, -5.0719481, 5.0540190
7: -3.1906345, 2.4533336, -3.0508156, 2.3572173, -5.5478516, 5.5041494
8: -3.6262946, 2.2966175, -3.4683330, 2.2102256, -5.8365202, 5.7649508
9: -2.6735749, 2.9805646, -2.5671434, 2.8555505, -5.5291252, 5.5477080

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287969, upper bound: 7.6292791
time: 3.53 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321597, upper bound: 7.6320081
time: 4.85 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -6.5288081, 4.9421573, -2.8331883, 2.2484591, -8.7772675, 7.7753458
1: -5.0947704, 4.5255833, -2.2209358, 2.0997846, -7.1945553, 6.7465191
2: -8.5502148, 3.3421032, -3.5141330, 1.7623811, -10.3125954, 6.8562365
3: -7.4846377, 3.5748596, -3.0929608, 1.7534735, -9.2381115, 6.6678205
4: -7.7477999, 4.7842016, -3.4009986, 2.2383063, -9.9861059, 8.1851997
5: -5.8207183, 5.1025615, -2.6139581, 2.3332529, -8.1539707, 7.7165194
6: -6.1181431, 5.2099872, -2.7103734, 2.3308320, -8.4489746, 7.9203606
7: -7.0698528, 5.1647873, -3.1056595, 2.3941813, -9.4640341, 8.2704468
8: -8.1121244, 4.7136359, -3.5300117, 2.2441573, -10.3562813, 8.2436476
9: -5.6560345, 6.5407004, -2.6089573, 2.9057474, -8.5617819, 9.1496582

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6260879, upper bound: 7.6292646
time: 2.20 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320022, upper bound: 7.6320022
time: 3.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 7.07 seconds
IS_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6291916, upper bound: 7.6289111
IS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6291748, upper bound: 7.6261458
IS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6319163, upper bound: 7.6321226
IS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6319099, upper bound: 7.6319628
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6294232, upper bound: 7.6280912
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6292638, upper bound: 7.6280880
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6320095, upper bound: 7.6321788
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6320023, upper bound: 7.6320099
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6287596, upper bound: 7.6291903
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6321042, upper bound: 7.6319157
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6236163, upper bound: 7.6291746
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6319556, upper bound: 7.6319098
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6287969, upper bound: 7.6292791
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6321597, upper bound: 7.6320081
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6260879, upper bound: 7.6292646
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 7.07
Output dim: 2, lower bound: -7.6320022, upper bound: 7.6320022

## BFS IS instance: IS_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2379815, 0.2731996, -0.1132475, 0.1215424, -0.3595239, 0.3864471
1: -0.2875202, 0.3104078, -0.1526800, 0.1584020, -0.4459222, 0.4630878
2: 0.4685953, 1.0354400, 0.6785882, 1.0349971, -0.5664018, 0.3568518
3: -0.1525727, 0.3782607, -0.0402024, 0.2249212, -0.3774939, 0.4184631
4: -0.3133272, 0.3006780, -0.1671513, 0.1779443, -0.4912715, 0.4678293
5: -0.2822747, 0.3117039, -0.1449934, 0.1580993, -0.4403740, 0.4566973
6: -0.2300377, 0.3415463, -0.1237120, 0.1592468, -0.3892846, 0.4652583
7: -0.2726915, 0.3449843, -0.1509633, 0.1814716, -0.4541631, 0.4959476
8: -0.3266579, 0.4131007, -0.1539393, 0.2447422, -0.5714000, 0.5670401
9: -0.3268625, 0.2953696, -0.1675896, 0.1444645, -0.4713270, 0.4629592

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4846840, upper bound: 7.4675718
time: 3.91 seconds

## Relational analysis of IS_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4962444, upper bound: 7.4671136
time: 2.95 seconds

## BFS IS instance: IS_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2446983, 0.2837725, -0.4164814, 0.4933273, -0.7380256, 0.7002539
1: -0.2946914, 0.3184751, -0.5114074, 0.5323352, -0.8270266, 0.8298826
2: 0.4566767, 1.0357324, 0.1280796, 1.0795386, -0.6228619, 0.9076529
3: -0.1592769, 0.3854914, -0.3162123, 0.6252201, -0.7844969, 0.7017037
4: -0.3220270, 0.3081238, -0.5334423, 0.4849846, -0.8070116, 0.8415661
5: -0.2898727, 0.3185282, -0.4898922, 0.5701591, -0.8600318, 0.8084204
6: -0.2367440, 0.3505608, -0.3642105, 0.6720750, -0.9088191, 0.7147714
7: -0.2800073, 0.3551775, -0.4493598, 0.5967471, -0.8767544, 0.8045373
8: -0.3355582, 0.4226448, -0.6060826, 0.6788703, -1.0144285, 1.0287274
9: -0.3354108, 0.3031562, -0.5546980, 0.5537082, -0.8891190, 0.8578542

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B1_B2_B1

### Relational analysis result of IS_B1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4969150, upper bound: 7.4472287
time: 3.47 seconds

## Relational analysis of IS_B1_A1_B1_B2_B2

### Relational analysis result of IS_B1_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4955314, upper bound: 7.4357553
time: 3.57 seconds

## BFS IS instance: IS_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.5866330, 1.3143128, -0.9353335, 0.8511229, -2.4377558, 2.2496462
1: -1.3027472, 1.2211740, -0.8175396, 0.7957641, -2.0985112, 2.0387137
2: -1.5813577, 1.3477395, -0.5720702, 1.1814234, -2.7627811, 1.9198097
3: -1.6077533, 1.1008241, -0.8197039, 0.7736508, -2.3814039, 1.9205281
4: -1.8643041, 1.3342454, -1.0789416, 0.8781754, -2.7424793, 2.4131870
5: -1.4595919, 1.3637612, -0.8611336, 0.9212211, -2.3808129, 2.2248948
6: -1.4598886, 1.3474193, -0.8441429, 0.8717217, -2.3316102, 2.1915622
7: -1.6985624, 1.4292846, -0.9691219, 0.9360417, -2.6346040, 2.3984065
8: -1.8720826, 1.3924133, -1.0313163, 0.9569674, -2.8290501, 2.4237294
9: -1.4891031, 1.6440150, -0.9164167, 1.0140761, -2.5031791, 2.5604317

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308864, upper bound: 7.6312515
time: 3.29 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312654, upper bound: 7.6315311
time: 4.56 seconds

## BFS IS instance: IS_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.6257542, 1.3436644, -3.6510129, 2.8811831, -4.5069375, 4.9946775
1: -1.3326128, 1.2479542, -2.8946528, 2.6472743, -3.9798870, 4.1426072
2: -1.6440675, 1.3586938, -4.8685274, 1.9304826, -3.5745502, 6.2272215
3: -1.6551979, 1.1211823, -4.1251383, 2.1756678, -3.8308656, 5.2463207
4: -1.9112455, 1.3621800, -4.3829761, 2.8268471, -4.7380924, 5.7451563
5: -1.4964209, 1.3924699, -3.4089243, 2.8624074, -4.3588285, 4.8013945
6: -1.4978595, 1.3774397, -3.4646440, 2.9439175, -4.4417772, 4.8420839
7: -1.7430850, 1.4588296, -4.0728316, 3.0179319, -4.7610168, 5.5316610
8: -1.9256550, 1.4189680, -4.7205510, 2.8004777, -4.7261329, 6.1395187
9: -1.5242820, 1.6845804, -3.3833418, 3.7830412, -5.3073235, 5.0679221

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A1_B2_B2_B1

### Relational analysis result of IS_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310176, upper bound: 7.6309220
time: 13.15 seconds

## Relational analysis of IS_B1_A1_B2_B2_B2

### Relational analysis result of IS_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312592, upper bound: 7.6313475
time: 4.37 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2661006, 0.3152484, -0.1416977, 0.1528119, -0.4189125, 0.4569461
1: -0.3170650, 0.3430721, -0.1856451, 0.1943974, -0.5114624, 0.5287172
2: 0.4198602, 1.0421411, 0.6268879, 1.0365411, -0.6166809, 0.4152532
3: -0.1795464, 0.4073124, -0.0642537, 0.2636984, -0.4432448, 0.4715661
4: -0.3486221, 0.3322037, -0.2010126, 0.2034111, -0.5520332, 0.5332163
5: -0.3122098, 0.3420056, -0.1792469, 0.1961785, -0.5083883, 0.5212525
6: -0.2575864, 0.3778756, -0.1476069, 0.2025774, -0.4601637, 0.5254825
7: -0.3029760, 0.3860695, -0.1757886, 0.2211840, -0.5241600, 0.5618581
8: -0.3632139, 0.4510142, -0.1925030, 0.2866032, -0.6498171, 0.6435171
9: -0.3616272, 0.3283661, -0.2050914, 0.1798155, -0.5414426, 0.5334575

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5151575, upper bound: 7.4419108
time: 3.55 seconds

## Relational analysis of IS_B1_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5342344, upper bound: 7.4419196
time: 4.96 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.9384831, 1.0517688, -0.1472265, 0.1588885, -1.0973716, 1.1989954
1: -0.9238402, 0.9863105, -0.1920513, 0.2013925, -1.1252327, 1.1783619
2: -0.6325741, 1.2177261, 0.6168409, 1.0368104, -1.6693845, 0.6008852
3: -0.7034311, 0.9182838, -0.0689276, 0.2712340, -0.9746652, 0.9872113
4: -0.9761102, 1.1201308, -0.2075929, 0.2083600, -1.1844702, 1.3277237
5: -0.9297940, 1.1191204, -0.1859033, 0.2035788, -1.1333728, 1.3050237
6: -0.8542085, 1.0254276, -0.1522505, 0.2109977, -1.0652062, 1.1776781
7: -0.9455384, 1.1827632, -0.1806129, 0.2289014, -1.1744398, 1.3633761
8: -1.0257618, 1.1284152, -0.1999972, 0.2947378, -1.3204997, 1.3284124
9: -0.9812577, 1.2057805, -0.2123792, 0.1866853, -1.1679430, 1.4181597

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5041666, upper bound: 7.4419076
time: 3.58 seconds

## Relational analysis of IS_B1_A2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5220109, upper bound: 7.4419159
time: 9.44 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1.9588089, 1.5919471, -1.1281862, 0.9799848, -2.9387937, 2.7201333
1: -1.5855298, 1.4819349, -0.9571196, 0.9112316, -2.4967613, 2.4390545
2: -2.1747639, 1.4559625, -0.8632092, 1.2206585, -3.3954225, 2.3191719
3: -2.0565834, 1.2989622, -1.0527874, 0.8643552, -2.9209385, 2.3517497
4: -2.3105314, 1.6050969, -1.3110116, 1.0093508, -3.3198822, 2.9161086
5: -1.8067981, 1.6450913, -1.0327834, 1.0475168, -2.8543148, 2.6778746
6: -1.8403213, 1.6371074, -1.0238867, 1.0043300, -2.8446512, 2.6609941
7: -2.1161170, 1.7208706, -1.1862996, 1.0749807, -3.1910977, 2.9071703
8: -2.3825436, 1.6448383, -1.2512217, 1.0833719, -3.4659154, 2.8960600
9: -1.8267667, 2.0261462, -1.0800052, 1.1867352, -3.0135019, 3.1061513

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309455, upper bound: 7.6312866
time: 4.88 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314647, upper bound: 7.6316516
time: 16.67 seconds

## BFS IS instance: IS_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1.9948628, 1.6185592, -3.8997025, 3.0625699, -5.0574327, 5.5182619
1: -1.6127424, 1.5069458, -3.0566573, 2.8700478, -4.4827900, 4.5636034
2: -2.2307801, 1.4661918, -5.2345376, 2.0108633, -4.2416434, 6.7007294
3: -2.1002522, 1.3180138, -4.4231310, 2.3413947, -4.4416466, 5.7411447
4: -2.3533025, 1.6309984, -4.6415496, 3.0376654, -5.3909678, 6.2725477
5: -1.8406832, 1.6713388, -3.6295485, 3.0485644, -4.8892479, 5.3008871
6: -1.8781304, 1.6649204, -3.8675854, 3.1669660, -5.0450964, 5.5325060
7: -2.1560678, 1.7485297, -4.3199239, 3.2579510, -5.4140186, 6.0684538
8: -2.4314761, 1.6689086, -5.1053476, 2.9638915, -5.3953676, 6.7742562
9: -1.8588388, 2.0631368, -3.6050820, 4.0294018, -5.8882408, 5.6682186

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 30

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B1_A2_B2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311260, upper bound: 7.6309564
time: 4.53 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314573, upper bound: 7.6314615
time: 10.66 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.1995944, 0.2174338, -0.2595449, 0.3065541, -0.5061485, 0.4769787
1: -0.2491403, 0.2636724, -0.3103957, 0.3364988, -0.5856391, 0.5740681
2: 0.5297765, 1.0383856, 0.4308966, 1.0402372, -0.5104607, 0.6074890
3: -0.1158119, 0.3357134, -0.1740130, 0.4013138, -0.5171257, 0.5097264
4: -0.2693106, 0.2574984, -0.3414182, 0.3246129, -0.5939235, 0.5989166
5: -0.2417872, 0.2698011, -0.3060089, 0.3340737, -0.5758609, 0.5758100
6: -0.1934911, 0.2898504, -0.2512261, 0.3705648, -0.5640558, 0.5410764
7: -0.2295029, 0.2969043, -0.2961744, 0.3775878, -0.6070907, 0.5930787
8: -0.2720990, 0.3651975, -0.3558174, 0.4433050, -0.7154040, 0.7210149
9: -0.2778537, 0.2505693, -0.3545634, 0.3199743, -0.5978280, 0.6051328

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4642970, upper bound: 7.4848014
time: 3.08 seconds

## Relational analysis of IS_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4639163, upper bound: 7.4962760
time: 3.35 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -1.7130049, 1.4097342, -1.7172199, 1.4120986, -3.1251035, 3.1269541
1: -1.3990321, 1.3091495, -1.4022477, 1.3123848, -2.7114170, 2.7113972
2: -1.7798905, 1.3839961, -1.7946317, 1.3851808, -3.1650715, 3.1786280
3: -1.7612841, 1.1672256, -1.7653654, 1.1697099, -2.9309940, 2.9325910
4: -2.0171897, 1.4256068, -2.0210781, 1.4290791, -3.4462688, 3.4466848
5: -1.5770222, 1.4548714, -1.5815157, 1.4642093, -3.0412316, 3.0363870
6: -1.5851051, 1.4447801, -1.5895538, 1.4485526, -3.0336576, 3.0343339
7: -1.8426656, 1.5267886, -1.8464636, 1.5310097, -3.3736753, 3.3732522
8: -2.0468912, 1.4774591, -2.0508938, 1.4819559, -3.5288472, 3.5283527
9: -1.6040356, 1.7756433, -1.6079849, 1.7779759, -3.3820114, 3.3836284

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312283, upper bound: 7.6308860
time: 11.93 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315188, upper bound: 7.6312649
time: 4.62 seconds

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1.8591766, 1.5143209, -0.2695360, 0.3188527, -2.1780293, 1.7838569
1: -1.5095743, 1.4084179, -0.3200483, 0.3459480, -1.8555224, 1.7284663
2: -2.0151904, 1.4237691, 0.4149232, 1.0428796, -3.0580699, 1.0088459
3: -1.9349462, 1.2416153, -0.1821734, 0.4097490, -2.3446951, 1.4237888
4: -2.1890073, 1.5281169, -0.3516026, 0.3357753, -2.5247827, 1.8797195
5: -1.7122488, 1.5667179, -0.3148242, 0.3456423, -2.0578911, 1.8815420
6: -1.7377836, 1.5635941, -0.2604537, 0.3811101, -2.1188936, 1.8240478
7: -2.0070655, 1.6375288, -0.3062764, 0.3894889, -2.3965545, 1.9438052
8: -2.2442336, 1.5756983, -0.3663616, 0.4544183, -2.6986518, 1.9420598
9: -1.7366849, 1.9138919, -0.3645805, 0.3322079, -2.0688930, 2.2784724

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4448704, upper bound: 7.4969907
time: 3.62 seconds

## Relational analysis of IS_B2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4343384, upper bound: 7.4956484
time: 6.20 seconds

## BFS IS instance: IS_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -5.0890627, 3.8981214, -1.7629669, 1.4463377, -6.5354004, 5.6610880
1: -3.9565568, 3.5843825, -1.4368813, 1.3444830, -5.3010397, 5.0212641
2: -6.6078606, 2.7252810, -1.8676517, 1.3980610, -8.0059214, 4.5929327
3: -5.7886763, 2.8609350, -1.8206711, 1.1943238, -6.9829998, 4.6816063
4: -6.0692754, 3.7865472, -2.0758796, 1.4627073, -7.5319824, 5.8624268
5: -4.5849409, 4.0347872, -1.6245269, 1.4981408, -6.0830817, 5.6593142
6: -4.7948704, 4.0911818, -1.6375400, 1.4844061, -6.2792764, 5.7287216
7: -5.5338178, 4.0887442, -1.8980465, 1.5670801, -7.1008978, 5.9867907
8: -6.3314624, 3.7561386, -2.1136193, 1.5132816, -7.8447437, 5.8697577
9: -4.4762974, 5.1205540, -1.6493992, 1.8255215, -6.3018188, 6.7699533

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B1_A2_A2_A1

### Relational analysis result of IS_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309008, upper bound: 7.6310168
time: 3.97 seconds

## Relational analysis of IS_B2_B1_A2_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313429, upper bound: 7.6312592
time: 7.69 seconds

## BFS IS instance: IS_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2202931, 0.2472766, -0.3247063, 0.3822355, -0.6025286, 0.5719829
1: -0.2694451, 0.2886429, -0.3720619, 0.3962502, -0.6656953, 0.6607047
2: 0.4977174, 1.0383873, 0.3288334, 1.0576677, -0.5599504, 0.7095540
3: -0.1355165, 0.3591521, -0.2251894, 0.4536749, -0.5891913, 0.5843415
4: -0.2914917, 0.2807247, -0.4052041, 0.3982618, -0.6897535, 0.6859288
5: -0.2637063, 0.2925329, -0.3608262, 0.4120013, -0.6757076, 0.6533591
6: -0.2133014, 0.3174754, -0.3103114, 0.4362794, -0.6495808, 0.6277868
7: -0.2532171, 0.3204814, -0.3610576, 0.4524437, -0.7056608, 0.6815390
8: -0.3017149, 0.3898931, -0.4226224, 0.5117710, -0.8134859, 0.8125156
9: -0.3036644, 0.2753536, -0.4174433, 0.4009780, -0.7046424, 0.6927968

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A1_A1_B1

### Relational analysis result of IS_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4780310, upper bound: 7.5058220
time: 7.34 seconds

## Relational analysis of IS_B2_B2_A1_A1_B2

### Relational analysis result of IS_B2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4784843, upper bound: 7.5232687
time: 3.44 seconds

## BFS IS instance: IS_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -1.9323101, 1.5730417, -2.0881042, 1.6878912, -3.6202013, 3.6611459
1: -1.5654699, 1.4630164, -1.6834122, 1.5743941, -3.1398640, 3.1464286
2: -2.1269956, 1.4481250, -2.3801510, 1.4937940, -3.6207895, 3.8282762
3: -2.0252209, 1.2846531, -2.2153111, 1.3676896, -3.3929105, 3.4999642
4: -2.2795062, 1.5855339, -2.4647474, 1.6987475, -3.9782538, 4.0502815
5: -1.7815702, 1.6201873, -1.9274306, 1.7440679, -3.5256381, 3.5476179
6: -1.8129494, 1.6161802, -1.9746823, 1.7381868, -3.5511363, 3.5908625
7: -2.0876541, 1.6993060, -2.2592344, 1.8227265, -3.9103806, 3.9585404
8: -2.3480158, 1.6256492, -2.5588250, 1.7317889, -4.0798044, 4.1844740
9: -1.8028166, 2.0007310, -1.9437864, 2.1576128, -3.9604294, 3.9445174

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A1_A2_B1

### Relational analysis result of IS_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312680, upper bound: 7.6309450
time: 5.39 seconds

## Relational analysis of IS_B2_B2_A1_A2_B2

### Relational analysis result of IS_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316339, upper bound: 7.6314641
time: 3.38 seconds

## BFS IS instance: IS_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -2.0556910, 1.6601466, -0.3346260, 0.3932492, -2.4489403, 1.9947726
1: -1.6584194, 1.5467254, -0.3810177, 0.4056408, -2.0640602, 1.9277431
2: -2.3194780, 1.4812391, 0.3130428, 1.0599878, -3.3794658, 1.1681962
3: -2.1760583, 1.3459669, -0.2329082, 0.4612089, -2.6372671, 1.5788751
4: -2.4244022, 1.6700402, -0.4142653, 0.4099405, -2.8343427, 2.0843055
5: -1.8958774, 1.7102185, -0.3703410, 0.4232096, -2.3190870, 2.0805595
6: -1.9421315, 1.7166784, -0.3192514, 0.4456673, -2.3877988, 2.0359297
7: -2.2253118, 1.7905902, -0.3704677, 0.4642849, -2.6895967, 2.1610579
8: -2.5133042, 1.7068309, -0.4319888, 0.5218548, -3.0351591, 2.1388197
9: -1.9145848, 2.1148267, -0.4263469, 0.4142855, -2.3288703, 2.5411735

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A2_A1_A1

### Relational analysis result of IS_B2_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4586107, upper bound: 7.5224118
time: 3.98 seconds

## Relational analysis of IS_B2_B2_A2_A1_A2

### Relational analysis result of IS_B2_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4403583, upper bound: 7.5224176
time: 10.57 seconds

## BFS IS instance: IS_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -5.3931427, 4.1141248, -2.1298437, 1.7190328, -7.1121755, 6.2439685
1: -4.1945620, 3.7810287, -1.7149413, 1.6040608, -5.7986226, 5.4959698
2: -7.0169191, 2.8453822, -2.4453118, 1.5057995, -8.5227184, 5.2906942
3: -6.1421046, 3.0130203, -2.2668970, 1.3898013, -7.5319061, 5.2799172
4: -6.4218163, 3.9958444, -2.5143142, 1.7290806, -8.1508970, 6.5101585
5: -4.8395176, 4.2557406, -1.9666169, 1.7747307, -6.6142483, 6.2223577
6: -5.0763860, 4.3251190, -2.0183744, 1.7707344, -6.8471203, 6.3434935
7: -5.8583784, 4.3115883, -2.3056705, 1.8551675, -7.7135458, 6.6172590
8: -6.7025943, 3.9537749, -2.6156878, 1.7601178, -8.4627123, 6.5694628
9: -4.7231894, 5.4260774, -1.9813111, 2.2004497, -6.9236393, 7.4073887

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_B2_B2_A2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309396, upper bound: 7.6311259
time: 4.94 seconds

## Relational analysis of IS_B2_B2_A2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314572, upper bound: 7.6314572
time: 3.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 10.25 seconds
IS_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4846840, upper bound: 7.4675718
IS_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4962444, upper bound: 7.4671136
IS_B1_A1_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4969150, upper bound: 7.4472287
IS_B1_A1_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4955314, upper bound: 7.4357553
IS_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6308864, upper bound: 7.6312515
IS_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6312654, upper bound: 7.6315311
IS_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6310176, upper bound: 7.6309220
IS_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6312592, upper bound: 7.6313475
IS_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.5151575, upper bound: 7.4419108
IS_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.5342344, upper bound: 7.4419196
IS_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.5041666, upper bound: 7.4419076
IS_B1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.5220109, upper bound: 7.4419159
IS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6309455, upper bound: 7.6312866
IS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6314647, upper bound: 7.6316516
IS_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6311260, upper bound: 7.6309564
IS_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6314573, upper bound: 7.6314615
IS_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4642970, upper bound: 7.4848014
IS_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4639163, upper bound: 7.4962760
IS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6312283, upper bound: 7.6308860
IS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6315188, upper bound: 7.6312649
IS_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4448704, upper bound: 7.4969907
IS_B2_B1_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4343384, upper bound: 7.4956484
IS_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6309008, upper bound: 7.6310168
IS_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6313429, upper bound: 7.6312592
IS_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4780310, upper bound: 7.5058220
IS_B2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4784843, upper bound: 7.5232687
IS_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6312680, upper bound: 7.6309450
IS_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6316339, upper bound: 7.6314641
IS_B2_B2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4586107, upper bound: 7.5224118
IS_B2_B2_A2_A1_A2, status: Status.VERIFIED, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.4403583, upper bound: 7.5224176
IS_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6309396, upper bound: 7.6311259
IS_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 10.25
Output dim: 2, lower bound: -7.6314572, upper bound: 7.6314572

## BFS IS instance: IS_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3239588, 0.3819306, -0.2682765, 0.3179641, -0.6419230, 0.6502072
1: -0.3714838, 0.3951572, -0.3190314, 0.3444707, -0.7159544, 0.7141885
2: 0.3293114, 1.0567409, 0.4162268, 1.0419468, -0.7126353, 0.6405141
3: -0.2245854, 0.4531425, -0.1811827, 0.4088623, -0.6334478, 0.6343251
4: -0.4040554, 0.3974327, -0.3500963, 0.3344013, -0.7384567, 0.7475290
5: -0.3610101, 0.4105807, -0.3144615, 0.3437958, -0.7048059, 0.7250422
6: -0.3100581, 0.4352003, -0.2597300, 0.3796188, -0.6896768, 0.6949303
7: -0.3602476, 0.4516046, -0.3050051, 0.3881169, -0.7483644, 0.7566097
8: -0.4208643, 0.5113509, -0.3642974, 0.4534627, -0.8743269, 0.8756483
9: -0.4161788, 0.4007342, -0.3629770, 0.3312650, -0.7474437, 0.7637112

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_B1_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5369420, upper bound: 7.6259428
time: 5.07 seconds

## Relational analysis of IS_B1_A1_B2_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5369086, upper bound: 7.5855740
time: 3.14 seconds

## BFS IS instance: IS_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.6410213, 0.6598396, -0.3898779, 0.4545301, -1.0955513, 1.0497174
1: -0.6168981, 0.6297956, -0.4311941, 0.4588051, -1.0757031, 1.0609896
2: -0.1366729, 1.1262910, 0.2252468, 1.0740790, -1.2107519, 0.9010442
3: -0.4896091, 0.6444982, -0.2760097, 0.5036019, -0.9932110, 0.9205079
4: -0.7283221, 0.6882091, -0.4658522, 0.4754367, -1.2037588, 1.1540613
5: -0.6163859, 0.7267670, -0.4226711, 0.4871133, -1.1034992, 1.1494381
6: -0.5874796, 0.6767609, -0.3689227, 0.4987192, -1.0861988, 1.0456836
7: -0.6582704, 0.7427402, -0.4231745, 0.5309563, -1.1892266, 1.1659147
8: -0.7267694, 0.7769821, -0.4859657, 0.5778942, -1.3046637, 1.2629478
9: -0.6763505, 0.7568667, -0.4771733, 0.4882383, -1.1645888, 1.2340400

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_B2_B1_A2_A1

### Relational analysis result of IS_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6274661, upper bound: 7.6160603
time: 3.24 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2_A2

### Relational analysis result of IS_B1_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5766574, upper bound: 7.6136823
time: 2.61 seconds

## BFS IS instance: IS_B1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.4353773, 0.5041955, -1.7454914, 1.4346794, -1.8700566, 2.2496870
1: -0.4742696, 0.5024987, -1.4250760, 1.3244663, -1.7987360, 1.9275748
2: 0.1539970, 1.0861157, -1.8389237, 1.3864335, -1.2324364, 2.9250393
3: -0.3114301, 0.5380479, -1.8004802, 1.1814184, -1.4928485, 2.3385282
4: -0.5086535, 0.5286908, -2.0543787, 1.4453145, -1.9539680, 2.5830696
5: -0.4665967, 0.5393621, -1.6071024, 1.4765546, -1.9431514, 2.1464643
6: -0.4099407, 0.5423809, -1.6206919, 1.4705667, -1.8805075, 2.1630728
7: -0.4670005, 0.5847885, -1.8847228, 1.5422161, -2.0092165, 2.4695113
8: -0.5308172, 0.6259872, -2.0894268, 1.4966769, -2.0274942, 2.7154140
9: -0.5189776, 0.5492907, -1.6282035, 1.8109463, -2.3299241, 2.1774940

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_B2_B2_B1_B1

### Relational analysis result of IS_B1_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5604640, upper bound: 7.5930105
time: 10.33 seconds

## Relational analysis of IS_B1_A1_B2_B2_B1_B2

### Relational analysis result of IS_B1_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5596062, upper bound: 7.5512883
time: 6.79 seconds

## BFS IS instance: IS_B1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.6409354, 0.6595678, -2.6089602, 2.0907605, -2.7316959, 3.2685280
1: -0.6167530, 0.6304125, -2.0913095, 1.9240196, -2.5407724, 2.7217221
2: -0.1360852, 1.1267914, -3.2129159, 1.6330204, -1.7691056, 4.3397074
3: -0.4894736, 0.6443824, -2.8549159, 1.6325399, -2.1220136, 3.4992981
4: -0.7281373, 0.6884401, -3.1107025, 2.0715675, -2.7997048, 3.7991426
5: -0.6156490, 0.7269634, -2.4243677, 2.1050682, -2.7207172, 3.1513309
6: -0.5869542, 0.6769252, -2.4567699, 2.1384192, -2.7253733, 3.1336951
7: -0.6582588, 0.7428862, -2.8767850, 2.2114353, -2.8696942, 3.6196713
8: -0.7270193, 0.7769870, -3.2823582, 2.0879583, -2.8149776, 4.0593452
9: -0.6767960, 0.7562391, -2.4241617, 2.7058468, -3.3826427, 3.1804008

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A1_B2_B2_B2_A1

### Relational analysis result of IS_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6274700, upper bound: 7.6034038
time: 2.70 seconds

## Relational analysis of IS_B1_A1_B2_B2_B2_A2

### Relational analysis result of IS_B1_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5766532, upper bound: 7.6011934
time: 6.40 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3823647, 0.4461442, -0.3017268, 0.3566697, -0.7390345, 0.7478710
1: -0.4243349, 0.4515424, -0.3507084, 0.3745621, -0.7988969, 0.8022507
2: 0.2372525, 1.0721444, 0.3640389, 1.0507162, -0.8134637, 0.7081056
3: -0.2701420, 0.4977979, -0.2072446, 0.4355842, -0.7057263, 0.7050425
4: -0.4587916, 0.4664846, -0.3825622, 0.3719870, -0.8307786, 0.8490468
5: -0.4155137, 0.4783555, -0.3420393, 0.3839705, -0.7994843, 0.8203948
6: -0.3621371, 0.4914784, -0.2900561, 0.4131045, -0.7752417, 0.7815344
7: -0.4159945, 0.5218246, -0.3383509, 0.4258955, -0.8418900, 0.8601755
8: -0.4785854, 0.5702479, -0.3982256, 0.4884218, -0.9670072, 0.9684735
9: -0.4702211, 0.4781079, -0.3949633, 0.3726683, -0.8428894, 0.8730712

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5550090, upper bound: 7.6276469
time: 6.35 seconds

## Relational analysis of IS_B1_A2_B2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5549837, upper bound: 7.6014676
time: 5.09 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.0246823, 0.9109364, -0.4478168, 0.5186638, -1.5433460, 1.3587532
1: -0.8816651, 0.8467242, -0.4875610, 0.5141806, -1.3958457, 1.3342853
2: -0.7072873, 1.1971674, 0.1333809, 1.0889379, -1.7962252, 1.0637865
3: -0.9282823, 0.8152008, -0.3209277, 0.5478788, -1.4761610, 1.1361284
4: -1.1863526, 0.9373721, -0.5205733, 0.5434016, -1.7297542, 1.4579455
5: -0.9425974, 0.9786198, -0.4807937, 0.5535260, -1.4961234, 1.4594135
6: -0.9278010, 0.9314222, -0.4222606, 0.5540131, -1.4818141, 1.3536828
7: -1.0691464, 0.9982071, -0.4790470, 0.6001067, -1.6692531, 1.4772542
8: -1.1274880, 1.0157312, -0.5423865, 0.6410382, -1.7685262, 1.5581177
9: -0.9908210, 1.0951165, -0.5301092, 0.5679979, -1.5588188, 1.6252258

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283196, upper bound: 7.6276000
time: 5.51 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6219351, upper bound: 7.6275557
time: 3.66 seconds

## BFS IS instance: IS_B1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.5173736, 0.5839319, -1.8639722, 1.5226035, -2.0399771, 2.4479041
1: -0.5469872, 0.5690922, -1.5131114, 1.4108764, -1.9578636, 2.0822036
2: 0.0264650, 1.1051081, -2.0225852, 1.4218807, -1.3954158, 3.1276932
3: -0.3740690, 0.5962976, -1.9435842, 1.2452496, -1.6193186, 2.5398817
4: -0.6002001, 0.6070770, -2.1940918, 1.5326608, -2.1328609, 2.8011689
5: -0.5383090, 0.6369645, -1.7162189, 1.5628191, -2.1011281, 2.3531835
6: -0.4893194, 0.6071568, -1.7474803, 1.5637298, -2.0530491, 2.3546371
7: -0.5529546, 0.6712081, -2.0164437, 1.6369880, -2.1899426, 2.6876519
8: -0.6222173, 0.7057459, -2.2552936, 1.5763109, -2.1985283, 2.9610395
9: -0.5903021, 0.6496753, -1.7356515, 1.9317267, -2.5220289, 2.3853269

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_B2_B2_B1_B1

### Relational analysis result of IS_B1_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5901824, upper bound: 7.6129918
time: 4.97 seconds

## Relational analysis of IS_B1_A2_B2_B2_B1_B2

### Relational analysis result of IS_B1_A2_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5888788, upper bound: 7.5618857
time: 5.13 seconds

## BFS IS instance: IS_B1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -1.0068678, 0.8988023, -2.8819151, 2.2931175, -3.2999854, 3.7807174
1: -0.8687578, 0.8371143, -2.2850866, 2.1404648, -3.0092225, 3.1222010
2: -0.6800162, 1.1942241, -3.6296341, 1.7162683, -2.3962846, 4.8238583
3: -0.9065353, 0.8069235, -3.1843038, 1.7938750, -2.7004104, 3.9912271
4: -1.1649381, 0.9256589, -3.4187279, 2.2852733, -3.4502115, 4.3443871
5: -0.9260644, 0.9671379, -2.6736894, 2.3060520, -3.2321162, 3.6408272
6: -0.9103822, 0.9197000, -2.8082383, 2.3655257, -3.2759080, 3.7279382
7: -1.0491757, 0.9861502, -3.1686041, 2.4477589, -3.4969347, 4.1547542
8: -1.1074791, 1.0042604, -3.6808195, 2.2705264, -3.3780055, 4.6850801
9: -0.9764109, 1.0784175, -2.6707878, 2.9818234, -3.9582343, 3.7492054

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B1_A2_B2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283196, upper bound: 7.6254098
time: 3.97 seconds

## Relational analysis of IS_B1_A2_B2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6219306, upper bound: 7.6253801
time: 4.94 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4475907, 0.5178056, -0.3523251, 0.4130748, -0.8606656, 0.8701308
1: -0.4870196, 0.5142366, -0.3970310, 0.4222754, -0.9092950, 0.9112676
2: 0.1345344, 1.0894709, 0.2846345, 1.0638878, -0.9293534, 0.8048363
3: -0.3207788, 0.5475426, -0.2466751, 0.4746686, -0.7954473, 0.7942177
4: -0.5206046, 0.5430015, -0.4302880, 0.4307982, -0.9514028, 0.9732896
5: -0.4795387, 0.5536143, -0.3876134, 0.4430258, -0.9225645, 0.9412277
6: -0.4215408, 0.5540668, -0.3353378, 0.4622994, -0.8838402, 0.8894045
7: -0.4788717, 0.5995796, -0.3872269, 0.4854396, -0.9643114, 0.9868065
8: -0.5429765, 0.6402682, -0.4483669, 0.5399588, -1.0829353, 1.0886352
9: -0.5303094, 0.5667216, -0.4420482, 0.4382792, -0.9685886, 1.0087698

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_B2_B1_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6244378, upper bound: 7.5369419
time: 5.00 seconds

## Relational analysis of IS_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_B2_B1_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5837822, upper bound: 7.5369086
time: 4.14 seconds

## BFS IS instance: IS_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.7269943, 0.7164934, -0.7482264, 0.7302510, -1.4572453, 1.4647198
1: -0.6729398, 0.6763539, -0.6877987, 0.6864661, -1.3594059, 1.3641527
2: -0.2605641, 1.1419387, -0.2926546, 1.1446824, -1.4052465, 1.4345933
3: -0.5765934, 0.6812204, -0.6007017, 0.6898835, -1.2664769, 1.2819221
4: -0.8272586, 0.7446625, -0.8518100, 0.7581484, -1.5854070, 1.5964725
5: -0.6818824, 0.7863671, -0.7008744, 0.7995828, -1.4814652, 1.4872415
6: -0.6597082, 0.7313260, -0.6794712, 0.7445505, -1.4042587, 1.4107971
7: -0.7394357, 0.7950452, -0.7622214, 0.8081936, -1.5476294, 1.5572666
8: -0.8068534, 0.8264061, -0.8274885, 0.8388186, -1.6456720, 1.6538947
9: -0.7397996, 0.8355030, -0.7558935, 0.8549579, -1.5947576, 1.5913966

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 173

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6156383, upper bound: 7.6274758
time: 2.92 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6133970, upper bound: 7.5766572
time: 4.60 seconds

## BFS IS instance: IS_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -2.7101765, 2.1559739, -0.4669355, 0.5381297, -3.2483063, 2.6229093
1: -2.1371560, 2.0132544, -0.5059016, 0.5312607, -2.6684167, 2.5191560
2: -3.3394148, 1.7087181, 0.1051135, 1.0937037, -4.4331188, 1.6036046
3: -2.9518442, 1.6891221, -0.3347542, 0.5616105, -3.5134547, 2.0238762
4: -3.2508700, 2.1497006, -0.5383887, 0.5639286, -3.8147986, 2.6880894
5: -2.5018001, 2.2349582, -0.4993276, 0.5743460, -3.0761461, 2.7342858
6: -2.5995026, 2.2380886, -0.4395157, 0.5710144, -3.1705170, 2.6776042
7: -2.9705842, 2.2957532, -0.4969208, 0.6214012, -3.5919852, 2.7926741
8: -3.3759665, 2.1596341, -0.5605943, 0.6624042, -4.0383706, 2.7202284
9: -2.5030730, 2.7819991, -0.5465084, 0.5924900, -3.0955629, 3.3285074

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_A2_A2_A1_A1

### Relational analysis result of IS_B2_B1_A2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5912399, upper bound: 7.5604639
time: 3.30 seconds

## Relational analysis of IS_B2_B1_A2_A2_A1_A2

### Relational analysis result of IS_B2_B1_A2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5477099, upper bound: 7.5595937
time: 4.03 seconds

## BFS IS instance: IS_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -3.7021599, 2.8822079, -0.7513342, 0.7320132, -4.4341731, 3.6335421
1: -2.8541958, 2.6713879, -0.6897714, 0.6888507, -3.5430465, 3.3611593
2: -4.7235880, 2.1175005, -0.2966198, 1.1457840, -5.8693719, 2.4141204
3: -4.1396866, 2.1783953, -0.6039004, 0.6909699, -4.8306565, 2.7822957
4: -4.4358077, 2.8327031, -0.8556706, 0.7604434, -5.1962509, 3.6883736
5: -3.3772871, 2.9929986, -0.7026699, 0.8020324, -4.1793194, 3.6956685
6: -3.5200052, 3.0071807, -0.6816851, 0.7467055, -4.2667108, 3.6888657
7: -4.0489655, 3.0401571, -0.7661359, 0.8102193, -4.8591847, 3.8062930
8: -4.6076975, 2.8262239, -0.8313453, 0.8406028, -5.4483004, 3.6575692
9: -3.3292816, 3.7599328, -0.7589613, 0.8568943, -4.1861758, 4.5188942

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B1_A2_A2_A2_B1

### Relational analysis result of IS_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6029794, upper bound: 7.6274787
time: 3.33 seconds

## Relational analysis of IS_B2_B1_A2_A2_A2_B2

### Relational analysis result of IS_B2_B1_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6009003, upper bound: 7.5766536
time: 3.90 seconds

## BFS IS instance: IS_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4921442, 0.5650535, -0.4099852, 0.4764804, -0.9686246, 0.9750386
1: -0.5289851, 0.5530535, -0.4492188, 0.4779571, -1.0069422, 1.0022724
2: 0.0643819, 1.1002811, 0.1937335, 1.0791078, -1.0147259, 0.9065476
3: -0.3523544, 0.5819188, -0.2916525, 0.5187651, -0.8711195, 0.8735713
4: -0.5710276, 0.5886707, -0.4843456, 0.4989849, -1.0700126, 1.0730164
5: -0.5218976, 0.6100044, -0.4414243, 0.5099680, -1.0318656, 1.0514287
6: -0.4672188, 0.5911634, -0.3867645, 0.5178708, -0.9850896, 0.9779279
7: -0.5257271, 0.6500202, -0.4422677, 0.5547814, -1.0805085, 1.0922880
8: -0.5932413, 0.6879162, -0.5053763, 0.5981114, -1.1913526, 1.1932924
9: -0.5710349, 0.6235850, -0.4954197, 0.5146766, -1.0857115, 1.1190047

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_B2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275911, upper bound: 7.5550084
time: 4.33 seconds

## Relational analysis of IS_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_B2_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5977567, upper bound: 7.5549838
time: 3.44 seconds

## BFS IS instance: IS_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.9635218, 0.8702549, -1.1439162, 0.9912986, -1.9548204, 2.0141711
1: -0.8376286, 0.8116579, -0.9683510, 0.9193578, -1.7569864, 1.7800089
2: -0.6150783, 1.1862290, -0.8884141, 1.2225085, -1.8375869, 2.0746431
3: -0.8541907, 0.7868127, -1.0720148, 0.8722246, -1.7264153, 1.8588276
4: -1.1131660, 0.8965640, -1.3292112, 1.0193615, -2.1325274, 2.2257752
5: -0.8872388, 0.9393908, -1.0490258, 1.0566181, -1.9438570, 1.9884166
6: -0.8701240, 0.8903593, -1.0397248, 1.0140396, -1.8841636, 1.9300842
7: -1.0004109, 0.9558344, -1.2031451, 1.0851370, -2.0855479, 2.1589794
8: -1.0611064, 0.9755446, -1.2679164, 1.0941360, -2.1552424, 2.2434611
9: -0.9399716, 1.0401248, -1.0914520, 1.2037143, -2.1436858, 2.1315768

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 173

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_A1_A2_B2_B1

### Relational analysis result of IS_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275362, upper bound: 7.6283232
time: 3.24 seconds

## Relational analysis of IS_B2_B2_A1_A2_B2_B2

### Relational analysis result of IS_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6274915, upper bound: 7.6219352
time: 4.16 seconds

## BFS IS instance: IS_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -2.8653183, 2.2693853, -0.6006028, 0.6328853, -3.4982038, 2.8699882
1: -2.2437310, 2.1169322, -0.5944997, 0.6094292, -2.8531604, 2.7114320
2: -3.5572171, 1.7649485, -0.0808782, 1.1195838, -4.6768007, 1.8458266
3: -3.1362801, 1.7660882, -0.4514263, 0.6274495, -3.7637296, 2.2175145
4: -3.4399657, 2.2570770, -0.6834301, 0.6614878, -4.1014538, 2.9405072
5: -2.6393545, 2.3516316, -0.5885529, 0.6982213, -3.3375759, 2.9401846
6: -2.7442136, 2.3582408, -0.5557547, 0.6519861, -3.3961997, 2.9139955
7: -3.1413743, 2.4116096, -0.6238632, 0.7197533, -3.8611276, 3.0354729
8: -3.5696011, 2.2639859, -0.6928595, 0.7531024, -4.3227034, 2.9568453
9: -2.6343424, 2.9354413, -0.6480089, 0.7189380, -3.3532805, 3.5834503

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5550016, upper bound: 7.6273563
time: 3.15 seconds

## Relational analysis of IS_B2_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5549794, upper bound: 7.5888943
time: 4.09 seconds

## BFS IS instance: IS_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -4.0786510, 3.1563859, -1.1262608, 0.9788368, -5.0574875, 4.2826467
1: -3.1536202, 2.9181652, -0.9556907, 0.9091879, -4.0628080, 3.8738558
2: -5.2349486, 2.2713647, -0.8603307, 1.2194108, -6.4543595, 3.1316953
3: -4.5878630, 2.3647480, -1.0505003, 0.8638089, -5.4516716, 3.4152484
4: -4.8811369, 3.0912392, -1.3079689, 1.0079324, -5.8890696, 4.3992081
5: -3.7007418, 3.2711759, -1.0325291, 1.0452886, -4.7460303, 4.3037052
6: -3.8679643, 3.3004904, -1.0225586, 1.0019665, -4.8699307, 4.3230491
7: -4.4518661, 3.3229330, -1.1834735, 1.0729100, -5.5247760, 4.5064063
8: -5.0749717, 3.0766058, -1.2471352, 1.0827605, -6.1577320, 4.3237410
9: -3.6397560, 4.1355505, -1.0773183, 1.1861399, -4.8258958, 5.2128687

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B2_B2_A2_A2_A2_B1

### Relational analysis result of IS_B2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6219531, upper bound: 7.6283241
time: 7.95 seconds

## Relational analysis of IS_B2_B2_A2_A2_A2_B2

### Relational analysis result of IS_B2_B2_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6219305, upper bound: 7.5888947
time: 11.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 20.75 seconds
IS_B1_A1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5369420, upper bound: 7.6259428
IS_B1_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5369086, upper bound: 7.5855740
IS_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6274661, upper bound: 7.6160603
IS_B1_A1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5766574, upper bound: 7.6136823
IS_B1_A1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5604640, upper bound: 7.5930105
IS_B1_A1_B2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5596062, upper bound: 7.5512883
IS_B1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6274700, upper bound: 7.6034038
IS_B1_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5766532, upper bound: 7.6011934
IS_B1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5550090, upper bound: 7.6276469
IS_B1_A2_B2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5549837, upper bound: 7.6014676
IS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6283196, upper bound: 7.6276000
IS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6219351, upper bound: 7.6275557
IS_B1_A2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5901824, upper bound: 7.6129918
IS_B1_A2_B2_B2_B1_B2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5888788, upper bound: 7.5618857
IS_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6283196, upper bound: 7.6254098
IS_B1_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6219306, upper bound: 7.6253801
IS_B2_B1_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6244378, upper bound: 7.5369419
IS_B2_B1_A1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5837822, upper bound: 7.5369086
IS_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6156383, upper bound: 7.6274758
IS_B2_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6133970, upper bound: 7.5766572
IS_B2_B1_A2_A2_A1_A1, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5912399, upper bound: 7.5604639
IS_B2_B1_A2_A2_A1_A2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5477099, upper bound: 7.5595937
IS_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6029794, upper bound: 7.6274787
IS_B2_B1_A2_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6009003, upper bound: 7.5766536
IS_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6275911, upper bound: 7.5550084
IS_B2_B2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5977567, upper bound: 7.5549838
IS_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6275362, upper bound: 7.6283232
IS_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6274915, upper bound: 7.6219352
IS_B2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5550016, upper bound: 7.6273563
IS_B2_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.5549794, upper bound: 7.5888943
IS_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6219531, upper bound: 7.6283241
IS_B2_B2_A2_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 20.75
Output dim: 2, lower bound: -7.6219305, upper bound: 7.5888947

## BFS IS instance: IS_B1_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.4805171, 0.5542248, -0.3489351, 0.4094247, -0.8899418, 0.9031599
1: -0.5192206, 0.5442593, -0.3941882, 0.4195014, -0.9387220, 0.9384475
2: 0.0816415, 1.0975283, 0.2899190, 1.0636208, -0.9819793, 0.8076093
3: -0.3441941, 0.5735343, -0.2440985, 0.4723769, -0.8165709, 0.8176327
4: -0.5567777, 0.5786235, -0.4277675, 0.4271088, -0.9838865, 1.0063910
5: -0.5127346, 0.5955586, -0.3842000, 0.4399720, -0.9527066, 0.9797586
6: -0.4556398, 0.5830150, -0.3323273, 0.4594713, -0.9151111, 0.9153422
7: -0.5131117, 0.6379464, -0.3841677, 0.4818642, -0.9949758, 1.0221140
8: -0.5796117, 0.6775615, -0.4460101, 0.5365303, -1.1161420, 1.1235715
9: -0.5606736, 0.6104860, -0.4396170, 0.4338494, -0.9945230, 1.0501029

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_B1_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_B1_A1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5727495, upper bound: 7.5632176
time: 6.39 seconds

## Relational analysis of IS_B1_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_B1_A1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5696095, upper bound: 7.5516256
time: 6.66 seconds

## BFS IS instance: IS_B1_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4823263, 0.5555320, -2.4415584, 1.9637845, -2.4461107, 2.9970903
1: -0.5205733, 0.5459378, -1.9622638, 1.8083372, -2.3289106, 2.5082016
2: 0.0796480, 1.0983428, -2.9465361, 1.5857221, -1.5060741, 4.0448790
3: -0.3454218, 0.5746906, -2.6506939, 1.5451665, -1.8905883, 3.2253845
4: -0.5588046, 0.5802804, -2.9065363, 1.9504241, -2.5092287, 3.4868166
5: -0.5136160, 0.5978138, -2.2657266, 1.9837430, -2.4973590, 2.8635404
6: -0.4569070, 0.5846459, -2.2942541, 2.0091395, -2.4660466, 2.8789001
7: -0.5148425, 0.6396905, -2.6845551, 2.0824201, -2.5972626, 3.3242455
8: -0.5819590, 0.6790519, -3.0516958, 1.9733709, -2.5553298, 3.7307477
9: -0.5625308, 0.6118819, -2.2705088, 2.5322852, -3.0948160, 2.8823905

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_B1_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_B1_A1_B2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5730042, upper bound: 7.5523798
time: 6.93 seconds

## Relational analysis of IS_B1_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_B1_A1_B2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5697490, upper bound: 7.5361486
time: 3.18 seconds

## BFS IS instance: IS_B1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3415038, 0.4012203, -0.2457390, 0.2858540, -0.6273578, 0.6469593
1: -0.3874636, 0.4123655, -0.2958865, 0.3195068, -0.7069705, 0.7082520
2: 0.3016816, 1.0617303, 0.4544025, 1.0346715, -0.7329900, 0.6073277
3: -0.2383052, 0.4667034, -0.1603681, 0.3866938, -0.6249990, 0.6270714
4: -0.4208534, 0.4183263, -0.3231628, 0.3092097, -0.7300631, 0.7414891
5: -0.3771960, 0.4314094, -0.2915832, 0.3191196, -0.6963156, 0.7229926
6: -0.3256789, 0.4523474, -0.2380445, 0.3518392, -0.6775180, 0.6903919
7: -0.3770882, 0.4729431, -0.2811205, 0.3567753, -0.7338635, 0.7540636
8: -0.4387662, 0.5290135, -0.3363530, 0.4244609, -0.8632271, 0.8653665
9: -0.4328023, 0.4239506, -0.3364415, 0.3046568, -0.7374591, 0.7603920

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 122

## Relational analysis of IS_B1_A2_B2_B1_A1_B1_B1

### Relational analysis result of IS_B1_A2_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4857010, upper bound: 7.5845216
time: 13.73 seconds

## Relational analysis of IS_B1_A2_B2_B1_A1_B1_B2

### Relational analysis result of IS_B1_A2_B2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4856979, upper bound: 7.5807115
time: 6.44 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 21.70 seconds
IS_B1_A1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 21.70
Output dim: 2, lower bound: -7.5727495, upper bound: 7.5632176
IS_B1_A1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 21.70
Output dim: 2, lower bound: -7.5696095, upper bound: 7.5516256
IS_B1_A1_B2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 21.70
Output dim: 2, lower bound: -7.5730042, upper bound: 7.5523798
IS_B1_A1_B2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 21.70
Output dim: 2, lower bound: -7.5697490, upper bound: 7.5361486
IS_B1_A2_B2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 21.70
Output dim: 2, lower bound: -7.4857010, upper bound: 7.5845216
IS_B1_A2_B2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 21.70
Output dim: 2, lower bound: -7.4856979, upper bound: 7.5807115
IS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6283196, upper bound: 7.6276000
IS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6219351, upper bound: 7.6275557
IS_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6283196, upper bound: 7.6254098
IS_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6156383, upper bound: 7.6274758
IS_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6029794, upper bound: 7.6274787
IS_B2_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6275911, upper bound: 7.5550084
IS_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6275362, upper bound: 7.6283232
IS_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6274915, upper bound: 7.6219352
IS_B2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.5550016, upper bound: 7.6273563
IS_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.70
Output dim: 2, lower bound: -7.6219531, upper bound: 7.6283241
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=9.388381958007812
rel_dist={2: [-7.63358222082411, 7.633582568333054]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1821.86 seconds
