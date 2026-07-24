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
execution time: IAR + LP analysis = 1.46 + 5.66 = 7.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -7.6336112, upper bound: 7.6336109


# Binary Search by BASE starts (time budget: 1992.87 seconds, max iter: 100)

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
Binary search time: 18.76 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1974.11 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6334873, upper bound: 7.6329929
time: 3.93 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329542, upper bound: 7.6329546
time: 2.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 6.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 6.84
Output dim: 2, lower bound: -7.6334873, upper bound: 7.6329929
IS_A2, status: Status.UNKNOWN, split count: 1, time: 6.84
Output dim: 2, lower bound: -7.6329542, upper bound: 7.6329546

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.2686481, 3.3034801, -5.1286879, 3.9331427, -8.2017908, 8.4321680
1: -3.3155420, 3.0498059, -4.0091529, 3.6131139, -6.9286556, 7.0589590
2: -5.4949217, 2.3689001, -6.6639881, 2.7243943, -8.2193165, 9.0328884
3: -4.8110600, 2.4704089, -5.8328662, 2.9040592, -7.7151194, 8.3032751
4: -5.1011324, 3.2392945, -6.1130657, 3.8349376, -8.9360695, 9.3523598
5: -3.8673596, 3.4099646, -4.6064701, 4.0478497, -7.9152093, 8.0164347
6: -4.0397010, 3.4505873, -4.8427486, 4.1230493, -8.1627502, 8.2933359
7: -4.6541395, 3.4808147, -5.5768681, 4.1297841, -8.7839241, 9.0576830
8: -5.3186646, 3.2162716, -6.3826962, 3.8029628, -9.1216278, 9.5989676
9: -3.7986977, 4.3246951, -4.5053129, 5.1798744, -8.9785719, 8.8300076

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329542, upper bound: 7.6329546
time: 2.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329542, upper bound: 7.6329546
time: 2.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.3132782, 6.2759089, -5.0093522, 3.8459010, -12.1591797, 11.2852612
1: -6.5844011, 5.6741643, -3.9128380, 3.5350213, -10.1194229, 9.5870018
2: -10.9843225, 4.1436543, -6.5023088, 2.6747284, -13.6590509, 10.6459637
3: -9.5884886, 4.5238266, -5.6912117, 2.8437872, -12.4322758, 10.2150383
4: -9.8131104, 6.0509901, -5.9729309, 3.7523787, -13.5654888, 12.0239210
5: -7.3376122, 6.4407911, -4.5041556, 3.9591372, -11.2967491, 10.9449463
6: -7.8093801, 6.6083603, -4.7312536, 4.0299139, -11.8392944, 11.3396139
7: -8.9933872, 6.5240011, -5.4488769, 4.0398254, -13.0332127, 11.9728775
8: -10.3325996, 5.9963574, -6.2352085, 3.7215319, -14.0541315, 12.2315655
9: -7.1123838, 8.2964840, -4.4072256, 5.0615902, -12.1739740, 12.7037096

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319716, upper bound: 7.6305064
time: 2.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325635, upper bound: 7.6325634
time: 2.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 6.58 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.58
Output dim: 2, lower bound: -7.6329542, upper bound: 7.6329546
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.58
Output dim: 2, lower bound: -7.6329542, upper bound: 7.6329546
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.58
Output dim: 2, lower bound: -7.6319716, upper bound: 7.6305064
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.58
Output dim: 2, lower bound: -7.6325635, upper bound: 7.6325634

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -4.2686481, 3.3034801, -4.2686481, 3.3034801, -7.5721283, 7.5721283
1: -3.3155420, 3.0498059, -3.3155420, 3.0498059, -6.3653479, 6.3653479
2: -5.4949217, 2.3689001, -5.4949217, 2.3689001, -7.8638220, 7.8638220
3: -4.8110600, 2.4704089, -4.8110600, 2.4704089, -7.2814689, 7.2814689
4: -5.1011324, 3.2392945, -5.1011324, 3.2392945, -8.3404274, 8.3404274
5: -3.8673596, 3.4099646, -3.8673596, 3.4099646, -7.2773242, 7.2773242
6: -4.0397010, 3.4505873, -4.0397010, 3.4505873, -7.4902883, 7.4902883
7: -4.6541395, 3.4808147, -4.6541395, 3.4808147, -8.1349545, 8.1349545
8: -5.3186646, 3.2162716, -5.3186646, 3.2162716, -8.5349360, 8.5349360
9: -3.7986977, 4.3246951, -3.7986977, 4.3246951, -8.1233931, 8.1233931

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315498, upper bound: 7.6320359
time: 4.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330834, upper bound: 7.6325995
time: 3.84 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -4.2686481, 3.3034801, -8.3132782, 6.2759089, -10.5445576, 11.6167583
1: -3.3155420, 3.0498059, -6.5844011, 5.6741643, -8.9897060, 9.6342068
2: -5.4949217, 2.3689001, -10.9843225, 4.1436543, -9.6385765, 13.3532228
3: -4.8110600, 2.4704089, -9.5884886, 4.5238266, -9.3348866, 12.0588970
4: -5.1011324, 3.2392945, -9.8131104, 6.0509901, -11.1521225, 13.0524044
5: -3.8673596, 3.4099646, -7.3376122, 6.4407911, -10.3081512, 10.7475767
6: -4.0397010, 3.4505873, -7.8093801, 6.6083603, -10.6480618, 11.2599678
7: -4.6541395, 3.4808147, -8.9933872, 6.5240011, -11.1781406, 12.4742022
8: -5.3186646, 3.2162716, -10.3325996, 5.9963574, -11.3150215, 13.5488710
9: -3.7986977, 4.3246951, -7.1123838, 8.2964840, -12.0951815, 11.4370785

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315498, upper bound: 7.6320360
time: 4.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330834, upper bound: 7.6326000
time: 2.67 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -7.2852178, 5.5220227, -0.7259468, 0.7153356, -8.0005531, 6.2479696
1: -5.7576551, 5.0055475, -0.6719762, 0.6762322, -6.4338875, 5.6775236
2: -9.6020679, 3.6971219, -0.2580050, 1.1420835, -10.7441511, 3.9551268
3: -8.3731556, 4.0032072, -0.5749285, 0.6805566, -9.0537119, 4.5781355
4: -8.6154976, 5.3355112, -0.8254209, 0.7441289, -9.3596268, 6.1609321
5: -6.4577227, 5.6718278, -0.6799997, 0.7856332, -7.2433558, 6.3518276
6: -6.8556104, 5.8060336, -0.6583025, 0.7306366, -7.5862470, 6.4643364
7: -7.8934011, 5.7522421, -0.7384548, 0.7942975, -8.6876984, 6.4906969
8: -9.0573540, 5.2909641, -0.8057206, 0.8257499, -9.8831043, 6.0966849
9: -6.2698946, 7.2849755, -0.7390342, 0.8336050, -7.1034994, 8.0240097

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309564, upper bound: 7.5922043
time: 6.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311825, upper bound: 7.5922194
time: 4.81 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.3132782, 6.2759089, -3.8228638, 2.9682674, -11.2815456, 10.0987730
1: -6.5844011, 5.6741643, -2.9605062, 2.7579441, -9.3423452, 8.6346703
2: -10.9843225, 4.1436543, -4.8636980, 2.1269107, -13.1112328, 9.0073528
3: -9.5884886, 4.5238266, -4.2921581, 2.2437789, -11.8322678, 8.8159847
4: -9.8131104, 6.0509901, -4.5939498, 2.9203625, -12.7334728, 10.6449394
5: -7.3376122, 6.4407911, -3.4740767, 3.0589762, -10.3965883, 9.9148674
6: -7.8093801, 6.6083603, -3.6341593, 3.1020687, -10.9114485, 10.2425194
7: -8.9933872, 6.5240011, -4.1732802, 3.1400962, -12.1334839, 10.6972809
8: -10.3325996, 5.9963574, -4.7621560, 2.9090209, -13.2416210, 10.7585135
9: -7.1123838, 8.2964840, -3.4344893, 3.9014049, -11.0137882, 11.7309732

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305064, upper bound: 7.6319716
time: 4.04 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305064, upper bound: 7.6325640
time: 3.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 9.13 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 9.13
Output dim: 2, lower bound: -7.6315498, upper bound: 7.6320359
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 9.13
Output dim: 2, lower bound: -7.6330834, upper bound: 7.6325995
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 9.13
Output dim: 2, lower bound: -7.6315498, upper bound: 7.6320360
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 9.13
Output dim: 2, lower bound: -7.6330834, upper bound: 7.6326000
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 9.13
Output dim: 2, lower bound: -7.6309564, upper bound: 7.5922043
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 9.13
Output dim: 2, lower bound: -7.6311825, upper bound: 7.5922194
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 9.13
Output dim: 2, lower bound: -7.6305064, upper bound: 7.6319716
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 9.13
Output dim: 2, lower bound: -7.6305064, upper bound: 7.6325640

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4509927, 0.5204877, -3.2438252, 2.5486033, -2.9995961, 3.7643130
1: -0.4899695, 0.5177615, -2.5038247, 2.3772981, -2.8672676, 3.0215862
2: 0.1305590, 1.0913000, -4.0814538, 1.9088887, -1.7783297, 5.1727538
3: -0.3232668, 0.5498731, -3.5954280, 1.9594202, -2.2826869, 4.1453009
4: -0.5242356, 0.5465826, -3.8999550, 2.5270867, -3.0513225, 4.4465375
5: -0.4814742, 0.5579521, -2.9747262, 2.6377053, -3.1191795, 3.5326784
6: -0.4239199, 0.5575832, -3.0937598, 2.6506190, -3.0745389, 3.6513429
7: -0.4820627, 0.6032573, -3.5537663, 2.7052164, -3.1872790, 4.1570234
8: -0.5474737, 0.6434382, -4.0455661, 2.5213039, -3.0687776, 4.6890044
9: -0.5339578, 0.5699098, -2.9547215, 3.3173528, -3.8513105, 3.5246313

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302874, upper bound: 7.6319114
time: 3.14 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303034, upper bound: 7.6320658
time: 1.86 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -4.2686481, 3.3034801, -6.4329743, 6.7303185
1: -2.4251304, 2.2983003, -3.3155420, 3.0498059, -5.4749365, 5.6138420
2: -3.9117460, 1.8518873, -5.4949217, 2.3689001, -6.2806463, 7.3468089
3: -3.4586265, 1.9007586, -4.8110600, 2.4704089, -5.9290352, 6.7118187
4: -3.7661660, 2.4433300, -5.1011324, 3.2392945, -7.0054607, 7.5444622
5: -2.8676689, 2.5432758, -3.8673596, 3.4099646, -6.2776337, 6.4106355
6: -2.9870868, 2.5610185, -4.0397010, 3.4505873, -6.4376740, 6.6007195
7: -3.4249156, 2.6162057, -4.6541395, 3.4808147, -6.9057302, 7.2703452
8: -3.9026318, 2.4418151, -5.3186646, 3.2162716, -7.1189032, 7.7604799
9: -2.8588891, 3.2085273, -3.7986977, 4.3246951, -7.1835842, 7.0072250

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6327430, upper bound: 7.6317041
time: 3.13 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6327430, upper bound: 7.6332056
time: 3.89 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4509927, 0.5204877, -7.2852178, 5.5220227, -5.9730153, 7.8057055
1: -0.4899695, 0.5177615, -5.7576551, 5.0055475, -5.4955168, 6.2754169
2: 0.1305590, 1.0913000, -9.6020679, 3.6971219, -3.5665629, 10.6933680
3: -0.3232668, 0.5498731, -8.3731556, 4.0032072, -4.3264742, 8.9230289
4: -0.5242356, 0.5465826, -8.6154976, 5.3355112, -5.8597469, 9.1620798
5: -0.4814742, 0.5579521, -6.4577227, 5.6718278, -6.1533022, 7.0156746
6: -0.4239199, 0.5575832, -6.8556104, 5.8060336, -6.2299538, 7.4131937
7: -0.4820627, 0.6032573, -7.8934011, 5.7522421, -6.2343049, 8.4966583
8: -0.5474737, 0.6434382, -9.0573540, 5.2909641, -5.8384380, 9.7007923
9: -0.5339578, 0.5699098, -6.2698946, 7.2849755, -7.8189335, 6.8398042

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6301062, upper bound: 7.6310329
time: 5.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6301212, upper bound: 7.6312667
time: 3.00 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -8.3132782, 6.2759089, -9.4054031, 10.7749481
1: -2.4251304, 2.2983003, -6.5844011, 5.6741643, -8.0992947, 8.8827019
2: -3.9117460, 1.8518873, -10.9843225, 4.1436543, -8.0554008, 12.8362103
3: -3.4586265, 1.9007586, -9.5884886, 4.5238266, -7.9824533, 11.4892473
4: -3.7661660, 2.4433300, -9.8131104, 6.0509901, -9.8171558, 12.2564402
5: -2.8676689, 2.5432758, -7.3376122, 6.4407911, -9.3084602, 9.8808880
6: -2.9870868, 2.5610185, -7.8093801, 6.6083603, -9.5954475, 10.3703985
7: -3.4249156, 2.6162057, -8.9933872, 6.5240011, -9.9489164, 11.6095924
8: -3.9026318, 2.4418151, -10.3325996, 5.9963574, -9.8989887, 12.7744150
9: -2.8588891, 3.2085273, -7.1123838, 8.2964840, -11.1553726, 10.3209114

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324748, upper bound: 7.6305415
time: 3.16 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324748, upper bound: 7.6326001
time: 3.12 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.8171968, 3.7123764, -0.6433449, 0.6610191, -5.4782162, 4.3557215
1: -3.7711031, 3.4006510, -0.6181645, 0.6322801, -4.4033833, 4.0188155
2: -6.2611356, 2.6112471, -0.1390246, 1.1276726, -7.3888083, 2.7502716
3: -5.4633665, 2.7536240, -0.4917378, 0.6454276, -6.1087942, 3.2453618
4: -5.7398520, 3.6268482, -0.7307219, 0.6902690, -6.4301209, 4.3575702
5: -4.3415751, 3.8120444, -0.6167204, 0.7288454, -5.0704203, 4.4287648
6: -4.5638123, 3.8829851, -0.5885282, 0.6786891, -5.2425013, 4.4715133
7: -5.2494936, 3.8954582, -0.6604567, 0.7445323, -5.9940257, 4.5559149
8: -6.0030179, 3.6076202, -0.7293729, 0.7784748, -6.7814927, 4.3369932
9: -4.2465315, 4.8631821, -0.6789867, 0.7580122, -5.0045438, 5.5421686

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309567, upper bound: 7.5922043
time: 2.49 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309567, upper bound: 7.5922042
time: 4.39 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.8320446, 4.4546814, -0.5655657, 0.6116728, -6.4437175, 5.0202470
1: -4.5844078, 4.0617747, -0.5749698, 0.5930765, -5.1774845, 4.6367445
2: -7.6369982, 3.0454421, -0.0352542, 1.1142453, -8.7512436, 3.0806963
3: -6.6629758, 3.2644382, -0.4186786, 0.6149403, -7.2779160, 3.6831167
4: -6.9285240, 4.3250151, -0.6493949, 0.6377388, -7.5662627, 4.9744101
5: -5.2116289, 4.5786300, -0.5665693, 0.6737299, -5.8853588, 5.1451993
6: -5.5055594, 4.6724954, -0.5275273, 0.6334106, -6.1389699, 5.2000227
7: -6.3358030, 4.6573563, -0.5953674, 0.7003088, -7.0361118, 5.2527237
8: -7.2561264, 4.2907152, -0.6651223, 0.7330123, -7.9891386, 4.9558372
9: -5.0778112, 5.8646336, -0.6249480, 0.6886615, -5.7664728, 6.4895816

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311825, upper bound: 7.5922201
time: 2.79 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311825, upper bound: 7.5922200
time: 6.85 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -3.4289303, 2.6780100, -3.8228638, 2.9682674, -6.3971977, 6.5008736
1: -2.6615841, 2.4935343, -2.9605062, 2.7579441, -5.4195280, 5.4540405
2: -4.3062325, 1.9688954, -4.8636980, 2.1269107, -6.4331431, 6.8325934
3: -3.8253829, 2.0555108, -4.2921581, 2.2437789, -6.0691619, 6.3476686
4: -4.1241121, 2.6504107, -4.5939498, 2.9203625, -7.0444746, 7.2443604
5: -3.1364799, 2.7556326, -3.4740767, 3.0589762, -6.1954560, 6.2297096
6: -3.2842293, 2.7937129, -3.6341593, 3.1020687, -6.3862982, 6.4278722
7: -3.7546012, 2.8442411, -4.1732802, 3.1400962, -6.8946972, 7.0175214
8: -4.2720189, 2.6389854, -4.7621560, 2.9090209, -7.1810398, 7.4011412
9: -3.1114426, 3.4878597, -3.4344893, 3.9014049, -7.0128474, 6.9223490

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304360, upper bound: 7.6319718
time: 2.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304361, upper bound: 7.6319716
time: 2.84 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -7.0665984, 5.3569503, -3.8228638, 2.9682674, -10.0348663, 9.1798143
1: -5.5791984, 4.8653116, -2.9605062, 2.7579441, -8.3371429, 7.8258181
2: -9.2917576, 3.5834966, -4.8636980, 2.1269107, -11.4186687, 8.4471951
3: -8.1203499, 3.8885853, -4.2921581, 2.2437789, -10.3641291, 8.1807432
4: -8.3703003, 5.1766109, -4.5939498, 2.9203625, -11.2906628, 9.7705612
5: -6.2708530, 5.5036459, -3.4740767, 3.0589762, -9.3298292, 8.9777222
6: -6.6540017, 5.6346269, -3.6341593, 3.1020687, -9.7560701, 9.2687864
7: -7.6584935, 5.5861216, -4.1732802, 3.1400962, -10.7985897, 9.7594013
8: -8.7819738, 5.1313066, -4.7621560, 2.9090209, -11.6909943, 9.8934631
9: -6.0929327, 7.0746951, -3.4344893, 3.9014049, -9.9943371, 10.5091839

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304360, upper bound: 7.6325641
time: 4.79 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304361, upper bound: 7.6304359
time: 1.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 8.21 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6302874, upper bound: 7.6319114
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6303034, upper bound: 7.6320658
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6327430, upper bound: 7.6317041
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6327430, upper bound: 7.6332056
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6301062, upper bound: 7.6310329
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6301212, upper bound: 7.6312667
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6324748, upper bound: 7.6305415
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6324748, upper bound: 7.6326001
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6309567, upper bound: 7.5922043
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6309567, upper bound: 7.5922042
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6311825, upper bound: 7.5922201
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6311825, upper bound: 7.5922200
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6304360, upper bound: 7.6319718
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6304361, upper bound: 7.6319716
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6304360, upper bound: 7.6325641
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 8.21
Output dim: 2, lower bound: -7.6304361, upper bound: 7.6304359

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4271084, 0.4942832, -1.1053064, 0.9632372, -1.3903457, 1.5995896
1: -0.4654395, 0.4951848, -0.9407901, 0.9000777, -1.3655173, 1.4359748
2: 0.1681221, 1.0850841, -0.8257809, 1.2177006, -1.0495784, 1.9108649
3: -0.3050309, 0.5317209, -1.0239100, 0.8545653, -1.1595962, 1.5556309
4: -0.5014338, 0.5190079, -1.2820165, 0.9964127, -1.4978465, 1.8010244
5: -0.4564503, 0.5308417, -1.0126833, 1.0311239, -1.4875742, 1.5435250
6: -0.4015422, 0.5350480, -0.9999775, 0.9875838, -1.3891259, 1.5350256
7: -0.4590012, 0.5750179, -1.1591082, 1.0613741, -1.5203753, 1.7341261
8: -0.5243088, 0.6157832, -1.2228985, 1.0707030, -1.5950118, 1.8386817
9: -0.5123287, 0.5364254, -1.0614384, 1.1637571, -1.6760857, 1.5978638

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5711355, upper bound: 7.6310386
time: 3.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5862020, upper bound: 7.6310677
time: 2.18 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4032527, 0.4682322, -1.9570563, 1.5919262, -1.9951789, 2.4252884
1: -0.4429742, 0.4721122, -1.5842605, 1.4806807, -1.9236549, 2.0563726
2: 0.2054854, 1.0785766, -2.1600404, 1.4551675, -1.2496821, 3.2386169
3: -0.2864389, 0.5135676, -2.0553217, 1.2985270, -1.5849659, 2.5688891
4: -0.4788212, 0.4909663, -2.3081710, 1.6043817, -2.0832028, 2.7991374
5: -0.4336670, 0.5031477, -1.8053162, 1.6323397, -2.0660067, 2.3084641
6: -0.3800945, 0.5120108, -1.8393650, 1.6339417, -2.0140362, 2.3513758
7: -0.4360116, 0.5466110, -2.1152077, 1.7178895, -2.1539011, 2.6618185
8: -0.5004768, 0.5907781, -2.3821626, 1.6418972, -2.1423740, 2.9729407
9: -0.4901814, 0.5045211, -1.8232844, 2.0292912, -2.5194726, 2.3278055

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5711556, upper bound: 7.6313769
time: 2.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5862259, upper bound: 7.6314080
time: 3.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -0.4509927, 0.5204877, -3.6499820, 2.9126630
1: -2.4251304, 2.2983003, -0.4899695, 0.5177615, -2.9428918, 2.7882698
2: -3.9117460, 1.8518873, 0.1305590, 1.0913000, -5.0030460, 1.7213284
3: -3.4586265, 1.9007586, -0.3232668, 0.5498731, -4.0084996, 2.2240255
4: -3.7661660, 2.4433300, -0.5242356, 0.5465826, -4.3127484, 2.9675655
5: -2.8676689, 2.5432758, -0.4814742, 0.5579521, -3.4256210, 3.0247500
6: -2.9870868, 2.5610185, -0.4239199, 0.5575832, -3.5446701, 2.9849384
7: -3.4249156, 2.6162057, -0.4820627, 0.6032573, -4.0281730, 3.0982683
8: -3.9026318, 2.4418151, -0.5474737, 0.6434382, -4.5460701, 2.9892888
9: -2.8588891, 3.2085273, -0.5339578, 0.5699098, -3.4287989, 3.7424850

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319111, upper bound: 7.6302874
time: 6.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320658, upper bound: 7.6303034
time: 4.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -3.1294942, 2.4616704, -5.5911646, 5.5911646
1: -2.4251304, 2.2983003, -2.4251304, 2.2983003, -4.7234306, 4.7234306
2: -3.9117460, 1.8518873, -3.9117460, 1.8518873, -5.7636333, 5.7636333
3: -3.4586265, 1.9007586, -3.4586265, 1.9007586, -5.3593850, 5.3593850
4: -3.7661660, 2.4433300, -3.7661660, 2.4433300, -6.2094960, 6.2094960
5: -2.8676689, 2.5432758, -2.8676689, 2.5432758, -5.4109449, 5.4109449
6: -2.9870868, 2.5610185, -2.9870868, 2.5610185, -5.5481052, 5.5481052
7: -3.4249156, 2.6162057, -3.4249156, 2.6162057, -6.0411215, 6.0411215
8: -3.9026318, 2.4418151, -3.9026318, 2.4418151, -6.3444471, 6.3444471
9: -2.8588891, 3.2085273, -2.8588891, 3.2085273, -6.0674162, 6.0674162

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319111, upper bound: 7.6327096
time: 3.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320655, upper bound: 7.6327805
time: 2.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4271084, 0.4942832, -4.8171968, 3.7123764, -4.1394849, 5.3114800
1: -0.4654395, 0.4951848, -3.7711031, 3.4006510, -3.8660905, 4.2662878
2: 0.1681221, 1.0850841, -6.2611356, 2.6112471, -2.4431250, 7.3462195
3: -0.3050309, 0.5317209, -5.4633665, 2.7536240, -3.0586548, 5.9950876
4: -0.5014338, 0.5190079, -5.7398520, 3.6268482, -4.1282821, 6.2588596
5: -0.4564503, 0.5308417, -4.3415751, 3.8120444, -4.2684946, 4.8724170
6: -0.4015422, 0.5350480, -4.5638123, 3.8829851, -4.2845273, 5.0988603
7: -0.4590012, 0.5750179, -5.2494936, 3.8954582, -4.3544593, 5.8245115
8: -0.5243088, 0.6157832, -6.0030179, 3.6076202, -4.1319289, 6.6188011
9: -0.5123287, 0.5364254, -4.2465315, 4.8631821, -5.3755107, 4.7829571

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5709278, upper bound: 7.6299130
time: 2.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5859831, upper bound: 7.6299526
time: 3.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4032527, 0.4682322, -5.8320446, 4.4546814, -4.8579340, 6.3002768
1: -0.4429742, 0.4721122, -4.5844078, 4.0617747, -4.5047488, 5.0565200
2: 0.2054854, 1.0785766, -7.6369982, 3.0454421, -2.8399568, 8.7155743
3: -0.2864389, 0.5135676, -6.6629758, 3.2644382, -3.5508771, 7.1765432
4: -0.4788212, 0.4909663, -6.9285240, 4.3250151, -4.8038363, 7.4194903
5: -0.4336670, 0.5031477, -5.2116289, 4.5786300, -5.0122972, 5.7147765
6: -0.3800945, 0.5120108, -5.5055594, 4.6724954, -5.0525899, 6.0175705
7: -0.4360116, 0.5466110, -6.3358030, 4.6573563, -5.0933681, 6.8824139
8: -0.5004768, 0.5907781, -7.2561264, 4.2907152, -4.7911921, 7.8469043
9: -0.4901814, 0.5045211, -5.0778112, 5.8646336, -6.3548150, 5.5823321

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5709383, upper bound: 7.6303880
time: 2.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5859980, upper bound: 7.6304315
time: 4.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -3.4289303, 2.6780100, -5.8075042, 5.8906007
1: -2.4251304, 2.2983003, -2.6615841, 2.4935343, -4.9186649, 4.9598846
2: -3.9117460, 1.8518873, -4.3062325, 1.9688954, -5.8806415, 6.1581197
3: -3.4586265, 1.9007586, -3.8253829, 2.0555108, -5.5141373, 5.7261415
4: -3.7661660, 2.4433300, -4.1241121, 2.6504107, -6.4165764, 6.5674419
5: -2.8676689, 2.5432758, -3.1364799, 2.7556326, -5.6233015, 5.6797557
6: -2.9870868, 2.5610185, -3.2842293, 2.7937129, -5.7807999, 5.8452477
7: -3.4249156, 2.6162057, -3.7546012, 2.8442411, -6.2691565, 6.3708067
8: -3.9026318, 2.4418151, -4.2720189, 2.6389854, -6.5416174, 6.7138338
9: -2.8588891, 3.2085273, -3.1114426, 3.4878597, -6.3467488, 6.3199701

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315607, upper bound: 7.5922434
time: 3.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317560, upper bound: 7.5922579
time: 3.26 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -3.1294942, 2.4616704, -7.0665984, 5.3569503, -8.4864445, 9.5282688
1: -2.4251304, 2.2983003, -5.5791984, 4.8653116, -7.2904420, 7.8774986
2: -3.9117460, 1.8518873, -9.2917576, 3.5834966, -7.4952426, 11.1436453
3: -3.4586265, 1.9007586, -8.1203499, 3.8885853, -7.3472118, 10.0211086
4: -3.7661660, 2.4433300, -8.3703003, 5.1766109, -8.9427767, 10.8136301
5: -2.8676689, 2.5432758, -6.2708530, 5.5036459, -8.3713150, 8.8141289
6: -2.9870868, 2.5610185, -6.6540017, 5.6346269, -8.6217136, 9.2150202
7: -3.4249156, 2.6162057, -7.6584935, 5.5861216, -9.0110369, 10.2746992
8: -3.9026318, 2.4418151, -8.7819738, 5.1313066, -9.0339384, 11.2237892
9: -2.8588891, 3.2085273, -6.0929327, 7.0746951, -9.9335842, 9.3014603

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315607, upper bound: 7.6320343
time: 4.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317560, upper bound: 7.5922584
time: 3.54 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.8171968, 3.7123764, -0.4271084, 0.4942832, -5.3114800, 4.1394849
1: -3.7711031, 3.4006510, -0.4654395, 0.4951848, -4.2662878, 3.8660905
2: -6.2611356, 2.6112471, 0.1681221, 1.0850841, -7.3462195, 2.4431250
3: -5.4633665, 2.7536240, -0.3050309, 0.5317209, -5.9950876, 3.0586548
4: -5.7398520, 3.6268482, -0.5014338, 0.5190079, -6.2588596, 4.1282821
5: -4.3415751, 3.8120444, -0.4564503, 0.5308417, -4.8724170, 4.2684946
6: -4.5638123, 3.8829851, -0.4015422, 0.5350480, -5.0988603, 4.2845273
7: -5.2494936, 3.8954582, -0.4590012, 0.5750179, -5.8245115, 4.3544593
8: -6.0030179, 3.6076202, -0.5243088, 0.6157832, -6.6188011, 4.1319289
9: -4.2465315, 4.8631821, -0.5123287, 0.5364254, -4.7829571, 5.3755107

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286609, upper bound: 7.5921309
time: 2.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286609, upper bound: 7.5922040
time: 2.18 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.8171968, 3.7123764, -2.4689381, 1.9027667, -6.7199636, 6.1813145
1: -3.7711031, 3.4006510, -1.7500299, 1.6195354, -5.3906384, 5.1506810
2: -6.2611356, 2.6112471, -2.7192602, 1.4355766, -7.6967120, 5.3305073
3: -5.4633665, 2.7536240, -2.2868671, 1.4499850, -6.9133515, 5.0404911
4: -5.7398520, 3.6268482, -2.7850626, 1.8980682, -7.6379204, 6.4119110
5: -4.3415751, 3.8120444, -1.9056885, 2.0160394, -6.3576145, 5.7177329
6: -4.5638123, 3.8829851, -2.0554390, 1.8403050, -6.4041171, 5.9384241
7: -5.2494936, 3.8954582, -2.2998633, 1.7988111, -7.0483046, 6.1953216
8: -6.0030179, 3.6076202, -2.2995026, 1.8573328, -7.8603506, 5.9071226
9: -4.2465315, 4.8631821, -1.9954029, 2.4681754, -6.7147069, 6.8585849

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286609, upper bound: 7.5921308
time: 2.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286609, upper bound: 7.5922038
time: 2.12 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.8320446, 4.4546814, -0.4032527, 0.4682322, -6.3002768, 4.8579340
1: -4.5844078, 4.0617747, -0.4429742, 0.4721122, -5.0565200, 4.5047488
2: -7.6369982, 3.0454421, 0.2054854, 1.0785766, -8.7155743, 2.8399568
3: -6.6629758, 3.2644382, -0.2864389, 0.5135676, -7.1765432, 3.5508771
4: -6.9285240, 4.3250151, -0.4788212, 0.4909663, -7.4194903, 4.8038363
5: -5.2116289, 4.5786300, -0.4336670, 0.5031477, -5.7147765, 5.0122972
6: -5.5055594, 4.6724954, -0.3800945, 0.5120108, -6.0175705, 5.0525899
7: -6.3358030, 4.6573563, -0.4360116, 0.5466110, -6.8824139, 5.0933681
8: -7.2561264, 4.2907152, -0.5004768, 0.5907781, -7.8469043, 4.7911921
9: -5.0778112, 5.8646336, -0.4901814, 0.5045211, -5.5823321, 6.3548150

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5921082, upper bound: 7.5921087
time: 2.28 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5921082, upper bound: 7.5922201
time: 2.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.8320446, 4.4546814, -2.2509675, 1.5110464, -7.3430910, 6.7056489
1: -4.5844078, 4.0617747, -1.5152408, 1.4072399, -5.9916477, 5.5770154
2: -7.6369982, 3.0454421, -2.1035364, 1.4072264, -9.0442247, 5.1489782
3: -6.6629758, 3.2644382, -1.9666181, 1.1921334, -7.8551092, 5.2310562
4: -6.9285240, 4.3250151, -2.2221379, 1.7674819, -8.6960058, 6.5471530
5: -5.2116289, 4.5786300, -1.5763423, 1.8232632, -7.0348921, 6.1549721
6: -5.5055594, 4.6724954, -1.8593901, 1.4595182, -6.9650774, 6.5318856
7: -6.3358030, 4.6573563, -1.9520111, 1.6470019, -7.9828048, 6.6093674
8: -7.2561264, 4.2907152, -2.0160816, 1.6421969, -8.8983231, 6.3067970
9: -5.0778112, 5.8646336, -1.7353462, 2.0069900, -7.0848012, 7.5999799

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5921082, upper bound: 7.5921085
time: 2.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5921082, upper bound: 7.5922197
time: 2.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.4289303, 2.6780100, -3.1294942, 2.4616704, -5.8906007, 5.8075042
1: -2.6615841, 2.4935343, -2.4251304, 2.2983003, -4.9598846, 4.9186649
2: -4.3062325, 1.9688954, -3.9117460, 1.8518873, -6.1581197, 5.8806415
3: -3.8253829, 2.0555108, -3.4586265, 1.9007586, -5.7261415, 5.5141373
4: -4.1241121, 2.6504107, -3.7661660, 2.4433300, -6.5674419, 6.4165764
5: -3.1364799, 2.7556326, -2.8676689, 2.5432758, -5.6797557, 5.6233015
6: -3.2842293, 2.7937129, -2.9870868, 2.5610185, -5.8452477, 5.7807999
7: -3.7546012, 2.8442411, -3.4249156, 2.6162057, -6.3708067, 6.2691565
8: -4.2720189, 2.6389854, -3.9026318, 2.4418151, -6.7138338, 6.5416174
9: -3.1114426, 3.4878597, -2.8588891, 3.2085273, -6.3199701, 6.3467488

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6288477, upper bound: 7.6312674
time: 2.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5922201, upper bound: 7.6311824
time: 2.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.4289303, 2.6780100, -6.6670465, 5.0439548, -8.4728851, 9.3450565
1: -2.6615841, 2.4935343, -5.2122169, 4.6210995, -7.2826834, 7.7057514
2: -4.3062325, 1.9688954, -8.7217026, 3.3665295, -7.6727619, 10.6905975
3: -3.8253829, 2.0555108, -7.6515274, 3.6416976, -7.4670806, 9.7070379
4: -4.1241121, 2.6504107, -7.9333062, 4.8617139, -8.9858265, 10.5837173
5: -3.1364799, 2.7556326, -5.9372015, 5.1964560, -8.3329353, 8.6928339
6: -3.2842293, 2.7937129, -6.2557740, 5.3262753, -8.6105042, 9.0494871
7: -3.7546012, 2.8442411, -7.2187166, 5.2745743, -9.0291758, 10.0629578
8: -4.2720189, 2.6389854, -8.2725658, 4.8120418, -9.0840607, 10.9115515
9: -3.1114426, 3.4878597, -5.7735968, 6.6979618, -9.8094044, 9.2614565

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6288477, upper bound: 7.6312674
time: 2.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5922201, upper bound: 7.6311821
time: 3.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -7.0665984, 5.3569503, -3.1294942, 2.4616704, -9.5282688, 8.4864445
1: -5.5791984, 4.8653116, -2.4251304, 2.2983003, -7.8774986, 7.2904420
2: -9.2917576, 3.5834966, -3.9117460, 1.8518873, -11.1436453, 7.4952426
3: -8.1203499, 3.8885853, -3.4586265, 1.9007586, -10.0211086, 7.3472118
4: -8.3703003, 5.1766109, -3.7661660, 2.4433300, -10.8136301, 8.9427767
5: -6.2708530, 5.5036459, -2.8676689, 2.5432758, -8.8141289, 8.3713150
6: -6.6540017, 5.6346269, -2.9870868, 2.5610185, -9.2150202, 8.6217136
7: -7.6584935, 5.5861216, -3.4249156, 2.6162057, -10.2746992, 9.0110369
8: -8.7819738, 5.1313066, -3.9026318, 2.4418151, -11.2237892, 9.0339384
9: -6.0929327, 7.0746951, -2.8588891, 3.2085273, -9.3014603, 9.9335842

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318068, upper bound: 7.6320035
time: 2.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320720, upper bound: 7.6320716
time: 2.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -7.0665984, 5.3569503, -6.6670465, 5.0439548, -12.1105537, 12.0239964
1: -5.5791984, 4.8653116, -5.2122169, 4.6210995, -10.2002983, 10.0775280
2: -9.2917576, 3.5834966, -8.7217026, 3.3665295, -12.6582870, 12.3051987
3: -8.1203499, 3.8885853, -7.6515274, 3.6416976, -11.7620478, 11.5401125
4: -8.3703003, 5.1766109, -7.9333062, 4.8617139, -13.2320137, 13.1099167
5: -6.2708530, 5.5036459, -5.9372015, 5.1964560, -11.4673090, 11.4408474
6: -6.6540017, 5.6346269, -6.2557740, 5.3262753, -11.9802771, 11.8904009
7: -7.6584935, 5.5861216, -7.2187166, 5.2745743, -12.9330673, 12.8048382
8: -8.7819738, 5.1313066, -8.2725658, 4.8120418, -13.5940151, 13.4038725
9: -6.0929327, 7.0746951, -5.7735968, 6.6979618, -12.7908945, 12.8482914

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318068, upper bound: 7.6320036
time: 3.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320720, upper bound: 7.6320721
time: 2.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.03 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5711355, upper bound: 7.6310386
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5862020, upper bound: 7.6310677
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5711556, upper bound: 7.6313769
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5862259, upper bound: 7.6314080
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6319111, upper bound: 7.6302874
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6320658, upper bound: 7.6303034
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6319111, upper bound: 7.6327096
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6320655, upper bound: 7.6327805
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5709278, upper bound: 7.6299130
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5859831, upper bound: 7.6299526
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5709383, upper bound: 7.6303880
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5859980, upper bound: 7.6304315
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6315607, upper bound: 7.5922434
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6317560, upper bound: 7.5922579
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6315607, upper bound: 7.6320343
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6317560, upper bound: 7.5922584
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6286609, upper bound: 7.5921309
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6286609, upper bound: 7.5922040
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6286609, upper bound: 7.5921308
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6286609, upper bound: 7.5922038
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5921082, upper bound: 7.5921087
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5921082, upper bound: 7.5922201
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5921082, upper bound: 7.5921085
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5921082, upper bound: 7.5922197
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6288477, upper bound: 7.6312674
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5922201, upper bound: 7.6311824
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6288477, upper bound: 7.6312674
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.5922201, upper bound: 7.6311821
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6318068, upper bound: 7.6320035
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6320720, upper bound: 7.6320716
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6318068, upper bound: 7.6320036
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.03
Output dim: 2, lower bound: -7.6320720, upper bound: 7.6320721

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1874970, 0.2036436, -1.0129484, 0.9015839, -1.0890809, 1.2165920
1: -0.2372475, 0.2500956, -0.8736311, 0.8439565, -1.0812041, 1.1237267
2: 0.5474359, 1.0401566, -0.6866575, 1.1976128, -0.6501769, 1.7268142
3: -0.1039863, 0.3226792, -0.9128640, 0.8107654, -0.9147516, 1.2355433
4: -0.2557076, 0.2460805, -1.1710162, 0.9327608, -1.1884685, 1.4170966
5: -0.2304034, 0.2558567, -0.9303185, 0.9706960, -1.2010994, 1.1861752
6: -0.1839036, 0.2734807, -0.9137970, 0.9242778, -1.1081814, 1.1872778
7: -0.2174428, 0.2828646, -1.0554674, 0.9937288, -1.2111716, 1.3383319
8: -0.2556123, 0.3509589, -1.1143694, 1.0101497, -1.2657621, 1.4653283
9: -0.2636027, 0.2364774, -0.9832344, 1.0810742, -1.3446770, 1.2197118

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5710659, upper bound: 7.6154079
time: 2.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5710659, upper bound: 7.6310386
time: 2.46 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2237872, 0.2521938, -0.9811528, 0.8806326, -1.1044198, 1.2333467
1: -0.2728415, 0.2928494, -0.8507293, 0.8249878, -1.0978292, 1.1435788
2: 0.4923054, 1.0409260, -0.6388487, 1.1914215, -0.6991161, 1.6797746
3: -0.1387696, 0.3630214, -0.8746096, 0.7960746, -0.9348443, 1.2376310
4: -0.2952042, 0.2847379, -1.1329560, 0.9112531, -1.2064573, 1.4176939
5: -0.2674049, 0.2962787, -0.9021387, 0.9501402, -1.2175450, 1.1984173
6: -0.2165657, 0.3221747, -0.8848271, 0.9025967, -1.1191624, 1.2070018
7: -0.2571482, 0.3244728, -1.0198803, 0.9702842, -1.2274325, 1.3443531
8: -0.3065949, 0.3941213, -1.0796453, 0.9892951, -1.2958900, 1.4737666
9: -0.3081053, 0.2794375, -0.9562414, 1.0532637, -1.3613689, 1.2356788

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5861358, upper bound: 7.6154307
time: 4.22 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5861358, upper bound: 7.6310677
time: 2.30 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1732886, 0.1877893, -1.8474820, 1.5109500, -1.6842386, 2.0352712
1: -0.2217900, 0.2339419, -1.5010078, 1.4039141, -1.6257041, 1.7349497
2: 0.5704663, 1.0400932, -1.9920003, 1.4225836, -0.8521172, 3.0320935
3: -0.0909384, 0.3060919, -1.9237034, 1.2400682, -1.3310065, 2.2297952
4: -0.2391381, 0.2319911, -2.1777763, 1.5247482, -1.7638863, 2.4097674
5: -0.2162016, 0.2379208, -1.7030430, 1.5530461, -1.7692478, 1.9409637
6: -0.1733780, 0.2516658, -1.7257665, 1.5488582, -1.7222362, 1.9774324
7: -0.2036288, 0.2651914, -1.9934957, 1.6322765, -1.8359053, 2.2586870
8: -0.2355525, 0.3324037, -2.2320991, 1.5684639, -1.8040164, 2.5645027
9: -0.2461864, 0.2191157, -1.7242310, 1.9175248, -2.1637113, 1.9433467

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5710520, upper bound: 7.5861002
time: 2.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5710520, upper bound: 7.6313772
time: 2.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2107242, 0.2335768, -1.8322561, 1.4997232, -1.7104474, 2.0658329
1: -0.2599831, 0.2771797, -1.4894035, 1.3931156, -1.6530988, 1.7665832
2: 0.5127159, 1.0408612, -1.9687450, 1.4179409, -0.9052249, 3.0096064
3: -0.1264535, 0.3483922, -1.9054583, 1.2318822, -1.3583357, 2.2538505
4: -0.2813092, 0.2695442, -2.1596334, 1.5135698, -1.7948791, 2.4291778
5: -0.2534396, 0.2820977, -1.6888444, 1.5420341, -1.7954738, 1.9709421
6: -0.2042072, 0.3046420, -1.7101015, 1.5369971, -1.7412043, 2.0147433
7: -0.2422649, 0.3096581, -1.9766171, 1.6202283, -1.8624932, 2.2862751
8: -0.2881194, 0.3783378, -2.2112288, 1.5581900, -1.8463094, 2.5895667
9: -0.2916138, 0.2639759, -1.7103970, 1.9020607, -2.1936746, 1.9743730

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5861250, upper bound: 7.5861253
time: 2.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5861250, upper bound: 7.6314081
time: 2.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.9512004, 0.8617375, -0.4271084, 0.4942832, -1.4454837, 1.2888459
1: -0.8289196, 0.8051167, -0.4654395, 0.4951848, -1.3241044, 1.2705562
2: -0.5958006, 1.1843369, 0.1681221, 1.0850841, -1.6808846, 1.0162147
3: -0.8389760, 0.7814506, -0.3050309, 0.5317209, -1.3706968, 1.0864815
4: -1.0978392, 0.8889894, -0.5014338, 0.5190079, -1.6168470, 1.3904233
5: -0.8760652, 0.9310993, -0.4564503, 0.5308417, -1.4069068, 1.3875496
6: -0.8582798, 0.8818473, -0.4015422, 0.5350480, -1.3933278, 1.2833896
7: -0.9862645, 0.9477892, -0.4590012, 0.5750179, -1.5612824, 1.4067905
8: -1.0474089, 0.9680716, -0.5243088, 0.6157832, -1.6631922, 1.4923804
9: -0.9297719, 1.0286045, -0.5123287, 0.5364254, -1.4661973, 1.5409331

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310386, upper bound: 7.5711357
time: 2.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310677, upper bound: 7.5862014
time: 5.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.9039898, 1.5532383, -0.4032527, 0.4682322, -2.3722219, 1.9564910
1: -1.5435649, 1.4417903, -0.4429742, 0.4721122, -2.0156772, 1.8847646
2: -2.0798304, 1.4384233, 0.2054854, 1.0785766, -3.1584070, 1.2329378
3: -1.9923401, 1.2694449, -0.2864389, 0.5135676, -2.5059075, 1.5558839
4: -2.2457442, 1.5643042, -0.4788212, 0.4909663, -2.7367105, 2.0431254
5: -1.7552578, 1.5948195, -0.4336670, 0.5031477, -2.2584057, 2.0284865
6: -1.7851844, 1.5925593, -0.3800945, 0.5120108, -2.2971952, 1.9726539
7: -2.0569201, 1.6748843, -0.4360116, 0.5466110, -2.6035309, 2.1108959
8: -2.3099952, 1.6051259, -0.5004768, 0.5907781, -2.9007733, 2.1056027
9: -1.7752999, 1.9758629, -0.4901814, 0.5045211, -2.2798209, 2.4660442

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313772, upper bound: 7.5711554
time: 2.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314080, upper bound: 7.5862259
time: 2.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.9512004, 0.8617375, -2.9857378, 2.3552699, -3.3064704, 3.8474753
1: -0.8289196, 0.8051167, -2.3264577, 2.2012329, -3.0301526, 3.1315744
2: -0.5958006, 1.1843369, -3.7073407, 1.7960043, -2.3918049, 4.8916779
3: -0.8389760, 0.7814506, -3.2828996, 1.8291472, -2.6681232, 4.0643501
4: -1.0978392, 0.8889894, -3.5917492, 2.3429205, -3.4407597, 4.4807386
5: -0.8760652, 0.9310993, -2.7384005, 2.4314122, -3.3074775, 3.6694999
6: -0.8582798, 0.8818473, -2.8522229, 2.4484720, -3.3067517, 3.7340703
7: -0.9862645, 0.9477892, -3.2656670, 2.5073543, -3.4936187, 4.2134562
8: -1.0474089, 0.9680716, -3.7229731, 2.3437855, -3.3911943, 4.6910448
9: -0.9297719, 1.0286045, -2.7375462, 3.0652418, -3.9950137, 3.7661507

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325153, upper bound: 7.6325153
time: 3.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325153, upper bound: 7.6327092
time: 2.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.9039898, 1.5532383, -2.8944330, 2.2878461, -4.1918359, 4.4476714
1: -1.5435649, 1.4417903, -2.2637115, 2.1394439, -3.6830087, 3.7055018
2: -2.0798304, 1.4384233, -3.5772915, 1.7604439, -3.8402743, 5.0157146
3: -1.9923401, 1.2694449, -3.1713338, 1.7837385, -3.7760787, 4.4407787
4: -2.2457442, 1.5643042, -3.4811118, 2.2790785, -4.5248227, 5.0454159
5: -1.7552578, 1.5948195, -2.6561947, 2.3602545, -4.1155124, 4.2510142
6: -1.7851844, 1.5925593, -2.7668812, 2.3767941, -4.1619787, 4.3594408
7: -2.0569201, 1.6748843, -3.1645284, 2.4380031, -4.4949231, 4.8394127
8: -2.3099952, 1.6051259, -3.6090894, 2.2812717, -4.5912666, 5.2142153
9: -1.7752999, 1.9758629, -2.6603346, 2.9745643, -4.7498641, 4.6361976

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6322263, upper bound: 7.6321460
time: 2.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6322434, upper bound: 7.6322433
time: 1.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1874970, 0.2036436, -4.6851788, 3.6150053, -3.8025024, 4.8888226
1: -0.2372475, 0.2500956, -3.6643434, 3.3145466, -3.5517941, 3.9144390
2: 0.5474359, 1.0401566, -6.0804362, 2.5542157, -2.0067797, 7.1205931
3: -0.1039863, 0.3226792, -5.3071618, 2.6862574, -2.7902436, 5.6298409
4: -0.2557076, 0.2460805, -5.5854502, 3.5346391, -3.7903466, 5.8315306
5: -0.2304034, 0.2558567, -4.2283993, 3.7132673, -3.9436707, 4.4842558
6: -0.1839036, 0.2734807, -4.4409480, 3.7798781, -3.9637816, 4.7144289
7: -0.2174428, 0.2828646, -5.1078305, 3.7955496, -4.0129924, 5.3906951
8: -0.2556123, 0.3509589, -5.8389091, 3.5169821, -3.7725945, 6.1898680
9: -0.2636027, 0.2364774, -4.1383605, 4.7326288, -4.9962316, 4.3748379

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5708046, upper bound: 7.4749845
time: 2.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5708046, upper bound: 7.6299129
time: 3.17 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2237872, 0.2521938, -4.6448174, 3.5849605, -3.8087478, 4.8970113
1: -0.2728415, 0.2928494, -3.6314445, 3.2881994, -3.5610409, 3.9242940
2: 0.4923054, 1.0409260, -6.0245237, 2.5356998, -2.0433943, 7.0654497
3: -0.1387696, 0.3630214, -5.2596359, 2.6653738, -2.8041434, 5.6226573
4: -0.2952042, 0.2847379, -5.5386343, 3.5060480, -3.8012521, 5.8233724
5: -0.2674049, 0.2962787, -4.1936588, 3.6829088, -3.9503136, 4.4899373
6: -0.2165657, 0.3221747, -4.4033389, 3.7482924, -3.9648581, 4.7255135
7: -0.2571482, 0.3244728, -5.0643539, 3.7647316, -4.0218797, 5.3888268
8: -0.3065949, 0.3941213, -5.7885056, 3.4886718, -3.7952666, 6.1826267
9: -0.3081053, 0.2794375, -4.1053467, 4.6929584, -5.0010638, 4.3847842

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5858665, upper bound: 7.4750108
time: 22.28 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5858665, upper bound: 7.6299526
time: 2.44 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1732886, 0.1877893, -5.6864538, 4.3475809, -4.5208693, 5.8742433
1: -0.2217900, 0.2339419, -4.4665437, 3.9668982, -4.1886883, 4.7004857
2: 0.5704663, 1.0400932, -7.4393797, 2.9828057, -2.4123394, 8.4794731
3: -0.0909384, 0.3060919, -6.4905424, 3.1902401, -3.2811785, 6.7966342
4: -0.2391381, 0.2319911, -6.7582741, 4.2235236, -4.4626617, 6.9902654
5: -0.2162016, 0.2379208, -5.0870919, 4.4698582, -4.6860600, 5.3250127
6: -0.1733780, 0.2516658, -5.3699183, 4.5589018, -4.7322798, 5.6215839
7: -0.2036288, 0.2651914, -6.1797690, 4.5473480, -4.7509770, 6.4449606
8: -0.2355525, 0.3324037, -7.0752330, 4.1909747, -4.4265270, 7.4076366
9: -0.2461864, 0.2191157, -4.9582987, 5.7208052, -5.9669914, 5.1774144

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5707546, upper bound: 7.4419952
time: 2.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5707546, upper bound: 7.6303880
time: 2.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2107242, 0.2335768, -5.6752644, 4.3390865, -4.5498109, 5.9088411
1: -0.2599831, 0.2771797, -4.4573212, 3.9596417, -4.2196250, 4.7345009
2: 0.5127159, 1.0408612, -7.4234152, 2.9768519, -2.4641361, 8.4642763
3: -0.1264535, 0.3483922, -6.4776216, 3.1843197, -3.3107734, 6.8260136
4: -0.2813092, 0.2695442, -6.7457471, 4.2153034, -4.4966125, 7.0152912
5: -0.2534396, 0.2820977, -5.0774541, 4.4612989, -4.7147384, 5.3595519
6: -0.2042072, 0.3046420, -5.3595433, 4.5501533, -4.7543607, 5.6641855
7: -0.2422649, 0.3096581, -6.1677051, 4.5387144, -4.7809792, 6.4773631
8: -0.2881194, 0.3783378, -7.0611758, 4.1826892, -4.4708085, 7.4395137
9: -0.2916138, 0.2639759, -4.9492359, 5.7100382, -6.0016518, 5.2132120

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5858211, upper bound: 7.4420170
time: 3.21 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5858211, upper bound: 7.6304315
time: 4.16 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.9512004, 0.8617375, -3.2977858, 2.5801084, -3.5313089, 4.1595230
1: -0.8289196, 0.8051167, -2.5549891, 2.4059935, -3.2349131, 3.3601058
2: -0.5958006, 1.1843369, -4.1240735, 1.9145367, -2.5103374, 5.3084106
3: -0.8389760, 0.7814506, -3.6671894, 1.9898530, -2.8288291, 4.4486399
4: -1.0978392, 0.8889894, -3.9668655, 2.5596905, -3.6575298, 4.8558550
5: -0.8760652, 0.9310993, -3.0198171, 2.6551247, -3.5311899, 3.9509163
6: -0.8582798, 0.8818473, -3.1605859, 2.6907148, -3.5489945, 4.0424333
7: -0.9862645, 0.9477892, -3.6113768, 2.7429376, -3.7292020, 4.5591660
8: -1.0474089, 0.9680716, -4.1097202, 2.5496402, -3.5970492, 5.0777917
9: -0.9297719, 1.0286045, -3.0017576, 3.3567724, -4.2865443, 4.0303621

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305958, upper bound: 7.4238961
time: 4.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306062, upper bound: 7.4421049
time: 2.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.9039898, 1.5532383, -3.1586375, 2.4771242, -4.3811140, 4.7118759
1: -1.5435649, 1.4417903, -2.4542098, 2.3121591, -3.8557239, 3.8960001
2: -2.0798304, 1.4384233, -3.9287047, 1.8576940, -3.9375243, 5.3671279
3: -1.9923401, 1.2694449, -3.4990597, 1.9196427, -3.9119828, 4.7685046
4: -2.2457442, 1.5643042, -3.7996647, 2.4625001, -4.7082443, 5.3639688
5: -1.7552578, 1.5948195, -2.8943865, 2.5473220, -4.3025799, 4.4892063
6: -1.7851844, 1.5925593, -3.0284867, 2.5825002, -4.3676844, 4.6210461
7: -2.0569201, 1.6748843, -3.4581544, 2.6363523, -4.6932726, 5.1330385
8: -2.3099952, 1.6051259, -3.9372120, 2.4552727, -4.7652678, 5.5423379
9: -1.7752999, 1.9758629, -2.8850131, 3.2196531, -4.9949532, 4.8608761

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309652, upper bound: 7.4239188
time: 5.25 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309941, upper bound: 7.4421272
time: 3.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.9512004, 0.8617375, -6.9159265, 5.2465849, -6.1977854, 7.7776642
1: -0.8289196, 0.8051167, -5.4578466, 4.7673273, -5.5962467, 6.2629633
2: -0.5958006, 1.1843369, -9.0892286, 3.5174587, -4.1132593, 10.2735653
3: -0.8389760, 0.7814506, -7.9425392, 3.8123171, -4.6512933, 8.7239895
4: -1.0978392, 0.8889894, -8.1946430, 5.0723443, -6.1701837, 9.0836325
5: -0.8760652, 0.9310993, -6.1417737, 5.3905454, -6.2666106, 7.0728731
6: -0.8582798, 0.8818473, -6.5140371, 5.5172434, -6.3755231, 7.3958845
7: -0.9862645, 0.9477892, -7.4971714, 5.4728184, -6.4590831, 8.4449606
8: -1.0474089, 0.9680716, -8.5955410, 5.0282660, -6.0756750, 9.5636129
9: -0.9297719, 1.0286045, -5.9693260, 6.9268246, -7.8565965, 6.9979305

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323916, upper bound: 7.6318259
time: 5.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323916, upper bound: 7.6320347
time: 2.49 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.9039898, 1.5532383, -6.8161311, 5.1730833, -7.0770731, 8.3693695
1: -1.5435649, 1.4417903, -5.3769970, 4.7024412, -6.2460060, 6.8187876
2: -2.0798304, 1.4384233, -8.9536200, 3.4726906, -5.5525208, 10.3920431
3: -1.9923401, 1.2694449, -7.8250332, 3.7613795, -5.7537193, 9.0944786
4: -2.2457442, 1.5643042, -8.0787821, 5.0026836, -7.2484279, 9.6430864
5: -1.7552578, 1.5948195, -6.0561719, 5.3157234, -7.0709810, 7.6509914
6: -1.7851844, 1.5925593, -6.4210968, 5.4392571, -7.2244415, 8.0136566
7: -2.0569201, 1.6748843, -7.3899946, 5.3973846, -7.4543047, 9.0648785
8: -2.3099952, 1.6051259, -8.4715881, 4.9587560, -7.2687511, 10.0767136
9: -1.7752999, 1.9758629, -5.8873625, 6.8290477, -8.6043472, 7.8632255

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320978, upper bound: 7.6313298
time: 3.12 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321226, upper bound: 7.6315326
time: 3.20 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.5457392, 1.2827413, -0.4271084, 0.4942832, -2.0400224, 1.7098497
1: -1.2722951, 1.1856624, -0.4654395, 0.4951848, -1.7674799, 1.6511019
2: -1.5166521, 1.3316029, 0.1681221, 1.0850841, -2.6017361, 1.1634808
3: -1.5567853, 1.0775865, -0.3050309, 0.5317209, -2.0885062, 1.3826175
4: -1.8102188, 1.3010962, -0.5014338, 0.5190079, -2.3292267, 1.8025301
5: -1.4175453, 1.3305972, -0.4564503, 0.5308417, -1.9483870, 1.7870475
6: -1.4274405, 1.3146764, -0.4015422, 0.5350480, -1.9624885, 1.7162186
7: -1.6546564, 1.3896329, -0.4590012, 0.5750179, -2.2296743, 1.8486342
8: -1.8133596, 1.3593698, -0.5243088, 0.6157832, -2.4291430, 1.8836786
9: -1.4449017, 1.6012875, -0.5123287, 0.5364254, -1.9813271, 2.1136162

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4749843, upper bound: 7.5708046
time: 2.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4750111, upper bound: 7.5858665
time: 2.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.4959335, 3.4693289, -0.4271084, 0.4942832, -4.9902167, 3.8964374
1: -3.5105519, 3.1928778, -0.4654395, 0.4951848, -4.0057368, 3.6583173
2: -5.7974186, 2.4429598, 0.1681221, 1.0850841, -6.8825026, 2.2748377
3: -5.0915055, 2.5866036, -0.3050309, 0.5317209, -5.6232262, 2.8916345
4: -5.3767180, 3.3945243, -0.5014338, 0.5190079, -5.8957257, 3.8959582
5: -4.0632315, 3.5637112, -0.4564503, 0.5308417, -4.5940733, 4.0201616
6: -4.2668610, 3.6310551, -0.4015422, 0.5350480, -4.8019090, 4.0325975
7: -4.9015784, 3.6491079, -0.4590012, 0.5750179, -5.4765964, 4.1081090
8: -5.6003866, 3.3712251, -0.5243088, 0.6157832, -6.2161698, 3.8955340
9: -3.9861240, 4.5531340, -0.5123287, 0.5364254, -4.5225496, 5.0654626

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4749843, upper bound: 7.5709272
time: 4.24 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4750111, upper bound: 7.5859833
time: 3.50 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.5457392, 1.2827413, -2.4689381, 1.9027667, -3.4485059, 3.7516794
1: -1.2722951, 1.1856624, -1.7500299, 1.6195354, -2.8918304, 2.9356923
2: -1.5166521, 1.3316029, -2.7192602, 1.4355766, -2.9522285, 4.0508633
3: -1.5567853, 1.0775865, -2.2868671, 1.4499850, -3.0067704, 3.3644538
4: -1.8102188, 1.3010962, -2.7850626, 1.8980682, -3.7082870, 4.0861588
5: -1.4175453, 1.3305972, -1.9056885, 2.0160394, -3.4335847, 3.2362857
6: -1.4274405, 1.3146764, -2.0554390, 1.8403050, -3.2677455, 3.3701153
7: -1.6546564, 1.3896329, -2.2998633, 1.7988111, -3.4534674, 3.6894963
8: -1.8133596, 1.3593698, -2.2995026, 1.8573328, -3.6706924, 3.6588724
9: -1.4449017, 1.6012875, -1.9954029, 2.4681754, -3.9130771, 3.5966904

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4749256, upper bound: 7.4237612
time: 2.28 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4749515, upper bound: 7.4419752
time: 2.22 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.4959335, 3.4693289, -2.4689381, 1.9027667, -6.3987002, 5.9382668
1: -3.5105519, 3.1928778, -1.7500299, 1.6195354, -5.1300874, 4.9429078
2: -5.7974186, 2.4429598, -2.7192602, 1.4355766, -7.2329950, 5.1622200
3: -5.0915055, 2.5866036, -2.2868671, 1.4499850, -6.5414906, 4.8734708
4: -5.3767180, 3.3945243, -2.7850626, 1.8980682, -7.2747860, 6.1795869
5: -4.0632315, 3.5637112, -1.9056885, 2.0160394, -6.0792708, 5.4693995
6: -4.2668610, 3.6310551, -2.0554390, 1.8403050, -6.1071658, 5.6864939
7: -4.9015784, 3.6491079, -2.2998633, 1.7988111, -6.7003894, 5.9489713
8: -5.6003866, 3.3712251, -2.2995026, 1.8573328, -7.4577193, 5.6707277
9: -3.9861240, 4.5531340, -1.9954029, 2.4681754, -6.4542994, 6.5485368

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4749256, upper bound: 7.4238543
time: 3.23 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4749515, upper bound: 7.4420637
time: 2.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.5457392, 1.2827413, -2.9857378, 2.3552699, -3.9010091, 4.2684793
1: -1.2722951, 1.1856624, -2.3264577, 2.2012329, -3.4735279, 3.5121202
2: -1.5166521, 1.3316029, -3.7073407, 1.7960043, -3.3126564, 5.0389438
3: -1.5567853, 1.0775865, -3.2828996, 1.8291472, -3.3859324, 4.3604860
4: -1.8102188, 1.3010962, -3.5917492, 2.3429205, -4.1531391, 4.8928452
5: -1.4175453, 1.3305972, -2.7384005, 2.4314122, -3.8489575, 4.0689974
6: -1.4274405, 1.3146764, -2.8522229, 2.4484720, -3.8759127, 4.1668992
7: -1.6546564, 1.3896329, -3.2656670, 2.5073543, -4.1620107, 4.6553001
8: -1.8133596, 1.3593698, -3.7229731, 2.3437855, -4.1571450, 5.0823431
9: -1.4449017, 1.6012875, -2.7375462, 3.0652418, -4.5101433, 4.3388338

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5922432, upper bound: 7.6315609
time: 2.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5922432, upper bound: 7.6317560
time: 3.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -2.0306103, 1.6444827, -2.8944330, 2.2878461, -4.3184566, 4.5389156
1: -1.6409917, 1.5239146, -2.2637115, 2.1394439, -3.7804356, 3.7876260
2: -2.2718799, 1.4721837, -3.5772915, 1.7604439, -4.0323238, 5.0494752
3: -2.1456385, 1.3345530, -3.1713338, 1.7837385, -3.9293771, 4.5058870
4: -2.3923080, 1.6524204, -3.4811118, 2.2790785, -4.6713867, 5.1335320
5: -1.8730156, 1.6805904, -2.6561947, 2.3602545, -4.2332702, 4.3367853
6: -1.9222389, 1.6905149, -2.7668812, 2.3767941, -4.2990332, 4.4573960
7: -2.1995964, 1.7660645, -3.1645284, 2.4380031, -4.6375995, 4.9305930
8: -2.4799912, 1.6861659, -3.6090894, 2.2812717, -4.7612629, 5.2952552
9: -1.8849020, 2.0994213, -2.6603346, 2.9745643, -4.8594666, 4.7597561

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4420893, upper bound: 7.6308673
time: 4.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4421272, upper bound: 7.6309941
time: 3.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.5457392, 1.2827413, -6.5204420, 4.9371662, -6.4829054, 7.8031836
1: -1.2722951, 1.1856624, -5.0957966, 4.5248246, -5.7971196, 6.2814589
2: -1.5166521, 1.3316029, -8.5256052, 3.3044534, -4.8211055, 9.8572083
3: -1.5567853, 1.0775865, -7.4779720, 3.5694335, -5.1262188, 8.5555582
4: -1.8102188, 1.3010962, -7.7604628, 4.7614388, -6.5716577, 9.0615587
5: -1.4175453, 1.3305972, -5.8106794, 5.0877514, -6.5052967, 7.1412764
6: -1.4274405, 1.3146764, -6.1206427, 5.2115893, -6.6390300, 7.4353189
7: -1.6546564, 1.3896329, -7.0621114, 5.1642036, -6.8188601, 8.4517441
8: -1.8133596, 1.3593698, -8.0913353, 4.7138529, -6.5272126, 9.4507046
9: -1.4449017, 1.6012875, -5.6527481, 6.5530515, -7.9979534, 7.2540355

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5922043, upper bound: 7.6309564
time: 3.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5922043, upper bound: 7.6311825
time: 3.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.0306103, 1.6444827, -6.4242711, 4.8672237, -6.8978338, 8.0687542
1: -1.6409917, 1.5239146, -5.0193138, 4.4614639, -6.1024556, 6.5432281
2: -2.2718799, 1.4721837, -8.3971653, 3.2639806, -5.5358605, 9.8693485
3: -2.1456385, 1.3345530, -7.3640304, 3.5219560, -5.6675944, 8.6985836
4: -2.3923080, 1.6524204, -7.6470613, 4.6954670, -7.0877752, 9.2994814
5: -1.8730156, 1.6805904, -5.7274656, 5.0167637, -6.8897791, 7.4080563
6: -1.9222389, 1.6905149, -6.0319018, 5.1361675, -7.0584064, 7.7224169
7: -2.1995964, 1.7660645, -6.9592814, 5.0916572, -7.2912536, 8.7253456
8: -2.4799912, 1.6861659, -7.9724617, 4.6492639, -7.1292553, 9.6586275
9: -1.8849020, 2.0994213, -5.5732737, 6.4581347, -8.3430367, 7.6726952

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4420426, upper bound: 7.6301520
time: 3.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4420853, upper bound: 7.6303309
time: 2.54 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.4959335, 3.4693289, -2.9857378, 2.3552699, -6.8512034, 6.4550667
1: -3.5105519, 3.1928778, -2.3264577, 2.2012329, -5.7117848, 5.5193357
2: -5.7974186, 2.4429598, -3.7073407, 1.7960043, -7.5934229, 6.1503005
3: -5.0915055, 2.5866036, -3.2828996, 1.8291472, -6.9206529, 5.8695030
4: -5.3767180, 3.3945243, -3.5917492, 2.3429205, -7.7196388, 6.9862738
5: -4.0632315, 3.5637112, -2.7384005, 2.4314122, -6.4946437, 6.3021116
6: -4.2668610, 3.6310551, -2.8522229, 2.4484720, -6.7153330, 6.4832783
7: -4.9015784, 3.6491079, -3.2656670, 2.5073543, -7.4089327, 6.9147749
8: -5.6003866, 3.3712251, -3.7229731, 2.3437855, -7.9441719, 7.0941982
9: -3.9861240, 4.5531340, -2.7375462, 3.0652418, -7.0513659, 7.2906799

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318281, upper bound: 7.6323926
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318281, upper bound: 7.6323926
time: 3.31 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.6481009, 4.3143053, -2.8944330, 2.2878461, -7.9359469, 7.2087383
1: -4.4343729, 3.9440126, -2.2637115, 2.1394439, -6.5738168, 6.2077241
2: -7.3675647, 2.9456990, -3.5772915, 1.7604439, -9.1280088, 6.5229902
3: -6.4511962, 3.1674852, -3.1713338, 1.7837385, -8.2349348, 6.3388190
4: -6.7232747, 4.1900253, -3.4811118, 2.2790785, -9.0023537, 7.6711369
5: -5.0535412, 4.4354692, -2.6561947, 2.3602545, -7.4137955, 7.0916638
6: -5.3355384, 4.5281334, -2.7668812, 2.3767941, -7.7123327, 7.2950144
7: -6.1367617, 4.5165505, -3.1645284, 2.4380031, -8.5747643, 7.6810789
8: -7.0238547, 4.1519985, -3.6090894, 2.2812717, -9.3051262, 7.7610879
9: -4.9293156, 5.6875744, -2.6603346, 2.9745643, -7.9038801, 8.3479090

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315024, upper bound: 7.6320219
time: 2.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315332, upper bound: 7.6321227
time: 2.52 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.4959335, 3.4693289, -6.5204420, 4.9371662, -9.4330997, 9.9897709
1: -3.5105519, 3.1928778, -5.0957966, 4.5248246, -8.0353765, 8.2886744
2: -5.7974186, 2.4429598, -8.5256052, 3.3044534, -9.1018715, 10.9685650
3: -5.0915055, 2.5866036, -7.4779720, 3.5694335, -8.6609392, 10.0645752
4: -5.3767180, 3.3945243, -7.7604628, 4.7614388, -10.1381569, 11.1549873
5: -4.0632315, 3.5637112, -5.8106794, 5.0877514, -9.1509829, 9.3743906
6: -4.2668610, 3.6310551, -6.1206427, 5.2115893, -9.4784508, 9.7516975
7: -4.9015784, 3.6491079, -7.0621114, 5.1642036, -10.0657825, 10.7112198
8: -5.6003866, 3.3712251, -8.0913353, 4.7138529, -10.3142395, 11.4625607
9: -3.9861240, 4.5531340, -5.6527481, 6.5530515, -10.5391750, 10.2058821

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317945, upper bound: 7.6317944
time: 3.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317945, upper bound: 7.6317943
time: 3.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.6481009, 4.3143053, -6.4242711, 4.8672237, -10.5153246, 10.7385769
1: -4.4343729, 3.9440126, -5.0193138, 4.4614639, -8.8958368, 8.9633265
2: -7.3675647, 2.9456990, -8.3971653, 3.2639806, -10.6315451, 11.3428640
3: -6.4511962, 3.1674852, -7.3640304, 3.5219560, -9.9731522, 10.5315151
4: -6.7232747, 4.1900253, -7.6470613, 4.6954670, -11.4187412, 11.8370867
5: -5.0535412, 4.4354692, -5.7274656, 5.0167637, -10.0703049, 10.1629353
6: -5.3355384, 4.5281334, -6.0319018, 5.1361675, -10.4717064, 10.5600357
7: -6.1367617, 4.5165505, -6.9592814, 5.0916572, -11.2284184, 11.4758320
8: -7.0238547, 4.1519985, -7.9724617, 4.6492639, -11.6731186, 12.1244602
9: -4.9293156, 5.6875744, -5.5732737, 6.4581347, -11.3874502, 11.2608480

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314631, upper bound: 7.6312969
time: 2.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314993, upper bound: 7.6314988
time: 2.10 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.25 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5710659, upper bound: 7.6154079
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5710659, upper bound: 7.6310386
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5861358, upper bound: 7.6154307
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5861358, upper bound: 7.6310677
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5710520, upper bound: 7.5861002
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5710520, upper bound: 7.6313772
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5861250, upper bound: 7.5861253
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5861250, upper bound: 7.6314081
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6310386, upper bound: 7.5711357
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6310677, upper bound: 7.5862014
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6313772, upper bound: 7.5711554
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6314080, upper bound: 7.5862259
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6325153, upper bound: 7.6325153
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6325153, upper bound: 7.6327092
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6322263, upper bound: 7.6321460
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6322434, upper bound: 7.6322433
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5708046, upper bound: 7.4749845
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5708046, upper bound: 7.6299129
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5858665, upper bound: 7.4750108
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5858665, upper bound: 7.6299526
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5707546, upper bound: 7.4419952
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5707546, upper bound: 7.6303880
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5858211, upper bound: 7.4420170
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5858211, upper bound: 7.6304315
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6305958, upper bound: 7.4238961
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6306062, upper bound: 7.4421049
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6309652, upper bound: 7.4239188
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6309941, upper bound: 7.4421272
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6323916, upper bound: 7.6318259
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6323916, upper bound: 7.6320347
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6320978, upper bound: 7.6313298
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6321226, upper bound: 7.6315326
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4749843, upper bound: 7.5708046
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4750111, upper bound: 7.5858665
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4749843, upper bound: 7.5709272
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4750111, upper bound: 7.5859833
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4749256, upper bound: 7.4237612
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4749515, upper bound: 7.4419752
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4749256, upper bound: 7.4238543
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4749515, upper bound: 7.4420637
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5922432, upper bound: 7.6315609
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5922432, upper bound: 7.6317560
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4420893, upper bound: 7.6308673
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4421272, upper bound: 7.6309941
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5922043, upper bound: 7.6309564
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.5922043, upper bound: 7.6311825
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4420426, upper bound: 7.6301520
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.4420853, upper bound: 7.6303309
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6318281, upper bound: 7.6323926
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6318281, upper bound: 7.6323926
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6315024, upper bound: 7.6320219
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6315332, upper bound: 7.6321227
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6317945, upper bound: 7.6317944
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6317945, upper bound: 7.6317943
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6314631, upper bound: 7.6312969
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.25
Output dim: 2, lower bound: -7.6314993, upper bound: 7.6314988

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1874970, 0.2036436, -0.8578385, 0.8006364, -0.9881334, 1.0614821
1: -0.2372475, 0.2500956, -0.7633286, 0.7502429, -0.9874904, 1.0134243
2: 0.5474359, 1.0401566, -0.4552889, 1.1661792, -0.6187433, 1.4954455
3: -0.1039863, 0.3226792, -0.7271672, 0.7382299, -0.8422161, 1.0498464
4: -0.2557076, 0.2460805, -0.9849149, 0.8285998, -1.0843074, 1.2309954
5: -0.2304034, 0.2558567, -0.7943538, 0.8712278, -1.1016312, 1.0502105
6: -0.1839036, 0.2734807, -0.7742565, 0.8176720, -1.0015755, 1.0477371
7: -0.2174428, 0.2828646, -0.8833599, 0.8818660, -1.0993088, 1.1662245
8: -0.2556123, 0.3509589, -0.9463471, 0.9070480, -1.1626604, 1.2973061
9: -0.2636027, 0.2364774, -0.8492008, 0.9478745, -1.2114773, 1.0856782

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5710659, upper bound: 7.6310386
time: 1.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5710659, upper bound: 7.6310385
time: 3.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2237872, 0.2521938, -0.8357258, 0.7863177, -1.0101049, 1.0879196
1: -0.2728415, 0.2928494, -0.7479779, 0.7372131, -1.0100546, 1.0408273
2: 0.4923054, 1.0409260, -0.4221525, 1.1619259, -0.6696205, 1.4630785
3: -0.1387696, 0.3630214, -0.7009888, 0.7281405, -0.8669101, 1.0640101
4: -0.2952042, 0.2847379, -0.9581282, 0.8145348, -1.1097389, 1.2428660
5: -0.2674049, 0.2962787, -0.7753685, 0.8570936, -1.1244985, 1.0716472
6: -0.2165657, 0.3221747, -0.7548549, 0.8024798, -1.0190455, 1.0770295
7: -0.2571482, 0.3244728, -0.8590951, 0.8664851, -1.1236333, 1.1835679
8: -0.3065949, 0.3941213, -0.9225346, 0.8927326, -1.1993275, 1.3166559
9: -0.3081053, 0.2794375, -0.8301646, 0.9290464, -1.2371516, 1.1096021

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5856709, upper bound: 7.6310677
time: 3.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5856709, upper bound: 7.6310677
time: 3.25 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1732886, 0.1877893, -1.7931721, 1.4713500, -1.6446385, 1.9809613
1: -0.2217900, 0.2339419, -1.4593767, 1.3642769, -1.5860668, 1.6933186
2: 0.5704663, 1.0400932, -1.9097934, 1.4054903, -0.8350239, 2.9498866
3: -0.0909384, 0.3060919, -1.8591875, 1.2102835, -1.3012218, 2.1652794
4: -0.2391381, 0.2319911, -2.1138225, 1.4837347, -1.7228729, 2.3458135
5: -0.2162016, 0.2379208, -1.6517661, 1.5146888, -1.7308905, 1.8896868
6: -0.1733780, 0.2516658, -1.6702858, 1.5064675, -1.6798455, 1.9219517
7: -0.2036288, 0.2651914, -1.9338150, 1.5883403, -1.7919691, 2.1990063
8: -0.2355525, 0.3324037, -2.1581810, 1.5308346, -1.7663870, 2.4905846
9: -0.2461864, 0.2191157, -1.6750757, 1.8628128, -2.1089993, 1.8941914

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5710521, upper bound: 7.6313769
time: 3.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5710521, upper bound: 7.6310826
time: 2.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2107242, 0.2335768, -1.7814255, 1.4625376, -1.6732619, 2.0150023
1: -0.2599831, 0.2771797, -1.4504319, 1.3557357, -1.6157188, 1.7276117
2: 0.5127159, 1.0408612, -1.8914535, 1.4019721, -0.8892561, 2.9323149
3: -0.1264535, 0.3483922, -1.8450061, 1.2038801, -1.3303336, 2.1933985
4: -0.2813092, 0.2695442, -2.0996571, 1.4749513, -1.7562605, 2.3692012
5: -0.2534396, 0.2820977, -1.6406740, 1.5061277, -1.7595674, 1.9227717
6: -0.2042072, 0.3046420, -1.6582147, 1.4971110, -1.7013181, 1.9628567
7: -0.2422649, 0.3096581, -1.9206433, 1.5788145, -1.8210794, 2.2303014
8: -0.2881194, 0.3783378, -2.1418192, 1.5227557, -1.8108752, 2.5201571
9: -0.2916138, 0.2639759, -1.6643058, 1.8507265, -2.1423402, 1.9282818

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5856572, upper bound: 7.6314081
time: 3.28 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5856572, upper bound: 7.6311423
time: 3.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.8578385, 0.8006364, -0.1874970, 0.2036436, -1.0614821, 0.9881334
1: -0.7633286, 0.7502429, -0.2372475, 0.2500956, -1.0134243, 0.9874904
2: -0.4552889, 1.1661792, 0.5474359, 1.0401566, -1.4954455, 0.6187433
3: -0.7271672, 0.7382299, -0.1039863, 0.3226792, -1.0498464, 0.8422161
4: -0.9849149, 0.8285998, -0.2557076, 0.2460805, -1.2309954, 1.0843074
5: -0.7943538, 0.8712278, -0.2304034, 0.2558567, -1.0502105, 1.1016312
6: -0.7742565, 0.8176720, -0.1839036, 0.2734807, -1.0477371, 1.0015755
7: -0.8833599, 0.8818660, -0.2174428, 0.2828646, -1.1662245, 1.0993088
8: -0.9463471, 0.9070480, -0.2556123, 0.3509589, -1.2973061, 1.1626604
9: -0.8492008, 0.9478745, -0.2636027, 0.2364774, -1.0856782, 1.2114773

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310168, upper bound: 7.5691159
time: 3.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310168, upper bound: 7.5691180
time: 4.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.8357258, 0.7863177, -0.2237872, 0.2521938, -1.0879196, 1.0101049
1: -0.7479779, 0.7372131, -0.2728415, 0.2928494, -1.0408273, 1.0100546
2: -0.4221525, 1.1619259, 0.4923054, 1.0409260, -1.4630785, 0.6696205
3: -0.7009888, 0.7281405, -0.1387696, 0.3630214, -1.0640101, 0.8669101
4: -0.9581282, 0.8145348, -0.2952042, 0.2847379, -1.2428660, 1.1097389
5: -0.7753685, 0.8570936, -0.2674049, 0.2962787, -1.0716472, 1.1244985
6: -0.7548549, 0.8024798, -0.2165657, 0.3221747, -1.0770295, 1.0190455
7: -0.8590951, 0.8664851, -0.2571482, 0.3244728, -1.1835679, 1.1236333
8: -0.9225346, 0.8927326, -0.3065949, 0.3941213, -1.3166559, 1.1993275
9: -0.8301646, 0.9290464, -0.3081053, 0.2794375, -1.1096021, 1.2371516

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310247, upper bound: 7.5861726
time: 3.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310247, upper bound: 7.5862014
time: 10.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.7931721, 1.4713500, -0.1732886, 0.1877893, -1.9809613, 1.6446385
1: -1.4593767, 1.3642769, -0.2217900, 0.2339419, -1.6933186, 1.5860668
2: -1.9097934, 1.4054903, 0.5704663, 1.0400932, -2.9498866, 0.8350239
3: -1.8591875, 1.2102835, -0.0909384, 0.3060919, -2.1652794, 1.3012218
4: -2.1138225, 1.4837347, -0.2391381, 0.2319911, -2.3458135, 1.7228729
5: -1.6517661, 1.5146888, -0.2162016, 0.2379208, -1.8896868, 1.7308905
6: -1.6702858, 1.5064675, -0.1733780, 0.2516658, -1.9219517, 1.6798455
7: -1.9338150, 1.5883403, -0.2036288, 0.2651914, -2.1990063, 1.7919691
8: -2.1581810, 1.5308346, -0.2355525, 0.3324037, -2.4905846, 1.7663870
9: -1.6750757, 1.8628128, -0.2461864, 0.2191157, -1.8941914, 2.1089993

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312631, upper bound: 7.5691296
time: 4.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312631, upper bound: 7.5691386
time: 4.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.7814255, 1.4625376, -0.2107242, 0.2335768, -2.0150023, 1.6732619
1: -1.4504319, 1.3557357, -0.2599831, 0.2771797, -1.7276117, 1.6157188
2: -1.8914535, 1.4019721, 0.5127159, 1.0408612, -2.9323149, 0.8892561
3: -1.8450061, 1.2038801, -0.1264535, 0.3483922, -2.1933985, 1.3303336
4: -2.0996571, 1.4749513, -0.2813092, 0.2695442, -2.3692012, 1.7562605
5: -1.6406740, 1.5061277, -0.2534396, 0.2820977, -1.9227717, 1.7595674
6: -1.6582147, 1.4971110, -0.2042072, 0.3046420, -1.9628567, 1.7013181
7: -1.9206433, 1.5788145, -0.2422649, 0.3096581, -2.2303014, 1.8210794
8: -2.1418192, 1.5227557, -0.2881194, 0.3783378, -2.5201571, 1.8108752
9: -1.6643058, 1.8507265, -0.2916138, 0.2639759, -1.9282818, 2.1423402

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312638, upper bound: 7.5861833
time: 2.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312638, upper bound: 7.5862259
time: 2.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.9512004, 0.8617375, -0.9512004, 0.8617375, -1.8129380, 1.8129380
1: -0.8289196, 0.8051167, -0.8289196, 0.8051167, -1.6340363, 1.6340363
2: -0.5958006, 1.1843369, -0.5958006, 1.1843369, -1.7801375, 1.7801375
3: -0.8389760, 0.7814506, -0.8389760, 0.7814506, -1.6204265, 1.6204265
4: -1.0978392, 0.8889894, -1.0978392, 0.8889894, -1.9868287, 1.9868287
5: -0.8760652, 0.9310993, -0.8760652, 0.9310993, -1.8071644, 1.8071644
6: -0.8582798, 0.8818473, -0.8582798, 0.8818473, -1.7401271, 1.7401271
7: -0.9862645, 0.9477892, -0.9862645, 0.9477892, -1.9340537, 1.9340537
8: -1.0474089, 0.9680716, -1.0474089, 0.9680716, -2.0154805, 2.0154805
9: -0.9297719, 1.0286045, -0.9297719, 1.0286045, -1.9583764, 1.9583764

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318300, upper bound: 7.6318354
time: 2.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318422, upper bound: 7.6318419
time: 2.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.9512004, 0.8617375, -1.9039898, 1.5532383, -2.5044386, 2.7657273
1: -0.8289196, 0.8051167, -1.5435649, 1.4417903, -2.2707100, 2.3486814
2: -0.5958006, 1.1843369, -2.0798304, 1.4384233, -2.0342238, 3.2641673
3: -0.8389760, 0.7814506, -1.9923401, 1.2694449, -2.1084208, 2.7737906
4: -1.0978392, 0.8889894, -2.2457442, 1.5643042, -2.6621435, 3.1347337
5: -0.8760652, 0.9310993, -1.7552578, 1.5948195, -2.4708848, 2.6863570
6: -0.8582798, 0.8818473, -1.7851844, 1.5925593, -2.4508390, 2.6670318
7: -0.9862645, 0.9477892, -2.0569201, 1.6748843, -2.6611488, 3.0047092
8: -1.0474089, 0.9680716, -2.3099952, 1.6051259, -2.6525350, 3.2780666
9: -0.9297719, 1.0286045, -1.7752999, 1.9758629, -2.9056349, 2.8039045

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318300, upper bound: 7.6321468
time: 2.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318422, upper bound: 7.6318419
time: 3.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.7931721, 1.4713500, -1.4709172, 1.2278349, -3.0210071, 2.9422672
1: -1.4593767, 1.3642769, -1.2135494, 1.1389586, -2.5983353, 2.5778263
2: -1.9097934, 1.4054903, -1.3999332, 1.3126146, -3.2224078, 2.8054235
3: -1.8591875, 1.2102835, -1.4673119, 1.0405695, -2.8997569, 2.6775954
4: -2.1138225, 1.4837347, -1.7235618, 1.2495437, -3.3633661, 3.2072964
5: -1.6517661, 1.5146888, -1.3503978, 1.2799013, -2.9316673, 2.8650866
6: -1.6702858, 1.5064675, -1.3505573, 1.2570488, -2.9273348, 2.8570247
7: -1.9338150, 1.5883403, -1.5667136, 1.3377533, -3.2715683, 3.1550539
8: -2.1581810, 1.5308346, -1.7121266, 1.3123361, -3.4705172, 3.2429612
9: -1.6750757, 1.8628128, -1.3806609, 1.5288044, -3.2038801, 3.2434735

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321401, upper bound: 7.6321404
time: 2.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321401, upper bound: 7.6321456
time: 2.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.7814255, 1.4625376, -1.8399742, 1.5048447, -3.2862701, 3.3025117
1: -1.4504319, 1.3557357, -1.4949507, 1.3972118, -2.8476439, 2.8506863
2: -1.8914535, 1.4019721, -1.9793522, 1.4194803, -3.3109338, 3.3813243
3: -1.8450061, 1.2038801, -1.9149847, 1.2351707, -3.0801768, 3.1188648
4: -2.0996571, 1.4749513, -2.1686149, 1.5179479, -3.6176050, 3.6435661
5: -1.6406740, 1.5061277, -1.6967078, 1.5467526, -3.1874266, 3.2028356
6: -1.6582147, 1.4971110, -1.7185755, 1.5432223, -3.2014370, 3.2156863
7: -1.9206433, 1.5788145, -1.9850460, 1.6251473, -3.5457907, 3.5638604
8: -2.1418192, 1.5227557, -2.2211101, 1.5631864, -3.7050056, 3.7438660
9: -1.6643058, 1.8507265, -1.7177238, 1.9085150, -3.5728207, 3.5684505

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321454, upper bound: 7.6322246
time: 2.52 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321454, upper bound: 7.6322440
time: 2.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1874970, 0.2036436, -4.3632569, 3.3707278, -3.5582249, 4.5669007
1: -0.2372475, 0.2500956, -3.4030342, 3.1059103, -3.3431578, 3.6531298
2: 0.5474359, 1.0401566, -5.6140189, 2.3850830, -1.8376471, 6.6541758
3: -0.1039863, 0.3226792, -4.9341564, 2.5192974, -2.6232836, 5.2568355
4: -0.2557076, 0.2460805, -5.2207289, 3.3012640, -3.5569715, 5.4668093
5: -0.2304034, 0.2558567, -3.9486096, 3.4647114, -3.6951149, 4.2044663
6: -0.1839036, 0.2734807, -4.1437016, 3.5267272, -3.7106307, 4.4171824
7: -0.2174428, 0.2828646, -4.7590876, 3.5478082, -3.7652509, 5.0419521
8: -0.2556123, 0.3509589, -5.4348106, 3.2795084, -3.5351207, 5.7857695
9: -0.2636027, 0.2364774, -3.8768587, 4.4211211, -4.6847239, 4.1133361

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5708046, upper bound: 7.6299130
time: 2.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5708046, upper bound: 7.6299131
time: 7.20 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 11.30 seconds
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5710659, upper bound: 7.6310386
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5710659, upper bound: 7.6310385
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5856709, upper bound: 7.6310677
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5856709, upper bound: 7.6310677
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5710521, upper bound: 7.6313769
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5710521, upper bound: 7.6310826
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5856572, upper bound: 7.6314081
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5856572, upper bound: 7.6311423
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6310168, upper bound: 7.5691159
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6310168, upper bound: 7.5691180
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6310247, upper bound: 7.5861726
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6310247, upper bound: 7.5862014
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6312631, upper bound: 7.5691296
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6312631, upper bound: 7.5691386
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6312638, upper bound: 7.5861833
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6312638, upper bound: 7.5862259
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6318300, upper bound: 7.6318354
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6318422, upper bound: 7.6318419
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6318300, upper bound: 7.6321468
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6318422, upper bound: 7.6318419
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6321401, upper bound: 7.6321404
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6321401, upper bound: 7.6321456
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6321454, upper bound: 7.6322246
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.6321454, upper bound: 7.6322440
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5708046, upper bound: 7.6299130
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.30
Output dim: 2, lower bound: -7.5708046, upper bound: 7.6299131
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.5858665, upper bound: 7.6299526
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.5707546, upper bound: 7.6303880
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.5858211, upper bound: 7.6304315
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6305958, upper bound: 7.4238961
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6306062, upper bound: 7.4421049
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6309652, upper bound: 7.4239188
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6309941, upper bound: 7.4421272
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6323916, upper bound: 7.6318259
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6323916, upper bound: 7.6320347
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6320978, upper bound: 7.6313298
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6321226, upper bound: 7.6315326
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.5922432, upper bound: 7.6315609
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.5922432, upper bound: 7.6317560
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.4420893, upper bound: 7.6308673
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.4421272, upper bound: 7.6309941
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.5922043, upper bound: 7.6309564
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.5922043, upper bound: 7.6311825
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.4420426, upper bound: 7.6301520
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.4420853, upper bound: 7.6303309
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6318281, upper bound: 7.6323926
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6318281, upper bound: 7.6323926
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6315024, upper bound: 7.6320219
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6315332, upper bound: 7.6321227
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6317945, upper bound: 7.6317944
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6317945, upper bound: 7.6317943
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6314631, upper bound: 7.6312969
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 11.30
Output dim: 2, lower bound: -7.6314993, upper bound: 7.6314988
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=9.388381958007812
rel_dist={2: [-7.633606580437654, 7.633606672931759]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6333416, upper bound: 7.6329650
time: 3.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329440, upper bound: 7.6329440
time: 2.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 5.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 5.91
Output dim: 2, lower bound: -7.6333416, upper bound: 7.6329650
IS_A2, status: Status.UNKNOWN, split count: 1, time: 5.91
Output dim: 2, lower bound: -7.6329440, upper bound: 7.6329440

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.2686481, 3.3034801, -4.8439870, 3.7249637, -7.9936118, 8.1474667
1: -3.3155420, 3.0498059, -3.7795856, 3.4268079, -6.7423496, 6.8293915
2: -5.4949217, 2.3689001, -6.2778797, 2.6069093, -8.1018314, 8.6467800
3: -4.8110600, 2.4704089, -5.4946775, 2.7603803, -7.5714402, 7.9650865
4: -5.1011324, 3.2392945, -5.7783298, 3.6380610, -8.7391930, 9.0176239
5: -3.8673596, 3.4099646, -4.3622389, 3.8365352, -7.7038946, 7.7722034
6: -4.0397010, 3.4505873, -4.5767317, 3.9007261, -7.9404268, 8.0273190
7: -4.6541395, 3.4808147, -5.2714081, 3.9153130, -8.5694523, 8.7522230
8: -5.3186646, 3.2162716, -6.0306888, 3.6089582, -8.9276228, 9.2469606
9: -3.7986977, 4.3246951, -4.2714772, 4.8971300, -8.6958275, 8.5961723

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317630, upper bound: 7.6304837
time: 10.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329432, upper bound: 7.6325742
time: 3.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.3132782, 6.2759089, -4.7834363, 3.6805861, -11.9938641, 11.0593452
1: -6.5844011, 5.6741643, -3.7305417, 3.3870606, -9.9714622, 9.4047060
2: -10.9843225, 4.1436543, -6.1959019, 2.5812473, -13.5655699, 10.3395557
3: -9.5884886, 4.5238266, -5.4231091, 2.7295809, -12.3180695, 9.9469357
4: -9.8131104, 6.0509901, -5.7074418, 3.5959396, -13.4090500, 11.7584324
5: -7.3376122, 6.4407911, -4.3102756, 3.7915919, -11.1292038, 10.7510662
6: -7.8093801, 6.6083603, -4.5202670, 3.8534420, -11.6628218, 11.1286278
7: -8.9933872, 6.5240011, -5.2064905, 3.8693681, -12.8627548, 11.7304916
8: -10.3325996, 5.9963574, -5.9557323, 3.5676150, -13.9002151, 11.9520893
9: -7.1123838, 8.2964840, -4.2217007, 4.8374529, -11.9498367, 12.5181847

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314270, upper bound: 7.6304627
time: 4.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325541, upper bound: 7.6325542
time: 2.73 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 8.80 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 8.80
Output dim: 2, lower bound: -7.6317630, upper bound: 7.6304837
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 8.80
Output dim: 2, lower bound: -7.6329432, upper bound: 7.6325742
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 8.80
Output dim: 2, lower bound: -7.6314270, upper bound: 7.6304627
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 8.80
Output dim: 2, lower bound: -7.6325541, upper bound: 7.6325542

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -2.0544791, 1.6642331, -0.6327204, 0.6538154, -2.7082944, 2.2969534
1: -1.6587214, 1.5523050, -0.6122586, 0.6269299, -2.2856512, 2.1645637
2: -2.3107984, 1.4857168, -0.1243145, 1.1261820, -3.4369802, 1.6100314
3: -2.1744158, 1.3505430, -0.4816597, 0.6408203, -2.8152361, 1.8322027
4: -2.4259250, 1.6757801, -0.7189755, 0.6833813, -3.1093063, 2.3947556
5: -1.8946854, 1.7060803, -0.6090171, 0.7215949, -2.6162803, 2.3150973
6: -1.9384396, 1.7116561, -0.5798560, 0.6723020, -2.6107416, 2.2915120
7: -2.2237339, 1.7968422, -0.6512892, 0.7386155, -2.9623494, 2.4481316
8: -2.5171628, 1.7071279, -0.7206078, 0.7722126, -3.2893753, 2.4277358
9: -1.9137287, 2.1263843, -0.6718641, 0.7479298, -2.6616585, 2.7982483

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304686, upper bound: 7.5921574
time: 4.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306432, upper bound: 7.5921658
time: 5.16 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.9733973, 3.0839028, -3.6603279, 2.8497655, -6.8231630, 6.7442307
1: -3.0780487, 2.8561590, -2.8303285, 2.6510715, -5.7291203, 5.6864872
2: -5.0836267, 2.2324247, -4.6422482, 2.0604300, -7.1440568, 6.8746729
3: -4.4626904, 2.3210492, -4.0982494, 2.1634922, -6.6261826, 6.4192986
4: -4.7573214, 3.0308218, -4.4014063, 2.8090172, -7.5663385, 7.4322281
5: -3.6105978, 3.1864536, -3.3333907, 2.9390576, -6.5496554, 6.5198441
6: -3.7670946, 3.2184243, -3.4833982, 2.9751120, -6.7422066, 6.7018223
7: -4.3368220, 3.2556291, -3.9991078, 3.0176325, -7.3544545, 7.2547369
8: -4.9507952, 3.0113065, -4.5607572, 2.8001130, -7.7509079, 7.5720634
9: -3.5559435, 4.0345802, -3.3003826, 3.7390604, -7.2950039, 7.3349628

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6322382, upper bound: 7.6319984
time: 3.48 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325112, upper bound: 7.6320795
time: 4.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.9475546, 4.5396948, -0.6061128, 0.6363257, -6.5838804, 5.1458077
1: -4.6825829, 4.1357999, -0.5976955, 0.6135504, -5.2961330, 4.7334952
2: -7.7880030, 3.1085556, -0.0878957, 1.1216575, -8.9096603, 3.1964512
3: -6.7936106, 3.3253481, -0.4566112, 0.6298405, -7.4234509, 3.7819593
4: -7.0588965, 4.4039311, -0.6896520, 0.6657025, -7.7245989, 5.0935831
5: -5.3121061, 4.6658125, -0.5910856, 0.7027930, -6.0148993, 5.2568979
6: -5.6150656, 4.7624488, -0.5592645, 0.6559900, -6.2710557, 5.3217134
7: -6.4609408, 4.7470765, -0.6287217, 0.7235727, -7.1845136, 5.3757982
8: -7.3980551, 4.3743696, -0.6983821, 0.7564668, -8.1545219, 5.0727520
9: -5.1743474, 5.9693584, -0.6531427, 0.7232587, -5.8976059, 6.6225014

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6300484, upper bound: 7.5921343
time: 7.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302708, upper bound: 7.5921424
time: 2.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.0024185, 6.0469885, -3.5986392, 2.8047452, -10.8071632, 9.6456280
1: -6.3337159, 5.4724278, -2.7810116, 2.6103392, -8.9440556, 8.2534389
2: -10.5640163, 4.0049324, -4.5582352, 2.0346484, -12.5986652, 8.5631676
3: -9.2221451, 4.3655596, -4.0249224, 2.1329720, -11.3551168, 8.3904819
4: -9.4531078, 5.8331246, -4.3285012, 2.7667048, -12.2198124, 10.1616259
5: -7.0714955, 6.2077842, -3.2799673, 2.8933988, -9.9648943, 9.4877510
6: -7.5212197, 6.3655143, -3.4265873, 2.9268625, -10.4480820, 9.7921019
7: -8.6606464, 6.2903137, -3.9330668, 2.9709082, -11.6315546, 10.2233810
8: -9.9460192, 5.7810035, -4.4842110, 2.7588148, -12.7048340, 10.2652149
9: -6.8580933, 7.9917693, -3.2494252, 3.6777515, -10.5358448, 11.2411947

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317888, upper bound: 7.6319766
time: 2.25 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320576, upper bound: 7.6320576
time: 4.05 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 7.84 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 2, lower bound: -7.6304686, upper bound: 7.5921574
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 2, lower bound: -7.6306432, upper bound: 7.5921658
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 2, lower bound: -7.6322382, upper bound: 7.6319984
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 2, lower bound: -7.6325112, upper bound: 7.6320795
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 2, lower bound: -7.6300484, upper bound: 7.5921343
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 2, lower bound: -7.6302708, upper bound: 7.5921424
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 2, lower bound: -7.6317888, upper bound: 7.6319766
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.84
Output dim: 2, lower bound: -7.6320576, upper bound: 7.6320576

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4375579, 0.5057535, -0.3759645, 0.4386618, -0.8762197, 0.8817180
1: -0.4761628, 0.5051691, -0.4183383, 0.4453653, -0.9215281, 0.9235073
2: 0.1516788, 1.0878110, 0.2479169, 1.0706682, -0.9189894, 0.8398941
3: -0.3131579, 0.5396056, -0.2651235, 0.4927065, -0.8058645, 0.8047291
4: -0.5113816, 0.5311555, -0.4527672, 0.4586921, -0.9700737, 0.9839227
5: -0.4673158, 0.5427294, -0.4089996, 0.4708560, -0.9381718, 0.9517289
6: -0.4112628, 0.5450190, -0.3561119, 0.4853476, -0.8966104, 0.9011309
7: -0.4692430, 0.5872211, -0.4098575, 0.5138015, -0.9830444, 0.9970787
8: -0.5345352, 0.6277806, -0.4724873, 0.5635416, -1.0980768, 1.1002679
9: -0.5218457, 0.5510181, -0.4643419, 0.4690111, -0.9908568, 1.0153600

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286447, upper bound: 7.4237966
time: 2.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287959, upper bound: 7.4420110
time: 5.34 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.9222797, 0.8429014, -0.3931107, 0.4580549, -1.3803346, 1.2360121
1: -0.8082142, 0.7876428, -0.4340487, 0.4617651, -1.2699792, 1.2216915
2: -0.5528372, 1.1784894, 0.2201818, 1.0747434, -1.6275806, 0.9583076
3: -0.8042040, 0.7677456, -0.2785083, 0.5059881, -1.3101921, 1.0462539
4: -1.0633701, 0.8693500, -0.4686782, 0.4791518, -1.5425220, 1.3380282
5: -0.8503823, 0.9127451, -0.4257610, 0.4905806, -1.3409629, 1.3385061
6: -0.8323878, 0.8622503, -0.3718149, 0.5017007, -1.3340886, 1.2340653
7: -0.9543545, 0.9268708, -0.4262078, 0.5346959, -1.4890504, 1.3530786
8: -1.0166404, 0.9484719, -0.4888583, 0.5811551, -1.5977955, 1.4373301
9: -0.9048309, 1.0035747, -0.4799495, 0.4924874, -1.3973184, 1.4835242

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6293055, upper bound: 7.4238101
time: 3.83 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6294149, upper bound: 7.4420255
time: 3.45 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6240726, 1.3434670, -2.5611894, 2.0513279, -3.6754005, 3.9046564
1: -1.3315597, 1.2480187, -2.0364418, 1.9174681, -3.2490277, 3.2844605
2: -1.6417361, 1.3585764, -3.0912013, 1.6502118, -3.2919478, 4.4497776
3: -1.6529676, 1.1216012, -2.7892103, 1.6188232, -3.2717907, 3.9108114
4: -1.9091322, 1.3627229, -3.0788908, 2.0503390, -3.9594712, 4.4416137
5: -1.4943283, 1.3903141, -2.3653483, 2.0939534, -3.5882816, 3.7556624
6: -1.4967098, 1.3748986, -2.4584308, 2.1190734, -3.6157832, 3.8333292
7: -1.7411511, 1.4582176, -2.8043394, 2.1896694, -3.9308205, 4.2625570
8: -1.9237050, 1.4178429, -3.2000084, 2.0574296, -3.9811344, 4.6178513
9: -1.5208721, 1.6871395, -2.3791962, 2.6529346, -4.1738067, 4.0663357

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315136, upper bound: 7.6311990
time: 2.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315409, upper bound: 7.6313845
time: 3.66 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.6027346, 2.0802090, -2.8696129, 2.2713099, -4.8740444, 4.9498219
1: -2.0643280, 1.9450513, -2.2473054, 2.1226027, -4.1869307, 4.1923566
2: -3.1670213, 1.6637118, -3.5330350, 1.7515802, -4.9186015, 5.1967468
3: -2.8332653, 1.6402210, -3.1437230, 1.7713864, -4.6046515, 4.7839441
4: -3.1266012, 2.0799649, -3.4531260, 2.2621088, -5.3887100, 5.5330906
5: -2.4012480, 2.1384859, -2.6349540, 2.3312769, -4.7325249, 4.7734399
6: -2.4963412, 2.1501756, -2.7461295, 2.3565311, -4.8528724, 4.8963051
7: -2.8478332, 2.2213492, -3.1390204, 2.4175413, -5.2653742, 5.3603697
8: -3.2487485, 2.0868356, -3.5804396, 2.2638538, -5.5126023, 5.6672754
9: -2.4127998, 2.6921189, -2.6388662, 2.9542093, -5.3670092, 5.3309851

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319107, upper bound: 7.6312999
time: 3.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319687, upper bound: 7.6315047
time: 2.81 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -3.5589991, 2.7780263, -0.3679520, 0.4299458, -3.9889450, 3.1459785
1: -2.7609611, 2.5787771, -0.4111218, 0.4376045, -3.1985655, 2.9898989
2: -4.5058522, 2.0404634, 0.2604243, 1.0684789, -5.5743313, 1.7800391
3: -3.9786603, 2.1189060, -0.2588759, 0.4866097, -4.4652700, 2.3777819
4: -4.2717552, 2.7448380, -0.4452316, 0.4492521, -4.7210073, 3.1900697
5: -3.2518902, 2.8606019, -0.4016528, 0.4615302, -3.7134204, 3.2622547
6: -3.4003422, 2.8951626, -0.3490357, 0.4775978, -3.8779399, 3.2441983
7: -3.8963063, 2.9406056, -0.4022073, 0.5042163, -4.4005227, 3.3428130
8: -4.4369383, 2.7339518, -0.4644707, 0.5555166, -4.9924550, 3.1984224
9: -3.2147672, 3.6201253, -0.4568843, 0.4585291, -3.6732965, 4.0770097

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6089962, upper bound: 7.4237736
time: 2.99 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6200372, upper bound: 7.4419872
time: 2.80 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.4387751, 3.4268723, -0.3849972, 0.4492262, -4.8880014, 3.8118694
1: -3.4659214, 3.1547296, -0.4267410, 0.4539097, -3.9198310, 3.5814705
2: -5.7159557, 2.4193392, 0.2328508, 1.0725321, -6.7884879, 2.1864884
3: -5.0219154, 2.5579576, -0.2721821, 0.4998149, -5.5217304, 2.8301399
4: -5.3099284, 3.3515816, -0.4610519, 0.4695930, -5.7795215, 3.8126335
5: -4.0128536, 3.5236270, -0.4183160, 0.4811420, -4.4939957, 3.9419432
6: -4.2143545, 3.5845549, -0.3646470, 0.4938565, -4.7082109, 3.9492018
7: -4.8394384, 3.6060505, -0.4184621, 0.5249913, -5.3644300, 4.0245128
8: -5.5272703, 3.3279989, -0.4807488, 0.5730268, -6.1002970, 3.8087478
9: -3.9385438, 4.4950681, -0.4724032, 0.4818683, -4.4204121, 4.9674711

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6285847, upper bound: 7.4237860
time: 5.21 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287830, upper bound: 7.4420011
time: 5.75 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.4747863, 4.1956425, -2.5134125, 2.0175185, -7.4923048, 6.7090549
1: -4.2982330, 3.8287520, -2.0037293, 1.8853480, -6.1835809, 5.8324814
2: -7.1613092, 2.8987954, -3.0233560, 1.6344966, -8.7958059, 5.9221516
3: -6.2395878, 3.0862188, -2.7364335, 1.5950335, -7.8346214, 5.8226523
4: -6.5063496, 4.0839391, -3.0209806, 2.0171900, -8.5235395, 7.1049194
5: -4.9062939, 4.3081031, -2.3234725, 2.0581331, -6.9644270, 6.6315756
6: -5.1734214, 4.3960547, -2.4139771, 2.0824580, -7.2558794, 6.8100319
7: -5.9546199, 4.3898392, -2.7523909, 2.1545715, -8.1091919, 7.1422300
8: -6.8181944, 4.0572567, -3.1409383, 2.0256248, -8.8438187, 7.1981950
9: -4.7843189, 5.5114660, -2.3387520, 2.6069543, -7.3912735, 7.8502178

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309685, upper bound: 7.6311769
time: 2.40 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309910, upper bound: 7.6313600
time: 2.34 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.5732942, 4.9984035, -2.8079538, 2.2269444, -8.8002386, 7.8063574
1: -5.1796169, 4.5439219, -2.2048202, 2.0812802, -7.2608972, 6.7487421
2: -8.6435480, 3.3709190, -3.4461098, 1.7310233, -10.3745708, 6.8170290
3: -7.5380139, 3.6394167, -3.0683122, 1.7407813, -9.2787952, 6.7077289
4: -7.7917151, 4.8401198, -3.3783178, 2.2194538, -10.0111694, 8.2184372
5: -5.8471150, 5.1372399, -2.5808668, 2.2844152, -8.1315308, 7.7181067
6: -6.1933317, 5.2504425, -2.6886446, 2.3084738, -8.5018053, 7.9390869
7: -7.1297507, 5.2142563, -3.0719848, 2.3709774, -9.5007286, 8.2862415
8: -8.1741972, 4.7977457, -3.5041375, 2.2218099, -10.3960075, 8.3018837
9: -5.6849499, 6.5934253, -2.5868192, 2.8926830, -8.5776329, 9.1802444

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314259, upper bound: 7.6312818
time: 6.40 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314853, upper bound: 7.6314852
time: 2.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 10.76 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6286447, upper bound: 7.4237966
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6287959, upper bound: 7.4420110
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6293055, upper bound: 7.4238101
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6294149, upper bound: 7.4420255
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6315136, upper bound: 7.6311990
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6315409, upper bound: 7.6313845
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6319107, upper bound: 7.6312999
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6319687, upper bound: 7.6315047
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6089962, upper bound: 7.4237736
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6200372, upper bound: 7.4419872
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6285847, upper bound: 7.4237860
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6287830, upper bound: 7.4420011
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6309685, upper bound: 7.6311769
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6309910, upper bound: 7.6313600
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6314259, upper bound: 7.6312818
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.76
Output dim: 2, lower bound: -7.6314853, upper bound: 7.6314852

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3215374, 0.3785565, -0.1655774, 0.1786179, -0.5001553, 0.5441339
1: -0.3690412, 0.3930303, -0.2128506, 0.2244805, -0.5935217, 0.6058809
2: 0.3340648, 1.0567021, 0.5842205, 1.0453693, -0.7113044, 0.4724817
3: -0.2226866, 0.4510706, -0.0841027, 0.2959728, -0.5186595, 0.5351733
4: -0.4020062, 0.3943269, -0.2298062, 0.2244284, -0.6264346, 0.6241331
5: -0.3577445, 0.4080103, -0.2075235, 0.2276342, -0.5853786, 0.6155338
6: -0.3073493, 0.4330929, -0.1673270, 0.2388721, -0.5462214, 0.6004199
7: -0.3579624, 0.4483341, -0.1962765, 0.2546783, -0.6126407, 0.6446106
8: -0.4192457, 0.5084815, -0.2243754, 0.3214177, -0.7406634, 0.7328569
9: -0.4142912, 0.3965317, -0.2364042, 0.2089900, -0.6232812, 0.6329359

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3278260, upper bound: 7.0120853
time: 5.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2280558, upper bound: 7.0119845
time: 3.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3260155, 0.3836042, -0.1901525, 0.2063102, -0.5323257, 0.5737567
1: -0.3731359, 0.3973123, -0.2399131, 0.2529218, -0.6260577, 0.6372254
2: 0.3268425, 1.0577728, 0.5435508, 1.0462335, -0.7193910, 0.5142220
3: -0.2261810, 0.4545308, -0.1065318, 0.3254724, -0.5516534, 0.5610626
4: -0.4061590, 0.3996609, -0.2587124, 0.2485973, -0.6547563, 0.6583732
5: -0.3621021, 0.4131572, -0.2328232, 0.2589746, -0.6210767, 0.6459804
6: -0.3114386, 0.4373646, -0.1857919, 0.2771498, -0.5885884, 0.6231564
7: -0.3622308, 0.4537770, -0.2199289, 0.2859460, -0.6481768, 0.6737059
8: -0.4235272, 0.5130725, -0.2591266, 0.3541500, -0.7776772, 0.7721992
9: -0.4183669, 0.4026374, -0.2667136, 0.2394815, -0.6578484, 0.6693509

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3380506, upper bound: 7.0288897
time: 3.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2380407, upper bound: 7.0287932
time: 6.50 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4921039, 0.5650793, -0.1671329, 0.1802153, -0.6723191, 0.7322122
1: -0.5289126, 0.5527412, -0.2145346, 0.2263891, -0.7553017, 0.7672758
2: 0.0643053, 1.0998832, 0.5815797, 1.0453516, -0.9810463, 0.5183035
3: -0.3522720, 0.5818375, -0.0853313, 0.2980142, -0.6502862, 0.6671689
4: -0.5708414, 0.5884474, -0.2316887, 0.2257293, -0.7965707, 0.8201361
5: -0.5220425, 0.6096491, -0.2092742, 0.2295879, -0.7516304, 0.8189233
6: -0.4672987, 0.5908822, -0.1685476, 0.2411903, -0.7084889, 0.7594298
7: -0.5257089, 0.6498010, -0.1975447, 0.2567992, -0.7825081, 0.8473457
8: -0.5927641, 0.6878866, -0.2263505, 0.3236338, -0.9163979, 0.9142370
9: -0.5707085, 0.6236483, -0.2383775, 0.2107961, -0.7815046, 0.8620258

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3758649, upper bound: 7.0121016
time: 3.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2622634, upper bound: 7.0119943
time: 4.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5505497, 0.6035465, -0.2087835, 0.2307984, -0.7813481, 0.8123299
1: -0.5662493, 0.5851125, -0.2580642, 0.2748550, -0.8411042, 0.8431766
2: -0.0169831, 1.1109116, 0.5157579, 1.0462120, -1.0631950, 0.5951537
3: -0.4047529, 0.6095151, -0.1246156, 0.3462098, -0.7509627, 0.7341306
4: -0.6348512, 0.6277246, -0.2792440, 0.2672769, -0.9021281, 0.9069686
5: -0.5581580, 0.6628464, -0.2513576, 0.2799813, -0.8381393, 0.9142039
6: -0.5161189, 0.6252654, -0.2023629, 0.3020389, -0.8181578, 0.8276283
7: -0.5826356, 0.6913940, -0.2400438, 0.3074629, -0.8900986, 0.9314378
8: -0.6523023, 0.7245587, -0.2853622, 0.3759942, -1.0282965, 1.0099208
9: -0.6140503, 0.6774718, -0.2891697, 0.2616687, -0.8757190, 0.9666415

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3872771, upper bound: 7.0289056
time: 3.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2758285, upper bound: 7.0288028
time: 4.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.0956531, 0.9566654, -1.2009926, 1.0301166, -2.1257696, 2.1576581
1: -0.9337584, 0.8934159, -1.0100741, 0.9581370, -1.8918953, 1.9034901
2: -0.8110871, 1.2148153, -0.9754078, 1.2378676, -2.0489547, 2.1902232
3: -1.0121615, 0.8501461, -1.1392158, 0.9014394, -1.9136009, 1.9893619
4: -1.2694221, 0.9897659, -1.3962607, 1.0601850, -2.3296070, 2.3860266
5: -1.0051379, 1.0237787, -1.1012683, 1.0928749, -2.0980129, 2.1250470
6: -0.9914132, 0.9800230, -1.0916661, 1.0550939, -2.0465071, 2.0716891
7: -1.1477908, 1.0535733, -1.2657177, 1.1291806, -2.2769713, 2.3192911
8: -1.2098949, 1.0649045, -1.3418822, 1.1332012, -2.3430963, 2.4067867
9: -1.0522729, 1.1554426, -1.1393110, 1.2582780, -2.3105509, 2.2947536

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459815, upper bound: 7.6283250
time: 3.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459815, upper bound: 7.6311681
time: 3.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.1488414, 0.9928353, -1.5339347, 1.2759864, -2.4248278, 2.5267701
1: -0.9724360, 0.9268545, -1.2617304, 1.1808023, -2.1532383, 2.1885848
2: -0.8927552, 1.2267164, -1.5018319, 1.3291522, -2.2219074, 2.7285483
3: -1.0762475, 0.8760806, -1.5441580, 1.0736284, -2.1498759, 2.4202385
4: -1.3337193, 1.0261621, -1.7992783, 1.2947860, -2.6285052, 2.8254404
5: -1.0531402, 1.0587689, -1.4100418, 1.3249989, -2.3781390, 2.4688106
6: -1.0410404, 1.0177692, -1.4129441, 1.3044155, -2.3454559, 2.4307132
7: -1.2073554, 1.0927880, -1.6390570, 1.3843610, -2.5917163, 2.7318449
8: -1.2735596, 1.0998133, -1.7980084, 1.3548675, -2.6284270, 2.8978219
9: -1.0970556, 1.2054052, -1.4364104, 1.5963377, -2.6933932, 2.6418157

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
time: 4.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5517384, upper bound: 7.4419719
time: 4.20 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.9732730, 1.6038033, -1.4180212, 1.1890336, -3.1623068, 3.0218244
1: -1.5965267, 1.4923103, -1.1727695, 1.1016750, -2.6982017, 2.6650798
2: -2.1917653, 1.4594777, -1.3183019, 1.2962567, -3.4880219, 2.7777796
3: -2.0742850, 1.3075709, -1.4024230, 1.0132606, -3.0875456, 2.7099938
4: -2.3261719, 1.6167358, -1.6583762, 1.2114739, -3.5376458, 3.2751122
5: -1.8212241, 1.6484563, -1.3012912, 1.2423633, -3.0635874, 2.9497476
6: -1.8570006, 1.6461083, -1.3009728, 1.2157518, -3.0727525, 2.9470811
7: -2.1325765, 1.7308207, -1.5070120, 1.2953182, -3.4278946, 3.2378325
8: -2.4028466, 1.6540153, -1.6387653, 1.2756732, -3.6785197, 3.2927806
9: -1.8369036, 2.0462775, -1.3308322, 1.4771473, -3.3140509, 3.3771098

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391196, upper bound: 7.6282341
time: 3.21 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391196, upper bound: 7.6313000
time: 5.06 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -2.1051967, 1.7020730, -1.8130972, 1.4855680, -3.5907648, 3.5151701
1: -1.6964428, 1.5865093, -1.4742785, 1.3773447, -3.0737877, 3.0607877
2: -2.3994079, 1.4982717, -1.9406636, 1.4102278, -3.8096356, 3.4389353
3: -2.2362418, 1.3777044, -1.8830147, 1.2208179, -3.4570599, 3.2607191
4: -2.4838617, 1.7124313, -2.1357846, 1.4981339, -3.9819956, 3.8482161
5: -1.9441192, 1.7480804, -1.6717939, 1.5273182, -3.4714375, 3.4198742
6: -1.9940128, 1.7491753, -1.6923141, 1.5212839, -3.5152967, 3.4414895
7: -2.2790396, 1.8346171, -1.9555091, 1.6024905, -3.8815303, 3.7901263
8: -2.5833616, 1.7426432, -2.1839557, 1.5447130, -4.1280746, 3.9265990
9: -1.9564302, 2.1810212, -1.6915098, 1.8835993, -3.8400295, 3.8725309

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
time: 4.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
time: 2.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.6776330, 2.8633990, -0.1634205, 0.1764030, -3.8540361, 3.0268195
1: -2.8502903, 2.6549621, -0.2105156, 0.2218339, -3.0721242, 2.8654776
2: -4.6704416, 2.0901721, 0.5878826, 1.0447383, -5.7151799, 1.5022895
3: -4.1177406, 2.1732159, -0.0823990, 0.2931424, -4.4108829, 2.2556150
4: -4.4131274, 2.8189101, -0.2271958, 0.2226245, -4.6357517, 3.0461059
5: -3.3544755, 2.9570847, -0.2050961, 0.2249250, -3.5794005, 3.1621807
6: -3.5080936, 2.9867780, -0.1656345, 0.2356572, -3.7437508, 3.1524124
7: -4.0226574, 3.0261099, -0.1945181, 0.2517378, -4.2743950, 3.2206280
8: -4.5789042, 2.8084664, -0.2216369, 0.3183449, -4.8972492, 3.0301034
9: -3.3112485, 3.7392657, -0.2336679, 0.2064860, -3.5177345, 3.9729335

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3453898, upper bound: 7.0120707
time: 3.48 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2414656, upper bound: 7.0119753
time: 4.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.8581655, 2.9957039, -0.2035037, 0.2232151, -4.0813808, 3.1992078
1: -2.9952476, 2.7734189, -0.2528629, 0.2685096, -3.2637572, 3.0262818
2: -4.9156423, 2.1640007, 0.5239875, 1.0458157, -5.9614582, 1.6400132
3: -4.3332272, 2.2631242, -0.1196031, 0.3402539, -4.6734810, 2.3827274
4: -4.6276579, 2.9428308, -0.2736076, 0.2612221, -4.8888798, 3.2164383
5: -3.5100358, 3.0905473, -0.2457275, 0.2742047, -3.7842405, 3.3362749
6: -3.6757510, 3.1281235, -0.1973287, 0.2949744, -3.9707253, 3.3254523
7: -4.2159624, 3.1625907, -0.2339984, 0.3014715, -4.5174341, 3.3965890
8: -4.8029947, 2.9293351, -0.2778364, 0.3696542, -5.1726489, 3.2071714
9: -3.4604352, 3.9192615, -0.2825514, 0.2553706, -3.7158058, 4.2018127

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3575303, upper bound: 7.0288756
time: 4.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2567094, upper bound: 7.0287855
time: 3.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.7718534, 3.6773858, -1.1628827, 1.0031489, -5.7750025, 4.8402686
1: -3.7295480, 3.3703914, -0.9822426, 0.9331744, -4.6627226, 4.3526340
2: -6.2022815, 2.5954933, -0.9160359, 1.2279017, -7.4301834, 3.5115292
3: -5.4078465, 2.7274580, -1.0935287, 0.8823293, -6.2901759, 3.8209867
4: -5.6844730, 3.5928669, -1.3500848, 1.0339899, -6.7184629, 4.9429517
5: -4.3040648, 3.7830396, -1.0665252, 1.0674635, -5.3715281, 4.8495646
6: -4.5192666, 3.8471582, -1.0561177, 1.0272465, -5.5465131, 4.9032760
7: -5.2007318, 3.8579092, -1.2233033, 1.1003927, -6.3011246, 5.0812125
8: -5.9438753, 3.5751503, -1.2902727, 1.1081519, -7.0520272, 4.8654232
9: -4.2081399, 4.8166890, -1.1070288, 1.2204630, -5.4286032, 5.9237180

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
time: 4.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311509
time: 4.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.8307295, 3.7204826, -1.4929535, 1.2450616, -6.0757914, 5.2134361
1: -3.7762842, 3.4090035, -1.2301675, 1.1522005, -4.9284849, 4.6391711
2: -6.2812252, 2.6188145, -1.4371502, 1.3172511, -7.5984764, 4.0559645
3: -5.4783545, 2.7568951, -1.4943238, 1.0521894, -6.5305438, 4.2512188
4: -5.7546797, 3.6332157, -1.7495033, 1.2649043, -7.0195837, 5.3827190
5: -4.3544202, 3.8268292, -1.3713409, 1.2955949, -5.6500149, 5.1981702
6: -4.5738249, 3.8931198, -1.3735831, 1.2727892, -5.8466139, 5.2667027
7: -5.2637720, 3.9021881, -1.5921925, 1.3525438, -6.6163158, 5.4943805
8: -6.0169573, 3.6140194, -1.7415417, 1.3267570, -7.3437142, 5.3555613
9: -4.2567363, 4.8756843, -1.3987327, 1.5542634, -5.8109999, 6.2744169

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
time: 2.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313076
time: 2.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.7915745, 4.4235950, -1.3771781, 1.1589034, -6.9504776, 5.8007731
1: -4.5464697, 4.0345554, -1.1416976, 1.0740740, -5.6205435, 5.1762533
2: -7.5860028, 3.0361753, -1.2538208, 1.2846625, -8.8706656, 4.2899961
3: -6.6116295, 3.2410007, -1.3523948, 0.9921165, -7.6037459, 4.5933952
4: -6.8767323, 4.2952223, -1.6088462, 1.1824472, -8.0591793, 5.9040685
5: -5.1787548, 4.5547438, -1.2636000, 1.2135365, -6.3922911, 5.8183436
6: -5.4647703, 4.6403966, -1.2616413, 1.1848114, -6.6495819, 5.9020376
7: -6.2922392, 4.6238313, -1.4614695, 1.2636461, -7.5558853, 6.0853009
8: -7.2029243, 4.2633080, -1.5828717, 1.2480716, -8.4509954, 5.8461800
9: -5.0431585, 5.8210378, -1.2941175, 1.4359045, -6.4790630, 7.1151552

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
time: 6.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312817
time: 3.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.9926887, 4.5704522, -1.7683985, 1.4520612, -7.4447498, 6.3388510
1: -4.7087898, 4.1658034, -1.4402798, 1.3452897, -6.0540795, 5.6060834
2: -7.8552809, 3.1182122, -1.8708315, 1.3969896, -9.2522707, 4.9890437
3: -6.8510852, 3.3427217, -1.8287210, 1.1967047, -8.0477896, 5.1714430
4: -7.1139984, 4.4339471, -2.0820878, 1.4649547, -8.5789528, 6.5160351
5: -5.3504949, 4.7038383, -1.6293851, 1.4950441, -6.8455391, 6.3332233
6: -5.6523695, 4.7972584, -1.6457657, 1.4859622, -7.1383314, 6.4430242
7: -6.5074272, 4.7751551, -1.9053166, 1.5667646, -8.0741920, 6.6804714
8: -7.4520969, 4.3985987, -2.1217227, 1.5141705, -8.9662676, 6.5203214
9: -5.2086983, 6.0206003, -1.6505555, 1.8376806, -7.0463791, 7.6711559

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287827
time: 2.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314848
time: 3.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.39 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.3278260, upper bound: 7.0120853
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.2280558, upper bound: 7.0119845
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.3380506, upper bound: 7.0288897
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.2380407, upper bound: 7.0287932
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.3758649, upper bound: 7.0121016
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.2622634, upper bound: 7.0119943
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.3872771, upper bound: 7.0289056
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.2758285, upper bound: 7.0288028
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.5459815, upper bound: 7.6283250
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.5459815, upper bound: 7.6311681
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.5517384, upper bound: 7.4419719
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.5391196, upper bound: 7.6282341
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.5391196, upper bound: 7.6313000
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.3453898, upper bound: 7.0120707
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.2414656, upper bound: 7.0119753
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.3575303, upper bound: 7.0288756
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.2567094, upper bound: 7.0287855
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311509
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313076
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312817
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287827
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.39
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314848

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0999603, 0.1069386, -1.2009926, 1.0301166, -1.1300769, 1.3079312
1: -0.1372842, 0.1415909, -1.0100741, 0.9581370, -1.0954212, 1.1516651
2: 0.7027340, 1.0396814, -0.9754078, 1.2378676, -0.5351336, 2.0150893
3: -0.0289697, 0.2068110, -1.1392158, 0.9014394, -0.9304091, 1.3460268
4: -0.1513371, 0.1660503, -1.3962607, 1.0601850, -1.2115221, 1.5623109
5: -0.1289959, 0.1403148, -1.1012683, 1.0928749, -1.2218708, 1.2415831
6: -0.1125522, 0.1390101, -1.0916661, 1.0550939, -1.1676461, 1.2306762
7: -0.1393690, 0.1629245, -1.2657177, 1.1291806, -1.2685496, 1.4286423
8: -0.1359288, 0.2251918, -1.3418822, 1.1332012, -1.2691300, 1.5670741
9: -0.1500749, 0.1279543, -1.1393110, 1.2582780, -1.4083530, 1.2672652

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6283250
time: 7.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6283250
time: 8.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.5137417, 0.5817116, -1.2009926, 1.0301166, -1.5438583, 1.7827042
1: -0.5447940, 0.5669526, -1.0100741, 0.9581370, -1.5029309, 1.5770267
2: 0.0311265, 1.1043226, -0.9754078, 1.2378676, -1.2067411, 2.0797305
3: -0.3708899, 0.5945812, -1.1392158, 0.9014394, -1.2723293, 1.7337971
4: -0.5962261, 0.6047728, -1.3962607, 1.0601850, -1.6564111, 2.0010335
5: -0.5363327, 0.6336310, -1.1012683, 1.0928749, -1.6292076, 1.7348993
6: -0.4864335, 0.6049811, -1.0916661, 1.0550939, -1.5415274, 1.6966472
7: -0.5493956, 0.6686618, -1.2657177, 1.1291806, -1.6785762, 1.9343796
8: -0.6184307, 0.7035065, -1.3418822, 1.1332012, -1.7516320, 2.0453887
9: -0.5875289, 0.6464775, -1.1393110, 1.2582780, -1.8458070, 1.7857884

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6311681
time: 3.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5459817, upper bound: 7.4237516
time: 4.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0997177, 0.1066720, -1.5339347, 1.2759864, -1.3757041, 1.6406066
1: -0.1370031, 0.1412840, -1.2617304, 1.1808023, -1.3178054, 1.4030144
2: 0.7031748, 1.0396798, -1.5018319, 1.3291522, -0.6259775, 2.5415115
3: -0.0287646, 0.2064803, -1.5441580, 1.0736284, -1.1023930, 1.7506384
4: -0.1510483, 0.1658331, -1.7992783, 1.2947860, -1.4458343, 1.9651114
5: -0.1287037, 0.1399902, -1.4100418, 1.3249989, -1.4537026, 1.5500320
6: -0.1123485, 0.1386405, -1.4129441, 1.3044155, -1.4167640, 1.5515846
7: -0.1391573, 0.1625858, -1.6390570, 1.3843610, -1.5235183, 1.8016429
8: -0.1355999, 0.2248350, -1.7980084, 1.3548675, -1.4904673, 2.0228434
9: -0.1497552, 0.1276529, -1.4364104, 1.5963377, -1.7460929, 1.5640633

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
time: 2.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
time: 3.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1481730, 0.1599290, -1.4180212, 1.1890336, -1.3372066, 1.5779502
1: -0.1931481, 0.2025903, -1.1727695, 1.1016750, -1.2948232, 1.3753598
2: 0.6151205, 1.0425050, -1.3183019, 1.2962567, -0.6811361, 2.3608069
3: -0.0697279, 0.2725243, -1.4024230, 1.0132606, -1.0829885, 1.6749474
4: -0.2087196, 0.2092075, -1.6583762, 1.2114739, -1.4201934, 1.8675838
5: -0.1870432, 0.2048456, -1.3012912, 1.2423633, -1.4294065, 1.5061369
6: -0.1530457, 0.2124394, -1.3009728, 1.2157518, -1.3687974, 1.5134122
7: -0.1814390, 0.2302229, -1.5070120, 1.2953182, -1.4767573, 1.7372348
8: -0.2012803, 0.2961308, -1.6387653, 1.2756732, -1.4769534, 1.9348962
9: -0.2136270, 0.1878617, -1.3308322, 1.4771473, -1.6907743, 1.5186939

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6282343
time: 3.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6282341
time: 3.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.3306952, 1.1253389, -1.4180212, 1.1890336, -2.5197287, 2.5433602
1: -1.1061755, 1.0433774, -1.1727695, 1.1016750, -2.2078505, 2.2161469
2: -1.1810069, 1.2723873, -1.3183019, 1.2962567, -2.4772635, 2.5906892
3: -1.2967119, 0.9677966, -1.4024230, 1.0132606, -2.3099725, 2.3702197
4: -1.5545385, 1.1491244, -1.6583762, 1.2114739, -2.7660124, 2.8075006
5: -1.2197224, 1.1831168, -1.3012912, 1.2423633, -2.4620857, 2.4844079
6: -1.2164470, 1.1506547, -1.3009728, 1.2157518, -2.4321988, 2.4516275
7: -1.4102179, 1.2283047, -1.5070120, 1.2953182, -2.7055361, 2.7353168
8: -1.5208342, 1.2165208, -1.6387653, 1.2756732, -2.7965074, 2.8552861
9: -1.2542810, 1.3892401, -1.3308322, 1.4771473, -2.7314284, 2.7200723

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5391198, upper bound: 7.4237475
time: 3.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6313000
time: 3.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1587485, 0.1715083, -1.8130972, 1.4855680, -1.6443166, 1.9846056
1: -0.2053554, 0.2160324, -1.4742785, 1.3773447, -1.5827001, 1.6903110
2: 0.5959755, 1.0424958, -1.9406636, 1.4102278, -0.8142523, 2.9831595
3: -0.0786342, 0.2869375, -1.8830147, 1.2208179, -1.2994521, 2.1699522
4: -0.2215261, 0.2186382, -2.1357846, 1.4981339, -1.7196600, 2.3544228
5: -0.1997317, 0.2189467, -1.6717939, 1.5273182, -1.7270499, 1.8907406
6: -0.1618941, 0.2286291, -1.6923141, 1.5212839, -1.6831779, 1.9209433
7: -0.1906322, 0.2452627, -1.9555091, 1.6024905, -1.7931228, 2.2007718
8: -0.2155851, 0.3116325, -2.1839557, 1.5447130, -1.7602981, 2.4955883
9: -0.2276510, 0.2009524, -1.6915098, 1.8835993, -2.1112502, 1.8924623

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
time: 3.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
time: 6.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.4588451, 1.2203939, -1.8130972, 1.4855680, -2.9444132, 3.0334911
1: -1.2039182, 1.1288085, -1.4742785, 1.3773447, -2.5812631, 2.6030869
2: -1.3840015, 1.3078859, -1.9406636, 1.4102278, -2.7942293, 3.2485495
3: -1.4534780, 1.0339497, -1.8830147, 1.2208179, -2.6742959, 2.9169645
4: -1.7099203, 1.2396287, -2.1357846, 1.4981339, -3.2080541, 3.3754134
5: -1.3381892, 1.2731776, -1.6717939, 1.5273182, -2.8655076, 2.9449716
6: -1.3403305, 1.2472912, -1.6923141, 1.5212839, -2.8616142, 2.9396052
7: -1.5539688, 1.3266928, -1.9555091, 1.6024905, -3.1564593, 3.2822018
8: -1.6963817, 1.3024652, -2.1839557, 1.5447130, -3.2410946, 3.4864209
9: -1.3689091, 1.5193797, -1.6915098, 1.8835993, -3.2525084, 3.2108896

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
time: 2.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
time: 3.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.0393264, 0.9178565, -1.1628827, 1.0031489, -2.0424752, 2.0807393
1: -0.8928415, 0.8542848, -0.9822426, 0.9331744, -1.8260159, 1.8365273
2: -0.7276485, 1.1981939, -0.9160359, 1.2279017, -1.9555502, 2.1142297
3: -0.9435567, 0.8207595, -1.0935287, 0.8823293, -1.8258860, 1.9142883
4: -1.1971776, 0.9476389, -1.3500848, 1.0339899, -2.2311676, 2.2977238
5: -0.9528845, 0.9844075, -1.0665252, 1.0674635, -2.0203481, 2.0509326
6: -0.9440743, 0.9422595, -1.0561177, 1.0272465, -1.9713209, 1.9983771
7: -1.0863340, 1.0054618, -1.2233033, 1.1003927, -2.1867266, 2.2287650
8: -1.1422608, 1.0249797, -1.2902727, 1.1081519, -2.2504127, 2.3152523
9: -0.9996936, 1.1036472, -1.1070288, 1.2204630, -2.2201567, 2.2106762

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
time: 4.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
time: 2.53 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -3.7919588, 2.9467027, -1.1628827, 1.0031489, -4.7951078, 4.1095853
1: -2.9406590, 2.7310686, -0.9822426, 0.9331744, -3.8738334, 3.7133112
2: -4.8264527, 2.1360939, -0.9160359, 1.2279017, -6.0543547, 3.0521297
3: -4.2560358, 2.2299223, -1.0935287, 0.8823293, -5.1383653, 3.3234510
4: -4.5482984, 2.9002113, -1.3500848, 1.0339899, -5.5822883, 4.2502961
5: -3.4544127, 3.0389261, -1.0665252, 1.0674635, -4.5218763, 4.1054516
6: -3.6136959, 3.0776973, -1.0561177, 1.0272465, -4.6409426, 4.1338148
7: -4.1456871, 3.1120200, -1.2233033, 1.1003927, -5.2460799, 4.3353233
8: -4.7225423, 2.8873305, -1.2902727, 1.1081519, -5.8306942, 4.1776032
9: -3.4062939, 3.8535600, -1.1070288, 1.2204630, -4.6267567, 4.9605889

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311510
time: 3.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311510
time: 3.53 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.0531604, 0.9272488, -1.4929535, 1.2450616, -2.2982221, 2.4202023
1: -0.9028684, 0.8625062, -1.2301675, 1.1522005, -2.0550690, 2.0926738
2: -0.7486366, 1.2010515, -1.4371502, 1.3172511, -2.0658877, 2.6382017
3: -0.9602711, 0.8273244, -1.4943238, 1.0521894, -2.0124605, 2.3216481
4: -1.2139385, 0.9570729, -1.7495033, 1.2649043, -2.4788427, 2.7065761
5: -0.9654055, 0.9935039, -1.3713409, 1.2955949, -2.2610004, 2.3648448
6: -0.9571872, 0.9516574, -1.3735831, 1.2727892, -2.2299764, 2.3252406
7: -1.1019704, 1.0151452, -1.5921925, 1.3525438, -2.4545143, 2.6073377
8: -1.1584860, 1.0340253, -1.7415417, 1.3267570, -2.4852428, 2.7755671
9: -1.0113906, 1.1161186, -1.3987327, 1.5542634, -2.5656538, 2.5148511

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
time: 3.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
time: 2.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -3.8652687, 3.0007892, -1.4929535, 1.2450616, -5.1103306, 4.4937429
1: -2.9992514, 2.7792504, -1.2301675, 1.1522005, -4.1514521, 4.0094180
2: -4.9265528, 2.1662290, -1.4371502, 1.3172511, -6.2438040, 3.6033792
3: -4.3435698, 2.2665625, -1.4943238, 1.0521894, -5.3957591, 3.7608862
4: -4.6355057, 2.9509566, -1.7495033, 1.2649043, -5.9004097, 4.7004600
5: -3.5174999, 3.0933805, -1.3713409, 1.2955949, -4.8130951, 4.4647212
6: -3.6814132, 3.1353431, -1.3735831, 1.2727892, -4.9542027, 4.5089264
7: -4.2242031, 3.1676087, -1.5921925, 1.3525438, -5.5767469, 4.7598014
8: -4.8138123, 2.9366531, -1.7415417, 1.3267570, -6.1405692, 4.6781950
9: -3.4669244, 3.9269178, -1.3987327, 1.5542634, -5.0211878, 5.3256502

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313079
time: 2.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313078
time: 3.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.4499955, 1.2108288, -1.3771781, 1.1589034, -2.6088989, 2.5880070
1: -1.1970177, 1.1183782, -1.1416976, 1.0740740, -2.2710917, 2.2600758
2: -1.3697586, 1.3010416, -1.2538208, 1.2846625, -2.6544211, 2.5548625
3: -1.4400407, 1.0272079, -1.3523948, 0.9921165, -2.4321570, 2.3796027
4: -1.6930354, 1.2307545, -1.6088462, 1.1824472, -2.8754826, 2.8396006
5: -1.3290328, 1.2621032, -1.2636000, 1.2135365, -2.5425692, 2.5257032
6: -1.3356463, 1.2411077, -1.2616413, 1.1848114, -2.5204577, 2.5027490
7: -1.5448251, 1.3125018, -1.4614695, 1.2636461, -2.8084712, 2.7739713
8: -1.6804488, 1.2949226, -1.5828717, 1.2480716, -2.9285202, 2.8777943
9: -1.3553064, 1.5072877, -1.2941175, 1.4359045, -2.7912109, 2.8014052

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281545
time: 3.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
time: 7.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.8693132, 3.7401671, -1.3771781, 1.1589034, -6.0282164, 5.1173453
1: -3.8040197, 3.4359655, -1.1416976, 1.0740740, -4.8780937, 4.5776634
2: -6.3038125, 2.6092558, -1.2538208, 1.2846625, -7.5884752, 3.8630767
3: -5.5297933, 2.7702694, -1.3523948, 0.9921165, -6.5219097, 4.1226645
4: -5.8126616, 3.6458583, -1.6088462, 1.1824472, -6.9951086, 5.2547045
5: -4.3859935, 3.8539329, -1.2636000, 1.2135365, -5.5995302, 5.1175327
6: -4.6104403, 3.9202943, -1.2616413, 1.1848114, -5.7952518, 5.1819353
7: -5.3008614, 3.9267373, -1.4614695, 1.2636461, -6.5645075, 5.3882070
8: -6.0561352, 3.6161757, -1.5828717, 1.2480716, -7.3042068, 5.1990471
9: -4.2908692, 4.9178047, -1.2941175, 1.4359045, -5.7267737, 6.2119222

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312818
time: 2.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312818
time: 3.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.5572910, 1.2917411, -1.7683985, 1.4520612, -3.0093522, 3.0601397
1: -1.2789603, 1.1921618, -1.4402798, 1.3452897, -2.6242499, 2.6324415
2: -1.5395666, 1.3316808, -1.8708315, 1.3969896, -2.9365563, 3.2025123
3: -1.5709763, 1.0829575, -1.8287210, 1.1967047, -2.7676811, 2.9116786
4: -1.8234650, 1.3076975, -2.0820878, 1.4649547, -3.2884197, 3.3897853
5: -1.4298999, 1.3392417, -1.6293851, 1.4950441, -2.9249439, 2.9686270
6: -1.4390669, 1.3236289, -1.6457657, 1.4859622, -2.9250290, 2.9693947
7: -1.6675868, 1.3947434, -1.9053166, 1.5667646, -3.2343514, 3.3000600
8: -1.8282120, 1.3676143, -2.1217227, 1.5141705, -3.3423824, 3.4893370
9: -1.4533160, 1.6178205, -1.6505555, 1.8376806, -3.2909966, 3.2683759

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
time: 2.31 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
time: 4.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -5.0666490, 3.8846729, -1.7683985, 1.4520612, -6.5187101, 5.6530714
1: -3.9632299, 3.5649397, -1.4402798, 1.3452897, -5.3085194, 5.0052195
2: -6.5705166, 2.6902735, -1.8708315, 1.3969896, -7.9675064, 4.5611048
3: -5.7645931, 2.8702362, -1.8287210, 1.1967047, -6.9612980, 4.6989574
4: -6.0453620, 3.7824497, -2.0820878, 1.4649547, -7.5103168, 5.8645372
5: -4.5549335, 4.0001926, -1.6293851, 1.4950441, -6.0499778, 5.6295776
6: -4.7944660, 4.0743966, -1.6457657, 1.4859622, -6.2804279, 5.7201624
7: -5.5124435, 4.0756145, -1.9053166, 1.5667646, -7.0792084, 5.9809313
8: -6.3009572, 3.7499282, -2.1217227, 1.5141705, -7.8151278, 5.8716507
9: -4.4531837, 5.1136880, -1.6505555, 1.8376806, -6.2908640, 6.7642436

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314849
time: 2.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314849
time: 2.12 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.26 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6283250
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6283250
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6311681
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5459817, upper bound: 7.4237516
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6282343
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6282341
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5391198, upper bound: 7.4237475
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6313000
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311510
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311510
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313079
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313078
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281545
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312818
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312818
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314849
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.26
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314849

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0999603, 0.1069386, -0.8551247, 0.7983816, -0.8983418, 0.9620633
1: -0.1372842, 0.1415909, -0.7615308, 0.7482750, -0.8855592, 0.9031218
2: 0.7027340, 1.0396814, -0.4506930, 1.1650431, -0.4623091, 1.4903744
3: -0.0289697, 0.2068110, -0.7235693, 0.7370088, -0.7659785, 0.9303803
4: -0.1513371, 0.1660503, -0.9802259, 0.8269940, -0.9783311, 1.1462762
5: -0.1289959, 0.1403148, -0.7923279, 0.8684083, -0.9974041, 0.9326428
6: -0.1125522, 0.1390101, -0.7721305, 0.8151376, -0.9276898, 0.9111406
7: -0.1393690, 0.1629245, -0.8798913, 0.8797811, -1.0191501, 1.0428158
8: -0.1359288, 0.2251918, -0.9418252, 0.9057338, -1.0416626, 1.1670170
9: -0.1500749, 0.1279543, -0.8458427, 0.9452844, -1.0953593, 0.9737970

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3050140, upper bound: 7.2710082
time: 3.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1434126, upper bound: 7.2490772
time: 4.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0999603, 0.1069386, -3.6476877, 2.7839622, -2.8839226, 3.7546263
1: -0.1372842, 0.1415909, -2.8018668, 2.5745392, -2.7118235, 2.9434576
2: 0.7027340, 1.0396814, -4.8289142, 1.8660736, -1.1633396, 5.8685956
3: -0.0289697, 0.2068110, -4.0722418, 2.1583340, -2.1873038, 4.2790527
4: -0.1513371, 0.1660503, -4.3325849, 2.7284336, -2.8797708, 4.4986353
5: -0.1289959, 0.1403148, -3.3557072, 2.7416050, -2.8706009, 3.4960220
6: -0.1125522, 0.1390101, -3.4450753, 2.8407545, -2.9533067, 3.5840852
7: -0.1393690, 0.1629245, -3.9689660, 2.9865489, -3.1259179, 4.1318903
8: -0.1359288, 0.2251918, -4.6700106, 2.7224474, -2.8583763, 4.8952022
9: -0.1500749, 0.1279543, -3.2760303, 3.6902027, -3.8402777, 3.4039845

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3050140, upper bound: 7.2710083
time: 3.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1434126, upper bound: 7.2490772
time: 2.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5137417, 0.5817116, -0.8551247, 0.7983816, -1.3121233, 1.4368362
1: -0.5447940, 0.5669526, -0.7615308, 0.7482750, -1.2930690, 1.3284833
2: 0.0311265, 1.1043226, -0.4506930, 1.1650431, -1.1339166, 1.5550156
3: -0.3708899, 0.5945812, -0.7235693, 0.7370088, -1.1078987, 1.3181505
4: -0.5962261, 0.6047728, -0.9802259, 0.8269940, -1.4232202, 1.5849987
5: -0.5363327, 0.6336310, -0.7923279, 0.8684083, -1.4047409, 1.4259589
6: -0.4864335, 0.6049811, -0.7721305, 0.8151376, -1.3015711, 1.3771117
7: -0.5493956, 0.6686618, -0.8798913, 0.8797811, -1.4291768, 1.5485532
8: -0.6184307, 0.7035065, -0.9418252, 0.9057338, -1.5241646, 1.6453317
9: -0.5875289, 0.6464775, -0.8458427, 0.9452844, -1.5328133, 1.4923202

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292519, upper bound: 7.5765304
time: 4.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6065362, upper bound: 7.5755026
time: 4.34 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0997177, 0.1066720, -1.1652219, 1.0053251, -1.1050427, 1.2718939
1: -0.1370031, 0.1412840, -0.9838070, 0.9347872, -1.0717902, 1.1250910
2: 0.7031748, 1.0396798, -0.9202113, 1.2288338, -0.5256590, 1.9598911
3: -0.0287646, 0.2064803, -1.0969396, 0.8832272, -0.9119918, 1.3034199
4: -0.1510483, 0.1658331, -1.3542893, 1.0352956, -1.1863439, 1.5201224
5: -0.1287037, 0.1399902, -1.0680280, 1.0702395, -1.1989433, 1.2080182
6: -0.1123485, 0.1386405, -1.0581671, 1.0293008, -1.1416494, 1.1968076
7: -0.1391573, 0.1625858, -1.2263916, 1.1025243, -1.2416816, 1.3889774
8: -0.1355999, 0.2248350, -1.2947221, 1.1090962, -1.2446960, 1.5195570
9: -0.1497552, 0.1276529, -1.1098964, 1.2230521, -1.3728074, 1.2375493

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3103391, upper bound: 7.2959609
time: 3.28 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1471496, upper bound: 7.2742580
time: 2.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0997177, 0.1066720, -4.0253606, 3.1445298, -3.2442474, 4.1320324
1: -0.1370031, 0.1412840, -3.1651258, 2.9174316, -3.0544348, 3.3064098
2: 0.7031748, 1.0396798, -5.4192858, 2.0534806, -1.3503058, 6.4589653
3: -0.0287646, 0.2064803, -4.5677948, 2.3638389, -2.3926036, 4.7742753
4: -0.1510483, 0.1658331, -4.8053541, 3.0876324, -3.2386806, 4.9711871
5: -0.1287037, 0.1399902, -3.7480755, 3.1195178, -3.2482216, 3.8880658
6: -0.1123485, 0.1386405, -3.8025064, 3.2269757, -3.3393242, 3.9411469
7: -0.1391573, 0.1625858, -4.4761119, 3.2936349, -3.4327922, 4.6386976
8: -0.1355999, 0.2248350, -5.2307100, 3.0486403, -3.1842401, 5.4555449
9: -0.1497552, 0.1276529, -3.7011733, 4.1598730, -4.3096280, 3.8288262

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3103391, upper bound: 7.2959609
time: 3.51 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1471496, upper bound: 7.2742580
time: 2.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1481730, 0.1599290, -1.0619552, 0.9352350, -1.0834080, 1.2218843
1: -0.1931481, 0.2025903, -0.9091503, 0.8703966, -1.0635448, 1.1117406
2: 0.6151205, 1.0425050, -0.7625545, 1.2056801, -0.5905596, 1.8050596
3: -0.0697279, 0.2725243, -0.9726491, 0.8332772, -0.9030051, 1.2451735
4: -0.2087196, 0.2092075, -1.2301250, 0.9645168, -1.1732364, 1.4393325
5: -0.1870432, 0.2048456, -0.9752898, 1.0026104, -1.1896536, 1.1801354
6: -0.1530457, 0.2124394, -0.9618854, 0.9570676, -1.1101133, 1.1743248
7: -0.1814390, 0.2302229, -1.1108429, 1.0267212, -1.2081603, 1.3410658
8: -0.2012803, 0.2961308, -1.1707318, 1.0411508, -1.2424310, 1.4668627
9: -0.2136270, 0.1878617, -1.0229536, 1.1269178, -1.3405448, 1.2108153

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2866843, upper bound: 7.2516038
time: 2.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0755991, upper bound: 7.2190859
time: 7.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1481730, 0.1599290, -3.8284457, 2.9689906, -3.1171637, 3.9883747
1: -0.1931481, 0.2025903, -3.0065353, 2.7203574, -2.9135056, 3.2091255
2: 0.6151205, 1.0425050, -5.1298418, 1.9725821, -1.3574616, 6.1723471
3: -0.0697279, 0.2725243, -4.3575759, 2.2571936, -2.3269215, 4.6301003
4: -0.2087196, 0.2092075, -4.5779529, 2.9183955, -3.1271150, 4.7871604
5: -0.1870432, 0.2048456, -3.5277989, 2.9429975, -3.1300406, 3.7326446
6: -0.1530457, 0.2124394, -3.6316020, 3.0413005, -3.1943462, 3.8440413
7: -0.1814390, 0.2302229, -4.1981115, 3.1519873, -3.3334262, 4.4283342
8: -0.2012803, 0.2961308, -4.9361658, 2.9024711, -3.1037514, 5.2322965
9: -0.2136270, 0.1878617, -3.4911053, 3.9163158, -4.1299429, 3.6789670

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2866843, upper bound: 7.2516039
time: 3.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0755991, upper bound: 7.2190859
time: 2.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.3306952, 1.1253389, -3.8284457, 2.9689906, -4.2996855, 4.9537845
1: -1.1061755, 1.0433774, -3.0065353, 2.7203574, -3.8265328, 4.0499125
2: -1.1810069, 1.2723873, -5.1298418, 1.9725821, -3.1535890, 6.4022293
3: -1.2967119, 0.9677966, -4.3575759, 2.2571936, -3.5539055, 5.3253727
4: -1.5545385, 1.1491244, -4.5779529, 2.9183955, -4.4729338, 5.7270775
5: -1.2197224, 1.1831168, -3.5277989, 2.9429975, -4.1627197, 4.7109156
6: -1.2164470, 1.1506547, -3.6316020, 3.0413005, -4.2577477, 4.7822566
7: -1.4102179, 1.2283047, -4.1981115, 3.1519873, -4.5622053, 5.4264164
8: -1.5208342, 1.2165208, -4.9361658, 2.9024711, -4.4233055, 6.1526866
9: -1.2542810, 1.3892401, -3.4911053, 3.9163158, -5.1705971, 4.8803453

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6298158, upper bound: 7.5768564
time: 5.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279708, upper bound: 7.5767800
time: 18.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1587485, 0.1715083, -1.4410763, 1.2068056, -1.3655541, 1.6125846
1: -0.2053554, 0.2160324, -1.1903447, 1.1171834, -1.3225389, 1.4063771
2: 0.5959755, 1.0424958, -1.3553936, 1.3030133, -0.7070378, 2.3978896
3: -0.0786342, 0.2869375, -1.4313635, 1.0249015, -1.1035357, 1.7183011
4: -0.2215261, 0.2186382, -1.6877695, 1.2274487, -1.4489748, 1.9064077
5: -0.1997317, 0.2189467, -1.3219635, 1.2599572, -1.4596889, 1.5409102
6: -0.1618941, 0.2286291, -1.3230343, 1.2336626, -1.3955567, 1.5516634
7: -0.1906322, 0.2452627, -1.5333601, 1.3132799, -1.5039121, 1.7786229
8: -0.2155851, 0.3116325, -1.6715739, 1.2907040, -1.5062891, 1.9832064
9: -0.2276510, 0.2009524, -1.3525006, 1.5009515, -1.7286025, 1.5534530

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2940412, upper bound: 7.2844194
time: 3.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0772492, upper bound: 7.2574310
time: 2.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1587485, 0.1715083, -4.3890800, 3.3949044, -3.5536530, 4.5605884
1: -0.2053554, 0.2160324, -3.4329352, 3.1820660, -3.3874214, 3.6489677
2: 0.5959755, 1.0424958, -5.9047050, 2.1749189, -1.5789434, 6.9472008
3: -0.0786342, 0.2869375, -4.9883642, 2.5989501, -2.6775844, 5.2753019
4: -0.2215261, 0.2186382, -5.1978326, 3.3766735, -3.5981996, 5.4164705
5: -0.1997317, 0.2189467, -4.0887527, 3.3756824, -3.5754142, 4.3076992
6: -0.1618941, 0.2286291, -4.3802404, 3.5210381, -3.6829321, 4.6088696
7: -0.1906322, 0.2452627, -4.8307076, 3.6130788, -3.8037109, 5.0759702
8: -0.2155851, 0.3116325, -5.7200208, 3.2789156, -3.4945006, 6.0316534
9: -0.2276510, 0.2009524, -4.0253072, 4.5143766, -4.7420278, 4.2262597

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2940412, upper bound: 7.2844194
time: 2.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0772492, upper bound: 7.2574310
time: 2.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.4588451, 1.2203939, -1.4410763, 1.2068056, -2.6656508, 2.6614702
1: -1.2039182, 1.1288085, -1.1903447, 1.1171834, -2.3211017, 2.3191533
2: -1.3840015, 1.3078859, -1.3553936, 1.3030133, -2.6870148, 2.6632795
3: -1.4534780, 1.0339497, -1.4313635, 1.0249015, -2.4783795, 2.4653132
4: -1.7099203, 1.2396287, -1.6877695, 1.2274487, -2.9373689, 2.9273982
5: -1.3381892, 1.2731776, -1.3219635, 1.2599572, -2.5981464, 2.5951412
6: -1.3403305, 1.2472912, -1.3230343, 1.2336626, -2.5739932, 2.5703254
7: -1.5539688, 1.3266928, -1.5333601, 1.3132799, -2.8672485, 2.8600531
8: -1.6963817, 1.3024652, -1.6715739, 1.2907040, -2.9870858, 2.9740391
9: -1.3689091, 1.5193797, -1.3525006, 1.5009515, -2.8698606, 2.8718803

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6299813, upper bound: 7.6255394
time: 3.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281657, upper bound: 7.6254628
time: 2.30 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.4588451, 1.2203939, -4.3890800, 3.3949044, -4.8537493, 5.6094742
1: -1.2039182, 1.1288085, -3.4329352, 3.1820660, -4.3859844, 4.5617437
2: -1.3840015, 1.3078859, -5.9047050, 2.1749189, -3.5589204, 7.2125912
3: -1.4534780, 1.0339497, -4.9883642, 2.5989501, -4.0524282, 6.0223141
4: -1.7099203, 1.2396287, -5.1978326, 3.3766735, -5.0865936, 6.4374614
5: -1.3381892, 1.2731776, -4.0887527, 3.3756824, -4.7138715, 5.3619304
6: -1.3403305, 1.2472912, -4.3802404, 3.5210381, -4.8613687, 5.6275315
7: -1.5539688, 1.3266928, -4.8307076, 3.6130788, -5.1670475, 6.1574001
8: -1.6963817, 1.3024652, -5.7200208, 3.2789156, -4.9752975, 7.0224857
9: -1.3689091, 1.5193797, -4.0253072, 4.5143766, -5.8832855, 5.5446868

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6299813, upper bound: 7.6255393
time: 3.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281657, upper bound: 7.6254617
time: 2.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.0393264, 0.9178565, -0.8551247, 0.7983816, -1.8377080, 1.7729812
1: -0.8928415, 0.8542848, -0.7615308, 0.7482750, -1.6411166, 1.6158156
2: -0.7276485, 1.1981939, -0.4506930, 1.1650431, -1.8926916, 1.6488869
3: -0.9435567, 0.8207595, -0.7235693, 0.7370088, -1.6805655, 1.5443289
4: -1.1971776, 0.9476389, -0.9802259, 0.8269940, -2.0241716, 1.9278648
5: -0.9528845, 0.9844075, -0.7923279, 0.8684083, -1.8212928, 1.7767354
6: -0.9440743, 0.9422595, -0.7721305, 0.8151376, -1.7592120, 1.7143900
7: -1.0863340, 1.0054618, -0.8798913, 0.8797811, -1.9661151, 1.8853531
8: -1.1422608, 1.0249797, -0.9418252, 0.9057338, -2.0479946, 1.9668050
9: -0.9996936, 1.1036472, -0.8458427, 0.9452844, -1.9449780, 1.9494900

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2582690, upper bound: 7.2687311
time: 4.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1068228, upper bound: 7.2489089
time: 2.32 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.0393264, 0.9178565, -3.6216893, 2.7327540, -3.7720804, 4.5395460
1: -0.8928415, 0.8542848, -2.7621019, 2.5255127, -3.4183543, 3.6163867
2: -0.7276485, 1.1981939, -4.7444296, 1.8587983, -2.5864468, 5.9426236
3: -0.9435567, 0.8207595, -4.0338159, 2.1043594, -3.0479159, 4.8545752
4: -1.1971776, 0.9476389, -4.3264675, 2.7139330, -3.9111106, 5.2741065
5: -0.9528845, 0.9844075, -3.2967334, 2.6955271, -3.6484115, 4.2811408
6: -0.9440743, 0.9422595, -3.3324790, 2.8237011, -3.7677755, 4.2747383
7: -1.0863340, 1.0054618, -3.9608104, 2.9322450, -4.0185790, 4.9662724
8: -1.1422608, 1.0249797, -4.5993862, 2.7224474, -3.8647082, 5.6243658
9: -0.9996936, 1.1036472, -3.1571589, 3.6648762, -4.6645699, 4.2608061

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2582690, upper bound: 7.2687306
time: 6.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1068228, upper bound: 7.2489089
time: 3.88 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -3.7919588, 2.9467027, -0.8551247, 0.7983816, -4.5903406, 3.8018274
1: -2.9406590, 2.7310686, -0.7615308, 0.7482750, -3.6889341, 3.4925995
2: -4.8264527, 2.1360939, -0.4506930, 1.1650431, -5.9914961, 2.5867867
3: -4.2560358, 2.2299223, -0.7235693, 0.7370088, -4.9930449, 2.9534917
4: -4.5482984, 2.9002113, -0.9802259, 0.8269940, -5.3752923, 3.8804374
5: -3.4544127, 3.0389261, -0.7923279, 0.8684083, -4.3228211, 3.8312540
6: -3.6136959, 3.0776973, -0.7721305, 0.8151376, -4.4288335, 3.8498278
7: -4.1456871, 3.1120200, -0.8798913, 0.8797811, -5.0254683, 3.9919114
8: -4.7225423, 2.8873305, -0.9418252, 0.9057338, -5.6282759, 3.8291557
9: -3.4062939, 3.8535600, -0.8458427, 0.9452844, -4.3515782, 4.6994028

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287345, upper bound: 7.5764697
time: 2.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5620579, upper bound: 7.5754063
time: 2.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -3.7919588, 2.9467027, -3.6216893, 2.7327540, -6.5247126, 6.5683918
1: -2.9406590, 2.7310686, -2.7621019, 2.5255127, -5.4661717, 5.4931707
2: -4.8264527, 2.1360939, -4.7444296, 1.8587983, -6.6852512, 6.8805237
3: -4.2560358, 2.2299223, -4.0338159, 2.1043594, -6.3603954, 6.2637382
4: -4.5482984, 2.9002113, -4.3264675, 2.7139330, -7.2622313, 7.2266788
5: -3.4544127, 3.0389261, -3.2967334, 2.6955271, -6.1499395, 6.3356595
6: -3.6136959, 3.0776973, -3.3324790, 2.8237011, -6.4373970, 6.4101763
7: -4.1456871, 3.1120200, -3.9608104, 2.9322450, -7.0779324, 7.0728302
8: -4.7225423, 2.8873305, -4.5993862, 2.7224474, -7.4449897, 7.4867167
9: -3.4062939, 3.8535600, -3.1571589, 3.6648762, -7.0711699, 7.0107188

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287345, upper bound: 7.5764697
time: 3.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5620579, upper bound: 7.5754063
time: 4.26 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.0531604, 0.9272488, -1.1652219, 1.0053251, -2.0584855, 2.0924706
1: -0.9028684, 0.8625062, -0.9838070, 0.9347872, -1.8376555, 1.8463132
2: -0.7486366, 1.2010515, -0.9202113, 1.2288338, -1.9774704, 2.1212628
3: -0.9602711, 0.8273244, -1.0969396, 0.8832272, -1.8434983, 1.9242640
4: -1.2139385, 0.9570729, -1.3542893, 1.0352956, -2.2492342, 2.3113623
5: -0.9654055, 0.9935039, -1.0680280, 1.0702395, -2.0356450, 2.0615320
6: -0.9571872, 0.9516574, -1.0581671, 1.0293008, -1.9864880, 2.0098245
7: -1.1019704, 1.0151452, -1.2263916, 1.1025243, -2.2044947, 2.2415366
8: -1.1584860, 1.0340253, -1.2947221, 1.1090962, -2.2675822, 2.3287473
9: -1.0113906, 1.1161186, -1.1098964, 1.2230521, -2.2344427, 2.2260151

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2642471, upper bound: 7.2940383
time: 3.55 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1116652, upper bound: 7.2736696
time: 2.90 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.0531604, 0.9272488, -3.9990139, 3.1387386, -4.1918993, 4.9262629
1: -0.9028684, 0.8625062, -3.1564875, 2.8728855, -3.7757540, 4.0189939
2: -0.7486366, 1.2010515, -5.3902931, 2.0323970, -2.7810335, 6.5913448
3: -0.9602711, 0.8273244, -4.5473056, 2.3561730, -3.3164442, 5.3746300
4: -1.2139385, 0.9570729, -4.7793722, 3.0876324, -4.3015709, 5.7364450
5: -0.9654055, 0.9935039, -3.7325683, 3.0704665, -4.0358720, 4.7260723
6: -0.9571872, 0.9516574, -3.7873683, 3.1964283, -4.1536155, 4.7390256
7: -1.1019704, 1.0151452, -4.4685564, 3.2716124, -4.3735828, 5.4837017
8: -1.1584860, 1.0340253, -5.1899042, 3.0368407, -4.1953268, 6.2239294
9: -1.0113906, 1.1161186, -3.7011733, 4.1113167, -5.1227074, 4.8172917

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2642471, upper bound: 7.2940383
time: 3.12 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1116652, upper bound: 7.2736696
time: 4.51 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -3.8652687, 3.0007892, -1.1652219, 1.0053251, -4.8705940, 4.1660109
1: -2.9992514, 2.7792504, -0.9838070, 0.9347872, -3.9340386, 3.7630572
2: -4.9265528, 2.1662290, -0.9202113, 1.2288338, -6.1553864, 3.0864403
3: -4.3435698, 2.2665625, -1.0969396, 0.8832272, -5.2267971, 3.3635020
4: -4.6355057, 2.9509566, -1.3542893, 1.0352956, -5.6708012, 4.3052459
5: -3.5174999, 3.0933805, -1.0680280, 1.0702395, -4.5877395, 4.1614084
6: -3.6814132, 3.1353431, -1.0581671, 1.0293008, -4.7107139, 4.1935101
7: -4.2242031, 3.1676087, -1.2263916, 1.1025243, -5.3267274, 4.3940001
8: -4.8138123, 2.9366531, -1.2947221, 1.1090962, -5.9229083, 4.2313752
9: -3.4669244, 3.9269178, -1.1098964, 1.2230521, -4.6899767, 5.0368142

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289030, upper bound: 7.6223652
time: 3.49 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5621397, upper bound: 7.6201651
time: 3.56 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 8.66 seconds
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.3050140, upper bound: 7.2710082
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.1434126, upper bound: 7.2490772
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.3050140, upper bound: 7.2710083
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.1434126, upper bound: 7.2490772
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6292519, upper bound: 7.5765304
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6065362, upper bound: 7.5755026
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.3103391, upper bound: 7.2959609
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.1471496, upper bound: 7.2742580
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.3103391, upper bound: 7.2959609
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.1471496, upper bound: 7.2742580
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.2866843, upper bound: 7.2516038
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.0755991, upper bound: 7.2190859
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.2866843, upper bound: 7.2516039
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.0755991, upper bound: 7.2190859
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6298158, upper bound: 7.5768564
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6279708, upper bound: 7.5767800
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.2940412, upper bound: 7.2844194
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.0772492, upper bound: 7.2574310
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.2940412, upper bound: 7.2844194
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.0772492, upper bound: 7.2574310
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6299813, upper bound: 7.6255394
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6281657, upper bound: 7.6254628
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6299813, upper bound: 7.6255393
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6281657, upper bound: 7.6254617
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.2582690, upper bound: 7.2687311
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.1068228, upper bound: 7.2489089
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.2582690, upper bound: 7.2687306
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.1068228, upper bound: 7.2489089
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6287345, upper bound: 7.5764697
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.5620579, upper bound: 7.5754063
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6287345, upper bound: 7.5764697
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.5620579, upper bound: 7.5754063
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.2642471, upper bound: 7.2940383
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.1116652, upper bound: 7.2736696
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.2642471, upper bound: 7.2940383
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.1116652, upper bound: 7.2736696
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.6289030, upper bound: 7.6223652
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.66
Output dim: 2, lower bound: -7.5621397, upper bound: 7.6201651
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313078
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281545
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312818
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312818
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314849
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.66
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314849
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=9.388381958007812
rel_dist={2: [-7.633595790761577, 7.633596173769259]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330848, upper bound: 7.6329360
time: 4.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329279, upper bound: 7.6329284
time: 3.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.33
Output dim: 2, lower bound: -7.6330848, upper bound: 7.6329360
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.33
Output dim: 2, lower bound: -7.6329279, upper bound: 7.6329284

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.2686481, 3.3034801, -4.4782391, 3.4574506, -7.7260990, 7.7817192
1: -3.3155420, 3.0498059, -3.4846888, 3.1874254, -6.5029674, 6.5344944
2: -5.4949217, 2.3689001, -5.7812233, 2.4561052, -7.9510269, 8.1501236
3: -4.8110600, 2.4704089, -5.0603476, 2.5757360, -7.3867960, 7.5307565
4: -5.1011324, 3.2392945, -5.3482561, 3.3850257, -8.4861584, 8.5875511
5: -3.8673596, 3.4099646, -4.0482831, 3.5652397, -7.4325991, 7.4582477
6: -4.0397010, 3.4505873, -4.2350430, 3.6150916, -7.6547928, 7.6856303
7: -4.6541395, 3.4808147, -4.8789215, 3.6397176, -8.2938576, 8.3597364
8: -5.3186646, 3.2162716, -5.5783753, 3.3597097, -8.6783743, 8.7946472
9: -3.7986977, 4.3246951, -3.9712863, 4.5337381, -8.3324356, 8.2959814

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309466, upper bound: 7.6304316
time: 5.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6326927, upper bound: 7.6325472
time: 5.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.3132782, 6.2759089, -4.5394058, 3.5020108, -11.8152885, 10.8153152
1: -6.5844011, 5.6741643, -3.5335712, 3.2272670, -9.8116684, 9.2077351
2: -10.9843225, 4.1436543, -5.8645182, 2.4798059, -13.4641285, 10.0081730
3: -9.5884886, 4.5238266, -5.1336527, 2.6061509, -12.1946392, 9.6574793
4: -9.8131104, 6.0509901, -5.4208002, 3.4268878, -13.2399979, 11.4717903
5: -7.3376122, 6.4407911, -4.1007547, 3.6104290, -10.9480410, 10.5415459
6: -7.8093801, 6.6083603, -4.2923889, 3.6629298, -11.4723101, 10.9007492
7: -8.9933872, 6.5240011, -4.9446640, 3.6851766, -12.6785641, 11.4686651
8: -10.3325996, 5.9963574, -5.6539040, 3.4011412, -13.7337408, 11.6502609
9: -7.1123838, 8.2964840, -4.0213695, 4.5954099, -11.7077942, 12.3178539

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308084, upper bound: 7.6304245
time: 5.55 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325393, upper bound: 7.6325398
time: 2.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 9.98
Output dim: 2, lower bound: -7.6309466, upper bound: 7.6304316
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 9.98
Output dim: 2, lower bound: -7.6326927, upper bound: 7.6325472
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 9.98
Output dim: 2, lower bound: -7.6308084, upper bound: 7.6304245
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 9.98
Output dim: 2, lower bound: -7.6325393, upper bound: 7.6325398

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.8520346, 0.7967304, -0.4896535, 0.5621493, -1.4141839, 1.2863839
1: -0.7590725, 0.7492665, -0.5266961, 0.5519304, -1.3110029, 1.2759626
2: -0.4458025, 1.1671515, 0.0695559, 1.1006848, -1.5464872, 1.0975956
3: -0.7199419, 0.7356373, -0.3504300, 0.5800169, -1.2999587, 1.0860673
4: -0.9795382, 0.8256910, -0.5676260, 0.5869345, -1.5664728, 1.3933170
5: -0.7867619, 0.8694988, -0.5190640, 0.6071957, -1.3939576, 1.3885629
6: -0.7677082, 0.8156086, -0.4638670, 0.5903727, -1.3580809, 1.2794756
7: -0.8784739, 0.8785231, -0.5224250, 0.6475274, -1.5260012, 1.4009480
8: -0.9435697, 0.9029079, -0.5908882, 0.6856712, -1.6292409, 1.4937961
9: -0.8469771, 0.9409963, -0.5695025, 0.6195772, -1.4665544, 1.5104988

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289800, upper bound: 7.5920820
time: 3.02 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6291709, upper bound: 7.5920844
time: 3.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.4381592, 2.6891713, -3.3042216, 2.5889869, -6.0271463, 5.9933929
1: -2.6496761, 2.5043685, -2.5444446, 2.4158349, -5.0655107, 5.0488129
2: -4.3456669, 1.9843242, -4.1555247, 1.9160061, -6.2616730, 6.1398487
3: -3.8290000, 2.0535426, -3.6713705, 1.9876379, -5.8166380, 5.7249131
4: -4.1329331, 2.6575460, -3.9775858, 2.5648346, -6.6977677, 6.6351318
5: -3.1428683, 2.7818649, -3.0231628, 2.6749330, -5.8178015, 5.8050280
6: -3.2745221, 2.8007100, -3.1521380, 2.6965775, -5.9710999, 5.9528480
7: -3.7619383, 2.8496456, -3.6162028, 2.7475872, -6.5095253, 6.4658484
8: -4.2851801, 2.6506751, -4.1197267, 2.5601616, -6.8453417, 6.7704020
9: -3.1155143, 3.5120399, -3.0054271, 3.3827658, -6.4982800, 6.5174670

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319503, upper bound: 7.6318756
time: 4.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6322169, upper bound: 7.6320486
time: 4.06 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -4.3667159, 3.3733566, -0.5051364, 0.5750864, -4.9418020, 3.8784931
1: -3.4129970, 3.1069968, -0.5388970, 0.5624546, -3.9754515, 3.6458938
2: -5.6105747, 2.3990748, 0.0450127, 1.1037982, -6.7143726, 2.3540621
3: -4.9307113, 2.5240188, -0.3636159, 0.5898122, -5.5205235, 2.8876348
4: -5.2215772, 3.2998452, -0.5862783, 0.5991106, -5.8206878, 3.8861237
5: -3.9523339, 3.4692559, -0.5301568, 0.6249512, -4.5772853, 3.9994128
6: -4.1502247, 3.5280318, -0.4788322, 0.6000625, -4.7502871, 4.0068641
7: -4.7635708, 3.5552168, -0.5404001, 0.6617785, -5.4253492, 4.0956168
8: -5.4358912, 3.2819157, -0.6092969, 0.6978468, -6.1337380, 3.8912125
9: -3.8811471, 4.4138784, -0.5819599, 0.6367363, -4.5178833, 4.9958382

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286966, upper bound: 7.5920747
time: 3.90 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289442, upper bound: 7.5920778
time: 3.88 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.4230795, 5.6199193, -3.3593738, 2.6297722, -10.0528517, 8.9792929
1: -5.8665671, 5.0965338, -2.5889730, 2.4524550, -8.3190222, 7.6855068
2: -9.7775850, 3.7450247, -4.2317634, 1.9361763, -11.7137613, 7.9767880
3: -8.5398102, 4.0703511, -3.7387998, 2.0149493, -10.5547600, 7.8091507
4: -8.7824516, 5.4268265, -4.0444012, 2.6028323, -11.3852844, 9.4712276
5: -6.5760260, 5.7724261, -3.0718691, 2.7159939, -9.2920198, 8.8442955
6: -6.9843092, 5.9130216, -3.2046323, 2.7399163, -9.7242260, 9.1176538
7: -8.0403557, 5.8544450, -3.6763098, 2.7895980, -10.8299541, 9.5307550
8: -9.2253551, 5.3789873, -4.1879663, 2.5979066, -11.8232613, 9.5669537
9: -6.3843107, 7.4237461, -3.0514688, 3.4390228, -9.8233337, 10.4752150

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317655, upper bound: 7.6318697
time: 5.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320412, upper bound: 7.6320417
time: 2.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 9.47 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 9.47
Output dim: 2, lower bound: -7.6289800, upper bound: 7.5920820
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 9.47
Output dim: 2, lower bound: -7.6291709, upper bound: 7.5920844
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 9.47
Output dim: 2, lower bound: -7.6319503, upper bound: 7.6318756
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 9.47
Output dim: 2, lower bound: -7.6322169, upper bound: 7.6320486
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 9.47
Output dim: 2, lower bound: -7.6286966, upper bound: 7.5920747
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 9.47
Output dim: 2, lower bound: -7.6289442, upper bound: 7.5920778
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 9.47
Output dim: 2, lower bound: -7.6317655, upper bound: 7.6318697
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 9.47
Output dim: 2, lower bound: -7.6320412, upper bound: 7.6320417

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2205492, 0.2476428, -0.2145411, 0.2390419, -0.4595911, 0.4621839
1: -0.2696983, 0.2889494, -0.2637576, 0.2817523, -0.5514506, 0.5527070
2: 0.4973161, 1.0366811, 0.5067328, 1.0421268, -0.5448107, 0.5299482
3: -0.1357589, 0.3594398, -0.1300689, 0.3526843, -0.4884433, 0.4895087
4: -0.2917641, 0.2810236, -0.2853710, 0.2740043, -0.5657684, 0.5663946
5: -0.2639807, 0.2928121, -0.2575352, 0.2862601, -0.5502409, 0.5503472
6: -0.2135446, 0.3178187, -0.2078350, 0.3097613, -0.5233058, 0.5256537
7: -0.2535100, 0.3207713, -0.2466339, 0.3139756, -0.5674856, 0.5674052
8: -0.3020786, 0.3902022, -0.2935427, 0.3829473, -0.6850259, 0.6837450
9: -0.3039868, 0.2756581, -0.2964208, 0.2685148, -0.5725015, 0.5720789

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5018252, upper bound: 7.4236857
time: 3.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5194747, upper bound: 7.4419308
time: 5.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3387598, 0.3983271, -0.2488750, 0.2907519, -0.6295117, 0.6472021
1: -0.3848763, 0.4092921, -0.2992351, 0.3233151, -0.7081914, 0.7085273
2: 0.3058115, 1.0603868, 0.4488762, 1.0421469, -0.7363354, 0.6115106
3: -0.2361171, 0.4644364, -0.1634964, 0.3900754, -0.6261925, 0.6279328
4: -0.4177365, 0.4149073, -0.3272711, 0.3127053, -0.7304419, 0.7421784
5: -0.3750773, 0.4275100, -0.2950729, 0.3223823, -0.6974597, 0.7225828
6: -0.3233609, 0.4493203, -0.2411511, 0.3560766, -0.6794375, 0.6904714
7: -0.3743301, 0.4693605, -0.2845450, 0.3615559, -0.7358860, 0.7539055
8: -0.4351169, 0.5263624, -0.3406039, 0.4288847, -0.8640016, 0.8669662
9: -0.4296470, 0.4205259, -0.3404891, 0.3082671, -0.7379141, 0.7610149

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5199846, upper bound: 7.4236980
time: 4.84 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5375920, upper bound: 7.4419389
time: 4.00 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.1982913, 1.0284003, -1.5546414, 1.2919626, -2.4902539, 2.5830417
1: -1.0081103, 0.9585188, -1.2780075, 1.1972733, -2.2053835, 2.2365263
2: -0.9708087, 1.2389424, -1.5339969, 1.3369559, -2.3077645, 2.7729392
3: -1.1362040, 0.9002963, -1.5695087, 1.0845689, -2.2207727, 2.4698050
4: -1.3948992, 1.0592468, -1.8259835, 1.3106649, -2.7055640, 2.8852303
5: -1.0974704, 1.0927470, -1.4283156, 1.3413596, -2.4388299, 2.5210626
6: -1.0873485, 1.0540254, -1.4312228, 1.3213468, -2.4086952, 2.4852481
7: -1.2630978, 1.1296008, -1.6630249, 1.4027084, -2.6658063, 2.7926257
8: -1.3402058, 1.1312919, -1.8285021, 1.3690095, -2.7092152, 2.9597940
9: -1.1391203, 1.2545627, -1.4573845, 1.6165507, -2.7556710, 2.7119472

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311495, upper bound: 7.6310472
time: 3.28 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311900, upper bound: 7.6311609
time: 4.10 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.1577408, 1.7416153, -2.1197672, 1.7132106, -3.8709514, 3.8613825
1: -1.7361618, 1.6240162, -1.7073935, 1.5960308, -3.3321927, 3.3314097
2: -2.4689713, 1.5144711, -2.4109194, 1.5028032, -3.9717746, 3.9253905
3: -2.3023322, 1.4051006, -2.2555776, 1.3845104, -3.6868424, 3.6606781
4: -2.5481567, 1.7500279, -2.5032301, 1.7215891, -4.2697458, 4.2532578
5: -1.9928129, 1.7783382, -1.9567810, 1.7511047, -3.7439175, 3.7351193
6: -2.0477276, 1.7907784, -2.0084088, 1.7610414, -3.8087690, 3.7991872
7: -2.3388524, 1.8752346, -2.2963800, 1.8448875, -4.1837397, 4.1716146
8: -2.6572475, 1.7767478, -2.6057034, 1.7500829, -4.4073305, 4.3824511
9: -2.0059767, 2.2346249, -1.9710488, 2.1956310, -4.2016077, 4.2056737

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315330, upper bound: 7.6312674
time: 3.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316553, upper bound: 7.6314733
time: 3.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2.1885741, 1.7609359, -0.2209295, 0.2481877, -2.4367619, 1.9818654
1: -1.7629075, 1.6416467, -0.2700744, 0.2894053, -2.0523129, 1.9117211
2: -2.5126140, 1.5234710, 0.4967197, 1.0424923, -3.5551062, 1.0267513
3: -2.3413196, 1.4207150, -0.1361193, 0.3598676, -2.7011871, 1.5568342
4: -2.5818830, 1.7702460, -0.2921689, 0.2814680, -2.8633509, 2.0624149
5: -2.0216606, 1.7985059, -0.2643890, 0.2932269, -2.3148875, 2.0628948
6: -2.0838110, 1.8155094, -0.2139063, 0.3183290, -2.4021401, 2.0294156
7: -2.3744147, 1.8966178, -0.2539455, 0.3212013, -2.6956160, 2.1505632
8: -2.6947701, 1.7951880, -0.3026190, 0.3906619, -3.0854321, 2.0978069
9: -2.0328469, 2.2539618, -0.3044658, 0.2761103, -2.3089571, 2.5584276

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4887528, upper bound: 7.4236761
time: 10.54 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5057976, upper bound: 7.4419231
time: 4.58 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -2.8361928, 2.2453942, -0.2551380, 0.3005347, -3.1367276, 2.5005322
1: -2.2293370, 2.0941019, -0.3059239, 0.3309205, -2.5602574, 2.4000258
2: -3.4818637, 1.7393060, 0.4378377, 1.0425038, -4.5243673, 1.3014683
3: -3.1068420, 1.7546443, -0.1697445, 0.3968294, -3.5036714, 1.9243888
4: -3.4110479, 2.2357273, -0.3354770, 0.3196872, -3.7307351, 2.5712042
5: -2.6059992, 2.3024545, -0.3020431, 0.3288990, -2.9348984, 2.6044977
6: -2.7226186, 2.3309517, -0.2473560, 0.3645401, -3.0871587, 2.5783076
7: -3.1057186, 2.3884215, -0.2913842, 0.3711050, -3.4768236, 2.6798058
8: -3.5368958, 2.2372046, -0.3490940, 0.4377204, -3.9746161, 2.5862985
9: -2.6101732, 2.9096589, -0.3485736, 0.3154778, -2.9256511, 3.2582326

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5061050, upper bound: 7.4236872
time: 2.65 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5236440, upper bound: 7.4419307
time: 5.06 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.8668013, 3.7452598, -1.5953858, 1.3224719, -6.1892729, 5.3406458
1: -3.8087857, 3.4339492, -1.3090289, 1.2252156, -5.0340014, 4.7429781
2: -6.3173323, 2.6174135, -1.5982044, 1.3484243, -7.6657567, 4.2156181
3: -5.5267076, 2.7757373, -1.6189672, 1.1056010, -6.6323085, 4.3947043
4: -5.8049173, 3.6561496, -1.8749120, 1.3397231, -7.1446404, 5.5310616
5: -4.3832550, 3.8465714, -1.4667497, 1.3705004, -5.7537556, 5.3133211
6: -4.6102710, 3.9210453, -1.4706067, 1.3526012, -5.9628720, 5.3916521
7: -5.3012629, 3.9304001, -1.7093735, 1.4334108, -6.7346735, 5.6397734
8: -6.0620918, 3.6320658, -1.8842999, 1.3966324, -7.4587240, 5.5163655
9: -4.2884970, 4.9166679, -1.4940381, 1.6587461, -5.9472432, 6.4107060

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309339, upper bound: 7.6310417
time: 4.92 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309685, upper bound: 7.6311565
time: 5.63 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.0014453, 4.5757880, -2.1543965, 1.7391416, -7.7405868, 6.7301846
1: -4.7189260, 4.1730919, -1.7335373, 1.6207361, -6.3396621, 5.9066291
2: -7.8566380, 3.1094055, -2.4643276, 1.5129402, -9.3695784, 5.5737333
3: -6.8659134, 3.3477540, -2.2984104, 1.4028828, -8.2687960, 5.6461644
4: -7.1309137, 4.4384460, -2.5443385, 1.7468638, -8.8777771, 6.9827843
5: -5.3569093, 4.7041316, -1.9893712, 1.7760545, -7.1329637, 6.6935029
6: -5.6631021, 4.8039093, -2.0446391, 1.7881677, -7.4512701, 6.8485484
7: -6.5162878, 4.7832007, -2.3353224, 1.8717459, -8.3880339, 7.1185231
8: -7.4630480, 4.3993421, -2.6528804, 1.7737849, -9.2368326, 7.0522223
9: -5.2177725, 6.0333385, -2.0026376, 2.2311971, -7.4489698, 8.0359764

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313492, upper bound: 7.6312612
time: 3.73 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314665, upper bound: 7.6314664
time: 5.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 10.71 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.5018252, upper bound: 7.4236857
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.5194747, upper bound: 7.4419308
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.5199846, upper bound: 7.4236980
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.5375920, upper bound: 7.4419389
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.6311495, upper bound: 7.6310472
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.6311900, upper bound: 7.6311609
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.6315330, upper bound: 7.6312674
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.6316553, upper bound: 7.6314733
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.4887528, upper bound: 7.4236761
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.5057976, upper bound: 7.4419231
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.5061050, upper bound: 7.4236872
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.5236440, upper bound: 7.4419307
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.6309339, upper bound: 7.6310417
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.6309685, upper bound: 7.6311565
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.6313492, upper bound: 7.6312612
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.71
Output dim: 2, lower bound: -7.6314665, upper bound: 7.6314664

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4625722, 0.5336000, -0.4696606, 0.5422779, -1.0048501, 1.0032606
1: -0.5017515, 0.5272101, -0.5089221, 0.5337111, -1.0354626, 1.0361322
2: 0.1117309, 1.0924790, 0.0991134, 1.0937690, -0.9820380, 0.9933656
3: -0.3315934, 0.5583498, -0.3367108, 0.5642843, -0.8958776, 0.8950606
4: -0.5338256, 0.5592457, -0.5421544, 0.5670723, -1.1008978, 1.1014001
5: -0.4952784, 0.5689606, -0.5031556, 0.5788096, -1.0740880, 1.0721161
6: -0.4353285, 0.5670644, -0.4437426, 0.5731939, -1.0085224, 1.0108070
7: -0.4924747, 0.6165509, -0.5006634, 0.6248332, -1.1173079, 1.1172143
8: -0.5557243, 0.6576089, -0.5641593, 0.6660566, -1.2217809, 1.2217681
9: -0.5425168, 0.5870508, -0.5489531, 0.5971509, -1.1396677, 1.1360040

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6077113, upper bound: 7.5619575
time: 4.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5673752, upper bound: 7.5611534
time: 5.00 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5024400, 0.5736826, -0.6282070, 0.6511669, -1.1536069, 1.2018895
1: -0.5370338, 0.5596935, -0.6094639, 0.6221560, -1.1591898, 1.1691574
2: 0.0477228, 1.1018881, -0.1189492, 1.1233741, -1.0756513, 1.2208374
3: -0.3611769, 0.5883497, -0.4773216, 0.6387691, -0.9999460, 1.0656712
4: -0.5833184, 0.5966606, -0.7136748, 0.6793727, -1.2626910, 1.3103354
5: -0.5294029, 0.6216102, -0.6080068, 0.7171799, -1.2465827, 1.2296171
6: -0.4770798, 0.5975037, -0.5777106, 0.6684265, -1.1455064, 1.1752143
7: -0.5376810, 0.6593258, -0.6471093, 0.7348012, -1.2724822, 1.3064351
8: -0.6053193, 0.6959015, -0.7150979, 0.7693464, -1.3746657, 1.4109994
9: -0.5791189, 0.6350346, -0.6664374, 0.7451652, -1.3242841, 1.3014719

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6272667, upper bound: 7.5969655
time: 4.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5826985, upper bound: 7.5957541
time: 6.22 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.1123949, 0.9691405, -0.7822293, 0.7518853, -1.8642802, 1.7513697
1: -0.9456803, 0.9008625, -0.7112108, 0.7048125, -1.6504928, 1.6120732
2: -0.8386081, 1.2163860, -0.3437538, 1.1506436, -1.9892517, 1.5601398
3: -1.0334216, 0.8575515, -0.6398640, 0.7040042, -1.7374258, 1.4974154
4: -1.2904196, 0.9990215, -0.8927307, 0.7801383, -2.0705578, 1.8917522
5: -1.0208443, 1.0353072, -0.7304159, 0.8218032, -1.8426476, 1.7657231
6: -1.0092809, 0.9916465, -0.7094733, 0.7667816, -1.7760625, 1.7011197
7: -1.1672823, 1.0630490, -0.7998806, 0.8304386, -1.9977210, 1.8629296
8: -1.2295002, 1.0744916, -0.8641673, 0.8588963, -2.0883965, 1.9386590
9: -1.0652089, 1.1732512, -0.7843866, 0.8843923, -1.9496012, 1.9576378

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279761, upper bound: 7.5767675
time: 3.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6136819, upper bound: 7.5767426
time: 3.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.3690962, 1.1535109, -1.1771061, 1.0143533, -2.3834496, 2.3306170
1: -1.1354026, 1.0683064, -0.9922112, 0.9408300, -2.0762324, 2.0605178
2: -1.2416596, 1.2823570, -0.9399185, 1.2306919, -2.4723516, 2.2222755
3: -1.3430893, 0.9877124, -1.1116887, 0.8886083, -2.2316976, 2.0994010
4: -1.6000596, 1.1762615, -1.3690233, 1.0421231, -2.6421828, 2.5452847
5: -1.2557890, 1.2087688, -1.0789764, 1.0783556, -2.3341446, 2.2877452
6: -1.2539032, 1.1788383, -1.0704420, 1.0378861, -2.2917893, 2.2492802
7: -1.4527948, 1.2572690, -1.2400658, 1.1097755, -2.5625703, 2.4973350
8: -1.5725781, 1.2421076, -1.3109221, 1.1160008, -2.6885788, 2.5530298
9: -1.2873644, 1.4283305, -1.1191056, 1.2359917, -2.5233560, 2.5474362

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6285447, upper bound: 7.6254535
time: 3.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275772, upper bound: 7.6254284
time: 3.11 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -3.5899358, 2.7988777, -0.4815736, 0.5557019, -4.1456375, 3.2804513
1: -2.7758296, 2.5978343, -0.5198990, 0.5441664, -3.3199959, 3.1177335
2: -4.5569472, 2.0598550, 0.0794509, 1.0964409, -5.6533880, 1.9804041
3: -4.0130262, 2.1281404, -0.3446735, 0.5742067, -4.5872331, 2.4728138
4: -4.3058276, 2.7608035, -0.5575433, 0.5790929, -4.8849206, 3.3183467
5: -3.2816665, 2.8938518, -0.5146049, 0.5959001, -3.8775666, 3.4084568
6: -3.4247901, 2.9197795, -0.4573561, 0.5828813, -4.0076714, 3.3771358
7: -3.9299448, 2.9579639, -0.5142486, 0.6386405, -4.5685854, 3.4722126
8: -4.4706292, 2.7526364, -0.5793705, 0.6787799, -5.1494093, 3.3320069
9: -3.2384691, 3.6497135, -0.5606321, 0.6120674, -3.8505366, 4.2103457

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5949630, upper bound: 7.5619050
time: 4.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5535319, upper bound: 7.5611521
time: 3.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -3.7515831, 2.9173248, -0.6558924, 0.6698543, -4.4214373, 3.5732172
1: -2.9049392, 2.7041187, -0.6262835, 0.6366116, -3.5415509, 3.3304024
2: -4.7765498, 2.1265461, -0.1578429, 1.1276948, -5.9042444, 2.2843890
3: -4.2059674, 2.2084582, -0.5042613, 0.6508727, -4.8568401, 2.7127194
4: -4.4977694, 2.8720152, -0.7444631, 0.6974427, -5.1952119, 3.6164784
5: -3.4208970, 3.0134940, -0.6280682, 0.7363418, -4.1572390, 3.6415622
6: -3.5742073, 3.0459709, -0.6003168, 0.6856104, -4.2598176, 3.6462877
7: -4.1030588, 3.0801744, -0.6714709, 0.7506628, -4.8537216, 3.7516453
8: -4.6719160, 2.8604052, -0.7384924, 0.7856265, -5.4575424, 3.5988977
9: -3.3722229, 3.8108823, -0.6857758, 0.7711719, -4.1433949, 4.4966583

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6156093, upper bound: 7.5968608
time: 3.65 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5621084, upper bound: 7.5956366
time: 2.57 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.5843315, 3.5311565, -0.8147995, 0.7727441, -5.3570757, 4.3459558
1: -3.5717161, 3.2488558, -0.7336716, 0.7230520, -4.2947683, 3.9825275
2: -5.9232411, 2.4968042, -0.3918052, 1.1563548, -7.0795960, 2.8886094
3: -5.1884871, 2.6247156, -0.6769700, 0.7185080, -5.9069948, 3.3016856
4: -5.4736729, 3.4484034, -0.9314679, 0.8005912, -6.2742643, 4.3798714
5: -4.1425505, 3.6452763, -0.7586553, 0.8423169, -4.9848671, 4.4039316
6: -4.3433504, 3.6976895, -0.7380998, 0.7876204, -5.1309710, 4.4357891
7: -4.9954219, 3.7102790, -0.8355120, 0.8515114, -5.8469334, 4.5457911
8: -5.7014828, 3.4247549, -0.8981539, 0.8793000, -6.5807829, 4.3229089
9: -4.0556574, 4.6325731, -0.8107872, 0.9122278, -4.9678850, 5.4433603

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276820, upper bound: 7.5767607
time: 3.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6011934, upper bound: 7.5767381
time: 3.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.9714561, 3.8152957, -1.1991655, 1.0299470, -6.0014029, 5.0144610
1: -3.8841245, 3.5018420, -1.0081986, 0.9548118, -4.8389363, 4.5100408
2: -6.4478512, 2.6574645, -0.9743596, 1.2360606, -7.6839118, 3.6318240
3: -5.6491642, 2.8210771, -1.1381700, 0.8996330, -6.5487971, 3.9592471
4: -5.9300790, 3.7166936, -1.3953825, 1.0569937, -6.9870729, 5.1120763
5: -4.4743466, 3.9333258, -1.0991230, 1.0928780, -5.5672245, 5.0324488
6: -4.7043953, 4.0000753, -1.0911827, 1.0537223, -5.7581177, 5.0912580
7: -5.4109282, 4.0026569, -1.2645509, 1.1257879, -6.5367160, 5.2672081
8: -6.1822510, 3.6874590, -1.3403615, 1.1304289, -7.3126798, 5.0278206
9: -4.3738637, 5.0170984, -1.1375577, 1.2580594, -5.6319232, 6.1546564

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283511, upper bound: 7.6254457
time: 2.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6254234, upper bound: 7.6254237
time: 4.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 8.47 seconds
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6077113, upper bound: 7.5619575
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.5673752, upper bound: 7.5611534
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6272667, upper bound: 7.5969655
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.5826985, upper bound: 7.5957541
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6279761, upper bound: 7.5767675
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6136819, upper bound: 7.5767426
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6285447, upper bound: 7.6254535
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6275772, upper bound: 7.6254284
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.5949630, upper bound: 7.5619050
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.5535319, upper bound: 7.5611521
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6156093, upper bound: 7.5968608
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.5621084, upper bound: 7.5956366
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6276820, upper bound: 7.5767607
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6011934, upper bound: 7.5767381
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6283511, upper bound: 7.6254457
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 8.47
Output dim: 2, lower bound: -7.6254234, upper bound: 7.6254237

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4278082, 0.4960156, -0.5086889, 0.5784118, -1.0062201, 1.0047045
1: -0.4666455, 0.4954525, -0.5414183, 0.5633469, -1.0299923, 1.0368707
2: 0.1657513, 1.0843160, 0.0379857, 1.1027468, -0.9369955, 1.0463303
3: -0.3055763, 0.5324805, -0.3664355, 0.5918876, -0.8974639, 0.8989160
4: -0.5016934, 0.5200617, -0.5904943, 0.6010363, -1.1027298, 1.1105560
5: -0.4587701, 0.5310978, -0.5335972, 0.6282147, -1.0869849, 1.0646950
6: -0.4029726, 0.5353070, -0.4824983, 0.6013255, -1.0042981, 1.0178053
7: -0.4596553, 0.5761859, -0.5442566, 0.6644984, -1.1241536, 1.1204426
8: -0.5237176, 0.6173710, -0.6122275, 0.7001230, -1.2238406, 1.2295985
9: -0.5123248, 0.5388985, -0.5835451, 0.6418800, -1.1542048, 1.1224437

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6216892, upper bound: 7.5928675
time: 3.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6228622, upper bound: 7.5902842
time: 6.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.8493292, 0.7951629, -0.6368303, 0.6569757, -1.5063050, 1.4319932
1: -0.7574638, 0.7445990, -0.6143054, 0.6274344, -1.3848982, 1.3589044
2: -0.4427773, 1.1640846, -0.1307541, 1.1254783, -1.5682555, 1.2948387
3: -0.7171650, 0.7342048, -0.4855078, 0.6426219, -1.3597870, 1.2197126
4: -0.9743586, 0.8229182, -0.7235493, 0.6853958, -1.6597544, 1.5464675
5: -0.7874734, 0.8654689, -0.6134947, 0.7237360, -1.5112095, 1.4789636
6: -0.7670898, 0.8115633, -0.5841527, 0.6740503, -1.4411401, 1.3957160
7: -0.8738267, 0.8755987, -0.6545212, 0.7402246, -1.6140513, 1.5301199
8: -0.9366871, 0.9013976, -0.7230582, 0.7744986, -1.7111857, 1.6244558
9: -0.8412702, 0.9410121, -0.6732154, 0.7529572, -1.5942273, 1.6142275

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279253, upper bound: 7.5767656
time: 4.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279730, upper bound: 7.5766833
time: 4.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.0916662, 0.9557519, -1.0069849, 0.8990698, -1.9907360, 1.9627368
1: -0.9306177, 0.8879112, -0.8687755, 0.8362195, -1.7668372, 1.7566867
2: -0.8083154, 1.2119373, -0.6806989, 1.1934825, -2.0017979, 1.8926362
3: -1.0090628, 0.8472285, -0.9068658, 0.8067932, -1.8158560, 1.7540944
4: -1.2670503, 0.9842363, -1.1649270, 0.9251761, -2.1922264, 2.1491632
5: -1.0017354, 1.0230112, -0.9268495, 0.9668901, -1.9686255, 1.9498608
6: -0.9899466, 0.9778934, -0.9109936, 0.9193062, -1.9092528, 1.8888869
7: -1.1445892, 1.0481541, -1.0490557, 0.9857498, -2.1303391, 2.0972099
8: -1.2066612, 1.0599850, -1.1069250, 1.0040816, -2.2107430, 2.1669102
9: -1.0485390, 1.1545253, -0.9757088, 1.0792583, -2.1277974, 2.1302342

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284980, upper bound: 7.6254127
time: 6.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6285139, upper bound: 7.6219606
time: 6.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -2.7552989, 2.1766465, -0.5654141, 0.6118160, -3.3671148, 2.7420607
1: -2.1918192, 1.9985526, -0.5751364, 0.5934767, -2.7852960, 2.5736890
2: -3.4288099, 1.6693366, -0.0357490, 1.1148301, -4.5436401, 1.7050855
3: -3.0239122, 1.7069172, -0.4186968, 0.6151429, -3.6390550, 2.1256139
4: -3.2728295, 2.1679316, -0.6499566, 0.6379957, -3.9108253, 2.8178883
5: -2.5335202, 2.1644850, -0.5666040, 0.6743338, -3.2078540, 2.7310889
6: -2.5853112, 2.2140257, -0.5273738, 0.6335727, -3.2188840, 2.7413995
7: -3.0006263, 2.3376069, -0.5950970, 0.7009176, -3.7015438, 2.9327040
8: -3.4771466, 2.1598251, -0.6659783, 0.7328625, -4.2100091, 2.8258033
9: -2.5140936, 2.8244228, -0.6256880, 0.6887164, -3.2028100, 3.4501109

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6274800, upper bound: 7.6253803
time: 12.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6274915, upper bound: 7.6219350
time: 2.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.2407613, 3.2780504, -0.6666971, 0.6771000, -4.9178615, 3.9447474
1: -3.2950144, 3.0236983, -0.6332484, 0.6431111, -3.9381256, 3.6569467
2: -5.4501910, 2.3496485, -0.1731361, 1.1301771, -6.5803680, 2.5227845
3: -4.7813640, 2.4521825, -0.5149528, 0.6556394, -5.4370031, 2.9671354
4: -5.0699949, 3.2091148, -0.7569447, 0.7048343, -5.7748294, 3.9660594
5: -3.8454256, 3.3885055, -0.6355621, 0.7444168, -4.5898423, 4.0240674
6: -4.0241718, 3.4280524, -0.6089704, 0.6927239, -4.7168956, 4.0370226
7: -4.6268601, 3.4497516, -0.6810646, 0.7574558, -5.3843160, 4.1308165
8: -5.2757783, 3.1887689, -0.7486245, 0.7920060, -6.0677843, 3.9373934
9: -3.7733064, 4.2909212, -0.6942241, 0.7810121, -4.5543184, 4.9851456

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276567, upper bound: 7.5767585
time: 2.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276722, upper bound: 7.5766759
time: 2.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.6208653, 3.5579624, -1.0302452, 0.9146022, -5.5354676, 4.5882077
1: -3.6019747, 3.2726798, -0.8857372, 0.8496523, -4.4516268, 4.1584172
2: -5.9670162, 2.5070601, -0.7157158, 1.1979773, -7.1649933, 3.2227759
3: -5.2344351, 2.6437426, -0.9350193, 0.8177509, -6.0521860, 3.5787618
4: -5.5197001, 3.4733799, -1.1927602, 0.9410757, -6.4607759, 4.6661401
5: -4.1725993, 3.6703367, -0.9477898, 0.9820168, -5.1546164, 4.6181264
6: -4.3776374, 3.7263746, -0.9333333, 0.9350080, -5.3126454, 4.6597080
7: -5.0340872, 3.7380505, -1.0753430, 1.0018730, -6.0359602, 4.8133936
8: -5.7484298, 3.4466157, -1.1335349, 1.0193632, -6.7677927, 4.5801506
9: -4.0867720, 4.6698923, -0.9951282, 1.1002202, -5.1869922, 5.6650205

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283196, upper bound: 7.6254091
time: 3.53 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283240, upper bound: 7.6219530
time: 5.53 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 10.68 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6216892, upper bound: 7.5928675
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6228622, upper bound: 7.5902842
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6279253, upper bound: 7.5767656
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6279730, upper bound: 7.5766833
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6284980, upper bound: 7.6254127
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6285139, upper bound: 7.6219606
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6274800, upper bound: 7.6253803
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6274915, upper bound: 7.6219350
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6276567, upper bound: 7.5767585
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6276722, upper bound: 7.5766759
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6283196, upper bound: 7.6254091
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.68
Output dim: 2, lower bound: -7.6283240, upper bound: 7.6219530

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.5073439, 0.5771849, -0.3093551, 0.3654963, -0.8728402, 0.8865399
1: -0.5407943, 0.5640316, -0.3579322, 0.3814245, -0.9222187, 0.9219638
2: 0.0406539, 1.1044482, 0.3521376, 1.0527161, -1.0120623, 0.7523106
3: -0.3657401, 0.5914554, -0.2131880, 0.4416780, -0.8074181, 0.8046434
4: -0.5895249, 0.6010549, -0.3899660, 0.3805581, -0.9700829, 0.9910209
5: -0.5320530, 0.6281106, -0.3483281, 0.3931325, -0.9251855, 0.9764387
6: -0.4810388, 0.6016073, -0.2969718, 0.4207407, -0.9017795, 0.8985791
7: -0.5430807, 0.6642898, -0.3459551, 0.4345112, -0.9775919, 1.0102448
8: -0.6128296, 0.6994808, -0.4059630, 0.4963941, -1.1092236, 1.1054437
9: -0.5840932, 0.6396583, -0.4022577, 0.3821099, -0.9662030, 1.0419160

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5936434, upper bound: 7.5068749
time: 5.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5907086, upper bound: 7.5068731
time: 4.48 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.6442428, 0.6620308, -0.4687729, 0.5406632, -1.1849060, 1.1308036
1: -0.6189097, 0.6325734, -0.5080763, 0.5335001, -1.1524098, 1.1406498
2: -0.1411142, 1.1278018, 0.1014883, 1.0946541, -1.2357683, 1.0263135
3: -0.4927609, 0.6460165, -0.3361239, 0.5635918, -1.0563526, 0.9821404
4: -0.7323890, 0.6908078, -0.5414606, 0.5663489, -1.2987379, 1.2322683
5: -0.6179341, 0.7297772, -0.5012788, 0.5781109, -1.1960449, 1.2310560
6: -0.5895183, 0.6792805, -0.4419703, 0.5730454, -1.1625637, 1.1212507
7: -0.6611860, 0.7452606, -0.4993629, 0.6241986, -1.2853847, 1.2446235
8: -0.7304710, 0.7789569, -0.5640032, 0.6647018, -1.3951728, 1.3429601
9: -0.6797493, 0.7595273, -0.5489745, 0.5953931, -1.2751424, 1.3085018

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6011992, upper bound: 7.5068845
time: 3.49 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5952399, upper bound: 7.5068818
time: 3.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.6936756, 0.6952389, -0.3982733, 0.4640101, -1.1576858, 1.0935123
1: -0.6507333, 0.6595012, -0.4387545, 0.4665037, -1.1172370, 1.0982556
2: -0.2121348, 1.1367357, 0.2116622, 1.0756475, -1.2877823, 0.9250734
3: -0.5423652, 0.6675700, -0.2825190, 0.5099267, -1.0522919, 0.9500890
4: -0.7896075, 0.7233227, -0.4732133, 0.4852598, -1.2748673, 1.1965361
5: -0.6547884, 0.7646694, -0.4310868, 0.4961923, -1.1509807, 1.1957562
6: -0.6305402, 0.7108054, -0.3766272, 0.5064379, -1.1369781, 1.0874326
7: -0.7065127, 0.7750162, -0.4310646, 0.5408897, -1.2474024, 1.2060808
8: -0.7755193, 0.8078921, -0.4933100, 0.5865431, -1.3620625, 1.3012021
9: -0.7160686, 0.8058119, -0.4843490, 0.4997417, -1.2158103, 1.2901609

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6231293, upper bound: 7.5456198
time: 5.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6209543, upper bound: 7.5456173
time: 3.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.8677661, 0.8074765, -0.7120996, 0.7070189, -1.5747850, 1.5195761
1: -0.7702035, 0.7557676, -0.6629508, 0.6681193, -1.4383228, 1.4187183
2: -0.4708281, 1.1680708, -0.2393737, 1.1390544, -1.6098825, 1.4074445
3: -0.7393001, 0.7427031, -0.5611549, 0.6752155, -1.4145157, 1.3038580
4: -0.9976406, 0.8346007, -0.8103188, 0.7347640, -1.7324047, 1.6449194
5: -0.8031855, 0.8779873, -0.6702131, 0.7762345, -1.5794201, 1.5482004
6: -0.7831936, 0.8246396, -0.6469102, 0.7219813, -1.5051749, 1.4715497
7: -0.8943638, 0.8886888, -0.7241107, 0.7857598, -1.6801236, 1.6127994
8: -0.9575963, 0.9131548, -0.7921805, 0.8181977, -1.7757940, 1.7053354
9: -0.8579020, 0.9569830, -0.7284876, 0.8227521, -1.6806540, 1.6854706

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6271655, upper bound: 7.5456202
time: 4.60 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6233520, upper bound: 7.5456172
time: 4.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -2.3311834, 1.8629612, -0.2811019, 0.3327285, -2.6639118, 2.1440630
1: -1.8691416, 1.7197866, -0.3312132, 0.3562236, -2.2253652, 2.0509999
2: -2.7575264, 1.5553240, 0.3963075, 1.0456005, -3.8031268, 1.1590164
3: -2.5090852, 1.4878309, -0.1911981, 0.4191783, -2.9282634, 1.6790290
4: -2.7630734, 1.8678424, -0.3627889, 0.3489076, -3.1119812, 2.2306314
5: -2.1404088, 1.8739445, -0.3248592, 0.3595167, -2.4999256, 2.1988037
6: -2.1742172, 1.8993367, -0.2712896, 0.3926253, -2.5668426, 2.1706264
7: -2.5273204, 2.0130928, -0.3178436, 0.4027565, -2.9300768, 2.3309364
8: -2.8976383, 1.8798313, -0.3777367, 0.4667999, -3.3644381, 2.2575679
9: -2.1426761, 2.3936751, -0.3755207, 0.3470617, -2.4897377, 2.7691958

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5775302, upper bound: 7.5455941
time: 3.56 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5684238, upper bound: 7.5455913
time: 4.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -2.5155118, 1.9993739, -0.4432190, 0.5129762, -3.0284879, 2.4425929
1: -2.0094461, 1.8409771, -0.4828086, 0.5111356, -2.5205817, 2.3237855
2: -3.0494967, 1.6048877, 0.1415188, 1.0895958, -4.1390924, 1.4633689
3: -2.7329822, 1.5830270, -0.3176759, 0.5446444, -3.2776265, 1.9007030
4: -2.9848380, 1.9982697, -0.5176203, 0.5385432, -3.5233812, 2.5158899
5: -2.3112085, 2.0003707, -0.4741660, 0.5503200, -2.8615284, 2.4745369
6: -2.3529913, 2.0362535, -0.4172766, 0.5508780, -2.9038694, 2.4535301
7: -2.7332218, 2.1541569, -0.4750789, 0.5950803, -3.3283019, 2.6292357
8: -3.1497359, 2.0014629, -0.5407672, 0.6349654, -3.7847013, 2.5422301
9: -2.3042498, 2.5809188, -0.5277300, 0.5604059, -2.8646555, 3.1086488

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5831101, upper bound: 7.5455935
time: 3.21 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5700294, upper bound: 7.5455904
time: 4.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -3.6721096, 2.8604829, -0.3191787, 0.3767370, -4.0488467, 3.1796615
1: -2.8366122, 2.6504431, -0.3671318, 0.3904011, -3.2270133, 3.0175748
2: -4.6755552, 2.1086442, 0.3367534, 1.0552629, -5.7308183, 1.7718909
3: -4.1052580, 2.1676152, -0.2208421, 0.4494383, -4.5546961, 2.3884573
4: -4.3987827, 2.8153043, -0.3993945, 0.3917339, -4.7905169, 3.2146988
5: -3.3521674, 2.9658918, -0.3567209, 0.4047997, -3.7569671, 3.3226128
6: -3.4953353, 2.9821358, -0.3058530, 0.4304657, -3.9258010, 3.2879889
7: -4.0174017, 3.0191591, -0.3556391, 0.4457759, -4.4631777, 3.3747983
8: -4.5720606, 2.8047125, -0.4158162, 0.5065815, -5.0786419, 3.2205288
9: -3.3047163, 3.7265346, -0.4115468, 0.3945007, -3.6992171, 4.1380816

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5829796, upper bound: 7.5068687
time: 4.49 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5791783, upper bound: 7.5068645
time: 3.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -3.9296107, 3.0493443, -0.4798092, 0.5534415, -4.4830523, 3.5291533
1: -3.0434480, 2.8193166, -0.5184959, 0.5434657, -3.5869136, 3.3378124
2: -5.0270333, 2.2183111, 0.0827682, 1.0971183, -6.1241517, 2.1355429
3: -4.4111824, 2.2959566, -0.3437128, 0.5728488, -4.9840312, 2.6396694
4: -4.7025361, 2.9929466, -0.5556556, 0.5778013, -5.2803373, 3.5486021
5: -3.5753989, 3.1577442, -0.5121725, 0.5942703, -4.1696692, 3.6699166
6: -3.7345390, 3.1837709, -0.4548765, 0.5822806, -4.3168197, 3.6386473
7: -4.2934089, 3.2137694, -0.5123043, 0.6369258, -4.9303346, 3.7260737
8: -4.8905416, 2.9780774, -0.5783926, 0.6768193, -5.5673609, 3.5564699
9: -3.5168152, 3.9815712, -0.5597212, 0.6095916, -4.1264067, 4.5412922

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5880461, upper bound: 7.5068761
time: 3.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5813409, upper bound: 7.5068740
time: 8.33 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.0497718, 3.1380341, -0.4069447, 0.4736280, -4.5233998, 3.5449789
1: -3.1408772, 2.8983588, -0.4465693, 0.4746880, -3.6155653, 3.3449280
2: -5.1880193, 2.2647421, 0.1978725, 1.0776505, -6.2656698, 2.0668695
3: -4.5557938, 2.3568554, -0.2892631, 0.5164984, -5.0722923, 2.6461186
4: -4.8465595, 3.0767579, -0.4811037, 0.4954486, -5.3420081, 3.5578616
5: -3.6779361, 3.2447910, -0.4394092, 0.5059435, -4.1838794, 3.6842003
6: -3.8461862, 3.2784352, -0.3844360, 0.5146235, -4.3608098, 3.6628711
7: -4.4217825, 3.3054461, -0.4392821, 0.5512171, -4.9729996, 3.7447283
8: -5.0411348, 3.0586181, -0.5014551, 0.5953540, -5.6364889, 3.5600731
9: -3.6166792, 4.1023526, -0.4921012, 0.5113614, -4.1280403, 4.5944538

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6118442, upper bound: 7.5456126
time: 3.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6088200, upper bound: 7.5456103
time: 8.61 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.3034153, 3.3245623, -0.7364562, 0.7229400, -5.0263553, 4.0610185
1: -3.3452098, 3.0646753, -0.6796516, 0.6808056, -4.0260153, 3.7443271
2: -5.5341101, 2.3724208, -0.2752293, 1.1431739, -6.6772842, 2.6476502
3: -4.8572111, 2.4839611, -0.5873816, 0.6852629, -5.5424738, 3.0713427
4: -5.1458821, 3.2526169, -0.8384616, 0.7506718, -5.8965540, 4.0910788
5: -3.8977792, 3.4342339, -0.6906475, 0.7924618, -4.6902409, 4.1248813
6: -4.0818677, 3.4775779, -0.6686466, 0.7372376, -4.8191051, 4.1462245
7: -4.6936154, 3.4974504, -0.7493359, 0.8010107, -5.4946260, 4.2467861
8: -5.3553104, 3.2300212, -0.8160836, 0.8318840, -6.1871943, 4.0461049
9: -3.8256102, 4.3542295, -0.7469913, 0.8446686, -4.6702785, 5.1012206

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 122

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6140850, upper bound: 7.5456118
time: 5.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6098194, upper bound: 7.5456101
time: 3.12 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 10.52 seconds
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5936434, upper bound: 7.5068749
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5907086, upper bound: 7.5068731
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6011992, upper bound: 7.5068845
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5952399, upper bound: 7.5068818
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6231293, upper bound: 7.5456198
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6209543, upper bound: 7.5456173
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6271655, upper bound: 7.5456202
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6233520, upper bound: 7.5456172
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5775302, upper bound: 7.5455941
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5684238, upper bound: 7.5455913
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5831101, upper bound: 7.5455935
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5700294, upper bound: 7.5455904
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5829796, upper bound: 7.5068687
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5791783, upper bound: 7.5068645
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5880461, upper bound: 7.5068761
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.5813409, upper bound: 7.5068740
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6118442, upper bound: 7.5456126
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6088200, upper bound: 7.5456103
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6140850, upper bound: 7.5456118
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 10.52
Output dim: 2, lower bound: -7.6098194, upper bound: 7.5456101

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5985383, 0.6319085, -0.5317153, 0.5929395, -1.1914778, 1.1636238
1: -0.5937583, 0.6100436, -0.5552891, 0.5771642, -1.1709225, 1.1653327
2: -0.0788212, 1.1207938, 0.0067632, 1.1084762, -1.1872973, 1.1140306
3: -0.4497680, 0.6270344, -0.3871472, 0.6028113, -1.0525793, 1.0141816
4: -0.6824765, 0.6608098, -0.6165308, 0.6156935, -1.2981701, 1.2773407
5: -0.5866217, 0.6981541, -0.5462289, 0.6498425, -1.2364643, 1.2443830
6: -0.5535516, 0.6516100, -0.5006011, 0.6158985, -1.1694502, 1.1522110
7: -0.6221524, 0.7199712, -0.5663413, 0.6811838, -1.3033363, 1.2863126
8: -0.6930979, 0.7519839, -0.6371295, 0.7139606, -1.4070586, 1.3891134
9: -0.6484404, 0.7170151, -0.6020362, 0.6620489, -1.3104894, 1.3190513

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456198
time: 3.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456204
time: 2.95 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 10.31 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 10.31
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456198
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 10.31
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456204

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.5985383, 0.6319085, -0.4776491, 0.5510796, -1.1496179, 1.1095575
1: -0.5937583, 0.6100436, -0.5168119, 0.5422386, -1.1359969, 1.1268555
2: -0.0788212, 1.1207938, 0.0862644, 1.0973593, -1.1761806, 1.0345294
3: -0.4497680, 0.6270344, -0.3424403, 0.5712816, -1.0210496, 0.9694747
4: -0.6824765, 0.6608098, -0.5533221, 0.5760981, -1.2585747, 1.2141320
5: -0.5866217, 0.6981541, -0.5098339, 0.5920675, -1.1786892, 1.2079880
6: -0.5535516, 0.6516100, -0.4523359, 0.5811517, -1.1347033, 1.1039459
7: -0.6221524, 0.7199712, -0.5100273, 0.6348365, -1.2569890, 1.2299986
8: -0.6930979, 0.7519839, -0.5766814, 0.6746255, -1.3677235, 1.3286653
9: -0.6484404, 0.7170151, -0.5582496, 0.6070217, -1.2554622, 1.2752647

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.8631825, upper bound: 6.8025508
time: 2.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456209
time: 6.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.5985383, 0.6319085, -2.0817614, 1.5531367, -2.1516750, 2.7136698
1: -0.5937583, 0.6100436, -1.4751073, 1.3709676, -1.9647260, 2.0851510
2: -0.0788212, 1.1207938, -2.0823576, 1.3812394, -1.4600606, 3.2031515
3: -0.4497680, 0.6270344, -1.7702906, 1.2397952, -1.6895633, 2.3973250
4: -0.6824765, 0.6608098, -2.3445997, 1.5198758, -2.2023523, 3.0054095
5: -0.5866217, 0.6981541, -1.4669044, 1.8866613, -2.4732831, 2.1650586
6: -0.5535516, 0.6516100, -1.6953988, 1.5102943, -2.0638459, 2.3470087
7: -0.6221524, 0.7199712, -1.9573610, 1.6453135, -2.2674661, 2.6773322
8: -0.6930979, 0.7519839, -2.0502787, 1.5960993, -2.2891972, 2.8022625
9: -0.6484404, 0.7170151, -1.7823358, 1.9226786, -2.5711191, 2.4993510

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.8631825, upper bound: 6.8025508
time: 2.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456205
time: 5.24 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 9.81 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 9.81
Output dim: 2, lower bound: -6.8631825, upper bound: 6.8025508
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 9.81
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456209
IS_A1_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 9.81
Output dim: 2, lower bound: -6.8631825, upper bound: 6.8025508
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 9.81
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456205

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4837789, 0.5568939, -0.4478325, 0.5181130, -1.0018919, 1.0047264
1: -0.5220600, 0.5478675, -0.4875114, 0.5150856, -1.0371456, 1.0353789
2: 0.0776960, 1.0997318, 0.1341423, 1.0903531, -1.0126572, 0.9655895
3: -0.3465161, 0.5760734, -0.3210046, 0.5480850, -0.8946010, 0.8970780
4: -0.5611830, 0.5821255, -0.5216784, 0.5436506, -1.1048336, 1.1038040
5: -0.5143865, 0.6006936, -0.4794147, 0.5550247, -1.0694113, 1.0801084
6: -0.4581960, 0.5864503, -0.4217477, 0.5548294, -1.0130255, 1.0081980
7: -0.5164225, 0.6419054, -0.4792748, 0.6004871, -1.1169095, 1.1211802
8: -0.5849478, 0.6804364, -0.5444833, 0.6405246, -1.2254725, 1.2249198
9: -0.5647837, 0.6134681, -0.5314144, 0.5670922, -1.1318759, 1.1448824

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5768738, upper bound: 7.4995664
time: 10.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6285950, upper bound: 7.6122264
time: 5.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4837789, 0.5568939, -2.0155506, 1.5119531, -1.9957321, 2.5724444
1: -0.5220600, 0.5478675, -1.4357650, 1.3374898, -1.8595498, 1.9836326
2: 0.0776960, 1.0997318, -1.9928634, 1.3699087, -1.2922127, 3.0925951
3: -0.3465161, 0.5760734, -1.7111957, 1.2125834, -1.5590994, 2.2872691
4: -0.5611830, 0.5821255, -2.2707388, 1.4813899, -2.0425730, 2.8528643
5: -0.5143865, 0.6006936, -1.4272532, 1.8338859, -2.3482723, 2.0279469
6: -0.4581960, 0.5864503, -1.6441319, 1.4722534, -1.9304495, 2.2305822
7: -0.5164225, 0.6419054, -1.8980334, 1.6042032, -2.1206257, 2.5399387
8: -0.5849478, 0.6804364, -1.9900948, 1.5584579, -2.1434057, 2.6705313
9: -0.5647837, 0.6134681, -1.7321892, 1.8684200, -2.4332037, 2.3456573

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4610872, upper bound: 7.3713066
time: 3.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456204
time: 3.50 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 8.97 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 8.97
Output dim: 2, lower bound: -7.5768738, upper bound: 7.4995664
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.97
Output dim: 2, lower bound: -7.6285950, upper bound: 7.6122264
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 8.97
Output dim: 2, lower bound: -7.4610872, upper bound: 7.3713066
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 8.97
Output dim: 2, lower bound: -7.6271656, upper bound: 7.5456204

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4636906, 0.5342022, -0.4200259, 0.4872412, -0.9509318, 0.9542281
1: -0.5029705, 0.5296311, -0.4587606, 0.4887868, -0.9917573, 0.9883917
2: 0.1108653, 1.0948642, 0.1783437, 1.0833732, -0.9725078, 0.9165205
3: -0.3325118, 0.5596430, -0.2996125, 0.5268791, -0.8593909, 0.8592555
4: -0.5366448, 0.5610071, -0.4952341, 0.5113420, -1.0479869, 1.0562413
5: -0.4946487, 0.5726001, -0.4498202, 0.5235435, -1.0181923, 1.0224203
6: -0.4357741, 0.5693220, -0.3954605, 0.5285601, -0.9643342, 0.9647825
7: -0.4938597, 0.6186857, -0.4522538, 0.5675663, -1.0614259, 1.0709395
8: -0.5599283, 0.6583075, -0.5177622, 0.6081575, -1.1680858, 1.1760697
9: -0.5455149, 0.5875890, -0.5063204, 0.5276097, -1.0731246, 1.0939095

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0959814, upper bound: 7.1015279
time: 2.95 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0959814, upper bound: 7.6122262
time: 2.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4636906, 0.5342022, -1.9475679, 1.4697357, -1.9334264, 2.4817700
1: -0.5029705, 0.5296311, -1.3953890, 1.3028538, -1.8058243, 1.9250201
2: 0.1108653, 1.0948642, -1.9010763, 1.3581123, -1.2472470, 2.9959407
3: -0.3325118, 0.5596430, -1.6504903, 1.1846484, -1.5171602, 2.2101333
4: -0.5366448, 0.5610071, -2.1949139, 1.4417797, -1.9784245, 2.7559209
5: -0.4946487, 0.5726001, -1.3867016, 1.7796571, -2.2743058, 1.9593017
6: -0.4357741, 0.5693220, -1.5915793, 1.4330817, -1.8688560, 2.1609013
7: -0.4938597, 0.6186857, -1.8370190, 1.5619674, -2.0558271, 2.4557047
8: -0.5599283, 0.6583075, -1.9282060, 1.5197517, -2.0796800, 2.5865135
9: -0.5455149, 0.5875890, -1.6805456, 1.8129101, -2.3584251, 2.2681346

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0465635, upper bound: 7.0417455
time: 3.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0465635, upper bound: 7.5456206
time: 7.46 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 12.99 seconds
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.99
Output dim: 2, lower bound: -7.0959814, upper bound: 7.1015279
IS_A1_B2_A2_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 12.99
Output dim: 2, lower bound: -7.0959814, upper bound: 7.6122262
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.99
Output dim: 2, lower bound: -7.0465635, upper bound: 7.0417455
IS_A1_B2_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 12.99
Output dim: 2, lower bound: -7.0465635, upper bound: 7.5456206
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=9.388381958007812
rel_dist={2: [-7.63358222082411, 7.633582568333054]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332224, upper bound: 7.6329509
time: 3.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329364, upper bound: 7.6329368
time: 4.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.72
Output dim: 2, lower bound: -7.6332224, upper bound: 7.6329509
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.72
Output dim: 2, lower bound: -7.6329364, upper bound: 7.6329368

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -4.2686481, 3.3034801, -4.6746330, 3.6010723, -7.8697205, 7.9781132
1: -3.3155420, 3.0498059, -3.6430438, 3.3159542, -6.6314964, 6.6928496
2: -5.4949217, 2.3689001, -6.0480118, 2.5371723, -8.0320940, 8.4169121
3: -4.8110600, 2.4704089, -5.2935281, 2.6748891, -7.4859490, 7.7639370
4: -5.1011324, 3.2392945, -5.5791483, 3.5209072, -8.6220398, 8.8184433
5: -3.8673596, 3.4099646, -4.2168908, 3.7109375, -7.5782971, 7.6268554
6: -4.0397010, 3.4505873, -4.4185152, 3.7684367, -7.8081379, 7.8691025
7: -4.6541395, 3.4808147, -5.0896683, 3.7877028, -8.4418421, 8.5704832
8: -5.3186646, 3.2162716, -5.8212180, 3.4935989, -8.8122635, 9.0374899
9: -3.7986977, 4.3246951, -4.1324453, 4.7288523, -8.5275497, 8.4571400

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313988, upper bound: 7.6304591
time: 6.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6328322, upper bound: 7.6325611
time: 3.73 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.3132782, 6.2759089, -4.6710958, 3.5983722, -11.9116507, 10.9470043
1: -6.5844011, 5.6741643, -3.6398854, 3.3134885, -9.8978901, 9.3140497
2: -10.9843225, 4.1436543, -6.0434790, 2.5347314, -13.5190544, 10.1871338
3: -9.5884886, 4.5238266, -5.2897911, 2.6727831, -12.2612715, 9.8136177
4: -9.8131104, 6.0509901, -5.5754151, 3.5181444, -13.3312550, 11.6264057
5: -7.3376122, 6.4407911, -4.2138567, 3.7082560, -11.0458679, 10.6546478
6: -7.8093801, 6.6083603, -4.4153509, 3.7657030, -11.5750828, 11.0237112
7: -8.9933872, 6.5240011, -5.0859556, 3.7845984, -12.7779856, 11.6099567
8: -10.3325996, 5.9963574, -5.8167553, 3.4910483, -13.8236485, 11.8131123
9: -7.1123838, 8.2964840, -4.1294432, 4.7259936, -11.8383770, 12.4259272

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311448, upper bound: 7.6304453
time: 4.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325473, upper bound: 7.6325479
time: 3.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 9.69 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 9.69
Output dim: 2, lower bound: -7.6313988, upper bound: 7.6304591
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 9.69
Output dim: 2, lower bound: -7.6328322, upper bound: 7.6325611
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 9.69
Output dim: 2, lower bound: -7.6311448, upper bound: 7.6304453
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 9.69
Output dim: 2, lower bound: -7.6325473, upper bound: 7.6325479

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -1.5154194, 1.2626250, -0.5501110, 0.6029903, -2.1184096, 1.8127360
1: -1.2484305, 1.1720226, -0.5661048, 0.5866237, -1.8350542, 1.7381275
2: -1.4715331, 1.3274641, -0.0157294, 1.1122110, -2.5837440, 1.3431935
3: -1.5217247, 1.0639236, -0.4041489, 0.6095768, -2.1313014, 1.4680725
4: -1.7800013, 1.2829766, -0.6346319, 0.6276587, -2.4076600, 1.9176085
5: -1.3891407, 1.3145685, -0.5565606, 0.6632382, -2.0523789, 1.8711292
6: -1.3918734, 1.2927294, -0.5146390, 0.6260328, -2.0179062, 1.8073684
7: -1.6191303, 1.3746485, -0.5825213, 0.6918657, -2.3109961, 1.9571698
8: -1.7768812, 1.3416964, -0.6528623, 0.7244771, -2.5013583, 1.9945587
9: -1.4237670, 1.5738322, -0.6151910, 0.6758682, -2.0996351, 2.1890230

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6298486, upper bound: 7.5921209
time: 4.98 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6300558, upper bound: 7.5921268
time: 6.22 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.7173996, 2.8950174, -3.4936070, 2.7281554, -6.4455547, 6.3886242
1: -2.8730814, 2.6877975, -2.6967399, 2.5414128, -5.4144945, 5.3845377
2: -4.7310686, 2.1151354, -4.4148378, 1.9919628, -6.7230315, 6.5299730
3: -4.1597300, 2.1926591, -3.8993604, 2.0811436, -6.2408733, 6.0920196
4: -4.4585991, 2.8515229, -4.2039127, 2.6949525, -7.1535516, 7.0554357
5: -3.3874247, 2.9933815, -3.1889741, 2.8159170, -6.2033415, 6.1823559
6: -3.5310397, 3.0180154, -3.3291271, 2.8448286, -6.3758683, 6.3471422
7: -4.0621114, 3.0612557, -3.8204763, 2.8918672, -6.9539785, 6.8817320
8: -4.6329117, 2.8379836, -4.3540449, 2.6883829, -7.3212948, 7.1920285
9: -3.3451328, 3.7843564, -3.1628327, 3.5725718, -6.9177046, 6.9471893

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321111, upper bound: 7.6319544
time: 2.99 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323775, upper bound: 7.6320638
time: 3.96 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -5.2606316, 4.0340438, -0.5557878, 0.6062570, -5.8668885, 4.5898314
1: -4.1306477, 3.6891751, -0.5694659, 0.5890498, -4.7196975, 4.2586412
2: -6.8474569, 2.8021827, -0.0229675, 1.1130555, -7.9605122, 2.8251503
3: -5.9835024, 2.9768682, -0.4095631, 0.6116136, -6.5951161, 3.3864312
4: -6.2607584, 3.9249229, -0.6402311, 0.6313424, -6.8921008, 4.5651541
5: -4.7226477, 4.1463680, -0.5602109, 0.6672503, -5.3898978, 4.7065792
6: -4.9779630, 4.2266397, -0.5193539, 0.6288786, -5.6068416, 4.7459936
7: -5.7240138, 4.2302332, -0.5874051, 0.6950583, -6.4190722, 4.8176384
8: -6.5460443, 3.9008555, -0.6575504, 0.7276987, -7.2737432, 4.5584059
9: -4.6121783, 5.2939205, -0.6189623, 0.6805410, -5.2927194, 5.9128828

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295175, upper bound: 7.5921053
time: 2.57 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6297318, upper bound: 7.5921115
time: 3.70 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -7.7303872, 5.8465171, -3.4877551, 2.7238524, -10.4542398, 9.3342724
1: -6.1143317, 5.2959018, -2.6922045, 2.5373626, -8.6516943, 7.9881063
2: -10.1951923, 3.8833094, -4.4070063, 1.9889437, -12.1841364, 8.2903156
3: -8.9016380, 4.2269974, -3.8927171, 2.0781806, -10.9798183, 8.1197147
4: -9.1380529, 5.6424289, -4.1971903, 2.6908107, -11.8288631, 9.8396187
5: -6.8389053, 6.0036030, -3.1839147, 2.8114188, -9.6503239, 9.1875172
6: -7.2690620, 6.1530428, -3.3240852, 2.8402147, -10.1092768, 9.4771280
7: -8.3693933, 6.0856738, -3.8142872, 2.8871946, -11.2565880, 9.8999615
8: -9.6076632, 5.5922937, -4.3467073, 2.6845250, -12.2921886, 9.9390011
9: -6.6355920, 7.7249413, -3.1579185, 3.5671012, -10.2026930, 10.8828602

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317781, upper bound: 7.6319400
time: 3.11 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320497, upper bound: 7.6320491
time: 4.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 8.99 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.99
Output dim: 2, lower bound: -7.6298486, upper bound: 7.5921209
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.99
Output dim: 2, lower bound: -7.6300558, upper bound: 7.5921268
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.99
Output dim: 2, lower bound: -7.6321111, upper bound: 7.6319544
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.99
Output dim: 2, lower bound: -7.6323775, upper bound: 7.6320638
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 8.99
Output dim: 2, lower bound: -7.6295175, upper bound: 7.5921053
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 8.99
Output dim: 2, lower bound: -7.6297318, upper bound: 7.5921115
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 8.99
Output dim: 2, lower bound: -7.6317781, upper bound: 7.6319400
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 8.99
Output dim: 2, lower bound: -7.6320497, upper bound: 7.6320491

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3333976, 0.3914432, -0.2905789, 0.3437701, -0.6771677, 0.6820221
1: -0.3798088, 0.4047638, -0.3401514, 0.3645336, -0.7443423, 0.7449152
2: 0.3155932, 1.0602779, 0.3814314, 1.0477935, -0.7322004, 0.6788466
3: -0.2319664, 0.4602215, -0.1985590, 0.4266787, -0.6586452, 0.6587806
4: -0.4134810, 0.4084299, -0.3717424, 0.3594608, -0.7729418, 0.7801722
5: -0.3684323, 0.4222377, -0.3328483, 0.3705817, -0.7390140, 0.7550860
6: -0.3177894, 0.4447805, -0.2799492, 0.4019449, -0.7197343, 0.7247297
7: -0.3693723, 0.4627319, -0.3272378, 0.4133050, -0.7826774, 0.7899697
8: -0.4316288, 0.5203241, -0.3869185, 0.4767710, -0.9083998, 0.9072425
9: -0.4256843, 0.4120018, -0.3843033, 0.3588696, -0.7845539, 0.7963052

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5782457, upper bound: 7.4237524
time: 3.63 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5907426, upper bound: 7.4419723
time: 3.95 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4986709, 0.5706198, -0.3223923, 0.3802877, -0.8789586, 0.8930121
1: -0.5342549, 0.5575686, -0.3700377, 0.3934768, -0.9277316, 0.9276062
2: 0.0538166, 1.1017479, 0.3316635, 1.0560672, -1.0022506, 0.7700843
3: -0.3579127, 0.5861883, -0.2233460, 0.4518900, -0.8098027, 0.8095343
4: -0.5790919, 0.5939325, -0.4023728, 0.3955271, -0.9746190, 0.9963053
5: -0.5266190, 0.6177903, -0.3597596, 0.4084856, -0.9351045, 0.9775499
6: -0.4735512, 0.5954361, -0.3087335, 0.4335376, -0.9070888, 0.9041696
7: -0.5333483, 0.6562901, -0.3586983, 0.4496297, -0.9829780, 1.0149883
8: -0.6014456, 0.6930774, -0.4189287, 0.5098346, -1.1112802, 1.1120062
9: -0.5765086, 0.6309670, -0.4144814, 0.3987846, -0.9752933, 1.0454483

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6100688, upper bound: 7.4237646
time: 4.15 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6236375, upper bound: 7.4419830
time: 4.97 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.4185922, 1.1893823, -2.0892038, 1.6900302, -3.1086226, 3.2785861
1: -1.1736836, 1.1054025, -1.6843858, 1.5750431, -2.7487268, 2.7897882
2: -1.3178990, 1.2989743, -2.3631973, 1.4940966, -2.8119955, 3.6621716
3: -1.4027975, 1.0144144, -2.2173145, 1.3687757, -2.7715731, 3.2317290
4: -1.6602342, 1.2139227, -2.4658840, 1.7001911, -3.3604255, 3.6798067
5: -1.3011694, 1.2436323, -1.9289269, 1.7280935, -3.0292630, 3.1725593
6: -1.2992735, 1.2169538, -1.9765177, 1.7369328, -3.0362062, 3.1934714
7: -1.5074570, 1.2994499, -2.2617607, 1.8216281, -3.3290851, 3.5612106
8: -1.6410565, 1.2769331, -2.5631285, 1.7302958, -3.3713522, 3.8400617
9: -1.3334750, 1.4763566, -1.9429376, 2.1642482, -3.4977231, 3.4192944

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313577, upper bound: 7.6311425
time: 5.12 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313901, upper bound: 7.6312991
time: 5.17 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -2.3783522, 1.9143577, -2.5006492, 2.0082731, -4.3866253, 4.4150066
1: -1.9033251, 1.7878979, -1.9943868, 1.8757545, -3.7790794, 3.7822847
2: -2.8250113, 1.5877448, -3.0049865, 1.6298800, -4.4548912, 4.5927315
3: -2.5734646, 1.5246955, -2.7219296, 1.5882483, -4.1617126, 4.2466249
4: -2.8427198, 1.9167153, -3.0055821, 2.0071166, -4.8498363, 4.9222975
5: -2.1991792, 1.9594783, -2.3115456, 2.0490146, -4.2481937, 4.2710238
6: -2.2769432, 1.9716136, -2.4015241, 2.0722210, -4.3491640, 4.3731375
7: -2.5938721, 2.0508959, -2.7379434, 2.1445317, -4.7384038, 4.7888393
8: -2.9590118, 1.9327288, -3.1248202, 2.0161703, -4.9751821, 5.0575490
9: -2.2147775, 2.4671032, -2.3276129, 2.5942070, -4.8089848, 4.7947159

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317447, upper bound: 7.6312847
time: 3.56 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318296, upper bound: 7.6314898
time: 3.26 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -2.9300737, 2.3124051, -0.2925939, 0.3461017, -3.2761755, 2.6049991
1: -2.2954106, 2.1594157, -0.3420597, 0.3663462, -2.6617570, 2.5014753
2: -3.6209393, 1.7757378, 0.3782877, 1.0483220, -4.6692610, 1.3974501
3: -3.2197976, 1.8037931, -0.2001291, 0.4282880, -3.6480856, 2.0039222
4: -3.5212486, 2.3048115, -0.3736981, 0.3617249, -3.8829734, 2.6785097
5: -2.6923056, 2.3797889, -0.3345095, 0.3730020, -3.0653076, 2.7142982
6: -2.8103781, 2.4052782, -0.2817760, 0.4039619, -3.2143400, 2.6870542
7: -3.2086201, 2.4628880, -0.3292463, 0.4155810, -3.6242011, 2.7921343
8: -3.6521208, 2.3040445, -0.3889622, 0.4788768, -4.1309977, 2.6930068
9: -2.6906672, 2.9976594, -0.3862301, 0.3613638, -3.0520310, 3.3838897

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5520415, upper bound: 7.4237377
time: 3.75 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5658081, upper bound: 7.4419559
time: 2.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -3.7248497, 2.8976264, -0.3241472, 0.3822271, -4.1070766, 3.2217736
1: -2.8930321, 2.6865845, -0.3716250, 0.3951566, -3.2881887, 3.0582094
2: -4.7262278, 2.0969739, 0.3288835, 1.0565068, -5.7827344, 1.7680904
3: -4.1775885, 2.1988058, -0.2247138, 0.4532290, -4.6308174, 2.4235196
4: -4.4739814, 2.8517616, -0.4039997, 0.3975989, -4.8715801, 3.2557614
5: -3.3923213, 2.9846625, -0.3614194, 0.4104984, -3.8028197, 3.3460820
6: -3.5547693, 3.0238662, -0.3103067, 0.4352152, -3.9899845, 3.3341730
7: -4.0720811, 3.0637934, -0.3603692, 0.4517347, -4.5238156, 3.4241626
8: -4.6390657, 2.8392015, -0.4206290, 0.5116116, -5.1506772, 3.2598305
9: -3.3516498, 3.7903810, -0.4160843, 0.4011247, -3.7527745, 4.2064652

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5846924, upper bound: 7.4237490
time: 3.42 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5981088, upper bound: 7.4419668
time: 2.58 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -5.1899810, 3.9848611, -2.0865409, 1.6880025, -6.8779836, 6.0714021
1: -4.0689030, 3.6438177, -1.6823046, 1.5728048, -5.6417079, 5.3261223
2: -6.7670326, 2.7675748, -2.3592355, 1.4930514, -8.2600842, 5.1268101
3: -5.9054689, 2.9408338, -2.2140272, 1.3672574, -7.2727261, 5.1548610
4: -6.1776762, 3.8836732, -2.4625371, 1.6980692, -7.8757453, 6.3462105
5: -4.6614261, 4.0921469, -1.9265571, 1.7259963, -6.3874226, 6.0187039
6: -4.9095602, 4.1735258, -1.9740138, 1.7347361, -6.6442962, 6.1475396
7: -5.6487226, 4.1747627, -2.2588077, 1.8191582, -7.4678807, 6.4335704
8: -6.4638305, 3.8584113, -2.5593104, 1.7284242, -8.1922550, 6.4177217
9: -4.5519452, 5.2328434, -1.9402872, 2.1616285, -6.7135735, 7.1731305

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309548, upper bound: 7.6311293
time: 2.92 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309809, upper bound: 7.6312862
time: 2.51 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.3061767, 4.8010764, -2.4932714, 2.0027745, -8.3089514, 7.2943478
1: -4.9644089, 4.3706656, -1.9889964, 1.8703084, -6.8347173, 6.3596621
2: -8.2766457, 3.2493119, -2.9936328, 1.6272013, -9.9038467, 6.2429447
3: -7.2238617, 3.5032060, -2.7133799, 1.5844284, -8.8082905, 6.2165861
4: -7.4828362, 4.6525927, -2.9961898, 2.0015984, -9.4844341, 7.6487827
5: -5.6182480, 4.9351974, -2.3051155, 2.0427651, -7.6610131, 7.2403126
6: -5.9456511, 5.0418243, -2.3944769, 2.0661933, -8.0118446, 7.4363012
7: -6.8432727, 5.0129709, -2.7294922, 2.1386456, -8.9819183, 7.7424631
8: -7.8419909, 4.6119442, -3.1151116, 2.0110481, -9.8530388, 7.7270555
9: -5.4666567, 6.3316493, -2.3208833, 2.5869045, -8.0535612, 8.6525326

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314003, upper bound: 7.6312725
time: 6.32 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314765, upper bound: 7.6314766
time: 2.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 10.55 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.5782457, upper bound: 7.4237524
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.5907426, upper bound: 7.4419723
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6100688, upper bound: 7.4237646
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6236375, upper bound: 7.4419830
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6313577, upper bound: 7.6311425
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6313901, upper bound: 7.6312991
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6317447, upper bound: 7.6312847
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6318296, upper bound: 7.6314898
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.5520415, upper bound: 7.4237377
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.5658081, upper bound: 7.4419559
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.5846924, upper bound: 7.4237490
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.5981088, upper bound: 7.4419668
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6309548, upper bound: 7.6311293
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6309809, upper bound: 7.6312862
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6314003, upper bound: 7.6312725
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 10.55
Output dim: 2, lower bound: -7.6314765, upper bound: 7.6314766

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.7317178, 0.7185454, -0.7929900, 0.7583159, -1.4900336, 1.5115354
1: -0.6759588, 0.6797937, -0.7185289, 0.7115272, -1.3874860, 1.3983226
2: -0.2658774, 1.1428313, -0.3585106, 1.1527706, -1.4186480, 1.5013419
3: -0.5812554, 0.6831681, -0.6515709, 0.7089669, -1.2902223, 1.3347390
4: -0.8310788, 0.7485945, -0.9047503, 0.7874457, -1.6185246, 1.6533448
5: -0.6855746, 0.7883697, -0.7394320, 0.8281471, -1.5137217, 1.5278016
6: -0.6633064, 0.7334579, -0.7184473, 0.7731820, -1.4364884, 1.4519051
7: -0.7438910, 0.7988285, -0.8114254, 0.8378454, -1.5817363, 1.6102538
8: -0.8097368, 0.8300622, -0.8744253, 0.8662732, -1.6760101, 1.7044874
9: -0.7433736, 0.8382888, -0.7931163, 0.8927635, -1.6361370, 1.6314051

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284107, upper bound: 7.5747388
time: 5.22 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5898775, upper bound: 7.5738587
time: 3.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.8152153, 0.7724587, -1.0953263, 0.9577310, -1.7729464, 1.8677850
1: -0.7337672, 0.7260442, -0.9332901, 0.8896244, -1.6233915, 1.6593344
2: -0.3903053, 1.1581259, -0.8131747, 1.2119783, -1.6022837, 1.9713006
3: -0.6765458, 0.7192363, -1.0128919, 0.8492694, -1.5258152, 1.7321281
4: -0.9320459, 0.8022876, -1.2695751, 0.9869640, -1.9190099, 2.0718627
5: -0.7576710, 0.8430766, -1.0059943, 1.0237658, -1.7814368, 1.8490709
6: -0.7366171, 0.7879646, -0.9938402, 0.9795444, -1.7161615, 1.7818048
7: -0.8361003, 0.8533172, -1.1480978, 1.0498991, -1.8859994, 2.0014150
8: -0.8989328, 0.8807882, -1.2088687, 1.0631993, -1.9621320, 2.0896568
9: -0.8126513, 0.9107112, -1.0502301, 1.1578590, -1.9705102, 1.9609413

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286185, upper bound: 7.6199156
time: 3.30 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5995114, upper bound: 7.6181704
time: 5.49 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.5686719, 1.3020871, -1.1195989, 0.9739500, -2.5426221, 2.4216859
1: -1.2885816, 1.2065192, -0.9508948, 0.9049819, -2.1935635, 2.1574140
2: -1.5558890, 1.3404009, -0.8495792, 1.2176940, -2.7735829, 2.1899800
3: -1.5862117, 1.0920445, -1.0420500, 0.8608696, -2.4470813, 2.1340945
4: -1.8417703, 1.3207878, -1.2987974, 1.0037329, -2.8455033, 2.6195850
5: -1.4424627, 1.3503354, -1.0272074, 1.0398750, -2.4823377, 2.3775427
6: -1.4452633, 1.3315471, -1.0163536, 0.9966989, -2.4419622, 2.3479009
7: -1.6784607, 1.4126933, -1.1754885, 1.0678068, -2.7462676, 2.5881817
8: -1.8465075, 1.3790041, -1.2378874, 1.0790522, -2.9255598, 2.6168914
9: -1.4689999, 1.6314906, -1.0709772, 1.1798601, -2.6488600, 2.7024679

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290721, upper bound: 7.5768157
time: 4.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275580, upper bound: 7.5767642
time: 5.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.7579027, 1.4441255, -1.5165784, 1.2632952, -3.0211978, 2.9607038
1: -1.4326425, 1.3392296, -1.2482587, 1.1679488, -2.6005912, 2.5874882
2: -1.8536398, 1.3950622, -1.4751046, 1.3237660, -3.1774058, 2.8701668
3: -1.8157661, 1.1916077, -1.5235292, 1.0641059, -2.8798718, 2.7151370
4: -2.0698128, 1.4582922, -1.7789069, 1.2814001, -3.3512130, 3.2371993
5: -1.6194270, 1.4875571, -1.3930905, 1.3132815, -2.9327085, 2.8806477
6: -1.6338817, 1.4779567, -1.3965034, 1.2912085, -2.9250903, 2.8744602
7: -1.8932590, 1.5600474, -1.6197919, 1.3703225, -3.2635815, 3.1798391
8: -2.1075404, 1.5076389, -1.7747239, 1.3422369, -3.4497771, 3.2823629
9: -1.6417733, 1.8263435, -1.4207835, 1.5786940, -3.2204673, 3.2471271

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6293899, upper bound: 7.6254999
time: 3.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279418, upper bound: 7.6254488
time: 2.99 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.2271070, 3.2726207, -0.7937428, 0.7588124, -4.9859195, 4.0663633
1: -3.2897089, 3.0149086, -0.7190876, 0.7116735, -4.0013824, 3.7339962
2: -5.4455490, 2.3494608, -0.3597775, 1.1526936, -6.5982428, 2.7092383
3: -4.7652617, 2.4503338, -0.6525024, 0.7092580, -5.4745197, 3.1028361
4: -5.0502262, 3.2090654, -0.9055327, 0.7878084, -5.8380346, 4.1145983
5: -3.8339288, 3.3724794, -0.7402468, 0.8284810, -4.6624098, 4.1127262
6: -4.0143967, 3.4197440, -0.7192957, 0.7736220, -4.7880187, 4.1390400
7: -4.6151738, 3.4437671, -0.8121916, 0.8381969, -5.4533706, 4.2559586
8: -5.2651196, 3.1955051, -0.8750511, 0.8666887, -6.1318083, 4.0705562
9: -3.7618344, 4.2789698, -0.7935185, 0.8935554, -4.6553898, 5.0724883

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6278441, upper bound: 7.5746780
time: 3.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5612255, upper bound: 7.5738440
time: 5.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.3341198, 3.3515897, -1.0956788, 0.9580144, -5.2921343, 4.4472685
1: -3.3753531, 3.0853066, -0.9335394, 0.8895314, -4.2648845, 4.0188460
2: -5.5908589, 2.3929620, -0.8138413, 1.2117876, -6.8026466, 3.2068033
3: -4.8932047, 2.5038779, -1.0133798, 0.8493870, -5.7425919, 3.5172577
4: -5.1777296, 3.2832296, -1.2698984, 0.9870245, -6.1647539, 4.5531282
5: -3.9260447, 3.4519231, -1.0064586, 1.0238881, -4.9499331, 4.4583817
6: -4.1133571, 3.5039458, -0.9944484, 0.9797000, -5.0930572, 4.4983940
7: -4.7297664, 3.5249200, -1.1484995, 1.0498092, -5.7795753, 4.6734195
8: -5.3985310, 3.2671819, -1.2090654, 1.0633667, -6.4618979, 4.4762473
9: -3.8504019, 4.3863964, -1.0502619, 1.1584013, -5.0088034, 5.4366584

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281822, upper bound: 7.6198164
time: 3.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5621274, upper bound: 7.6180378
time: 4.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -5.2376146, 4.0147638, -1.1160524, 0.9716008, -6.2092152, 5.1308165
1: -4.0991101, 3.6742914, -0.9483142, 0.9025266, -5.0016365, 4.6226058
2: -6.8261819, 2.7894740, -0.8443509, 1.2166736, -8.0428553, 3.6338248
3: -5.9585352, 2.9583344, -1.0377991, 0.8591354, -6.8176708, 3.9961336
4: -6.2331538, 3.9071767, -1.2944229, 1.0011609, -7.2343149, 5.2015996
5: -4.7039447, 4.1378441, -1.0241930, 1.0374300, -5.7413750, 5.1620369
6: -4.9501162, 4.2079659, -1.0132357, 0.9941467, -5.9442630, 5.2212014
7: -5.6978645, 4.2052526, -1.1715020, 1.0649455, -6.7628098, 5.3767548
8: -6.5142603, 3.8792896, -1.2335163, 1.0766891, -7.5909495, 5.1128058
9: -4.5897474, 5.2761517, -1.0677705, 1.1767373, -5.7664847, 6.3439221

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287645, upper bound: 7.5767999
time: 3.26 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6172729, upper bound: 7.5767563
time: 3.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.5233407, 4.2238441, -1.5107791, 1.2588996, -6.7822404, 5.7346230
1: -4.3296781, 3.8607659, -1.2437270, 1.1636184, -5.4932966, 5.1044931
2: -7.2109175, 2.9068747, -1.4660275, 1.3218682, -8.5327854, 4.3729019
3: -6.2982945, 3.1030858, -1.5164950, 1.0610054, -7.3592997, 4.6195807
4: -6.5698738, 4.1046391, -1.7717253, 1.2770045, -7.8468781, 5.8763642
5: -4.9482679, 4.3498564, -1.3876824, 1.3090060, -6.2572737, 5.7375388
6: -5.2164607, 4.4309359, -1.3911073, 1.2866461, -6.5031071, 5.8220434
7: -6.0038457, 4.4205470, -1.6131645, 1.3654910, -7.3693366, 6.0337114
8: -6.8684826, 4.0721989, -1.7665839, 1.3381821, -8.2066650, 5.8387828
9: -4.8246260, 5.5596504, -1.4152484, 1.5728509, -6.3974771, 6.9748988

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290769, upper bound: 7.6254844
time: 3.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6254403, upper bound: 7.6254399
time: 3.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 7.69 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6284107, upper bound: 7.5747388
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.5898775, upper bound: 7.5738587
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6286185, upper bound: 7.6199156
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.5995114, upper bound: 7.6181704
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6290721, upper bound: 7.5768157
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6275580, upper bound: 7.5767642
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6293899, upper bound: 7.6254999
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6279418, upper bound: 7.6254488
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6278441, upper bound: 7.5746780
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.5612255, upper bound: 7.5738440
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6281822, upper bound: 7.6198164
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.5621274, upper bound: 7.6180378
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6287645, upper bound: 7.5767999
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6172729, upper bound: 7.5767563
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6290769, upper bound: 7.6254844
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.69
Output dim: 2, lower bound: -7.6254403, upper bound: 7.6254399

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.5238148, 0.5874259, -0.7141889, 0.7076795, -1.2314943, 1.3016148
1: -0.5506582, 0.5735485, -0.6641570, 0.6687328, -1.2193910, 1.2377055
2: 0.0187040, 1.1071550, -0.2415674, 1.1386582, -1.1199541, 1.3487225
3: -0.3798209, 0.5992303, -0.5630791, 0.6757982, -1.0556191, 1.1623094
4: -0.6070502, 0.6112440, -0.8113751, 0.7359273, -1.3429775, 1.4226191
5: -0.5412977, 0.6427230, -0.6719127, 0.7764802, -1.3177779, 1.3146358
6: -0.4938495, 0.6112430, -0.6490000, 0.7225909, -1.2164404, 1.2602431
7: -0.5591760, 0.6757990, -0.7260889, 0.7865530, -1.3457290, 1.4018879
8: -0.6289263, 0.7098529, -0.7925314, 0.8195826, -1.4485090, 1.5023842
9: -0.5958037, 0.6544506, -0.7287630, 0.8239583, -1.4197621, 1.3832135

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284107, upper bound: 7.5747385
time: 3.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284107, upper bound: 7.5747384
time: 4.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.5823771, 0.6210498, -1.0090517, 0.8999099, -1.4822869, 1.6301014
1: -0.5844750, 0.6021883, -0.8704338, 0.8377842, -1.4222592, 1.4726222
2: -0.0566150, 1.1177000, -0.6829252, 1.1939096, -1.2505246, 1.8006252
3: -0.4345541, 0.6209863, -0.9089651, 0.8080681, -1.2426223, 1.5299515
4: -0.6654260, 0.6497183, -1.1663246, 0.9271958, -1.5926218, 1.8160429
5: -0.5765622, 0.6857051, -0.9287852, 0.9674246, -1.5439868, 1.6144903
6: -0.5406960, 0.6420517, -0.9127439, 0.9202765, -1.4609725, 1.5547956
7: -0.6089383, 0.7104550, -1.0509160, 0.9875919, -1.5965302, 1.7613709
8: -0.6791611, 0.7426566, -1.1080711, 1.0061922, -1.6853533, 1.8507277
9: -0.6368469, 0.7021062, -0.9771293, 1.0805006, -1.7173475, 1.6792356

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286186, upper bound: 7.6199159
time: 4.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286186, upper bound: 7.6199158
time: 3.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.2915177, 1.0959535, -1.0340779, 0.9165971, -2.2081149, 2.1300313
1: -1.0766857, 1.0193396, -0.8886971, 0.8529612, -1.9296468, 1.9080367
2: -1.1180363, 1.2629619, -0.7204940, 1.1993576, -2.3173938, 1.9834559
3: -1.2487054, 0.9481695, -0.9392132, 0.8198888, -2.0685942, 1.8873827
4: -1.5071225, 1.1227316, -1.1966248, 0.9447076, -2.4518301, 2.3193564
5: -1.1836765, 1.1559740, -0.9507950, 0.9841447, -2.1678212, 2.1067691
6: -1.1775451, 1.1217510, -0.9362806, 0.9377007, -2.1152458, 2.0580316
7: -1.3660929, 1.2000407, -1.0794511, 1.0055071, -2.3716002, 2.2794919
8: -1.4672322, 1.1916207, -1.1375574, 1.0226184, -2.4898505, 2.3291781
9: -1.2204726, 1.3486145, -0.9987484, 1.1025630, -2.3230357, 2.3473628

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290721, upper bound: 7.5768157
time: 3.22 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290721, upper bound: 7.5768158
time: 4.24 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -3.0375018, 2.4050846, -0.6626386, 0.6742908, -3.7117925, 3.0677233
1: -2.4143798, 2.2464247, -0.6305395, 0.6425101, -3.0568900, 2.8769641
2: -3.8604703, 1.7793603, -0.1669347, 1.1309043, -4.9913745, 1.9462950
3: -3.3684554, 1.8570858, -0.5108564, 0.6539789, -4.0224342, 2.3679423
4: -3.6196191, 2.3890729, -0.7527900, 0.7029091, -4.3225284, 3.1418629
5: -2.8202264, 2.4120362, -0.6312702, 0.7425683, -3.5627947, 3.0433064
6: -2.8457251, 2.4687986, -0.6046268, 0.6908599, -3.5365849, 3.0734253
7: -3.3505766, 2.5562243, -0.6775585, 0.7559941, -4.1065707, 3.2337828
8: -3.8783236, 2.3839271, -0.7462359, 0.7897641, -4.6680875, 3.1301630
9: -2.8140163, 3.1375153, -0.6928753, 0.7765326, -3.5905490, 3.8303907

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275580, upper bound: 7.5767636
time: 3.97 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275580, upper bound: 7.5767633
time: 5.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.4733255, 1.2309141, -1.4282154, 1.1972916, -2.6706171, 2.6591296
1: -1.2154164, 1.1403990, -1.1802998, 1.1076736, -2.3230901, 2.3206987
2: -1.4060476, 1.3131672, -1.3353817, 1.2987655, -2.7048130, 2.6485491
3: -1.4708030, 1.0420879, -1.4157023, 1.0180531, -2.4888561, 2.4577904
4: -1.7274957, 1.2513086, -1.6719120, 1.2178168, -2.9453125, 2.9232206
5: -1.3518810, 1.2832844, -1.3102055, 1.2506785, -2.6025596, 2.5934899
6: -1.3533399, 1.2585435, -1.3111658, 1.2236434, -2.5769835, 2.5697093
7: -1.5700779, 1.3395661, -1.5189342, 1.3024961, -2.8725739, 2.8585005
8: -1.7164344, 1.3132712, -1.6536034, 1.2816886, -2.9981229, 2.9668746
9: -1.3826953, 1.5335649, -1.3403618, 1.4883040, -2.8709993, 2.8739266

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6293899, upper bound: 7.6254995
time: 4.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6293899, upper bound: 7.6255000
time: 3.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 10.16 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6284107, upper bound: 7.5747385
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6284107, upper bound: 7.5747384
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6286186, upper bound: 7.6199159
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6286186, upper bound: 7.6199158
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6290721, upper bound: 7.5768157
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6290721, upper bound: 7.5768158
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6275580, upper bound: 7.5767636
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6275580, upper bound: 7.5767633
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6293899, upper bound: 7.6254995
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.16
Output dim: 2, lower bound: -7.6293899, upper bound: 7.6255000
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 10.16
Output dim: 2, lower bound: -7.6279418, upper bound: 7.6254488
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.16
Output dim: 2, lower bound: -7.6278441, upper bound: 7.5746780
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 10.16
Output dim: 2, lower bound: -7.6281822, upper bound: 7.6198164
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 10.16
Output dim: 2, lower bound: -7.6287645, upper bound: 7.5767999
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 10.16
Output dim: 2, lower bound: -7.6290769, upper bound: 7.6254844
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=9.388381958007812
rel_dist={2: [-7.633589185222909, 7.633589239850782]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 1976.66 seconds
