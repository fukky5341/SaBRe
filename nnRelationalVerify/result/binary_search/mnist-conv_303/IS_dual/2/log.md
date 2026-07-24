## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.3120148312
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.5105915, -10.3324680, -12.5105915, -10.3324680, -2.1781235, 2.1781235)
1: (3.1649604, 4.4064875, 3.1649604, 4.4064875, -1.1255052, 1.1255053)
2: (-4.9406466, -3.7702384, -4.9406466, -3.7702384, -1.1704082, 1.1704082)
3: (-12.7448177, -10.8541327, -12.7448177, -10.8541327, -1.7759883, 1.7759886)
4: (-2.4278350, -0.9076300, -2.4278350, -0.9076300, -1.4544666, 1.4544665)
5: (-10.0812330, -8.6300726, -10.0812330, -8.6300726, -1.3241396, 1.3241397)
6: (-8.0952625, -6.3753734, -8.0952625, -6.3753734, -1.7115004, 1.7115003)
7: (-2.7799165, -1.9236844, -2.7799165, -1.9236844, -0.8193477, 0.8193477)
8: (-3.7933545, -2.3891320, -3.7933545, -2.3891320, -1.4042225, 1.4042225)
9: (-12.4935493, -10.9539299, -12.4935493, -10.9539299, -1.5232468, 1.5232465)

## BASE Result
execution time: IAR + LP analysis = 13.24 + 31.84 = 45.07 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3554.93 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=0.8663967847824097
rel_dist={1: [-0.5771277606023322, 0.5771254352696031]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=0.7368426322937012
rel_dist={1: [-0.38431871261449, 0.38431867244677376]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=0.6504731178283691
rel_dist={1: [-0.2375077651971229, 0.23750601401867089]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=0.6936578750610352
rel_dist={1: [-0.3128648053580161, 0.31286157369120016]}

## Binary Search Result
Binary search time: 187.89 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Individual Split (IS_dual) starts
Time budget: 3367.04 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start
Binary search (step 0): status=Status.ADV_EXAMPLE, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=None

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5815
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4501825, upper bound: 0.4439257
time: 3.47 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509254, upper bound: 0.4509245
time: 3.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.03
Output dim: 1, lower bound: -0.4501825, upper bound: 0.4439257
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.03
Output dim: 1, lower bound: -0.4509254, upper bound: 0.4509245

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.4172287, -10.4103403, -12.5045977, -10.3702469, -1.5056851, 1.5585427
1: 3.2003183, 4.3859591, 3.1689262, 4.3971572, -0.7352107, 0.7530195
2: -4.9007716, -3.8007307, -4.9384508, -3.7855086, -0.8675041, 0.8892967
3: -12.6646280, -10.9396000, -12.7049980, -10.8597231, -1.1990869, 1.1506343
4: -2.4103954, -0.9145901, -2.4237161, -0.9086497, -1.1086111, 1.1111283
5: -10.0592670, -8.6618805, -10.0707741, -8.6334591, -0.8511039, 0.8370783
6: -8.0244331, -6.4216833, -8.0865364, -6.3963509, -1.1513684, 1.1848718
7: -2.7749660, -1.9314392, -2.7793074, -1.9270303, -0.5802813, 0.5804886
8: -3.7502260, -2.4320984, -3.7903309, -2.4096999, -0.9258177, 0.9469641
9: -12.4648705, -10.9785194, -12.4826956, -10.9589033, -1.0685978, 1.0670943

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4495625, upper bound: 0.4378820
time: 3.44 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4501789, upper bound: 0.4439201
time: 3.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.5105877, -10.3325186, -12.5105896, -10.3324928, -1.5798602, 1.5929289
1: 3.1649652, 4.4064684, 3.1649628, 4.4064789, -0.7698855, 0.7726337
2: -4.9406443, -3.7702651, -4.9406452, -3.7702515, -0.9010482, 0.8920180
3: -12.7447414, -10.8541403, -12.7447805, -10.8541355, -1.2133043, 1.2334883
4: -2.4278302, -0.9076324, -2.4278333, -0.9076309, -1.1251149, 1.1248498
5: -10.0812159, -8.6300774, -10.0812254, -8.6300764, -0.8692052, 0.8717290
6: -8.0952511, -6.3753991, -8.0952568, -6.3753858, -1.2286313, 1.2398872
7: -2.7799153, -1.9236879, -2.7799168, -1.9236858, -0.5883070, 0.5891856
8: -3.7933502, -2.3891625, -3.7933521, -2.3891459, -0.9599640, 0.9696902
9: -12.4935284, -10.9539337, -12.4935379, -10.9539318, -1.1034970, 1.1041341

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4459480, upper bound: 0.4503078
time: 3.53 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509132, upper bound: 0.4509151
time: 3.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.72 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 19.72
Output dim: 1, lower bound: -0.4495625, upper bound: 0.4378820
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 19.72
Output dim: 1, lower bound: -0.4501789, upper bound: 0.4439201
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 19.72
Output dim: 1, lower bound: -0.4459480, upper bound: 0.4503078
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 19.72
Output dim: 1, lower bound: -0.4509132, upper bound: 0.4509151

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -12.3105316, -10.5074368, -12.4939947, -10.4178162, -1.3474171, 1.4554880
1: 3.2318249, 4.3631792, 3.1731215, 4.3865547, -0.6922501, 0.7225113
2: -4.8479691, -3.8461206, -4.9346752, -3.8079932, -0.7890183, 0.8392475
3: -12.6164761, -10.9975052, -12.6812191, -10.8673668, -1.1465831, 1.0602703
4: -2.3746874, -0.9398980, -2.4181414, -0.9201157, -1.0498829, 1.0756731
5: -10.0452938, -8.6808701, -10.0645313, -8.6365843, -0.8313280, 0.8095739
6: -7.9756212, -6.4619083, -8.0782871, -6.4150171, -1.0781410, 1.1248531
7: -2.7575972, -1.9554555, -2.7780828, -1.9384670, -0.5482519, 0.5534686
8: -3.7258692, -2.4545965, -3.7858667, -2.4185286, -0.8862617, 0.9157314
9: -12.4249992, -11.0154867, -12.4759989, -10.9762726, -1.0116637, 1.0229461

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4489434, upper bound: 0.4329081
time: 4.11 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4495508, upper bound: 0.4378699
time: 3.36 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -12.4172192, -10.4104376, -12.5045948, -10.3702936, -1.4432454, 1.5132728
1: 3.2003226, 4.3859491, 3.1689291, 4.3971524, -0.7251697, 0.7432228
2: -4.9007673, -3.8007407, -4.9384489, -3.7855136, -0.8385336, 0.8610290
3: -12.6646061, -10.9396076, -12.7049866, -10.8597279, -1.1838899, 1.1264391
4: -2.4103894, -0.9146023, -2.4237130, -0.9086592, -1.1066260, 1.1087301
5: -10.0592642, -8.6618843, -10.0707703, -8.6334600, -0.8486501, 0.8386961
6: -8.0244217, -6.4217129, -8.0865316, -6.3963633, -1.1438086, 1.1685044
7: -2.7749653, -1.9314508, -2.7793076, -1.9270363, -0.5753552, 0.5595282
8: -3.7502241, -2.4321170, -3.7903299, -2.4097066, -0.9211725, 0.9285527
9: -12.4648628, -10.9785519, -12.4826937, -10.9589233, -1.0685792, 1.0546296

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4495749, upper bound: 0.4389710
time: 3.55 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4501672, upper bound: 0.4439080
time: 3.67 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -12.5054913, -10.3411045, -12.5081387, -10.3369017, -1.5673213, 1.5788028
1: 3.1846991, 4.4036179, 3.1748090, 4.4050951, -0.7463422, 0.7537141
2: -4.9352884, -3.7796023, -4.9380336, -3.7748985, -0.8901377, 0.8796351
3: -12.7415752, -10.8856039, -12.7432547, -10.8698130, -1.1917057, 1.2008808
4: -2.4227071, -0.9344304, -2.4253223, -0.9209964, -1.1068554, 1.0950351
5: -10.0226917, -8.6355553, -10.0520935, -8.6327696, -0.8084671, 0.8290290
6: -8.0490026, -6.3774276, -8.0722084, -6.3763657, -1.1774492, 1.2025752
7: -2.7785952, -1.9371884, -2.7792561, -1.9304552, -0.5738448, 0.5696794
8: -3.7881212, -2.4324932, -3.7908154, -2.4106970, -0.9262002, 0.9236164
9: -12.4645138, -10.9591599, -12.4789753, -10.9564972, -1.0586574, 1.0665449

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400261, upper bound: 0.4498235
time: 3.58 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4459439, upper bound: 0.4503025
time: 3.50 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -12.5141678, -10.3311129, -12.5105896, -10.3325014, -1.5820775, 1.5919671
1: 3.1629658, 4.4210567, 3.1649809, 4.4064770, -0.7628355, 0.7725947
2: -4.9461975, -3.7676620, -4.9406428, -3.7702603, -0.9039221, 0.8934047
3: -12.7674427, -10.8529835, -12.7447786, -10.8541584, -1.2161362, 1.2253056
4: -2.4501011, -0.9069724, -2.4278319, -0.9076481, -1.1378977, 1.1151757
5: -10.0845270, -8.5760803, -10.0812063, -8.6300774, -0.8392289, 0.8726747
6: -8.0979557, -6.3389297, -8.0952396, -6.3753867, -1.2086594, 1.2384880
7: -2.7885222, -1.9220767, -2.7799156, -1.9236960, -0.5914397, 0.5849799
8: -3.8365717, -2.3891459, -3.7933497, -2.3891683, -0.9647344, 0.9496807
9: -12.4970646, -10.9363213, -12.4935312, -10.9539347, -1.0969818, 1.1009103

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6193

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4449915, upper bound: 0.4504308
time: 3.55 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509092, upper bound: 0.4509116
time: 3.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.62 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 19.62
Output dim: 1, lower bound: -0.4489434, upper bound: 0.4329081
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 19.62
Output dim: 1, lower bound: -0.4495508, upper bound: 0.4378699
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 19.62
Output dim: 1, lower bound: -0.4495749, upper bound: 0.4389710
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 19.62
Output dim: 1, lower bound: -0.4501672, upper bound: 0.4439080
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 19.62
Output dim: 1, lower bound: -0.4400261, upper bound: 0.4498235
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 19.62
Output dim: 1, lower bound: -0.4459439, upper bound: 0.4503025
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 19.62
Output dim: 1, lower bound: -0.4449915, upper bound: 0.4504308
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 19.62
Output dim: 1, lower bound: -0.4509092, upper bound: 0.4509116

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -12.3080730, -10.5118294, -12.4888926, -10.4263945, -1.3332477, 1.4441886
1: 3.2414255, 4.3617992, 3.1927662, 4.3837061, -0.6734921, 0.6993831
2: -4.8454027, -3.8507476, -4.9293575, -3.8173215, -0.7766843, 0.8294257
3: -12.6149521, -11.0131340, -12.6780539, -10.8988094, -1.1134195, 1.0386478
4: -2.3722448, -0.9532638, -2.4130685, -0.9469142, -1.0201402, 1.0574594
5: -10.0161638, -8.6836395, -10.0060110, -8.6420946, -0.7958139, 0.7489007
6: -7.9526148, -6.4629040, -8.0320005, -6.4170485, -1.0460198, 1.0742061
7: -2.7569575, -1.9621696, -2.7767856, -1.9519317, -0.5287608, 0.5390668
8: -3.7233586, -2.4761200, -3.7806568, -2.4618416, -0.8403540, 0.8884157
9: -12.4103851, -11.0180035, -12.4469862, -10.9814548, -0.9741831, 0.9783111

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4489393, upper bound: 0.4292143
time: 3.74 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4489410, upper bound: 0.4329024
time: 3.82 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -12.3105316, -10.5074463, -12.4975786, -10.4164066, -1.3464479, 1.4609184
1: 3.2318425, 4.3631773, 3.1711040, 4.4011412, -0.6922085, 0.7154669
2: -4.8479671, -3.8461289, -4.9402313, -3.8053916, -0.7904029, 0.8459030
3: -12.6164761, -10.9975300, -12.7039204, -10.8662281, -1.1383188, 1.0631018
4: -2.3746860, -0.9399152, -2.4404488, -0.9194562, -1.0402093, 1.0929854
5: -10.0452728, -8.6808720, -10.0678587, -8.5825472, -0.8395191, 0.7795955
6: -7.9756036, -6.4619083, -8.0808945, -6.3785462, -1.0819409, 1.1047401
7: -2.7575955, -1.9554651, -2.7866914, -1.9368539, -0.5440917, 0.5568155
8: -3.7258668, -2.4546194, -3.8290882, -2.4185205, -0.8662162, 0.9269298
9: -12.4249916, -11.0154896, -12.4794645, -10.9586496, -1.0143154, 1.0165372

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4495450, upper bound: 0.4341776
time: 3.36 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4495447, upper bound: 0.4378630
time: 3.49 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -12.4147615, -10.4148540, -12.4994955, -10.3788795, -1.4290819, 1.5007260
1: 3.2100120, 4.3845677, 3.1886129, 4.3943033, -0.7063659, 0.7200675
2: -4.8981762, -3.8053846, -4.9331112, -3.7948539, -0.8261623, 0.8500979
3: -12.6630802, -10.9552574, -12.7018223, -10.8911800, -1.1512690, 1.1048292
4: -2.4079621, -0.9279680, -2.4186492, -0.9354577, -1.0762920, 1.0881383
5: -10.0301323, -8.6645956, -10.0122452, -8.6389465, -0.8090863, 0.7779198
6: -8.0013990, -6.4227037, -8.0402260, -6.3983946, -1.1064756, 1.1178893
7: -2.7743151, -1.9381986, -2.7779989, -1.9405234, -0.5549849, 0.5451324
8: -3.7477002, -2.4536519, -3.7851105, -2.4530277, -0.8751352, 0.8947784
9: -12.4503250, -10.9810839, -12.4537067, -10.9641190, -1.0311391, 1.0098963

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4458812, upper bound: 0.4389642
time: 3.49 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4495682, upper bound: 0.4389620
time: 3.57 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -12.4172173, -10.4104462, -12.5081768, -10.3688831, -1.4422817, 1.5154867
1: 3.2003403, 4.3859491, 3.1669054, 4.4117403, -0.7251298, 0.7361894
2: -4.9007654, -3.8007488, -4.9440022, -3.7829120, -0.8399208, 0.8639028
3: -12.6646042, -10.9396305, -12.7276878, -10.8585825, -1.1757092, 1.1292713
4: -2.4103875, -0.9146199, -2.4460068, -0.9079995, -1.0970330, 1.1164784
5: -10.0592432, -8.6618862, -10.0740910, -8.5794420, -0.8527970, 0.8087317
6: -8.0244055, -6.4217124, -8.0891914, -6.3598938, -1.1424096, 1.1484418
7: -2.7749648, -1.9314604, -2.7879126, -1.9254270, -0.5687041, 0.5629606
8: -3.7502208, -2.4321384, -3.8335505, -2.4096947, -0.9011582, 0.9333216
9: -12.4648561, -10.9785528, -12.4861917, -10.9413052, -1.0611768, 1.0481703

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4501602, upper bound: 0.4402225
time: 3.51 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4501599, upper bound: 0.4438971
time: 3.49 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -12.4954987, -10.3886604, -12.4070768, -10.4338589, -1.4619460, 1.4204190
1: 3.1886330, 4.3929400, 3.2046795, 4.3821893, -0.7147671, 0.7080128
2: -4.9317207, -3.8021038, -4.8862982, -3.8204882, -0.8391991, 0.7989080
3: -12.7177887, -10.8928518, -12.6951733, -10.9245319, -1.1009343, 1.1473522
4: -2.4171791, -0.9459064, -2.3891451, -0.9463177, -1.0717378, 1.0360072
5: -10.0163803, -8.6385689, -10.0379782, -8.6514702, -0.7807205, 0.8085286
6: -8.0409145, -6.3962088, -8.0273275, -6.4173417, -1.1149323, 1.1250608
7: -2.7774129, -1.9485710, -2.7619030, -1.9548452, -0.5464092, 0.5379435
8: -3.7839699, -2.4413333, -3.7691393, -2.4328465, -0.8937297, 0.8836293
9: -12.4575930, -10.9764709, -12.4378881, -10.9928637, -1.0139832, 1.0076176

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4363307, upper bound: 0.4498201
time: 3.59 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400220, upper bound: 0.4498185
time: 3.72 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -12.5054865, -10.3411503, -12.5081339, -10.3369951, -1.5186722, 1.5202692
1: 3.1847000, 4.4036131, 3.1748128, 4.4050865, -0.7330227, 0.7427469
2: -4.9352875, -3.7796082, -4.9380298, -3.7749090, -0.8537059, 0.8495690
3: -12.7415657, -10.8856068, -12.7432346, -10.8698196, -1.1692935, 1.1805167
4: -2.4227047, -0.9344392, -2.4253178, -0.9210141, -1.0981264, 1.0912333
5: -10.0226879, -8.6355572, -10.0520897, -8.6327724, -0.8073438, 0.8227932
6: -8.0489979, -6.3774395, -8.0721989, -6.3763909, -1.1574438, 1.1878521
7: -2.7785947, -1.9371943, -2.7792554, -1.9304664, -0.5530952, 0.5643309
8: -3.7881184, -2.4325018, -3.7908130, -2.4107122, -0.9018562, 0.9193665
9: -12.4645100, -10.9591789, -12.4789696, -10.9565353, -1.0461912, 1.0575280

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4422500, upper bound: 0.4502977
time: 3.69 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4459387, upper bound: 0.4502978
time: 3.73 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -12.5041761, -10.3786755, -12.4095297, -10.4294796, -1.4766953, 1.4335852
1: 3.1669445, 4.4103780, 3.1949196, 4.3835692, -0.7313458, 0.7268647
2: -4.9426136, -3.7901745, -4.8888855, -3.8158648, -0.8529387, 0.8125947
3: -12.7436562, -10.8602467, -12.6966972, -10.9088974, -1.1253680, 1.1717889
4: -2.4445951, -0.9184484, -2.3916767, -0.9329693, -1.1018910, 1.0561659
5: -10.0782204, -8.5790529, -10.0670853, -8.6487484, -0.8114135, 0.8521863
6: -8.0898476, -6.3577118, -8.0504246, -6.4163580, -1.1461277, 1.1610181
7: -2.7873318, -1.9334850, -2.7625525, -1.9481149, -0.5638615, 0.5532359
8: -3.8324118, -2.3979979, -3.7716632, -2.4113274, -0.9322414, 0.9096525
9: -12.4901400, -10.9536390, -12.4525146, -10.9903202, -1.0523713, 1.0372453

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4412977, upper bound: 0.4504258
time: 3.39 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4449858, upper bound: 0.4504253
time: 3.45 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5141659, -10.3311586, -12.5105820, -10.3325939, -1.5334334, 1.5334368
1: 3.1629696, 4.4210515, 3.1649842, 4.4064674, -0.7494754, 0.7616018
2: -4.9461956, -3.7676671, -4.9406390, -3.7702708, -0.8674695, 0.8632817
3: -12.7674332, -10.8529873, -12.7447567, -10.8541641, -1.1937261, 1.2049463
4: -2.4500985, -0.9069810, -2.4278264, -0.9076660, -1.1264048, 1.1120520
5: -10.0845251, -8.5760822, -10.0812016, -8.6300802, -0.8381330, 0.8664505
6: -8.0979509, -6.3389411, -8.0952301, -6.3754134, -1.1887112, 1.2238224
7: -2.7885220, -1.9220824, -2.7799144, -1.9237070, -0.5706467, 0.5780585
8: -3.8365698, -2.3891549, -3.7933483, -2.3891850, -0.9403864, 0.9454330
9: -12.4970636, -10.9363403, -12.4935255, -10.9539719, -1.0845153, 1.0870844

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4472155, upper bound: 0.4509035
time: 3.39 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4509023, upper bound: 0.4509052
time: 3.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.72 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4489393, upper bound: 0.4292143
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4489410, upper bound: 0.4329024
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4495450, upper bound: 0.4341776
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4495447, upper bound: 0.4378630
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4458812, upper bound: 0.4389642
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4495682, upper bound: 0.4389620
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4501602, upper bound: 0.4402225
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4501599, upper bound: 0.4438971
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4363307, upper bound: 0.4498201
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4400220, upper bound: 0.4498185
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4422500, upper bound: 0.4502977
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4459387, upper bound: 0.4502978
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4412977, upper bound: 0.4504258
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4449858, upper bound: 0.4504253
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4472155, upper bound: 0.4509035
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 19.72
Output dim: 1, lower bound: -0.4509023, upper bound: 0.4509052

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.3025436, -10.5121193, -12.4888926, -10.4263945, -1.3274159, 1.4267979
1: 3.2416925, 4.3562431, 3.1927662, 4.3837061, -0.6717354, 0.6932969
2: -4.8453460, -3.8579822, -4.9293575, -3.8173215, -0.7730744, 0.7920753
3: -12.6142473, -11.0134525, -12.6780539, -10.8988094, -1.1175642, 1.0356387
4: -2.3706980, -0.9663267, -2.4130685, -0.9469142, -1.0172296, 1.0442085
5: -10.0138769, -8.6838465, -10.0060110, -8.6420946, -0.7934339, 0.7484376
6: -7.9516287, -6.4659610, -8.0320005, -6.4170485, -1.0445557, 1.0725675
7: -2.7511468, -1.9623013, -2.7767856, -1.9519317, -0.5152217, 0.5377479
8: -3.7227640, -2.4800091, -3.7806568, -2.4618416, -0.8495836, 0.8845149
9: -12.4039173, -11.0186110, -12.4469862, -10.9814548, -0.9697447, 0.9729497

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4452496, upper bound: 0.4292144
time: 3.41 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4452517, upper bound: 0.4292143
time: 3.56 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.3096638, -10.4717646, -12.4888792, -10.4263954, -1.3355949, 1.4501274
1: 3.2017856, 4.3649149, 3.1927681, 4.3837023, -0.6809334, 0.7088026
2: -4.8918471, -3.8480651, -4.9293571, -3.8173323, -0.7821200, 0.8434166
3: -12.6168613, -11.0033932, -12.6780519, -10.8988094, -1.1248053, 1.0432385
4: -2.4723935, -0.9503787, -2.4130640, -0.9469247, -1.0331364, 1.0657465
5: -10.0181770, -8.6689072, -10.0060072, -8.6420956, -0.7976551, 0.7606797
6: -7.9804659, -6.4582362, -8.0319977, -6.4170566, -1.0559025, 1.0874749
7: -2.7633843, -1.9148933, -2.7767701, -1.9519315, -0.5440148, 0.5450097
8: -3.7575970, -2.4750075, -3.7806549, -2.4618464, -0.8459423, 0.8901750
9: -12.4216671, -10.9843359, -12.4469795, -10.9814558, -1.0040901, 0.9918711

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4445830, upper bound: 0.4329004
time: 3.62 seconds

## Relational analysis of IS_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4445850, upper bound: 0.4329002
time: 3.49 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.3049974, -10.5077372, -12.4975786, -10.4164066, -1.3406162, 1.4435132
1: 3.2321167, 4.3576212, 3.1711040, 4.4011412, -0.6904454, 0.7095709
2: -4.8479066, -3.8533664, -4.9402313, -3.8053916, -0.7867970, 0.8085532
3: -12.6157703, -10.9978504, -12.7039204, -10.8662281, -1.1424038, 1.0600979
4: -2.3731463, -0.9529781, -2.4404488, -0.9194562, -1.0373039, 1.0796428
5: -10.0429802, -8.6810799, -10.0678587, -8.5825472, -0.8371394, 0.7791318
6: -7.9746127, -6.4649639, -8.0808945, -6.3785462, -1.0804586, 1.1030998
7: -2.7517867, -1.9555981, -2.7866914, -1.9368539, -0.5308553, 0.5554978
8: -3.7252703, -2.4585071, -3.8290882, -2.4185205, -0.8752792, 0.9230293
9: -12.4185305, -11.0161018, -12.4794645, -10.9586496, -1.0094392, 1.0111730

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4458572, upper bound: 0.4341779
time: 3.52 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4458572, upper bound: 0.4341774
time: 3.40 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.3121128, -10.4674034, -12.4975624, -10.4164047, -1.3487897, 1.4648860
1: 3.1921453, 4.3662953, 3.1711054, 4.4011383, -0.6996746, 0.7253729
2: -4.8944163, -3.8434315, -4.9402313, -3.8054016, -0.7958179, 0.8574183
3: -12.6183853, -10.9877281, -12.7039194, -10.8662281, -1.1492562, 1.0676732
4: -2.4748058, -0.9370310, -2.4404438, -0.9194667, -1.0539155, 1.0998440
5: -10.0472851, -8.6661444, -10.0678530, -8.5825462, -0.8413656, 0.7914773
6: -8.0034866, -6.4572520, -8.0808916, -6.3785539, -1.0918224, 1.1180403
7: -2.7639863, -1.9081886, -2.7866755, -1.9368544, -0.5576832, 0.5569860
8: -3.7601109, -2.4535050, -3.8290868, -2.4185252, -0.8719294, 0.9286969
9: -12.4362135, -10.9818172, -12.4794588, -10.9586535, -1.0336375, 1.0241603

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of IS_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4490640, upper bound: 0.4354574
time: 3.51 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4495349, upper bound: 0.4378529
time: 3.54 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -12.4147615, -10.4148540, -12.4939671, -10.3791809, -1.4114141, 1.4948871
1: 3.2100120, 4.3845677, 3.1889162, 4.3887420, -0.7001240, 0.7183478
2: -4.8981762, -3.8053846, -4.9330387, -3.8020957, -0.7887131, 0.8465192
3: -12.6630802, -10.9552574, -12.7011175, -10.8915215, -1.1482925, 1.1090662
4: -2.4079621, -0.9279680, -2.4170916, -0.9485211, -1.0629492, 1.0849643
5: -10.0301323, -8.6645956, -10.0099401, -8.6391735, -0.8085846, 0.7755156
6: -8.0013990, -6.4227037, -8.0392237, -6.4014449, -1.1048734, 1.1164300
7: -2.7743151, -1.9381986, -2.7722058, -1.9406614, -0.5536586, 0.5317653
8: -3.7477002, -2.4536519, -3.7845078, -2.4569130, -0.8712360, 0.9040838
9: -12.4503250, -10.9810839, -12.4472647, -10.9647455, -1.0257921, 1.0059915

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4403467, upper bound: 0.4389617
time: 4.69 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4403467, upper bound: 0.4389620
time: 3.64 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.4147472, -10.4148540, -12.5010900, -10.3388481, -1.4312639, 1.5030880
1: 3.2100120, 4.3845639, 3.1481428, 4.3974295, -0.7119048, 0.7287523
2: -4.8981767, -3.8053954, -4.9800091, -3.7921035, -0.8328378, 0.8558387
3: -12.6630783, -10.9552593, -12.7037315, -10.8811970, -1.1556730, 1.1097174
4: -2.4079566, -0.9279795, -2.5185740, -0.9325745, -1.0851450, 1.0937464
5: -10.0301285, -8.6645975, -10.0143147, -8.6242199, -0.8166555, 0.7798042
6: -8.0013981, -6.4227123, -8.0679502, -6.3938336, -1.1125457, 1.1389024
7: -2.7743001, -1.9381983, -2.7845457, -1.8931911, -0.5551815, 0.5621133
8: -3.7476993, -2.4536562, -3.8195982, -2.4519043, -0.8769350, 0.8987422
9: -12.4503202, -10.9810867, -12.4647751, -10.9304647, -1.0299296, 1.0398610

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440326, upper bound: 0.4389613
time: 6.30 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4440325, upper bound: 0.4389618
time: 3.80 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.4116869, -10.4107485, -12.5081768, -10.3688831, -1.4364438, 1.4978054
1: 3.2006273, 4.3803873, 3.1669054, 4.4117403, -0.7233996, 0.7302974
2: -4.9006977, -3.8079984, -4.9440022, -3.7829120, -0.8363287, 0.8264768
3: -12.6639013, -10.9399643, -12.7276878, -10.8585825, -1.1798123, 1.1262859
4: -2.4088709, -0.9276819, -2.4460068, -0.9079995, -1.0937746, 1.1031353
5: -10.0569334, -8.6620970, -10.0740910, -8.5794420, -0.8503976, 0.8082463
6: -8.0234165, -6.4247646, -8.0891914, -6.3598938, -1.1409431, 1.1468003
7: -2.7691636, -1.9316032, -2.7879126, -1.9254270, -0.5553702, 0.5616407
8: -3.7496214, -2.4360247, -3.8335505, -2.4096947, -0.9101765, 0.9294221
9: -12.4584141, -10.9791708, -12.4861917, -10.9413052, -1.0561793, 1.0428152

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4446122, upper bound: 0.4402228
time: 3.34 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4446121, upper bound: 0.4402227
time: 3.67 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.4188128, -10.3704433, -12.5081635, -10.3688822, -1.4446399, 1.5176680
1: 3.1601925, 4.3890743, 3.1669064, 4.4117365, -0.7326471, 0.7430471
2: -4.9474478, -3.7980330, -4.9440022, -3.7829216, -0.8454368, 0.8705370
3: -12.6665144, -10.9296894, -12.7276859, -10.8585835, -1.1805984, 1.1337006
4: -2.5104337, -0.9117346, -2.4460025, -0.9080117, -1.1027470, 1.1253312
5: -10.0612946, -8.6471748, -10.0740871, -8.5794430, -0.8546768, 0.8163301
6: -8.0523548, -6.4170985, -8.0891886, -6.3599019, -1.1522176, 1.1617882
7: -2.7814155, -1.8841653, -2.7878973, -1.9254270, -0.5797624, 0.5631579
8: -3.7845845, -2.4310255, -3.8335485, -2.4096994, -0.9049997, 0.9351077
9: -12.4758282, -10.9449558, -12.4861879, -10.9413080, -1.0806191, 1.0510373

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4446119, upper bound: 0.4438976
time: 3.44 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4446118, upper bound: 0.4438973
time: 3.72 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -12.4954987, -10.3886604, -12.4015474, -10.4341440, -1.4442859, 1.4145839
1: 3.1886330, 4.3929400, 3.2049904, 4.3766313, -0.7088892, 0.7062885
2: -4.9317207, -3.8021038, -4.8862286, -3.8277168, -0.8017880, 0.7953196
3: -12.7177887, -10.8928518, -12.6944723, -10.9248705, -1.0979486, 1.1515176
4: -2.4171791, -0.9459064, -2.3875551, -0.9593811, -1.0584872, 1.0330617
5: -10.0163803, -8.6385689, -10.0356846, -8.6516943, -0.7802215, 0.8061355
6: -8.0409145, -6.3962088, -8.0262957, -6.4203930, -1.1133807, 1.1235862
7: -2.7774129, -1.9485710, -2.7561045, -1.9549787, -0.5450884, 0.5243958
8: -3.7839699, -2.4413333, -3.7685366, -2.4367337, -0.8898300, 0.8928072
9: -12.4575930, -10.9764709, -12.4314461, -10.9934826, -1.0086303, 1.0029330

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4284948, upper bound: 0.4492878
time: 3.60 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4363262, upper bound: 0.4498167
time: 3.53 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -12.4954834, -10.3886604, -12.4086542, -10.3937998, -1.4641442, 1.4227684
1: 3.1886344, 4.3929367, 3.1644793, 4.3853111, -0.7203076, 0.7159343
2: -4.9317207, -3.8021140, -4.9330487, -3.8177376, -0.8458829, 0.8044845
3: -12.7177877, -10.8928528, -12.6970863, -10.9145699, -1.1053243, 1.1522425
4: -2.4171741, -0.9459171, -2.4890683, -0.9434328, -1.0800250, 1.0464509
5: -10.0163746, -8.6385679, -10.0400181, -8.6367903, -0.7887400, 0.8103833
6: -8.0409126, -6.3962188, -8.0550537, -6.4127660, -1.1209970, 1.1347103
7: -2.7773976, -1.9485722, -2.7684083, -1.9075017, -0.5520537, 0.5525542
8: -3.7839708, -2.4413381, -3.8035588, -2.4317212, -0.8955156, 0.8873932
9: -12.4575872, -10.9764700, -12.4491558, -10.9591370, -1.0228348, 1.0272752

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4321826, upper bound: 0.4492874
time: 3.55 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4400175, upper bound: 0.4498163
time: 3.96 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -12.5054865, -10.3411503, -12.5026007, -10.3372898, -1.5010028, 1.5144265
1: 3.1847000, 4.4036131, 3.1751318, 4.3995266, -0.7271371, 0.7410413
2: -4.9352875, -3.7796082, -4.9379554, -3.7821496, -0.8162898, 0.8459922
3: -12.7415657, -10.8856068, -12.7425308, -10.8701639, -1.1663225, 1.1846846
4: -2.4227047, -0.9344392, -2.4237585, -0.9340782, -1.0847836, 1.0880978
5: -10.0226879, -8.6355572, -10.0497780, -8.6330032, -0.8068441, 0.8203802
6: -8.0489979, -6.3774395, -8.0712137, -6.3794403, -1.1559381, 1.1863912
7: -2.7785947, -1.9371943, -2.7734659, -1.9306090, -0.5517726, 0.5506894
8: -3.7881184, -2.4325018, -3.7902088, -2.4145985, -0.8979580, 0.9284854
9: -12.4645100, -10.9591789, -12.4725437, -10.9571648, -1.0408483, 1.0528456

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4352754, upper bound: 0.4495706
time: 3.79 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4352754, upper bound: 0.4503006
time: 3.44 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -12.5054693, -10.3411503, -12.5097265, -10.2969751, -1.5208645, 1.5226374
1: 3.1847005, 4.4036093, 3.1342244, 4.4082193, -0.7385632, 0.7507147
2: -4.9352865, -3.7796173, -4.9849787, -3.7721357, -0.8604268, 0.8552115
3: -12.7415638, -10.8856077, -12.7451448, -10.8597431, -1.1735978, 1.1854073
4: -2.4226995, -0.9344509, -2.5251691, -0.9181304, -1.1069787, 1.0968637
5: -10.0226831, -8.6355562, -10.0541763, -8.6180248, -0.8147578, 0.8246862
6: -8.0489960, -6.3774490, -8.1000423, -6.3718514, -1.1634915, 1.1975245
7: -2.7785800, -1.9371946, -2.7858126, -1.8831110, -0.5589349, 0.5753849
8: -3.7881188, -2.4325056, -3.8253298, -2.4095912, -0.9036714, 0.9231653
9: -12.4645071, -10.9591808, -12.4899921, -10.9228764, -1.0509086, 1.0771025

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389613, upper bound: 0.4495679
time: 5.97 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4389613, upper bound: 0.4495688
time: 3.83 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -12.5041761, -10.3786755, -12.4039955, -10.4297705, -1.4590182, 1.4277508
1: 3.1669445, 4.4103780, 3.1952381, 4.3780107, -0.7256730, 0.7251356
2: -4.9426136, -3.7901745, -4.8888144, -3.8230956, -0.8155257, 0.8090098
3: -12.7436562, -10.8602467, -12.6959934, -10.9092350, -1.1223874, 1.1758945
4: -2.4445951, -0.9184484, -2.3900943, -0.9460325, -1.0885487, 1.0532265
5: -10.0782204, -8.5790529, -10.0647907, -8.6489735, -0.8109139, 0.8497931
6: -8.0898476, -6.3577118, -8.0493917, -6.4194083, -1.1445310, 1.1595275
7: -2.7873318, -1.9334850, -2.7567565, -1.9482496, -0.5625342, 0.5399903
8: -3.8324118, -2.3979979, -3.7710590, -2.4152131, -0.9283423, 0.9186652
9: -12.4901400, -10.9536390, -12.4460802, -10.9909372, -1.0470157, 1.0322375

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4334601, upper bound: 0.4498935
time: 3.53 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4412932, upper bound: 0.4504224
time: 3.51 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.5041618, -10.3786774, -12.4111023, -10.3894444, -1.4789011, 1.4359281
1: 3.1669464, 4.4103737, 3.1546750, 4.3866911, -0.7368853, 0.7348579
2: -4.9426126, -3.7901840, -4.9356380, -3.8130984, -0.8596275, 0.8181583
3: -12.7436543, -10.8602495, -12.6986094, -10.8988791, -1.1297483, 1.1766808
4: -2.4445906, -0.9184597, -2.4915755, -0.9300842, -1.1107438, 1.0673175
5: -10.0782146, -8.5790529, -10.0691290, -8.6340685, -0.8195207, 0.8540465
6: -8.0898457, -6.3577194, -8.0781498, -6.4117956, -1.1521847, 1.1706629
7: -2.7873163, -1.9334846, -2.7690253, -1.9007709, -0.5640219, 0.5662791
8: -3.8324118, -2.3980017, -3.8060856, -2.4102020, -0.9340323, 0.9134303
9: -12.4901323, -10.9536419, -12.4637260, -10.9565954, -1.0554185, 1.0568502

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4371421, upper bound: 0.4498931
time: 3.75 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4449813, upper bound: 0.4504197
time: 3.59 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -12.5141659, -10.3311586, -12.5050507, -10.3328924, -1.5157475, 1.5275941
1: 3.1629696, 4.4210515, 3.1653080, 4.4009070, -0.7437751, 0.7598915
2: -4.9461956, -3.7676671, -4.9405627, -3.7775130, -0.8300371, 0.8597082
3: -12.7674332, -10.8529873, -12.7440538, -10.8545132, -1.1907597, 1.2090540
4: -2.4500985, -0.9069810, -2.4262748, -0.9207296, -1.1130621, 1.1087527
5: -10.0845251, -8.5760822, -10.0788870, -8.6303110, -0.8376332, 0.8640380
6: -8.0979509, -6.3389411, -8.0942440, -6.3784609, -1.1871581, 1.2223458
7: -2.7885220, -1.9220824, -2.7741263, -1.9238498, -0.5693253, 0.5647207
8: -3.8365698, -2.3891549, -3.7927427, -2.3930678, -0.9364886, 0.9543877
9: -12.4970636, -10.9363403, -12.4871063, -10.9545975, -1.0791707, 1.0819969

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4402226, upper bound: 0.4501618
time: 3.74 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4402225, upper bound: 0.4501624
time: 3.75 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -12.5141487, -10.3311586, -12.5121737, -10.2925959, -1.5356336, 1.5357993
1: 3.1629696, 4.4210482, 3.1243582, 4.4096007, -0.7550163, 0.7698071
2: -4.9461951, -3.7676775, -4.9875889, -3.7674839, -0.8741759, 0.8689299
3: -12.7674313, -10.8529873, -12.7466698, -10.8440399, -1.1980176, 1.2098390
4: -2.4500954, -0.9069934, -2.5276551, -0.9047816, -1.1352572, 1.1177344
5: -10.0845203, -8.5760822, -10.0832882, -8.6153345, -0.8455515, 0.8683487
6: -8.0979481, -6.3389497, -8.1230841, -6.3708839, -1.1947522, 1.2335463
7: -2.7885067, -1.9220824, -2.7864385, -1.8763528, -0.5708952, 0.5891481
8: -3.8365679, -2.3891582, -3.8278666, -2.3880625, -0.9422067, 0.9492311
9: -12.4970579, -10.9363413, -12.5045195, -10.9203167, -1.0835547, 1.1065955

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4438973, upper bound: 0.4501591
time: 5.55 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4438973, upper bound: 0.4501599
time: 3.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.76 seconds
IS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4452496, upper bound: 0.4292144
IS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4452517, upper bound: 0.4292143
IS_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4445830, upper bound: 0.4329004
IS_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4445850, upper bound: 0.4329002
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4458572, upper bound: 0.4341779
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4458572, upper bound: 0.4341774
IS_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4490640, upper bound: 0.4354574
IS_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4495349, upper bound: 0.4378529
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4403467, upper bound: 0.4389617
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4403467, upper bound: 0.4389620
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4440326, upper bound: 0.4389613
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4440325, upper bound: 0.4389618
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4446122, upper bound: 0.4402228
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4446121, upper bound: 0.4402227
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4446119, upper bound: 0.4438976
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4446118, upper bound: 0.4438973
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4284948, upper bound: 0.4492878
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4363262, upper bound: 0.4498167
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4321826, upper bound: 0.4492874
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4400175, upper bound: 0.4498163
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4352754, upper bound: 0.4495706
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4352754, upper bound: 0.4503006
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4389613, upper bound: 0.4495679
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4389613, upper bound: 0.4495688
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4334601, upper bound: 0.4498935
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4412932, upper bound: 0.4504224
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4371421, upper bound: 0.4498931
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4449813, upper bound: 0.4504197
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4402226, upper bound: 0.4501618
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4402225, upper bound: 0.4501624
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4438973, upper bound: 0.4501591
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 21.76
Output dim: 1, lower bound: -0.4438973, upper bound: 0.4501599

## BFS IS instance: IS_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.3025436, -10.5121193, -12.4833641, -10.4266930, -1.3097935, 1.4210041
1: 3.2416925, 4.3562431, 3.1930742, 4.3781490, -0.6665508, 0.6929227
2: -4.8453460, -3.8579822, -4.9292841, -3.8245580, -0.7387729, 0.7916236
3: -12.6142473, -11.0134525, -12.6773500, -10.8991508, -1.1166558, 1.0419514
4: -2.3706980, -0.9663267, -2.4115031, -0.9599771, -1.0039785, 1.0412863
5: -10.0138769, -8.6838465, -10.0037117, -8.6423159, -0.7930149, 0.7461262
6: -7.9516287, -6.4659610, -8.0309973, -6.4200978, -1.0436988, 1.0718882
7: -2.7511468, -1.9623013, -2.7709928, -1.9520633, -0.5151477, 0.5256201
8: -3.7227640, -2.4800091, -3.7800531, -2.4657297, -0.8455865, 0.8937155
9: -12.4039173, -11.0186110, -12.4405346, -10.9820814, -0.9689648, 0.9736187

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4408949, upper bound: 0.4292143
time: 3.46 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4408949, upper bound: 0.4292144
time: 3.61 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.3025436, -10.5121193, -12.4903641, -10.3866882, -1.3293898, 1.4283462
1: 3.2416925, 4.3562431, 3.1523643, 4.3864403, -0.6727979, 0.7077788
2: -4.8453460, -3.8579822, -4.9754648, -3.8146935, -0.7724845, 0.8082539
3: -12.6142473, -11.0134525, -12.6797209, -10.8889818, -1.1285110, 1.0363553
4: -2.3706980, -0.9663267, -2.5129364, -0.9443910, -1.0198789, 1.0568185
5: -10.0138769, -8.6838465, -10.0079603, -8.6279078, -0.8005659, 0.7503216
6: -7.9516287, -6.4659610, -8.0595379, -6.4130116, -1.0476553, 1.0988653
7: -2.7511468, -1.9623013, -2.7828360, -1.9045796, -0.5208422, 0.5432361
8: -3.7227640, -2.4800091, -3.8147635, -2.4607210, -0.8502698, 0.8882747
9: -12.4039173, -11.0186110, -12.4573250, -10.9477911, -0.9829829, 0.9818084

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4428582, upper bound: 0.4287354
time: 3.56 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4452410, upper bound: 0.4292023
time: 6.19 seconds

## BFS IS instance: IS_A1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -12.3070116, -10.4759560, -12.4888792, -10.4263954, -1.3329408, 1.4458513
1: 3.2114835, 4.3634496, 3.1927681, 4.3837023, -0.6735290, 0.7059948
2: -4.8891401, -3.8527498, -4.9293571, -3.8173323, -0.7797058, 0.8395205
3: -12.6152153, -11.0191908, -12.6780519, -10.8988094, -1.1238316, 1.0312440
4: -2.4698770, -0.9638131, -2.4130640, -0.9469247, -1.0308552, 1.0518413
5: -9.9887848, -8.6717710, -10.0060072, -8.6420956, -0.7698112, 0.7579747
6: -7.9572816, -6.4592915, -8.0319977, -6.4170566, -1.0386815, 1.0837942
7: -2.7627766, -1.9215672, -2.7767701, -1.9519315, -0.5416742, 0.5387716
8: -3.7549229, -2.4967608, -3.7806549, -2.4618464, -0.8433664, 0.8689620
9: -12.4072781, -10.9869442, -12.4469795, -10.9814558, -0.9917111, 0.9840726

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of IS_A1_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4441053, upper bound: 0.4305061
time: 4.13 seconds

## Relational analysis of IS_A1_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.4445733, upper bound: 0.4328907
time: 3.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 32.01 seconds
IS_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 32.01
Output dim: 1, lower bound: -0.4408949, upper bound: 0.4292143
IS_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 32.01
Output dim: 1, lower bound: -0.4408949, upper bound: 0.4292144
IS_A1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 32.01
Output dim: 1, lower bound: -0.4428582, upper bound: 0.4287354
IS_A1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 32.01
Output dim: 1, lower bound: -0.4452410, upper bound: 0.4292023
IS_A1_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 32.01
Output dim: 1, lower bound: -0.4441053, upper bound: 0.4305061
IS_A1_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 32.01
Output dim: 1, lower bound: -0.4445733, upper bound: 0.4328907
IS_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4445850, upper bound: 0.4329002
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4458572, upper bound: 0.4341779
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4458572, upper bound: 0.4341774
IS_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4490640, upper bound: 0.4354574
IS_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4495349, upper bound: 0.4378529
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4403467, upper bound: 0.4389617
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4403467, upper bound: 0.4389620
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4440326, upper bound: 0.4389613
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4440325, upper bound: 0.4389618
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4446122, upper bound: 0.4402228
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4446121, upper bound: 0.4402227
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4446119, upper bound: 0.4438976
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4446118, upper bound: 0.4438973
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4284948, upper bound: 0.4492878
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4363262, upper bound: 0.4498167
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4321826, upper bound: 0.4492874
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4400175, upper bound: 0.4498163
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4352754, upper bound: 0.4495706
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4352754, upper bound: 0.4503006
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4389613, upper bound: 0.4495679
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4389613, upper bound: 0.4495688
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4334601, upper bound: 0.4498935
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4412932, upper bound: 0.4504224
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4371421, upper bound: 0.4498931
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4449813, upper bound: 0.4504197
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4402226, upper bound: 0.4501618
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4402225, upper bound: 0.4501624
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4438973, upper bound: 0.4501591
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 32.01
Output dim: 1, lower bound: -0.4438973, upper bound: 0.4501599
Binary search (step 1): status=Status.UNKNOWN, k_low=2, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=0.7800273895263672
rel_dist={1: [-0.4509328816559992, 0.4509326440462238]}

## Binary search (step 2) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5815
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 6193
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5815

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120336, upper bound: 0.3087195
time: 3.50 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128601, upper bound: 0.3128588
time: 3.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.51
Output dim: 1, lower bound: -0.3120336, upper bound: 0.3087195
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.51
Output dim: 1, lower bound: -0.3128601, upper bound: 0.3128588

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.4172287, -10.4103403, -12.5019283, -10.3858471, -1.3372064, 1.4017956
1: 3.2003183, 4.3859591, 3.1705866, 4.3933010, -0.6445777, 0.6660265
2: -4.9007716, -3.8007307, -4.9374738, -3.7914991, -0.7931604, 0.8201355
3: -12.6646280, -10.9396000, -12.6884060, -10.8620052, -1.0772204, 1.0171981
4: -2.4103954, -0.9145901, -2.4220924, -0.9091063, -1.0260217, 1.0276417
5: -10.0592670, -8.6618805, -10.0664368, -8.6348610, -0.7389917, 0.7213717
6: -8.0244331, -6.4216833, -8.0828323, -6.4050708, -1.0252571, 1.0652679
7: -2.7749660, -1.9314392, -2.7790475, -1.9283173, -0.5214313, 0.5227751
8: -3.7502260, -2.4320984, -3.7889771, -2.4180245, -0.8080220, 0.8375584
9: -12.4648705, -10.9785194, -12.4782801, -10.9611387, -0.9614844, 0.9575908

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3098871, upper bound: 0.3038427
time: 3.26 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120315, upper bound: 0.3087161
time: 3.40 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.5105877, -10.3325186, -12.5105896, -10.3325024, -1.4232864, 1.4255333
1: 3.1649652, 4.4064684, 3.1649628, 4.4064741, -0.6822556, 0.6822606
2: -4.9406443, -3.7702651, -4.9406447, -3.7702572, -0.8290207, 0.8177297
3: -12.7447414, -10.8541403, -12.7447643, -10.8541393, -1.0804958, 1.1122093
4: -2.4278302, -0.9076324, -2.4278324, -0.9076314, -1.0426164, 1.0424240
5: -10.0812159, -8.6300774, -10.0812206, -8.6300774, -0.7535467, 0.7580578
6: -8.0952511, -6.3753991, -8.0952549, -6.3753924, -1.1099367, 1.1149006
7: -2.7799153, -1.9236879, -2.7799163, -1.9236860, -0.5308601, 0.5314697
8: -3.7933502, -2.3891625, -3.7933526, -2.3891535, -0.8495405, 0.8541151
9: -12.4935284, -10.9539337, -12.4935312, -10.9539337, -0.9986632, 0.9993553

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104978, upper bound: 0.3125432
time: 5.02 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128541, upper bound: 0.3128565
time: 3.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.13 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 21.13
Output dim: 1, lower bound: -0.3098871, upper bound: 0.3038427
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 21.13
Output dim: 1, lower bound: -0.3120315, upper bound: 0.3087161
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 21.13
Output dim: 1, lower bound: -0.3104978, upper bound: 0.3125432
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 21.13
Output dim: 1, lower bound: -0.3128541, upper bound: 0.3128565

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -12.4172192, -10.4104376, -12.5019226, -10.3859158, -1.2747657, 1.3432517
1: 3.2003226, 4.3859491, 3.1705894, 4.3932943, -0.6341716, 0.6538830
2: -4.9007673, -3.8007407, -4.9374704, -3.7915075, -0.7627171, 0.7828542
3: -12.6646061, -10.9396076, -12.6883926, -10.8620100, -1.0551964, 0.9918807
4: -2.4103894, -0.9146023, -2.4220891, -0.9091201, -1.0203552, 1.0207348
5: -10.0592642, -8.6618843, -10.0664349, -8.6348648, -0.7348562, 0.7200511
6: -8.0244217, -6.4217129, -8.0828247, -6.4050913, -1.0168467, 1.0472722
7: -2.7749653, -1.9314508, -2.7790470, -1.9283252, -0.5157905, 0.4994594
8: -3.7502241, -2.4321170, -3.7889743, -2.4180374, -0.8011479, 0.8130301
9: -12.4648628, -10.9785519, -12.4782772, -10.9611664, -0.9614601, 0.9438852

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3116276, upper bound: 0.3063467
time: 3.55 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120233, upper bound: 0.3087103
time: 3.42 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -12.5054913, -10.3411045, -12.5070219, -10.3387260, -1.4087381, 1.4102328
1: 3.1846991, 4.4036179, 3.1790657, 4.4044690, -0.6574658, 0.6597614
2: -4.9352884, -3.7796023, -4.9368687, -3.7769217, -0.8162830, 0.8043442
3: -12.7415752, -10.8856039, -12.7425470, -10.8766079, -1.0531863, 1.0791881
4: -2.4227071, -0.9344304, -2.4242063, -0.9267859, -1.0183916, 1.0116317
5: -10.0226917, -8.6355553, -10.0394382, -8.6339579, -0.6916554, 0.7052054
6: -8.0490026, -6.3774276, -8.0622187, -6.3768134, -1.0572259, 1.0693108
7: -2.7785952, -1.9371884, -2.7789724, -1.9333627, -0.5128497, 0.5106189
8: -3.7881212, -2.4324932, -3.7896748, -2.4200621, -0.8083539, 0.8069345
9: -12.4645138, -10.9591599, -12.4727249, -10.9576330, -0.9494236, 0.9542668

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3091838, upper bound: 0.3085326
time: 3.40 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104960, upper bound: 0.3125422
time: 4.21 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -12.5141678, -10.3311129, -12.5105877, -10.3325148, -1.4251969, 1.4245706
1: 3.1629658, 4.4210567, 3.1649895, 4.4064722, -0.6728221, 0.6822209
2: -4.9461975, -3.7676620, -4.9406414, -3.7702689, -0.8318936, 0.8188879
3: -12.7674427, -10.8529835, -12.7447634, -10.8541698, -1.0833267, 1.1013961
4: -2.4501011, -0.9069724, -2.4278288, -0.9076540, -1.0524697, 1.0301149
5: -10.0845270, -8.5760803, -10.0811939, -8.6300793, -0.7148775, 0.7590023
6: -8.0979557, -6.3389297, -8.0952301, -6.3753924, -1.0838871, 1.1134992
7: -2.7885222, -1.9220767, -2.7799151, -1.9237006, -0.5328680, 0.5250924
8: -3.8365717, -2.3891459, -3.7933478, -2.3891840, -0.8543104, 0.8289405
9: -12.4970646, -10.9363213, -12.4935217, -10.9539356, -0.9865916, 0.9919752

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6193
type: B, layer: 1, pos: 6193
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6193

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3116827, upper bound: 0.3089862
time: 3.15 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128525, upper bound: 0.3128543
time: 4.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 20.47 seconds
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 20.47
Output dim: 1, lower bound: -0.3116276, upper bound: 0.3063467
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 20.47
Output dim: 1, lower bound: -0.3120233, upper bound: 0.3087103
IS_A2_A1_A1, status: Status.VERIFIED, split count: 3, time: 20.47
Output dim: 1, lower bound: -0.3091838, upper bound: 0.3085326
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 20.47
Output dim: 1, lower bound: -0.3104960, upper bound: 0.3125422
IS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 20.47
Output dim: 1, lower bound: -0.3116827, upper bound: 0.3089862
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 20.47
Output dim: 1, lower bound: -0.3128525, upper bound: 0.3128543

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -12.4172173, -10.4104500, -12.5055065, -10.3845053, -1.2738025, 1.3451560
1: 3.2003479, 4.3859482, 3.1685557, 4.4078822, -0.6341304, 0.6444563
2: -4.9007645, -3.8007519, -4.9430223, -3.7889049, -0.7638748, 0.7857275
3: -12.6646042, -10.9396410, -12.7110929, -10.8608685, -1.0443851, 0.9947112
4: -2.4103866, -0.9146252, -2.4443932, -0.9084606, -1.0081120, 1.0267031
5: -10.0592346, -8.6618862, -10.0697575, -8.5808353, -0.7357615, 0.6813805
6: -8.0243969, -6.4217134, -8.0854788, -6.3686209, -1.0154455, 1.0211562
7: -2.7749636, -1.9314649, -2.7876539, -1.9267089, -0.5069600, 0.5017449
8: -3.7502208, -2.4321480, -3.8321934, -2.4180269, -0.7759662, 0.8177997
9: -12.4648514, -10.9785538, -12.4817600, -10.9435482, -0.9499407, 0.9319563

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3119802, upper bound: 0.3061676
time: 3.61 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120217, upper bound: 0.3087082
time: 3.55 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -12.5054836, -10.3411942, -12.5070181, -10.3387928, -1.3501987, 1.3510010
1: 3.1847019, 4.4036102, 3.1790681, 4.4044638, -0.6465225, 0.6444521
2: -4.9352846, -3.7796123, -4.9368658, -3.7769282, -0.7842574, 0.7621140
3: -12.7415562, -10.8856077, -12.7425327, -10.8766136, -1.0270554, 1.0556508
4: -2.4227018, -0.9344487, -2.4242029, -0.9267993, -1.0077939, 1.0004277
5: -10.0226879, -8.6355581, -10.0394363, -8.6339607, -0.6836603, 0.7021979
6: -8.0489931, -6.3774509, -8.0622120, -6.3768339, -1.0424385, 1.0466622
7: -2.7785950, -1.9371998, -2.7789714, -1.9333720, -0.5052660, 0.4875934
8: -3.7881188, -2.4325080, -3.7896714, -2.4200740, -0.8006973, 0.7781866
9: -12.4645100, -10.9591980, -12.4727230, -10.9576616, -0.9414213, 0.9369478

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3078704, upper bound: 0.3125081
time: 5.19 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104942, upper bound: 0.3125435
time: 3.82 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -12.5141630, -10.3312073, -12.5105839, -10.3325825, -1.3666663, 1.3653419
1: 3.1629715, 4.4210482, 3.1649919, 4.4064665, -0.6618152, 0.6668814
2: -4.9461932, -3.7676718, -4.9406385, -3.7702768, -0.7998356, 0.7766387
3: -12.7674217, -10.8529902, -12.7447453, -10.8541737, -1.0571978, 1.0778667
4: -2.4500968, -0.9069896, -2.4278259, -0.9076672, -1.0410626, 1.0196404
5: -10.0845242, -8.5760822, -10.0811920, -8.6300831, -0.7068982, 0.7560134
6: -8.0979452, -6.3389530, -8.0952234, -6.3754115, -1.0691881, 1.0909183
7: -2.7885211, -1.9220881, -2.7799139, -1.9237087, -0.5202397, 0.5020454
8: -3.8365688, -2.3891621, -3.7933474, -2.3891954, -0.8466465, 0.8001934
9: -12.4970608, -10.9363594, -12.4935188, -10.9539633, -0.9727602, 0.9725995

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6113

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103135, upper bound: 0.3128310
time: 4.36 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128508, upper bound: 0.3128503
time: 4.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.86 seconds
IS_A1_A2_B2_A1, status: Status.VERIFIED, split count: 4, time: 21.86
Output dim: 1, lower bound: -0.3119802, upper bound: 0.3061676
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 1, lower bound: -0.3120217, upper bound: 0.3087082
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 1, lower bound: -0.3078704, upper bound: 0.3125081
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 1, lower bound: -0.3104942, upper bound: 0.3125435
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 1, lower bound: -0.3103135, upper bound: 0.3128310
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.86
Output dim: 1, lower bound: -0.3128508, upper bound: 0.3128503

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.4184742, -10.3707561, -12.5054846, -10.3845043, -1.2748296, 1.3471220
1: 3.1602192, 4.3879881, 3.1685567, 4.4078770, -0.6416204, 0.6450367
2: -4.9452906, -3.7983005, -4.9430227, -3.7889214, -0.7683311, 0.7863280
3: -12.6660109, -10.9300604, -12.7110910, -10.8608704, -1.0463673, 0.9989316
4: -2.5103366, -0.9128299, -2.4443855, -0.9084759, -1.0137410, 1.0263035
5: -10.0611200, -8.6487017, -10.0697489, -8.5808363, -0.7375883, 0.6879247
6: -8.0518608, -6.4182959, -8.0854759, -6.3686337, -1.0245361, 1.0301677
7: -2.7802153, -1.8841794, -2.7876301, -1.9267101, -0.5111599, 0.5019359
8: -3.7838583, -2.4310522, -3.8321919, -2.4180322, -0.7793858, 0.8179911
9: -12.4734325, -10.9449816, -12.4817524, -10.9435501, -0.9592202, 0.9301052

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3104423, upper bound: 0.3082346
time: 3.83 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3120159, upper bound: 0.3087024
time: 3.48 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -12.5041151, -10.3412848, -12.5014887, -10.3390951, -1.3291576, 1.3451037
1: 3.1847849, 4.4020519, 3.1793842, 4.3989019, -0.6414077, 0.6409242
2: -4.9352674, -3.7816844, -4.9367924, -3.7841682, -0.7487936, 0.7561997
3: -12.7413263, -10.8857021, -12.7418289, -10.8769569, -1.0231361, 1.0584939
4: -2.4222596, -0.9376776, -2.4226408, -0.9398627, -0.9939547, 0.9933105
5: -10.0220242, -8.6356182, -10.0371265, -8.6341915, -0.6824473, 0.6996909
6: -8.0487251, -6.3782778, -8.0612278, -6.3798814, -1.0394657, 1.0438589
7: -2.7771382, -1.9372387, -2.7731800, -1.9335132, -0.5019899, 0.4730258
8: -3.7879219, -2.4334621, -3.7890692, -2.4239607, -0.7965692, 0.7857759
9: -12.4625788, -10.9593744, -12.4662933, -10.9582901, -0.9333515, 0.9343141

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3034517, upper bound: 0.3118619
time: 3.33 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3078684, upper bound: 0.3125088
time: 3.73 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -12.5054598, -10.3411970, -12.5082769, -10.2990932, -1.3521712, 1.3520405
1: 3.1847019, 4.4036040, 3.1385202, 4.4065094, -0.6471069, 0.6523163
2: -4.9352846, -3.7796278, -4.9816580, -3.7744253, -0.7849455, 0.7666839
3: -12.7415533, -10.8856106, -12.7439394, -10.8669205, -1.0311592, 1.0576335
4: -2.4226944, -0.9344637, -2.5239661, -0.9250047, -1.0073946, 1.0059938
5: -10.0226784, -8.6355591, -10.0413542, -8.6207418, -0.6900363, 0.7040375
6: -8.0489893, -6.3774652, -8.0895634, -6.3734856, -1.0447953, 1.0556314
7: -2.7785707, -1.9372003, -2.7843428, -1.8860259, -0.5055028, 0.4980316
8: -3.7881169, -2.4325132, -3.8234634, -2.4189701, -0.8009157, 0.7815508
9: -12.4645004, -10.9591999, -12.4813566, -10.9240246, -0.9395697, 0.9463729

Time for backsubstitution: 12.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3060725, upper bound: 0.3118941
time: 3.69 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104922, upper bound: 0.3125383
time: 6.17 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -12.5127935, -10.3312979, -12.5050526, -10.3328800, -1.3456006, 1.3594408
1: 3.1630602, 4.4194884, 3.1653156, 4.4009047, -0.6568744, 0.6633474
2: -4.9461746, -3.7697465, -4.9405627, -3.7775183, -0.7643476, 0.7707249
3: -12.7671909, -10.8530846, -12.7440434, -10.8545218, -1.0532862, 1.0806231
4: -2.4496589, -0.9102192, -2.4262743, -0.9207304, -1.0272174, 1.0123506
5: -10.0838585, -8.5761433, -10.0788784, -8.6303139, -0.7056831, 0.7535044
6: -8.0976715, -6.3397789, -8.0942354, -6.3784595, -1.0661573, 1.0880940
7: -2.7870665, -1.9221284, -2.7741261, -1.9238515, -0.5169612, 0.4877813
8: -3.8363705, -2.3901134, -3.7927437, -2.3930798, -0.8425192, 0.8075389
9: -12.4951324, -10.9365320, -12.4870987, -10.9545908, -0.9646847, 0.9693775

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058365, upper bound: 0.3122131
time: 3.93 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103117, upper bound: 0.3128263
time: 3.88 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5141392, -10.3312054, -12.5118370, -10.2928886, -1.3686512, 1.3663745
1: 3.1629720, 4.4210424, 3.1243868, 4.4085112, -0.6623998, 0.6750474
2: -4.9461932, -3.7676871, -4.9854317, -3.7677543, -0.8005105, 0.7812147
3: -12.7674198, -10.8529911, -12.7461557, -10.8444128, -1.0612756, 1.0798517
4: -2.4500899, -0.9070067, -2.5275533, -0.9058723, -1.0406621, 1.0252607
5: -10.0845165, -8.5760841, -10.0831127, -8.6168652, -0.7132688, 0.7578588
6: -8.0979424, -6.3389664, -8.1225853, -6.3720813, -1.0715349, 1.0999570
7: -2.7884977, -1.9220886, -2.7852385, -1.8763647, -0.5204822, 0.5107267
8: -3.8365674, -2.3891659, -3.8271418, -2.3880925, -0.8468723, 0.8035634
9: -12.4970531, -10.9363632, -12.5021200, -10.9203320, -0.9709191, 0.9819350

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3083744, upper bound: 0.3122392
time: 3.80 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128489, upper bound: 0.3128482
time: 3.72 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 20.28 seconds
IS_A1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3104423, upper bound: 0.3082346
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3120159, upper bound: 0.3087024
IS_A2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3034517, upper bound: 0.3118619
IS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3078684, upper bound: 0.3125088
IS_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3060725, upper bound: 0.3118941
IS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3104922, upper bound: 0.3125383
IS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3058365, upper bound: 0.3122131
IS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3103117, upper bound: 0.3128263
IS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3083744, upper bound: 0.3122392
IS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 20.28
Output dim: 1, lower bound: -0.3128489, upper bound: 0.3128482

## BFS IS instance: IS_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.4184723, -10.3707581, -12.5054808, -10.3845081, -1.2702918, 1.3436265
1: 3.1602240, 4.3879733, 3.1685629, 4.4078536, -0.6327549, 0.6380477
2: -4.9452534, -3.7983050, -4.9429688, -3.7889254, -0.7509384, 0.7508357
3: -12.6660099, -10.9300671, -12.7110882, -10.8608789, -1.0415676, 0.9955395
4: -2.5103068, -0.9128327, -2.4443431, -0.9084792, -1.0015583, 1.0107200
5: -10.0611153, -8.6487083, -10.0697403, -8.5808449, -0.7294490, 0.6821445
6: -8.0518532, -6.4183068, -8.0854664, -6.3686514, -0.9996049, 1.0179842
7: -2.7802143, -1.8841825, -2.7876267, -1.9267132, -0.4989535, 0.4965404
8: -3.7838554, -2.4310541, -3.8321881, -2.4180346, -0.7739024, 0.8081127
9: -12.4734173, -10.9449854, -12.4817324, -10.9435577, -0.9539547, 0.9241023

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6113
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6193
type: A, layer: 1, pos: 4599

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 901

## Relational analysis of IS_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5815

## Relational analysis of IS_A1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3091497, upper bound: 0.3087032
time: 3.72 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3091495, upper bound: 0.3087028
time: 5.36 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.5041094, -10.3413420, -12.5524273, -10.3380909, -1.3251810, 1.3522851
1: 3.1847878, 4.4020472, 3.1660213, 4.3994002, -0.6416961, 0.6467071
2: -4.9352660, -3.7817042, -4.9577198, -3.7838552, -0.7463036, 0.7611004
3: -12.7413006, -10.8857069, -12.7422819, -10.8434563, -1.0283082, 1.0532577
4: -2.4198494, -0.9376779, -2.4224596, -0.9339805, -0.9946198, 0.9944487
5: -10.0220137, -8.6356220, -10.0375900, -8.6187706, -0.6854257, 0.6982459
6: -8.0487156, -6.3783092, -8.0901461, -6.3779922, -1.0403271, 1.0500495
7: -2.7771375, -1.9372442, -2.7776203, -1.9329667, -0.5014021, 0.4741378
8: -3.7879167, -2.4334965, -3.8123035, -2.4230251, -0.7964671, 0.7899412
9: -12.4625597, -10.9593792, -12.4692421, -10.9498682, -0.9348714, 0.9366982

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3063489, upper bound: 0.3120715
time: 3.90 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3078626, upper bound: 0.3125033
time: 3.77 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -12.5054541, -10.3412523, -12.5592117, -10.2980890, -1.3482034, 1.3592246
1: 3.1847057, 4.4035988, 3.1249785, 4.4070077, -0.6473970, 0.6583689
2: -4.9352827, -3.7796469, -5.0026684, -3.7740982, -0.7824717, 0.7715074
3: -12.7415276, -10.8856134, -12.7443953, -10.8333597, -1.0362651, 1.0523961
4: -2.4202859, -0.9344654, -2.5237589, -0.9191225, -1.0080597, 1.0071695
5: -10.0226688, -8.6355619, -10.0418224, -8.6053085, -0.6930064, 0.7025944
6: -8.0489807, -6.3774972, -8.1183882, -6.3716283, -1.0456293, 1.0618863
7: -2.7785699, -1.9372058, -2.7887657, -1.8854764, -0.5049273, 0.4994570
8: -3.7881136, -2.4325485, -3.8467498, -2.4180346, -0.8008175, 0.7857076
9: -12.4644842, -10.9592056, -12.4843206, -10.9155779, -0.9410641, 0.9487745

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3089675, upper bound: 0.3121080
time: 4.42 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4599

## Relational analysis of IS_A2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3100584, upper bound: 0.3110156
time: 4.31 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3125328
time: 6.71 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -12.5110054, -10.3463783, -12.5023146, -10.3557997, -1.3198781, 1.3388810
1: 3.1640916, 4.4167156, 3.1668983, 4.3966904, -0.6505164, 0.6579581
2: -4.9456096, -3.7752514, -4.9396963, -3.7858815, -0.7549460, 0.7632170
3: -12.7571850, -10.8545074, -12.7288380, -10.8566952, -1.0399388, 1.0638831
4: -2.4487576, -0.9103663, -2.4248965, -0.9209538, -1.0228128, 1.0070524
5: -10.0798950, -8.5768852, -10.0728512, -8.6314487, -0.6994609, 0.7458498
6: -8.0953283, -6.3479834, -8.0908175, -6.3909225, -1.0490067, 1.0735271
7: -2.7868967, -1.9238045, -2.7738674, -1.9263949, -0.5135327, 0.4861529
8: -3.8354259, -2.3971887, -3.7912955, -2.4038296, -0.8306055, 0.7981206
9: -12.4917212, -10.9380627, -12.4819298, -10.9569340, -0.9577918, 0.9618489

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3042544, upper bound: 0.3117461
time: 3.79 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058307, upper bound: 0.3122076
time: 3.84 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.5127850, -10.3313513, -12.5559912, -10.3318853, -1.3416269, 1.3666244
1: 3.1630640, 4.4194837, 3.1518946, 4.4014020, -0.6571631, 0.6692374
2: -4.9461727, -3.7697654, -4.9614906, -3.7772033, -0.7618678, 0.7756443
3: -12.7671661, -10.8530874, -12.7444973, -10.8210049, -1.0584536, 1.0753868
4: -2.4472566, -0.9102206, -2.4260988, -0.9148493, -1.0278819, 1.0134897
5: -10.0838480, -8.5761452, -10.0793438, -8.6148968, -0.7086549, 0.7520592
6: -8.0976639, -6.3398094, -8.1232262, -6.3765755, -1.0670142, 1.0943055
7: -2.7870662, -1.9221342, -2.7785633, -1.9233003, -0.5163748, 0.4890420
8: -3.8363676, -2.3901472, -3.8159781, -2.3921447, -0.8424189, 0.8117141
9: -12.4951143, -10.9365368, -12.4900541, -10.9461651, -0.9662051, 0.9717712

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A2_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3087307, upper bound: 0.3123593
time: 4.02 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103059, upper bound: 0.3128215
time: 3.88 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -12.5123501, -10.3462887, -12.5090981, -10.3158102, -1.3429325, 1.3458142
1: 3.1640024, 4.4182692, 3.1260090, 4.4042978, -0.6560334, 0.6696546
2: -4.9456267, -3.7731907, -4.9845448, -3.7761195, -0.7911088, 0.7736620
3: -12.7574120, -10.8544168, -12.7309513, -10.8466034, -1.0479221, 1.0631104
4: -2.4491868, -0.9071529, -2.5261889, -0.9060957, -1.0362582, 1.0199752
5: -10.0805511, -8.5768242, -10.0770798, -8.6180077, -0.7070481, 0.7501991
6: -8.0956039, -6.3471718, -8.1191101, -6.3845377, -1.0543859, 1.0854061
7: -2.7883286, -1.9237649, -2.7849615, -1.8789135, -0.5170586, 0.5079106
8: -3.8356237, -2.3962412, -3.8256836, -2.3988371, -0.8349562, 0.7941400
9: -12.4936438, -10.9378929, -12.4969578, -10.9226856, -0.9640176, 0.9744132

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3067892, upper bound: 0.3117721
time: 3.68 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3083686, upper bound: 0.3122342
time: 3.64 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -12.5141344, -10.3312607, -12.5627728, -10.2918997, -1.3646863, 1.3735607
1: 3.1629748, 4.4210377, 3.1107936, 4.4090109, -0.6626899, 0.6811388
2: -4.9461918, -3.7677057, -5.0064425, -3.7674236, -0.7980382, 0.7860135
3: -12.7673950, -10.8529968, -12.7466106, -10.8108377, -1.0663843, 1.0746142
4: -2.4476876, -0.9070072, -2.5273519, -0.8999906, -1.0413280, 1.0264361
5: -10.0845051, -8.5760851, -10.0835781, -8.6014156, -0.7162536, 0.7564162
6: -8.0979357, -6.3389974, -8.1514721, -6.3702278, -1.0723732, 1.1062300
7: -2.7884974, -1.9220939, -2.7896609, -1.8758101, -0.5199074, 0.5121660
8: -3.8365636, -2.3892016, -3.8504295, -2.3871536, -0.8467762, 0.8077343
9: -12.4970360, -10.9363661, -12.5050936, -10.9118834, -0.9724133, 0.9843463

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4599
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4599

## Relational analysis of IS_A2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3112656, upper bound: 0.3123796
time: 4.09 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3128431, upper bound: 0.3128428
time: 3.58 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 20.28 seconds
IS_A1_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3091497, upper bound: 0.3087032
IS_A1_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3091495, upper bound: 0.3087028
IS_A2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3063489, upper bound: 0.3120715
IS_A2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3078626, upper bound: 0.3125033
IS_A2_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3100584, upper bound: 0.3110156
IS_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3125328
IS_A2_A2_A2_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3042544, upper bound: 0.3117461
IS_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3058307, upper bound: 0.3122076
IS_A2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3087307, upper bound: 0.3123593
IS_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3103059, upper bound: 0.3128215
IS_A2_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3067892, upper bound: 0.3117721
IS_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3083686, upper bound: 0.3122342
IS_A2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3112656, upper bound: 0.3123796
IS_A2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 20.28
Output dim: 1, lower bound: -0.3128431, upper bound: 0.3128428

## BFS IS instance: IS_A2_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -12.5027809, -10.3435478, -12.5451612, -10.3428669, -1.3126612, 1.3418202
1: 3.1877723, 4.3952723, 3.1780119, 4.3881540, -0.6286991, 0.6258031
2: -4.9168348, -3.7842445, -4.9304032, -3.8056107, -0.6872251, 0.7307215
3: -12.7401619, -10.8904715, -12.7367668, -10.8512745, -1.0055568, 1.0405307
4: -2.4058814, -0.9397702, -2.4000244, -0.9496453, -0.9613838, 0.9682508
5: -10.0185966, -8.6428919, -10.0277872, -8.6294765, -0.6708784, 0.6761400
6: -8.0425863, -6.3898602, -8.0662251, -6.3947773, -1.0145473, 1.0076270
7: -2.7759855, -1.9395976, -2.7748976, -1.9394667, -0.4836698, 0.4616221
8: -3.7859521, -2.4349842, -3.8057146, -2.4258957, -0.7863939, 0.7801280
9: -12.4519892, -10.9614639, -12.4519844, -10.9599190, -0.9115875, 0.9180102

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A2_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3063490, upper bound: 0.3094860
time: 3.90 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3063490, upper bound: 0.3120717
time: 3.77 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -12.5041065, -10.3413429, -12.5524254, -10.3380938, -1.3207705, 1.3487933
1: 3.1847916, 4.4020300, 3.1660280, 4.3993769, -0.6327748, 0.6397442
2: -4.9352274, -3.7817068, -4.9576654, -3.7838604, -0.7289073, 0.7256082
3: -12.7412996, -10.8857136, -12.7422800, -10.8434629, -1.0235722, 1.0498639
4: -2.4198198, -0.9376810, -2.4224157, -0.9339848, -0.9824133, 0.9788730
5: -10.0220070, -8.6356277, -10.0375824, -8.6187820, -0.6772482, 0.6924653
6: -8.0487099, -6.3783207, -8.0901375, -6.3780103, -1.0152843, 1.0373178
7: -2.7771361, -1.9372468, -2.7776182, -1.9329703, -0.4891596, 0.4753869
8: -3.7879148, -2.4334984, -3.8123002, -2.4230280, -0.7912467, 0.7800541
9: -12.4625444, -10.9593830, -12.4692202, -10.9498730, -0.9295940, 0.9306817

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A2_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3078626, upper bound: 0.3099126
time: 3.81 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3078626, upper bound: 0.3125033
time: 3.75 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5054531, -10.3412571, -12.5592117, -10.2980919, -1.3447785, 1.3547826
1: 3.1847119, 4.4035759, 3.1249847, 4.4069901, -0.6403761, 0.6495981
2: -4.9352293, -3.7796521, -5.0026302, -3.7741020, -0.7470047, 0.7541115
3: -12.7415257, -10.8856211, -12.7443933, -10.8333635, -1.0328718, 1.0476733
4: -2.4202423, -0.9344673, -2.5237286, -0.9191251, -0.9924808, 0.9949656
5: -10.0226574, -8.6355715, -10.0418158, -8.6053143, -0.6872254, 0.6944431
6: -8.0489731, -6.3775148, -8.1183805, -6.3716416, -1.0328596, 1.0370014
7: -2.7785676, -1.9372101, -2.7887638, -1.8854797, -0.4995884, 0.4872891
8: -3.7881122, -2.4325495, -3.8467488, -2.4180365, -0.7909546, 0.7804739
9: -12.4644632, -10.9592104, -12.4843063, -10.9155798, -0.9350772, 0.9434873

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A2_A1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3064769
time: 7.04 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3125323
time: 5.88 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -12.5110025, -10.3463821, -12.5023117, -10.3558035, -1.3154194, 1.3352923
1: 3.1640964, 4.4166989, 3.1669035, 4.3966689, -0.6416235, 0.6509565
2: -4.9455724, -3.7752547, -4.9396443, -3.7858872, -0.7375546, 0.7277540
3: -12.7571850, -10.8545160, -12.7288370, -10.8567038, -1.0351808, 1.0604894
4: -2.4487271, -0.9103680, -2.4248528, -0.9209571, -1.0105999, 0.9914927
5: -10.0798883, -8.5768909, -10.0728445, -8.6314583, -0.6913331, 0.7400692
6: -8.0953236, -6.3479972, -8.0908089, -6.3909397, -1.0241804, 1.0608487
7: -2.7868955, -1.9238076, -2.7738657, -1.9263983, -0.5013133, 0.4840990
8: -3.8354244, -2.3971901, -3.7912922, -2.4038291, -0.8252881, 0.7882451
9: -12.4917078, -10.9380674, -12.4819107, -10.9569387, -0.9525144, 0.9558558

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A2_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3058306, upper bound: 0.3096841
time: 3.67 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058306, upper bound: 0.3122075
time: 3.67 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -12.5113392, -10.3335600, -12.5486469, -10.3366594, -1.3290820, 1.3559554
1: 3.1660357, 4.4126921, 3.1639128, 4.3901844, -0.6440895, 0.6482927
2: -4.9277167, -3.7722917, -4.9342885, -3.7989314, -0.7028909, 0.7452384
3: -12.7659674, -10.8579254, -12.7389088, -10.8288946, -1.0369051, 1.0623292
4: -2.4332569, -0.9122889, -2.4037213, -0.9304976, -0.9946764, 0.9873488
5: -10.0804539, -8.5833826, -10.0695782, -8.6256237, -0.6941240, 0.7298059
6: -8.0914602, -6.3513851, -8.0993004, -6.3932943, -1.0412102, 1.0519762
7: -2.7859483, -1.9244828, -2.7758710, -1.9298708, -0.4986372, 0.4765046
8: -3.8342876, -2.3916616, -3.8091736, -2.3950405, -0.8323483, 0.8016129
9: -12.4846163, -10.9386034, -12.4727831, -10.9561863, -0.9426908, 0.9529185

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A2_A2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3087306, upper bound: 0.3098478
time: 4.24 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3087306, upper bound: 0.3123609
time: 4.24 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -12.5127869, -10.3313522, -12.5559874, -10.3318882, -1.3371711, 1.3630402
1: 3.1630678, 4.4194660, 3.1518993, 4.4013791, -0.6482701, 0.6622840
2: -4.9461355, -3.7697699, -4.9614372, -3.7772083, -0.7444818, 0.7401849
3: -12.7671661, -10.8530941, -12.7444963, -10.8210135, -1.0536464, 1.0719923
4: -2.4472256, -0.9102230, -2.4260554, -0.9148526, -1.0156693, 0.9979300
5: -10.0838432, -8.5761518, -10.0793352, -8.6149063, -0.7004940, 0.7462784
6: -8.0976572, -6.3398237, -8.1232176, -6.3765936, -1.0421867, 1.0816543
7: -2.7870653, -1.9221370, -2.7785625, -1.9233048, -0.5041553, 0.4883994
8: -3.8363647, -2.3901501, -3.8159733, -2.3921461, -0.8371062, 0.8018405
9: -12.4951000, -10.9365425, -12.4900351, -10.9461699, -0.9609380, 0.9657787

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A2_A2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3103059, upper bound: 0.3103058
time: 3.68 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3103059, upper bound: 0.3128207
time: 3.59 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -12.5123520, -10.3462896, -12.5090952, -10.3158121, -1.3384938, 1.3422339
1: 3.1640072, 4.4182520, 3.1260138, 4.4042749, -0.6471395, 0.6627858
2: -4.9455900, -3.7731950, -4.9844913, -3.7761254, -0.7737480, 0.7381997
3: -12.7574110, -10.8544216, -12.7309504, -10.8466129, -1.0430214, 1.0597172
4: -2.4491572, -0.9071550, -2.5261462, -0.9060993, -1.0240452, 1.0044160
5: -10.0805464, -8.5768318, -10.0770721, -8.6180153, -0.6987702, 0.7444184
6: -8.0955973, -6.3471847, -8.1190996, -6.3845553, -1.0295148, 1.0729203
7: -2.7883272, -1.9237676, -2.7849600, -1.8789171, -0.5048392, 0.5026133
8: -3.8356209, -2.3962445, -3.8256803, -2.3988409, -0.8296633, 0.7842648
9: -12.4936285, -10.9378948, -12.4969387, -10.9226894, -0.9587665, 0.9684213

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of IS_A2_A2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3058306, upper bound: 0.3096821
time: 5.44 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3058306, upper bound: 0.3096866
time: 3.78 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -12.5126877, -10.3334723, -12.5554333, -10.2966890, -1.3521369, 1.3628881
1: 3.1659527, 4.4142447, 3.1227188, 4.3977852, -0.6498413, 0.6601224
2: -4.9277349, -3.7702339, -4.9791288, -3.7891150, -0.7391171, 0.7555926
3: -12.7661972, -10.8578358, -12.7410250, -10.8189421, -1.0447994, 1.0615926
4: -2.4336874, -0.9090750, -2.5050178, -0.9156396, -1.0081210, 1.0001173
5: -10.0811090, -8.5833244, -10.0737610, -8.6123857, -0.7015936, 0.7341142
6: -8.0917320, -6.3505721, -8.1275682, -6.3869781, -1.0465097, 1.0641544
7: -2.7873812, -1.9244447, -2.7870917, -1.8823709, -0.5021250, 0.5000730
8: -3.8344841, -2.3907156, -3.8436518, -2.3900566, -0.8367348, 0.7976139
9: -12.4865417, -10.9384336, -12.4878607, -10.9218493, -0.9489222, 0.9649698

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A2_A2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3067892, upper bound: 0.3079061
time: 4.32 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3067892, upper bound: 0.3079022
time: 4.64 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -12.5141315, -10.3312626, -12.5627689, -10.2919006, -1.3602493, 1.3699851
1: 3.1629801, 4.4210210, 3.1107988, 4.4089880, -0.6537960, 0.6743138
2: -4.9461536, -3.7677100, -5.0063896, -3.7674298, -0.7806829, 0.7505553
3: -12.7673931, -10.8530006, -12.7466087, -10.8108444, -1.0614383, 1.0712204
4: -2.4476557, -0.9070094, -2.5273099, -0.8999939, -1.0291147, 1.0108769
5: -10.0844994, -8.5760918, -10.0835724, -8.6014252, -0.7079459, 0.7506353
6: -8.0979271, -6.3390102, -8.1514626, -6.3702450, -1.0475018, 1.0937679
7: -2.7884965, -1.9220967, -2.7896585, -1.8758140, -0.5076873, 0.5068976
8: -3.8365631, -2.3892040, -3.8504257, -2.3871560, -0.8414871, 0.7978610
9: -12.4970207, -10.9363708, -12.5050726, -10.9118881, -0.9671717, 0.9783547

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A2_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3083686, upper bound: 0.3083685
time: 6.02 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3083686, upper bound: 0.3083696
time: 4.00 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 22.68 seconds
IS_A2_A1_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3063490, upper bound: 0.3094860
IS_A2_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3063490, upper bound: 0.3120717
IS_A2_A1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3078626, upper bound: 0.3099126
IS_A2_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3078626, upper bound: 0.3125033
IS_A2_A1_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3064769
IS_A2_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3104866, upper bound: 0.3125323
IS_A2_A2_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3058306, upper bound: 0.3096841
IS_A2_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3058306, upper bound: 0.3122075
IS_A2_A2_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3087306, upper bound: 0.3098478
IS_A2_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3087306, upper bound: 0.3123609
IS_A2_A2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3103059, upper bound: 0.3103058
IS_A2_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3103059, upper bound: 0.3128207
IS_A2_A2_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3058306, upper bound: 0.3096821
IS_A2_A2_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3058306, upper bound: 0.3096866
IS_A2_A2_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3067892, upper bound: 0.3079061
IS_A2_A2_A2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3067892, upper bound: 0.3079022
IS_A2_A2_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3083686, upper bound: 0.3083685
IS_A2_A2_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 7, time: 22.68
Output dim: 1, lower bound: -0.3083686, upper bound: 0.3083696

## BFS IS instance: IS_A2_A1_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.5052662, -10.3041487, -12.5451612, -10.3428669, -1.3141074, 1.3436170
1: 3.1471658, 4.3984118, 3.1780119, 4.3881540, -0.6379995, 0.6266392
2: -4.9607134, -3.7797897, -4.9304032, -3.8056107, -0.6945174, 0.7310083
3: -12.7415199, -10.8809729, -12.7367668, -10.8512745, -1.0062180, 1.0467395
4: -2.5059955, -0.9351730, -2.4000244, -0.9496453, -0.9672890, 0.9699621
5: -10.0210228, -8.6303473, -10.0277872, -8.6294765, -0.6731844, 0.6825436
6: -8.0699720, -6.3863053, -8.0662251, -6.3947773, -1.0242426, 1.0101936
7: -2.7823462, -1.8922251, -2.7748976, -1.9394667, -0.4882230, 0.4700814
8: -3.8195181, -2.4329324, -3.8057146, -2.4258957, -0.7897754, 0.7808633
9: -12.4616489, -10.9276581, -12.4519844, -10.9599190, -0.9178042, 0.9209099

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A2_A1_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3019115, upper bound: 0.3076146
time: 3.53 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3019115, upper bound: 0.3076145
time: 3.70 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.5065975, -10.3019352, -12.5524254, -10.3380938, -1.3222268, 1.3505976
1: 3.1441927, 4.4051743, 3.1660280, 4.3993769, -0.6422162, 0.6405858
2: -4.9791665, -3.7772696, -4.9576654, -3.7838604, -0.7362468, 0.7258953
3: -12.7426567, -10.8761044, -12.7422800, -10.8434629, -1.0242333, 1.0561905
4: -2.5199022, -0.9330845, -2.4224157, -0.9339848, -0.9883204, 0.9805841
5: -10.0244722, -8.6229658, -10.0375824, -8.6187820, -0.6795682, 0.6984961
6: -8.0761299, -6.3747578, -8.0901375, -6.3780103, -1.0250196, 1.0398960
7: -2.7834330, -1.8898687, -2.7776182, -1.9329703, -0.4935290, 0.4769312
8: -3.8214746, -2.4314513, -3.8123002, -2.4230280, -0.7946498, 0.7807885
9: -12.4721508, -10.9255943, -12.4692202, -10.9498730, -0.9362111, 0.9335904

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A2_A1_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3034458, upper bound: 0.3080756
time: 3.62 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A1_A2_B1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3034458, upper bound: 0.3080791
time: 3.36 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.5054531, -10.3412571, -12.5663586, -10.2905340, -1.3476586, 1.3607018
1: 3.1847119, 4.4035759, 3.1087556, 4.4235849, -0.6450901, 0.6550788
2: -4.9352293, -3.7796521, -5.0119352, -3.7648454, -0.7515495, 0.7603872
3: -12.7415257, -10.8856211, -12.7692785, -10.8096323, -1.0404509, 1.0519679
4: -2.4202423, -0.9344673, -2.5495751, -0.8993111, -0.9975233, 1.0045896
5: -10.0226574, -8.6355715, -10.0869446, -8.5474510, -0.6919322, 0.7027134
6: -8.0489731, -6.3775148, -8.1542091, -6.3338041, -1.0375118, 1.0463789
7: -2.7785676, -1.9372101, -2.7982533, -1.8741813, -0.5036244, 0.4877561
8: -3.7881122, -2.4325495, -3.8936286, -2.3871012, -0.7916031, 0.7889537
9: -12.4644632, -10.9592104, -12.5086746, -10.8942881, -0.9386652, 0.9511291

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 6113
type: B, layer: 1, pos: 6206
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 4599
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A2_A1_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3060669, upper bound: 0.3081163
time: 4.11 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3060669, upper bound: 0.3081164
time: 3.53 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -12.5134907, -10.3069954, -12.5023117, -10.3558035, -1.3168876, 1.3370862
1: 3.1234322, 4.4198461, 3.1669035, 4.3966689, -0.6513784, 0.6518039
2: -4.9894810, -3.7707841, -4.9396443, -3.7858872, -0.7448609, 0.7280344
3: -12.7585449, -10.8448009, -12.7288370, -10.8567038, -1.0358429, 1.0667851
4: -2.5487652, -0.9057703, -2.4248528, -0.9209571, -1.0165083, 0.9932041
5: -10.0823889, -8.5642538, -10.0728445, -8.6314583, -0.6936887, 0.7460645
6: -8.1228046, -6.3444524, -8.0908089, -6.3909397, -1.0339675, 1.0634086
7: -2.7931051, -1.8764286, -2.7738657, -1.9263983, -0.5057237, 0.4856362
8: -3.8689618, -2.3951349, -3.7912922, -2.4038291, -0.8286889, 0.7889926
9: -12.5013237, -10.9043036, -12.4819107, -10.9569387, -0.9590414, 0.9587781

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6206
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6206

## Relational analysis of IS_A2_A2_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058305, upper bound: 0.3120868
time: 3.82 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3058305, upper bound: 0.3122073
time: 3.67 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -12.5138235, -10.2941895, -12.5486469, -10.3366594, -1.3305402, 1.3577394
1: 3.1253405, 4.4158363, 3.1639128, 4.3901844, -0.6535809, 0.6491355
2: -4.9715805, -3.7678032, -4.9342885, -3.7989314, -0.7101734, 0.7455356
3: -12.7673302, -10.8483009, -12.7389088, -10.8288946, -1.0375669, 1.0685184
4: -2.5333061, -0.9076915, -2.4037213, -0.9304976, -1.0005829, 0.9890606
5: -10.0829191, -8.5708561, -10.0695782, -8.6256237, -0.6964678, 0.7361443
6: -8.1188927, -6.3478518, -8.0993004, -6.3932943, -1.0509384, 1.0545242
7: -2.7922323, -1.8771069, -2.7758710, -1.9298708, -0.5032032, 0.4830551
8: -3.8678403, -2.3896055, -3.8091736, -2.3950405, -0.8357408, 0.8023536
9: -12.4942751, -10.9048166, -12.4727831, -10.9561863, -0.9491200, 0.9558377

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 6206
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 4599
type: A, layer: 1, pos: 6206
type: B, layer: 1, pos: 5815
type: B, layer: 1, pos: 6193

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 901

## Relational analysis of IS_A2_A2_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3042544, upper bound: 0.3078842
time: 4.23 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3042542, upper bound: 0.3078812
time: 4.52 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 21.36 seconds
IS_A2_A1_A2_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3019115, upper bound: 0.3076146
IS_A2_A1_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3019115, upper bound: 0.3076145
IS_A2_A1_A2_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3034458, upper bound: 0.3080756
IS_A2_A1_A2_B1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3034458, upper bound: 0.3080791
IS_A2_A1_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3060669, upper bound: 0.3081163
IS_A2_A1_A2_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3060669, upper bound: 0.3081164
IS_A2_A2_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3058305, upper bound: 0.3120868
IS_A2_A2_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3058305, upper bound: 0.3122073
IS_A2_A2_A2_B1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3042544, upper bound: 0.3078842
IS_A2_A2_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 8, time: 21.36
Output dim: 1, lower bound: -0.3042542, upper bound: 0.3078812
IS_A2_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 21.36
Output dim: 1, lower bound: -0.3103059, upper bound: 0.3128207
Binary search (step 2): status=Status.UNKNOWN, k_low=2, k_high=3, k_mid=2, eps_mid=0.0078125, abs_max=0.6936578750610352
rel_dist={1: [-0.3128648053580161, 0.31286157369120016]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 1647.63 seconds
