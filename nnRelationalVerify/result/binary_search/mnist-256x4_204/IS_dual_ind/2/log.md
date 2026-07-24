## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 242.013661301
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-131.4713287, 104.7019348, -131.4713287, 104.7019348, -236.1732635, 236.1732635)
1: (-109.8579636, 92.8269119, -109.8579636, 92.8269119, -202.6848755, 202.6848755)
2: (-144.6704407, 94.3037720, -144.6704407, 94.3037720, -238.9741974, 238.9741974)
3: (-153.9682007, 81.8407135, -153.9682007, 81.8407135, -235.8089142, 235.8089142)
4: (-141.1737061, 108.4469757, -141.1737061, 108.4469757, -249.6206818, 249.6206818)
5: (-126.9361572, 99.3330688, -126.9361572, 99.3330688, -226.2692261, 226.2692261)
6: (-121.2382507, 116.3698807, -121.2382507, 116.3698807, -237.6081238, 237.6081238)
7: (-132.1308594, 111.5709610, -132.1308594, 111.5709610, -243.7018127, 243.7018127)
8: (-158.4042053, 108.2726822, -158.4042053, 108.2726822, -266.6768799, 266.6768799)
9: (-120.4584961, 118.8621521, -120.4584961, 118.8621521, -239.3206482, 239.3206482)

## BASE Result
execution time: IAR + LP analysis = 1.09 + 8.87 = 9.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -242.0420621, upper bound: 242.0420621


# Binary Search by BASE starts (time budget: 2690.04 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=243.70181274414062
rel_dist={7: [-242.04184490722724, 242.04184490722724]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=243.70181274414062
rel_dist={7: [-242.0414140657913, 242.04141406382126]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=243.70181274414062
rel_dist={7: [-242.04091628504267, 242.04091628504267]}

## Binary Search Result
Binary search time: 33.27 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 2656.77 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0353554, upper bound: 242.0334284
time: 6.88 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0395681, upper bound: 242.0395681
time: 6.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.51
Output dim: 7, lower bound: -242.0353554, upper bound: 242.0334284
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.51
Output dim: 7, lower bound: -242.0395681, upper bound: 242.0395681

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -119.5140991, 95.2666245, -129.6401062, 103.2556000, -222.7696991, 224.9067383
1: -99.9024429, 84.4575806, -108.3319550, 91.5456161, -191.4480591, 192.7895355
2: -131.5668793, 85.7744370, -142.6625977, 92.9989166, -224.5657959, 228.4370117
3: -140.1050415, 74.4418182, -151.8422852, 80.7104416, -220.8154907, 226.2840881
4: -128.4107208, 98.6364822, -139.2171631, 106.9458313, -235.3565521, 237.8536224
5: -115.4722366, 90.4375610, -125.1798477, 97.9702759, -213.4425049, 215.6173553
6: -110.2378006, 105.8749466, -119.5542221, 114.7615051, -224.9992981, 225.4291534
7: -120.2295685, 101.5678558, -130.3072815, 110.0384750, -230.2680206, 231.8751373
8: -143.9831390, 98.4588470, -156.1963348, 106.7714539, -250.7545776, 254.6551819
9: -109.6788559, 108.1721420, -118.8062668, 117.2257538, -226.9046021, 226.9784088

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0291404, upper bound: 242.0279245
time: 7.19 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0318040, upper bound: 242.0300418
time: 7.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.0534821, 99.6287537, -131.4713287, 104.7019348, -229.7554169, 231.1000671
1: -104.4979324, 88.3349228, -109.8579636, 92.8269119, -197.3248444, 198.1928711
2: -137.6306152, 89.7292252, -144.6704407, 94.3037720, -231.9343872, 234.3996582
3: -146.5207214, 77.8730164, -153.9682007, 81.8407135, -228.3614349, 231.8412170
4: -134.3097839, 103.1774673, -141.1737061, 108.4469757, -242.7567596, 244.3511658
5: -120.7826691, 94.5478973, -126.9361572, 99.3330688, -220.1157379, 221.4840546
6: -115.3268051, 110.7257385, -121.2382507, 116.3698807, -231.6966858, 231.9639893
7: -125.7305145, 106.1953125, -132.1308594, 111.5709610, -237.3014832, 238.3261719
8: -150.6563873, 102.9990997, -158.4042053, 108.2726822, -258.9290771, 261.4033203
9: -114.6585846, 113.1035004, -120.4584961, 118.8621521, -233.5207214, 233.5619965

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0334284, upper bound: 242.0353554
time: 6.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0334284, upper bound: 242.0395681
time: 6.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 14.27 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 14.27
Output dim: 7, lower bound: -242.0291404, upper bound: 242.0279245
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 14.27
Output dim: 7, lower bound: -242.0318040, upper bound: 242.0300418
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 14.27
Output dim: 7, lower bound: -242.0334284, upper bound: 242.0353554
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 14.27
Output dim: 7, lower bound: -242.0334284, upper bound: 242.0395681

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -117.5783234, 93.7364578, -111.9386520, 89.2590179, -206.8373108, 205.6751099
1: -98.2753830, 83.0844269, -93.4457550, 78.9929123, -177.2682953, 176.5301819
2: -129.4295807, 84.3873444, -123.1231079, 80.3174438, -209.7470093, 207.5104370
3: -137.8265839, 73.2234192, -131.0056610, 69.5862427, -207.4128113, 204.2290802
4: -126.3355255, 97.0400848, -120.2404709, 92.3546524, -218.6901550, 217.2805481
5: -113.6156311, 88.9872818, -108.1908722, 84.7078705, -198.3235016, 197.1781616
6: -108.4517746, 104.1599121, -103.2251053, 99.0864258, -207.5382080, 207.3850098
7: -118.2824936, 99.9351883, -112.5095215, 95.1105652, -213.3930664, 212.4447021
8: -141.6492004, 96.8635483, -134.8593140, 92.1977158, -233.8469238, 231.7228699
9: -107.9062424, 106.4206467, -102.6040039, 101.2240829, -209.1303253, 209.0246277

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0278688, upper bound: 242.0261593
time: 6.20 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0278688, upper bound: 242.0279245
time: 5.81 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -119.5140991, 95.2666245, -120.0655136, 95.6869736, -215.2010803, 215.3321381
1: -99.9024429, 84.4575806, -100.2791214, 84.7638321, -184.6662750, 184.7366943
2: -131.5668793, 85.7744370, -132.0803986, 86.1466446, -217.7135315, 217.8548279
3: -140.1050415, 74.4418182, -140.5962830, 74.7154922, -214.8205261, 215.0380859
4: -128.4107208, 98.6364822, -128.9444580, 99.0531616, -227.4638824, 227.5809174
5: -115.4722366, 90.4375610, -115.9812698, 90.8096008, -206.2818298, 206.4188232
6: -110.2378006, 105.8749466, -110.7351685, 106.2801132, -216.5179138, 216.6100769
7: -120.2295685, 101.5678558, -120.7009354, 101.9870148, -222.2165833, 222.2687988
8: -143.9831390, 98.4588470, -144.6401062, 98.8720703, -242.8552094, 243.0989532
9: -109.6788559, 108.1721420, -110.0743866, 108.5977478, -218.2766113, 218.2465057

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0291698, upper bound: 242.0270548
time: 6.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0291698, upper bound: 242.0300418
time: 6.67 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -125.0534821, 99.6287537, -119.5140991, 95.2666245, -220.3200989, 219.1428528
1: -104.4979324, 88.3349228, -99.9024429, 84.4575806, -188.9555054, 188.2373505
2: -137.6306152, 89.7292252, -131.5668793, 85.7744370, -223.4050598, 221.2960968
3: -146.5207214, 77.8730164, -140.1050415, 74.4418182, -220.9625244, 217.9780579
4: -134.3097839, 103.1774673, -128.4107208, 98.6364822, -232.9462433, 231.5881805
5: -120.7826691, 94.5478973, -115.4722366, 90.4375610, -211.2202148, 210.0201263
6: -115.3268051, 110.7257385, -110.2378006, 105.8749466, -221.2017212, 220.9635315
7: -125.7305145, 106.1953125, -120.2295685, 101.5678558, -227.2983704, 226.4248810
8: -150.6563873, 102.9990997, -143.9831390, 98.4588470, -249.1152344, 246.9822388
9: -114.6585846, 113.1035004, -109.6788559, 108.1721420, -222.8307037, 222.7823486

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0279245, upper bound: 242.0291403
time: 7.12 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0300418, upper bound: 242.0318040
time: 6.54 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -125.0534821, 99.6287537, -125.0534821, 99.6287537, -224.6822357, 224.6822357
1: -104.4979324, 88.3349228, -104.4979324, 88.3349228, -192.8328552, 192.8328552
2: -137.6306152, 89.7292252, -137.6306152, 89.7292252, -227.3598328, 227.3598328
3: -146.5207214, 77.8730164, -146.5207214, 77.8730164, -224.3937378, 224.3937378
4: -134.3097839, 103.1774673, -134.3097839, 103.1774673, -237.4872437, 237.4872437
5: -120.7826691, 94.5478973, -120.7826691, 94.5478973, -215.3305664, 215.3305664
6: -115.3268051, 110.7257385, -115.3268051, 110.7257385, -226.0525360, 226.0525360
7: -125.7305145, 106.1953125, -125.7305145, 106.1953125, -231.9258270, 231.9258270
8: -150.6563873, 102.9990997, -150.6563873, 102.9990997, -253.6554871, 253.6554871
9: -114.6585846, 113.1035004, -114.6585846, 113.1035004, -227.7620697, 227.7620697

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 185

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0279245, upper bound: 242.0333219
time: 7.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0300418, upper bound: 242.0318040
time: 7.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 15.67 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.67
Output dim: 7, lower bound: -242.0278688, upper bound: 242.0261593
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.67
Output dim: 7, lower bound: -242.0278688, upper bound: 242.0279245
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.67
Output dim: 7, lower bound: -242.0291698, upper bound: 242.0270548
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.67
Output dim: 7, lower bound: -242.0291698, upper bound: 242.0300418
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.67
Output dim: 7, lower bound: -242.0279245, upper bound: 242.0291403
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.67
Output dim: 7, lower bound: -242.0300418, upper bound: 242.0318040
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.67
Output dim: 7, lower bound: -242.0279245, upper bound: 242.0333219
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.67
Output dim: 7, lower bound: -242.0300418, upper bound: 242.0318040

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -101.4301071, 80.9681244, -111.9386520, 89.2590179, -190.6891174, 192.9067688
1: -84.6959610, 71.6358643, -93.4457550, 78.9929123, -163.6888733, 165.0816193
2: -111.6088715, 72.8202286, -123.1231079, 80.3174438, -191.9263000, 195.9433289
3: -118.8277435, 63.0818062, -131.0056610, 69.5862427, -188.4139862, 194.0874634
4: -109.0274353, 83.7317886, -120.2404709, 92.3546524, -201.3820801, 203.9722595
5: -98.1165009, 76.8906250, -108.1908722, 84.7078705, -182.8243408, 185.0814972
6: -93.5580368, 89.8654327, -103.2251053, 99.0864258, -192.6444702, 193.0905457
7: -102.0507278, 86.3209839, -112.5095215, 95.1105652, -197.1612854, 198.8305054
8: -122.1853104, 83.5736618, -134.8593140, 92.1977158, -214.3829956, 218.4329834
9: -93.1329193, 91.8287811, -102.6040039, 101.2240829, -194.3569336, 194.4327698

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0243797
time: 6.87 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0261593
time: 5.72 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -110.0347748, 87.7748642, -111.9386520, 89.2590179, -199.2937775, 199.7135162
1: -91.9302521, 77.7438660, -93.4457550, 78.9929123, -170.9231567, 171.1896210
2: -121.0906143, 78.9880219, -123.1231079, 80.3174438, -201.4080505, 202.1111298
3: -128.9739685, 68.5081482, -131.0056610, 69.5862427, -198.5601807, 199.5137787
4: -118.2392120, 90.8231964, -120.2404709, 92.3546524, -210.5938568, 211.0636597
5: -106.3668823, 83.3512802, -108.1908722, 84.7078705, -191.0747528, 191.5421448
6: -101.5052795, 97.4777069, -103.2251053, 99.0864258, -200.5917053, 200.7028046
7: -110.7176361, 93.5975189, -112.5095215, 95.1105652, -205.8282013, 206.1070251
8: -132.5403595, 90.6395264, -134.8593140, 92.1977158, -224.7380676, 225.4988403
9: -101.0330429, 99.6307449, -102.6040039, 101.2240829, -202.2571106, 202.2347260

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0261029
time: 6.15 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0279245
time: 6.14 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -101.4301071, 80.9681244, -120.0655136, 95.6869736, -197.1170807, 201.0336304
1: -84.6959610, 71.6358643, -100.2791214, 84.7638321, -169.4597931, 171.9149780
2: -111.6088715, 72.8202286, -132.0803986, 86.1466446, -197.7555237, 204.9006348
3: -118.8277435, 63.0818062, -140.5962830, 74.7154922, -193.5432434, 203.6780853
4: -109.0274353, 83.7317886, -128.9444580, 99.0531616, -208.0805969, 212.6762390
5: -98.1165009, 76.8906250, -115.9812698, 90.8096008, -188.9260864, 192.8718719
6: -93.5580368, 89.8654327, -110.7351685, 106.2801132, -199.8381500, 200.6006012
7: -102.0507278, 86.3209839, -120.7009354, 101.9870148, -204.0377350, 207.0219116
8: -122.1853104, 83.5736618, -144.6401062, 98.8720703, -221.0573730, 228.2137756
9: -93.1329193, 91.8287811, -110.0743866, 108.5977478, -201.7306061, 201.9031525

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0255023
time: 5.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0270548
time: 6.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -110.0347748, 87.7748642, -120.0655136, 95.6869736, -205.7217407, 207.8403778
1: -91.9302521, 77.7438660, -100.2791214, 84.7638321, -176.6940918, 178.0229797
2: -121.0906143, 78.9880219, -132.0803986, 86.1466446, -207.2372589, 211.0684204
3: -128.9739685, 68.5081482, -140.5962830, 74.7154922, -203.6894531, 209.1044006
4: -118.2392120, 90.8231964, -128.9444580, 99.0531616, -217.2923737, 219.7676544
5: -106.3668823, 83.3512802, -115.9812698, 90.8096008, -197.1764832, 199.3325500
6: -101.5052795, 97.4777069, -110.7351685, 106.2801132, -207.7854004, 208.2128296
7: -110.7176361, 93.5975189, -120.7009354, 101.9870148, -212.7046509, 214.2984619
8: -132.5403595, 90.6395264, -144.6401062, 98.8720703, -231.4124298, 235.2796326
9: -101.0330429, 99.6307449, -110.0743866, 108.5977478, -209.6307831, 209.7050781

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0284998
time: 5.61 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0300418
time: 6.69 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -107.5060654, 85.7508545, -117.5783234, 93.7364578, -201.2424927, 203.3291626
1: -89.7403564, 75.8904114, -98.2753830, 83.0844269, -172.8247833, 174.1657867
2: -118.2596817, 77.1554565, -129.4295807, 84.3873444, -202.6470337, 206.5850372
3: -125.8651352, 66.8420029, -137.8265839, 73.2234192, -199.0885468, 204.6685791
4: -115.4971466, 88.7138519, -126.3355255, 97.0400848, -212.5372009, 215.0493622
5: -103.9413300, 81.4007568, -113.6156311, 88.9872818, -192.9286041, 195.0163879
6: -99.1404343, 95.1850281, -108.4517746, 104.1599121, -203.3003387, 203.6367950
7: -108.0864563, 91.3960953, -118.2824936, 99.9351883, -208.0216370, 209.6785889
8: -129.5045929, 88.5489120, -141.6492004, 96.8635483, -226.3681335, 230.1981049
9: -98.5942459, 97.2404938, -107.9062424, 106.4206467, -205.0148621, 205.1467285

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0261593, upper bound: 242.0278688
time: 6.31 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0261593, upper bound: 242.0291404
time: 6.36 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -115.3595352, 91.9651337, -119.5140991, 95.2666245, -210.6261597, 211.4792328
1: -96.3469696, 81.4689941, -99.9024429, 84.4575806, -180.8045502, 181.3714294
2: -126.9190979, 82.7938156, -131.5668793, 85.7744370, -212.6935272, 214.3606873
3: -135.1389160, 71.8030090, -140.1050415, 74.4418182, -209.5807190, 211.9080505
4: -123.9117508, 95.1867371, -128.4107208, 98.6364822, -222.5482025, 223.5974274
5: -111.4708710, 87.2975616, -115.4722366, 90.4375610, -201.9084320, 202.7697906
6: -106.3994598, 102.1400833, -110.2378006, 105.8749466, -212.2743988, 212.3778839
7: -116.0046310, 98.0456085, -120.2295685, 101.5678558, -217.5724792, 218.2751770
8: -138.9573975, 95.0015106, -143.9831390, 98.4588470, -237.4162445, 238.9846497
9: -105.8180695, 104.3705444, -109.6788559, 108.1721420, -213.9902039, 214.0493927

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0270548, upper bound: 242.0291698
time: 6.52 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0270548, upper bound: 242.0318040
time: 5.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -107.5060654, 85.7508545, -123.1644440, 98.1351013, -205.6411285, 208.9152832
1: -89.7403564, 75.8904114, -102.9091339, 86.9944763, -176.7348328, 178.7995453
2: -118.2596817, 77.1554565, -135.5443726, 88.3741455, -206.6338196, 212.6998291
3: -125.8651352, 66.8420029, -144.2964325, 76.6839523, -202.5490570, 211.1384277
4: -115.4971466, 88.7138519, -132.2835083, 101.6194687, -217.1166077, 220.9973602
5: -103.9413300, 81.4007568, -118.9702682, 93.1325226, -197.0738220, 200.3710175
6: -99.1404343, 95.1850281, -113.5834732, 109.0513611, -208.1917877, 208.7684631
7: -108.0864563, 91.3960953, -123.8294983, 104.6010666, -212.6874695, 215.2255859
8: -129.5045929, 88.5489120, -148.3782959, 101.4420853, -230.9466858, 236.9272003
9: -98.5942459, 97.2404938, -112.9276276, 111.3937073, -209.9879303, 210.1680908

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0322595, upper bound: 242.0322969
time: 6.39 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0322595, upper bound: 242.0333219
time: 6.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -115.3595352, 91.9651337, -125.0534821, 99.6287537, -214.9882660, 217.0186157
1: -96.3469696, 81.4689941, -104.4979324, 88.3349228, -184.6818542, 185.9669189
2: -126.9190979, 82.7938156, -137.6306152, 89.7292252, -216.6483154, 220.4244385
3: -135.1389160, 71.8030090, -146.5207214, 77.8730164, -213.0119324, 218.3237152
4: -123.9117508, 95.1867371, -134.3097839, 103.1774673, -227.0892181, 229.4965210
5: -111.4708710, 87.2975616, -120.7826691, 94.5478973, -206.0187683, 208.0802307
6: -106.3994598, 102.1400833, -115.3268051, 110.7257385, -217.1251984, 217.4668732
7: -116.0046310, 98.0456085, -125.7305145, 106.1953125, -222.1999359, 223.7761230
8: -138.9573975, 95.0015106, -150.6563873, 102.9990997, -241.9564972, 245.6578979
9: -105.8180695, 104.3705444, -114.6585846, 113.1035004, -218.9215698, 219.0290833

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0332261, upper bound: 242.0337365
time: 8.15 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0332261, upper bound: 242.0356734
time: 7.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.47 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0243797
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0261593
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0261029
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0279245
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0255023
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0270548
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0284998
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0243797, upper bound: 242.0300418
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0261593, upper bound: 242.0278688
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0261593, upper bound: 242.0291404
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0270548, upper bound: 242.0291698
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0270548, upper bound: 242.0318040
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0322595, upper bound: 242.0322969
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0322595, upper bound: 242.0333219
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0332261, upper bound: 242.0337365
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.47
Output dim: 7, lower bound: -242.0332261, upper bound: 242.0356734

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -101.4301071, 80.9681244, -101.4403152, 80.9749603, -182.4050446, 182.4084473
1: -84.6959610, 71.6358643, -84.7008438, 71.6421127, -156.3380737, 156.3367004
2: -111.6088715, 72.8202286, -111.6212845, 72.8291626, -184.4380341, 184.4415131
3: -118.8277435, 63.0818062, -118.8426285, 63.0874023, -181.9151306, 181.9244385
4: -109.0274353, 83.7317886, -109.0396881, 83.7397614, -192.7671967, 192.7714844
5: -98.1165009, 76.8906250, -98.1256027, 76.8953857, -175.0118713, 175.0162048
6: -93.5580368, 89.8654327, -93.5700531, 89.8747025, -183.4327393, 183.4354858
7: -102.0507278, 86.3209839, -102.0631409, 86.3288956, -188.3796082, 188.3841248
8: -122.1853104, 83.5736618, -122.1987686, 83.5815735, -205.7668457, 205.7724304
9: -93.1329193, 91.8287811, -93.1418228, 91.8370972, -184.9699402, 184.9705811

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0067666, upper bound: 242.0086707
time: 6.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0021966, upper bound: 242.0021966
time: 6.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -101.4301071, 80.9681244, -107.5060654, 85.7508545, -187.1809540, 188.4741669
1: -84.6959610, 71.6358643, -89.7403564, 75.8904114, -160.5863647, 161.3762207
2: -111.6088715, 72.8202286, -118.2596817, 77.1554565, -188.7643280, 191.0799103
3: -118.8277435, 63.0818062, -125.8651352, 66.8420029, -185.6697388, 188.9469452
4: -109.0274353, 83.7317886, -115.4971466, 88.7138519, -197.7412872, 199.2289429
5: -98.1165009, 76.8906250, -103.9413300, 81.4007568, -179.5172424, 180.8318939
6: -93.5580368, 89.8654327, -99.1404343, 95.1850281, -188.7430725, 189.0058594
7: -102.0507278, 86.3209839, -108.0864563, 91.3960953, -193.4468231, 194.4074249
8: -122.1853104, 83.5736618, -129.5045929, 88.5489120, -210.7341766, 213.0782471
9: -93.1329193, 91.8287811, -98.5942459, 97.2404938, -190.3733521, 190.4230042

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 49

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0067666, upper bound: 242.0115359
time: 6.48 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0021966, upper bound: 242.0045905
time: 7.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -110.0347748, 87.7748642, -101.4403152, 80.9749603, -191.0097198, 189.2151794
1: -91.9302521, 77.7438660, -84.7008438, 71.6421127, -163.5723572, 162.4447021
2: -121.0906143, 78.9880219, -111.6212845, 72.8291626, -193.9197693, 190.6093140
3: -128.9739685, 68.5081482, -118.8426285, 63.0874023, -192.0613403, 187.3507538
4: -118.2392120, 90.8231964, -109.0396881, 83.7397614, -201.9789734, 199.8628693
5: -106.3668823, 83.3512802, -98.1256027, 76.8953857, -183.2622681, 181.4768829
6: -101.5052795, 97.4777069, -93.5700531, 89.8747025, -191.3799591, 191.0477295
7: -110.7176361, 93.5975189, -102.0631409, 86.3288956, -197.0465393, 195.6606140
8: -132.5403595, 90.6395264, -122.1987686, 83.5815735, -216.1219330, 212.8382874
9: -101.0330429, 99.6307449, -93.1418228, 91.8370972, -192.8701172, 192.7725220

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0078573, upper bound: 242.0101834
time: 5.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0035329, upper bound: 242.0037582
time: 5.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -110.0347748, 87.7748642, -107.5060654, 85.7508545, -195.7856293, 195.2808990
1: -91.9302521, 77.7438660, -89.7403564, 75.8904114, -167.8206635, 167.4842224
2: -121.0906143, 78.9880219, -118.2596817, 77.1554565, -198.2460632, 197.2477112
3: -128.9739685, 68.5081482, -125.8651352, 66.8420029, -195.8159637, 194.3732452
4: -118.2392120, 90.8231964, -115.4971466, 88.7138519, -206.9530640, 206.3203430
5: -106.3668823, 83.3512802, -103.9413300, 81.4007568, -187.7676392, 187.2925873
6: -101.5052795, 97.4777069, -99.1404343, 95.1850281, -196.6902924, 196.6181030
7: -110.7176361, 93.5975189, -108.0864563, 91.3960953, -202.1137238, 201.6839294
8: -132.5403595, 90.6395264, -129.5045929, 88.5489120, -221.0892639, 220.1441193
9: -101.0330429, 99.6307449, -98.5942459, 97.2404938, -198.2735291, 198.2249603

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0078573, upper bound: 242.0132639
time: 6.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0035329, upper bound: 242.0064131
time: 6.30 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -101.4301071, 80.9681244, -110.0320435, 87.7727509, -189.2028503, 191.0001678
1: -84.6959610, 71.6358643, -91.9280167, 77.7420349, -162.4379883, 163.5638733
2: -111.6088715, 72.8202286, -121.0876007, 78.9861374, -190.5950012, 193.9078369
3: -118.8277435, 63.0818062, -128.9708557, 68.5064926, -187.3342285, 192.0526581
4: -109.0274353, 83.7317886, -118.2362671, 90.8210144, -199.8484497, 201.9680481
5: -98.1165009, 76.8906250, -106.3643036, 83.3492813, -181.4657593, 183.2548828
6: -93.5580368, 89.8654327, -101.5029068, 97.4752808, -191.0333252, 191.3683472
7: -102.0507278, 86.3209839, -110.7149658, 93.5952530, -195.6459808, 197.0359497
8: -122.1853104, 83.5736618, -132.5371552, 90.6372757, -212.8225555, 216.1108093
9: -93.1329193, 91.8287811, -101.0305481, 99.6283188, -192.7611694, 192.8593292

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 49

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0085065, upper bound: 242.0102667
time: 7.20 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0037582, upper bound: 242.0035329
time: 7.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -101.4301071, 80.9681244, -115.3595352, 91.9651337, -193.3952332, 196.3276520
1: -84.6959610, 71.6358643, -96.3469696, 81.4689941, -166.1649475, 167.9828186
2: -111.6088715, 72.8202286, -126.9190979, 82.7938156, -194.4026794, 199.7393188
3: -118.8277435, 63.0818062, -135.1389160, 71.8030090, -190.6307526, 198.2207184
4: -109.0274353, 83.7317886, -123.9117508, 95.1867371, -204.2141724, 207.6435394
5: -98.1165009, 76.8906250, -111.4708710, 87.2975616, -185.4140472, 188.3614807
6: -93.5580368, 89.8654327, -106.3994598, 102.1400833, -195.6981201, 196.2648926
7: -102.0507278, 86.3209839, -116.0046310, 98.0456085, -200.0963135, 202.3256226
8: -122.1853104, 83.5736618, -138.9573975, 95.0015106, -217.1867828, 222.5310669
9: -93.1329193, 91.8287811, -105.8180695, 104.3705444, -197.5033875, 197.6468353

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0085065, upper bound: 242.0126476
time: 7.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0037582, upper bound: 242.0054488
time: 7.47 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -110.0347748, 87.7748642, -110.0320435, 87.7727509, -197.8075256, 197.8069153
1: -91.9302521, 77.7438660, -91.9280167, 77.7420349, -169.6722870, 169.6718750
2: -121.0906143, 78.9880219, -121.0876007, 78.9861374, -200.0767517, 200.0756226
3: -128.9739685, 68.5081482, -128.9708557, 68.5064926, -197.4804230, 197.4789429
4: -118.2392120, 90.8231964, -118.2362671, 90.8210144, -209.0602264, 209.0594635
5: -106.3668823, 83.3512802, -106.3643036, 83.3492813, -189.7161560, 189.7155762
6: -101.5052795, 97.4777069, -101.5029068, 97.4752808, -198.9805450, 198.9805756
7: -110.7176361, 93.5975189, -110.7149658, 93.5952530, -204.3128967, 204.3124847
8: -132.5403595, 90.6395264, -132.5371552, 90.6372757, -223.1776428, 223.1766815
9: -101.0330429, 99.6307449, -101.0305481, 99.6283188, -200.6613464, 200.6612701

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0112529, upper bound: 242.0138174
time: 7.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0070459, upper bound: 242.0070448
time: 6.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -110.0347748, 87.7748642, -115.3595352, 91.9651337, -201.9999084, 203.1343842
1: -91.9302521, 77.7438660, -96.3469696, 81.4689941, -173.3992462, 174.0908203
2: -121.0906143, 78.9880219, -126.9190979, 82.7938156, -203.8844299, 205.9071198
3: -128.9739685, 68.5081482, -135.1389160, 71.8030090, -200.7769470, 203.6470337
4: -118.2392120, 90.8231964, -123.9117508, 95.1867371, -213.4259491, 214.7349396
5: -106.3668823, 83.3512802, -111.4708710, 87.2975616, -193.6644440, 194.8221436
6: -101.5052795, 97.4777069, -106.3994598, 102.1400833, -203.6453552, 203.8771515
7: -110.7176361, 93.5975189, -116.0046310, 98.0456085, -208.7632446, 209.6021271
8: -132.5403595, 90.6395264, -138.9573975, 95.0015106, -227.5418701, 229.5969238
9: -101.0330429, 99.6307449, -105.8180695, 104.3705444, -205.4035645, 205.4487915

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0112529, upper bound: 242.0160426
time: 7.19 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0070459, upper bound: 242.0090294
time: 6.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -107.5060654, 85.7508545, -101.4301071, 80.9681244, -188.4741669, 187.1809540
1: -89.7403564, 75.8904114, -84.6959610, 71.6358643, -161.3762207, 160.5863647
2: -118.2596817, 77.1554565, -111.6088715, 72.8202286, -191.0799103, 188.7643280
3: -125.8651352, 66.8420029, -118.8277435, 63.0818062, -188.9469452, 185.6697388
4: -115.4971466, 88.7138519, -109.0274353, 83.7317886, -199.2289429, 197.7412872
5: -103.9413300, 81.4007568, -98.1165009, 76.8906250, -180.8318939, 179.5172424
6: -99.1404343, 95.1850281, -93.5580368, 89.8654327, -189.0058594, 188.7430725
7: -108.0864563, 91.3960953, -102.0507278, 86.3209839, -194.4074249, 193.4468231
8: -129.5045929, 88.5489120, -122.1853104, 83.5736618, -213.0782471, 210.7341766
9: -98.5942459, 97.2404938, -93.1329193, 91.8287811, -190.4230042, 190.3733521

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0084079, upper bound: 242.0118638
time: 6.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0045905, upper bound: 242.0069932
time: 5.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -107.5060654, 85.7508545, -110.0347748, 87.7748642, -195.2808990, 195.7856293
1: -89.7403564, 75.8904114, -91.9302521, 77.7438660, -167.4842224, 167.8206635
2: -118.2596817, 77.1554565, -121.0906143, 78.9880219, -197.2477112, 198.2460632
3: -125.8651352, 66.8420029, -128.9739685, 68.5081482, -194.3732452, 195.8159637
4: -115.4971466, 88.7138519, -118.2392120, 90.8231964, -206.3203430, 206.9530640
5: -103.9413300, 81.4007568, -106.3668823, 83.3512802, -187.2925873, 187.7676392
6: -99.1404343, 95.1850281, -101.5052795, 97.4777069, -196.6181030, 196.6902924
7: -108.0864563, 91.3960953, -110.7176361, 93.5975189, -201.6839294, 202.1137238
8: -129.5045929, 88.5489120, -132.5403595, 90.6395264, -220.1441193, 221.0892639
9: -98.5942459, 97.2404938, -101.0330429, 99.6307449, -198.2249603, 198.2735291

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0084079, upper bound: 242.0137572
time: 7.12 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0045905, upper bound: 242.0069932
time: 6.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -115.3595352, 91.9651337, -101.4301071, 80.9681244, -196.3276520, 193.3952332
1: -96.3469696, 81.4689941, -84.6959610, 71.6358643, -167.9828186, 166.1649475
2: -126.9190979, 82.7938156, -111.6088715, 72.8202286, -199.7393188, 194.4026794
3: -135.1389160, 71.8030090, -118.8277435, 63.0818062, -198.2207184, 190.6307526
4: -123.9117508, 95.1867371, -109.0274353, 83.7317886, -207.6435394, 204.2141724
5: -111.4708710, 87.2975616, -98.1165009, 76.8906250, -188.3614807, 185.4140472
6: -106.3994598, 102.1400833, -93.5580368, 89.8654327, -196.2648926, 195.6981201
7: -116.0046310, 98.0456085, -102.0507278, 86.3209839, -202.3256226, 200.0963135
8: -138.9573975, 95.0015106, -122.1853104, 83.5736618, -222.5310669, 217.1867828
9: -105.8180695, 104.3705444, -93.1329193, 91.8287811, -197.6468353, 197.5033875

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0092829, upper bound: 242.0131249
time: 7.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0054488, upper bound: 242.0081210
time: 6.18 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.3595352, 91.9651337, -110.0347748, 87.7748642, -203.1343842, 201.9999084
1: -96.3469696, 81.4689941, -91.9302521, 77.7438660, -174.0908203, 173.3992462
2: -126.9190979, 82.7938156, -121.0906143, 78.9880219, -205.9071198, 203.8844299
3: -135.1389160, 71.8030090, -128.9739685, 68.5081482, -203.6470337, 200.7769470
4: -123.9117508, 95.1867371, -118.2392120, 90.8231964, -214.7349396, 213.4259491
5: -111.4708710, 87.2975616, -106.3668823, 83.3512802, -194.8221436, 193.6644440
6: -106.3994598, 102.1400833, -101.5052795, 97.4777069, -203.8771515, 203.6453552
7: -116.0046310, 98.0456085, -110.7176361, 93.5975189, -209.6021271, 208.7632446
8: -138.9573975, 95.0015106, -132.5403595, 90.6395264, -229.5969238, 227.5418701
9: -105.8180695, 104.3705444, -101.0330429, 99.6307449, -205.4487915, 205.4035645

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0092829, upper bound: 242.0169296
time: 7.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0054488, upper bound: 242.0115529
time: 6.48 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -107.5060654, 85.7508545, -107.5060654, 85.7508545, -193.2568817, 193.2568817
1: -89.7403564, 75.8904114, -89.7403564, 75.8904114, -165.6307678, 165.6307678
2: -118.2596817, 77.1554565, -118.2596817, 77.1554565, -195.4151306, 195.4151306
3: -125.8651352, 66.8420029, -125.8651352, 66.8420029, -192.7071381, 192.7071381
4: -115.4971466, 88.7138519, -115.4971466, 88.7138519, -204.2109985, 204.2109985
5: -103.9413300, 81.4007568, -103.9413300, 81.4007568, -185.3420563, 185.3420563
6: -99.1404343, 95.1850281, -99.1404343, 95.1850281, -194.3254395, 194.3254395
7: -108.0864563, 91.3960953, -108.0864563, 91.3960953, -199.4825287, 199.4825134
8: -129.5045929, 88.5489120, -129.5045929, 88.5489120, -218.0534973, 218.0534973
9: -98.5942459, 97.2404938, -98.5942459, 97.2404938, -195.8347321, 195.8347321

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170175, upper bound: 242.0184183
time: 6.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0129464, upper bound: 242.0129696
time: 6.77 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -107.5060654, 85.7508545, -115.3595352, 91.9651337, -199.4711761, 201.1103668
1: -89.7403564, 75.8904114, -96.3469696, 81.4689941, -171.2093506, 172.2373505
2: -118.2596817, 77.1554565, -126.9190979, 82.7938156, -201.0534973, 204.0745544
3: -125.8651352, 66.8420029, -135.1389160, 71.8030090, -197.6681366, 201.9809265
4: -115.4971466, 88.7138519, -123.9117508, 95.1867371, -210.6838684, 212.6255951
5: -103.9413300, 81.4007568, -111.4708710, 87.2975616, -191.2388916, 192.8716278
6: -99.1404343, 95.1850281, -106.3994598, 102.1400833, -201.2805023, 201.5844879
7: -108.0864563, 91.3960953, -116.0046310, 98.0456085, -206.1320190, 207.4007263
8: -129.5045929, 88.5489120, -138.9573975, 95.0015106, -224.5061035, 227.5063019
9: -98.5942459, 97.2404938, -105.8180695, 104.3705444, -202.9647522, 203.0585632

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0170175, upper bound: 242.0196461
time: 6.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0129464, upper bound: 242.0139561
time: 6.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -115.3595352, 91.9651337, -107.5060654, 85.7508545, -201.1103668, 199.4711761
1: -96.3469696, 81.4689941, -89.7403564, 75.8904114, -172.2373505, 171.2093506
2: -126.9190979, 82.7938156, -118.2596817, 77.1554565, -204.0745544, 201.0534973
3: -135.1389160, 71.8030090, -125.8651352, 66.8420029, -201.9809265, 197.6681366
4: -123.9117508, 95.1867371, -115.4971466, 88.7138519, -212.6255951, 210.6838684
5: -111.4708710, 87.2975616, -103.9413300, 81.4007568, -192.8716278, 191.2388916
6: -106.3994598, 102.1400833, -99.1404343, 95.1850281, -201.5844879, 201.2805023
7: -116.0046310, 98.0456085, -108.0864563, 91.3960953, -207.4007263, 206.1320190
8: -138.9573975, 95.0015106, -129.5045929, 88.5489120, -227.5063019, 224.5061035
9: -105.8180695, 104.3705444, -98.5942459, 97.2404938, -203.0585632, 202.9647522

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0179848, upper bound: 242.0198597
time: 8.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0142749
time: 6.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -115.3595352, 91.9651337, -115.3595352, 91.9651337, -207.3246613, 207.3246613
1: -96.3469696, 81.4689941, -96.3469696, 81.4689941, -177.8159332, 177.8159332
2: -126.9190979, 82.7938156, -126.9190979, 82.7938156, -209.7129059, 209.7129059
3: -135.1389160, 71.8030090, -135.1389160, 71.8030090, -206.9419098, 206.9419098
4: -123.9117508, 95.1867371, -123.9117508, 95.1867371, -219.0984650, 219.0984650
5: -111.4708710, 87.2975616, -111.4708710, 87.2975616, -198.7684326, 198.7684326
6: -106.3994598, 102.1400833, -106.3994598, 102.1400833, -208.5395508, 208.5395508
7: -116.0046310, 98.0456085, -116.0046310, 98.0456085, -214.0502167, 214.0502167
8: -138.9573975, 95.0015106, -138.9573975, 95.0015106, -233.9589081, 233.9589081
9: -105.8180695, 104.3705444, -105.8180695, 104.3705444, -210.1885834, 210.1885834

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0179848, upper bound: 242.0221129
time: 6.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0162296
time: 5.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 13.92 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0067666, upper bound: 242.0086707
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0021966, upper bound: 242.0021966
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0067666, upper bound: 242.0115359
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0021966, upper bound: 242.0045905
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0078573, upper bound: 242.0101834
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0035329, upper bound: 242.0037582
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0078573, upper bound: 242.0132639
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0035329, upper bound: 242.0064131
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0085065, upper bound: 242.0102667
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0037582, upper bound: 242.0035329
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0085065, upper bound: 242.0126476
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0037582, upper bound: 242.0054488
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0112529, upper bound: 242.0138174
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0070459, upper bound: 242.0070448
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0112529, upper bound: 242.0160426
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0070459, upper bound: 242.0090294
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0084079, upper bound: 242.0118638
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0045905, upper bound: 242.0069932
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0084079, upper bound: 242.0137572
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0045905, upper bound: 242.0069932
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0092829, upper bound: 242.0131249
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0054488, upper bound: 242.0081210
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0092829, upper bound: 242.0169296
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0054488, upper bound: 242.0115529
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0170175, upper bound: 242.0184183
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0129464, upper bound: 242.0129696
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0170175, upper bound: 242.0196461
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0129464, upper bound: 242.0139561
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0179848, upper bound: 242.0198597
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0142749
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0179848, upper bound: 242.0221129
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.92
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0162296

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -104.5879517, 83.4538193, -110.0320435, 87.7727509, -192.3607025, 193.4858704
1: -87.3516312, 73.9139023, -91.9280167, 77.7420349, -165.0936584, 165.8419189
2: -115.0852432, 75.0903778, -121.0876007, 78.9861374, -194.0713501, 196.1779785
3: -122.6017914, 65.1172791, -128.9708557, 68.5064926, -191.1082611, 194.0881195
4: -112.3994446, 86.3296204, -118.2362671, 90.8210144, -203.2204590, 204.5658875
5: -101.1284027, 79.2552185, -106.3643036, 83.3492813, -184.4776611, 185.6195068
6: -96.4895096, 92.6721268, -101.5029068, 97.4752808, -193.9647369, 194.1750183
7: -105.2536469, 89.0225754, -110.7149658, 93.5952530, -198.8489075, 199.7375488
8: -125.9500122, 86.1149445, -132.5371552, 90.6372757, -216.5872803, 218.6520691
9: -96.0716019, 94.7195282, -101.0305481, 99.6283188, -195.6999207, 195.7500763

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0070459, upper bound: 242.0070448
time: 5.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0070459, upper bound: 242.0070448
time: 5.29 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -104.5879517, 83.4538193, -115.3595352, 91.9651337, -196.5530853, 198.8133392
1: -87.3516312, 73.9139023, -96.3469696, 81.4689941, -168.8206177, 170.2608337
2: -115.0852432, 75.0903778, -126.9190979, 82.7938156, -197.8790436, 202.0094757
3: -122.6017914, 65.1172791, -135.1389160, 71.8030090, -194.4047852, 200.2561951
4: -112.3994446, 86.3296204, -123.9117508, 95.1867371, -207.5861816, 210.2413635
5: -101.1284027, 79.2552185, -111.4708710, 87.2975616, -188.4259491, 190.7260895
6: -96.4895096, 92.6721268, -106.3994598, 102.1400833, -198.6295471, 199.0715790
7: -105.2536469, 89.0225754, -116.0046310, 98.0456085, -203.2992554, 205.0271759
8: -125.9500122, 86.1149445, -138.9573975, 95.0015106, -220.9515228, 225.0723267
9: -96.0716019, 94.7195282, -105.8180695, 104.3705444, -200.4421387, 200.5375977

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0116005, upper bound: 242.0090294
time: 6.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0116005, upper bound: 242.0090294
time: 7.25 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -102.0835495, 81.4505463, -110.0347748, 87.7748642, -189.8583984, 191.4853210
1: -85.1836395, 72.0771255, -91.9302521, 77.7438660, -162.9275055, 164.0073853
2: -112.2822037, 73.2767334, -121.0906143, 78.9880219, -191.2702179, 194.3673401
3: -119.5193939, 63.4695854, -128.9739685, 68.5081482, -188.0275269, 192.4435272
4: -109.6824722, 84.2406006, -118.2392120, 90.8231964, -200.5056610, 202.4798126
5: -98.7268448, 77.3238831, -106.3668823, 83.3512802, -182.0781097, 183.6907654
6: -94.1480026, 90.4007568, -101.5052795, 97.4777069, -191.6257019, 191.9060364
7: -102.6475754, 86.8412399, -110.7176361, 93.5975189, -196.2450714, 197.5588531
8: -122.9452896, 84.0488205, -132.5403595, 90.6395264, -213.5848083, 216.5891724
9: -93.6522980, 92.3525238, -101.0330429, 99.6307449, -193.2830200, 193.3855591

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0064131, upper bound: 242.0086153
time: 7.22 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0064131, upper bound: 242.0086153
time: 7.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -109.9153442, 87.6464081, -110.0347748, 87.7748642, -197.6902161, 197.6811523
1: -91.7697220, 77.6386414, -91.9302521, 77.7438660, -169.5135803, 169.5688934
2: -120.9149094, 78.8972702, -121.0906143, 78.9880219, -199.9029236, 199.9878845
3: -128.7644196, 68.4146500, -128.9739685, 68.5081482, -197.2725372, 197.3886108
4: -118.0710983, 90.6936493, -118.2392120, 90.8231964, -208.8942871, 208.9328613
5: -106.2345276, 83.2020416, -106.3668823, 83.3512802, -189.5858154, 189.5689240
6: -101.3837814, 97.3347244, -101.5052795, 97.4777069, -198.8614655, 198.8399963
7: -110.5419388, 93.4709015, -110.7176361, 93.5975189, -204.1394653, 204.1885376
8: -132.3679504, 90.4805450, -132.5403595, 90.6395264, -223.0074768, 223.0209045
9: -100.8562927, 99.4604187, -101.0330429, 99.6307449, -200.4869995, 200.4934692

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0089991, upper bound: 242.0115529
time: 6.51 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0089991, upper bound: 242.0115529
time: 7.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -102.0835495, 81.4505463, -107.5060654, 85.7508545, -187.8343811, 188.9565735
1: -85.1836395, 72.0771255, -89.7403564, 75.8904114, -161.0740356, 161.8174744
2: -112.2822037, 73.2767334, -118.2596817, 77.1554565, -189.4376526, 191.5364075
3: -119.5193939, 63.4695854, -125.8651352, 66.8420029, -186.3613892, 189.3347168
4: -109.6824722, 84.2406006, -115.4971466, 88.7138519, -198.3963013, 199.7377472
5: -98.7268448, 77.3238831, -103.9413300, 81.4007568, -180.1275787, 181.2652130
6: -94.1480026, 90.4007568, -99.1404343, 95.1850281, -189.3330383, 189.5411835
7: -102.6475754, 86.8412399, -108.0864563, 91.3960953, -194.0436707, 194.9276276
8: -122.9452896, 84.0488205, -129.5045929, 88.5489120, -211.4941711, 213.5534058
9: -93.6522980, 92.3525238, -98.5942459, 97.2404938, -190.8927917, 190.9467468

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0129464, upper bound: 242.0129696
time: 6.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0129464, upper bound: 242.0129696
time: 5.03 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -102.0835495, 81.4505463, -115.3595352, 91.9651337, -194.0486755, 196.8100586
1: -85.1836395, 72.0771255, -96.3469696, 81.4689941, -166.6526184, 168.4240723
2: -112.2822037, 73.2767334, -126.9190979, 82.7938156, -195.0760040, 200.1958313
3: -119.5193939, 63.4695854, -135.1389160, 71.8030090, -191.3224030, 198.6085052
4: -109.6824722, 84.2406006, -123.9117508, 95.1867371, -204.8691864, 208.1523438
5: -98.7268448, 77.3238831, -111.4708710, 87.2975616, -186.0243988, 188.7947540
6: -94.1480026, 90.4007568, -106.3994598, 102.1400833, -196.2880859, 196.8002167
7: -102.6475754, 86.8412399, -116.0046310, 98.0456085, -200.6931610, 202.8458252
8: -122.9452896, 84.0488205, -138.9573975, 95.0015106, -217.9468079, 223.0062256
9: -93.6522980, 92.3525238, -105.8180695, 104.3705444, -198.0228271, 198.1705780

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0142574, upper bound: 242.0139561
time: 6.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0142574, upper bound: 242.0139561
time: 5.98 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -95.3692551, 76.1016769, -112.3373184, 89.5667191, -184.9359589, 188.4389496
1: -79.4669876, 67.3152466, -93.8063049, 79.3451233, -158.8121033, 161.1215515
2: -104.8592224, 68.4101028, -123.5878067, 80.6319733, -185.4911804, 191.9978943
3: -111.7288055, 59.1629257, -131.6047516, 69.9236984, -181.6524963, 190.7676697
4: -102.4642410, 78.6306305, -120.6722412, 92.6951904, -195.1594238, 199.3028564
5: -92.2550278, 72.1868973, -108.5648956, 85.0265045, -177.2815094, 180.7517395
6: -87.9440308, 84.4696274, -103.6203537, 99.4739761, -187.4179535, 188.0899658
7: -95.8845520, 81.1856461, -112.9753723, 95.5079346, -191.3924866, 194.1610107
8: -114.7206879, 78.2988129, -135.3015747, 92.4914474, -207.2121277, 213.6003876
9: -87.5127258, 86.2015457, -103.0655899, 101.6471405, -189.1598511, 189.2671356

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0090212, upper bound: 242.0086339
time: 7.29 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0123675, upper bound: 242.0121917
time: 6.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -109.9153442, 87.6464081, -107.5060654, 85.7508545, -195.6661987, 195.1524048
1: -91.7697220, 77.6386414, -89.7403564, 75.8904114, -167.6601257, 167.3789978
2: -120.9149094, 78.8972702, -118.2596817, 77.1554565, -198.0703583, 197.1569519
3: -128.7644196, 68.4146500, -125.8651352, 66.8420029, -195.6064148, 194.2797852
4: -118.0710983, 90.6936493, -115.4971466, 88.7138519, -206.7849426, 206.1907959
5: -106.2345276, 83.2020416, -103.9413300, 81.4007568, -187.6352844, 187.1433716
6: -101.3837814, 97.3347244, -99.1404343, 95.1850281, -196.5688019, 196.4751587
7: -110.5419388, 93.4709015, -108.0864563, 91.3960953, -201.9380341, 201.5573578
8: -132.3679504, 90.4805450, -129.5045929, 88.5489120, -220.9168549, 219.9851379
9: -100.8562927, 99.4604187, -98.5942459, 97.2404938, -198.0967712, 198.0546570

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0142749
time: 6.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0142749
time: 8.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -103.3080063, 82.3943329, -104.5048904, 83.3694534, -186.6774292, 186.8992310
1: -86.1719894, 72.9629822, -87.2164688, 73.7811661, -159.9531555, 160.1794434
2: -113.6262817, 74.0985718, -114.9509430, 75.0110016, -188.6372833, 189.0494995
3: -121.1059265, 64.1785583, -122.3547440, 64.9770279, -186.0829163, 186.5332947
4: -110.9777451, 85.1802902, -112.2790985, 86.2391434, -197.2168732, 197.4593811
5: -99.8724747, 78.1572037, -101.0556412, 79.1451645, -179.0176392, 179.2128448
6: -95.2762222, 91.5114365, -96.3811035, 92.5364761, -187.8126831, 187.8925476
7: -103.8960419, 87.9007187, -105.0792847, 88.8780518, -192.7740631, 192.9799652
8: -124.2783966, 84.8235474, -125.8736420, 86.0557175, -210.3341064, 210.6971741
9: -94.8086548, 93.4204407, -95.8614044, 94.5357285, -189.3443909, 189.2818146

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0086102, upper bound: 242.0086402
time: 6.49 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0120284, upper bound: 242.0123840
time: 6.00 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -109.9153442, 87.6464081, -115.3595352, 91.9651337, -201.8804779, 203.0058899
1: -91.7697220, 77.6386414, -96.3469696, 81.4689941, -173.2387085, 173.9855804
2: -120.9149094, 78.8972702, -126.9190979, 82.7938156, -203.7087097, 205.8163757
3: -128.7644196, 68.4146500, -135.1389160, 71.8030090, -200.5674286, 203.5535583
4: -118.0710983, 90.6936493, -123.9117508, 95.1867371, -213.2578430, 214.6053925
5: -106.2345276, 83.2020416, -111.4708710, 87.2975616, -193.5320892, 194.6729126
6: -101.3837814, 97.3347244, -106.3994598, 102.1400833, -203.5238647, 203.7341919
7: -110.5419388, 93.4709015, -116.0046310, 98.0456085, -208.5875549, 209.4755249
8: -132.3679504, 90.4805450, -138.9573975, 95.0015106, -227.3694611, 229.4379425
9: -100.8562927, 99.4604187, -105.8180695, 104.3705444, -205.2268066, 205.2784882

Time for backsubstitution: 1.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0162011, upper bound: 242.0162296
time: 6.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0162011, upper bound: 242.0162296
time: 6.95 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 15.10 seconds
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0070459, upper bound: 242.0070448
IS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0070459, upper bound: 242.0070448
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0116005, upper bound: 242.0090294
IS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0116005, upper bound: 242.0090294
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0064131, upper bound: 242.0086153
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0064131, upper bound: 242.0086153
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0089991, upper bound: 242.0115529
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0089991, upper bound: 242.0115529
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0129464, upper bound: 242.0129696
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0129464, upper bound: 242.0129696
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0142574, upper bound: 242.0139561
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0142574, upper bound: 242.0139561
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0090212, upper bound: 242.0086339
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0123675, upper bound: 242.0121917
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0142749
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0142749
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0086102, upper bound: 242.0086402
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0120284, upper bound: 242.0123840
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0162011, upper bound: 242.0162296
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 15.10
Output dim: 7, lower bound: -242.0162011, upper bound: 242.0162296
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.10
Output dim: 7, lower bound: -242.0138685, upper bound: 242.0162296
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=243.70181274414062
rel_dist={7: [-242.04184490722724, 242.04184490722724]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0335970, upper bound: 242.0322714
time: 7.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0391004, upper bound: 242.0391004
time: 8.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.73
Output dim: 7, lower bound: -242.0335970, upper bound: 242.0322714
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.73
Output dim: 7, lower bound: -242.0391004, upper bound: 242.0391004

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -119.5140991, 95.2666245, -125.9023590, 100.3039017, -219.8179932, 221.1689758
1: -99.9024429, 84.4575806, -105.2167740, 88.9300232, -188.8324585, 189.6743469
2: -131.5668793, 85.7744370, -138.5635223, 90.3358002, -221.9026642, 224.3379517
3: -140.1050415, 74.4418182, -147.5020599, 78.4039536, -218.5090027, 221.9438629
4: -128.4107208, 98.6364822, -135.2237549, 103.8819733, -232.2926636, 233.8602295
5: -115.4722366, 90.4375610, -121.5957413, 95.1890106, -210.6612244, 212.0332794
6: -110.2378006, 105.8749466, -116.1174011, 111.4783478, -221.7161560, 221.9923096
7: -120.2295685, 101.5678558, -126.5848007, 106.9105453, -227.1401062, 228.1526489
8: -143.9831390, 98.4588470, -151.6892090, 103.7078094, -247.6909485, 250.1480560
9: -109.6788559, 108.1721420, -115.4343109, 113.8852005, -223.5640564, 223.6064148

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0270328, upper bound: 242.0262241
time: 7.72 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0302092, upper bound: 242.0289925
time: 9.21 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.0534821, 99.6287537, -129.5298157, 103.1674271, -228.2208862, 229.1585693
1: -104.4979324, 88.3349228, -108.2366486, 91.4681168, -195.9660492, 196.5715637
2: -137.6306152, 89.7292252, -142.5407410, 92.9196625, -230.5502777, 232.2699585
3: -146.5207214, 77.8730164, -151.7151794, 80.6408539, -227.1615753, 229.5881958
4: -134.3097839, 103.1774673, -139.0975952, 106.8526840, -241.1624756, 242.2750549
5: -120.7826691, 94.5478973, -125.0749054, 97.8855286, -218.6681976, 219.6228027
6: -115.3268051, 110.7257385, -119.4499283, 114.6628723, -229.9896851, 230.1756592
7: -125.7305145, 106.1953125, -130.1946716, 109.9449539, -235.6754608, 236.3899841
8: -150.6563873, 102.9990997, -156.0604553, 106.6781082, -257.3345032, 259.0595703
9: -114.6585846, 113.1035004, -118.7041702, 117.1202240, -231.7787933, 231.8076782

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0326259, upper bound: 242.0326553
time: 6.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0352706, upper bound: 242.0352706
time: 7.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.56 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.56
Output dim: 7, lower bound: -242.0270328, upper bound: 242.0262241
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.56
Output dim: 7, lower bound: -242.0302092, upper bound: 242.0289925
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.56
Output dim: 7, lower bound: -242.0326259, upper bound: 242.0326553
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.56
Output dim: 7, lower bound: -242.0352706, upper bound: 242.0352706

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -112.3835526, 89.6300430, -108.1289902, 86.2522430, -198.6358032, 197.7590027
1: -93.9085922, 79.3988342, -90.2710495, 76.3267822, -170.2353821, 169.6698761
2: -123.6942596, 80.6644363, -118.9460297, 77.6036224, -201.2978821, 199.6104736
3: -131.7135162, 69.9551315, -126.5837631, 67.2356873, -198.9492035, 196.5388947
4: -120.7650528, 92.7563553, -116.1709061, 89.2314682, -209.9965210, 208.9272614
5: -108.6327972, 85.0961761, -104.5385590, 81.8746262, -190.5074158, 189.6347198
6: -103.6588974, 99.5569992, -99.7218246, 95.7410736, -199.3999634, 199.2787781
7: -113.0569534, 95.5541916, -108.7163849, 91.9235458, -204.9804840, 204.2705688
8: -135.3855896, 92.5829773, -130.2655487, 89.0761719, -224.4617615, 222.8485260
9: -103.1496964, 101.7214737, -99.1681595, 97.8196564, -200.9693451, 200.8896179

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0091192, upper bound: 242.0083498
time: 7.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0063286, upper bound: 242.0044880
time: 7.01 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -116.4236908, 92.8242722, -116.3610077, 92.7625275, -209.1861877, 209.1852570
1: -97.3034897, 82.2674484, -97.1916809, 82.1721649, -179.4756470, 179.4591370
2: -128.1509705, 83.5611420, -128.0179443, 83.5070724, -211.6580048, 211.5790863
3: -136.4749451, 72.5059509, -136.2958984, 72.4303818, -208.9053345, 208.8018036
4: -125.0943222, 96.0886078, -124.9865723, 96.0168533, -221.1111603, 221.0751801
5: -112.5043640, 88.1265259, -112.4294205, 88.0547714, -200.5591431, 200.5559387
6: -107.3896790, 103.1371994, -107.3285751, 103.0263901, -210.4160614, 210.4657745
7: -117.1272736, 98.9687653, -117.0118942, 98.8881760, -216.0154419, 215.9806519
8: -140.2528381, 95.9094315, -140.1730194, 95.8361359, -236.0889587, 236.0824585
9: -106.8592377, 105.3858871, -106.7330780, 105.2875443, -212.1467743, 212.1189575

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0127955, upper bound: 242.0120996
time: 8.16 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0099088, upper bound: 242.0079949
time: 8.56 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -118.0925064, 94.1237640, -111.9002609, 89.2259216, -207.3184052, 206.0239868
1: -98.6433411, 83.3953018, -93.4104919, 78.9663162, -177.6096497, 176.8057861
2: -129.9424744, 84.7362595, -123.0802536, 80.2885742, -210.2310181, 207.8165131
3: -138.3243103, 73.4917984, -130.9627838, 69.5600357, -207.8843384, 204.4545898
4: -126.8427582, 97.4366455, -120.1974640, 92.3212814, -219.1640320, 217.6341095
5: -114.1037064, 89.3325882, -108.1543961, 84.6765366, -198.7802277, 197.4869690
6: -108.9029617, 104.5559311, -103.1875687, 99.0504608, -207.9534149, 207.7434998
7: -118.7249603, 100.3213043, -112.4688187, 95.0768356, -213.8017883, 212.7901306
8: -142.2622833, 97.2613144, -134.8107605, 92.1614151, -234.4237061, 232.0720825
9: -108.2814560, 106.8045425, -102.5658798, 101.1833191, -209.4647675, 209.3704224

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0152982, upper bound: 242.0161018
time: 6.40 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0131374, upper bound: 242.0132806
time: 7.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -121.8824310, 97.1216965, -119.9105988, 95.5625839, -217.4450073, 217.0322876
1: -101.8306732, 86.0882874, -100.1470718, 84.6550293, -186.4856873, 186.2353516
2: -134.1262360, 87.4604416, -131.9106750, 86.0364151, -220.1626434, 219.3711090
3: -142.7959747, 75.8865128, -140.4179230, 74.6174088, -217.4133911, 216.3044281
4: -130.9088287, 100.5630646, -128.7778625, 98.9234467, -229.8322601, 229.3408966
5: -117.7371979, 92.1752167, -115.8332901, 90.6909180, -208.4281158, 208.0085144
6: -112.4060745, 107.9170227, -110.5902023, 106.1421432, -218.5482178, 218.5072174
7: -122.5488586, 103.5288315, -120.5437164, 101.8562393, -224.4050903, 224.0725403
8: -146.8290710, 100.3827057, -144.4513702, 98.7412338, -245.5703125, 244.8340607
9: -111.7660751, 110.2448425, -109.9310684, 108.4529266, -220.2189941, 220.1759033

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0180252, upper bound: 242.0186776
time: 8.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0156868, upper bound: 242.0156868
time: 7.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.07 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 17.07
Output dim: 7, lower bound: -242.0091192, upper bound: 242.0083498
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 17.07
Output dim: 7, lower bound: -242.0063286, upper bound: 242.0044880
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 17.07
Output dim: 7, lower bound: -242.0127955, upper bound: 242.0120996
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 17.07
Output dim: 7, lower bound: -242.0099088, upper bound: 242.0079949
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.07
Output dim: 7, lower bound: -242.0152982, upper bound: 242.0161018
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 17.07
Output dim: 7, lower bound: -242.0131374, upper bound: 242.0132806
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.07
Output dim: 7, lower bound: -242.0180252, upper bound: 242.0186776
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.07
Output dim: 7, lower bound: -242.0156868, upper bound: 242.0156868

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -112.6317139, 89.7925034, -111.9002609, 89.2259216, -201.8576355, 201.6927338
1: -94.0531082, 79.5544891, -93.4104919, 78.9663162, -173.0194244, 172.9649506
2: -123.9218903, 80.8288345, -123.0802536, 80.2885742, -204.2104492, 203.9090881
3: -131.9329987, 70.0943756, -130.9627838, 69.5600357, -201.4930267, 201.0571442
4: -120.9868698, 92.9311295, -120.1974640, 92.3212814, -213.3081512, 213.1285400
5: -108.8516998, 85.2263260, -108.1543961, 84.6765366, -193.5282288, 193.3806915
6: -103.8736191, 99.7379532, -103.1875687, 99.0504608, -202.9240570, 202.9255219
7: -113.2460403, 95.7331543, -112.4688187, 95.0768356, -208.3228760, 208.2019348
8: -135.6550293, 92.7286301, -134.8107605, 92.1614151, -227.8164062, 227.5393524
9: -103.3043289, 101.8814468, -102.5658798, 101.1833191, -204.4876404, 204.4473267

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0058584, upper bound: 242.0081104
time: 6.79 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0058584, upper bound: 242.0161018
time: 7.81 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -116.4001999, 92.7732773, -119.9105988, 95.5625839, -211.9627533, 212.6838684
1: -97.2215118, 82.2316818, -100.1470718, 84.6550293, -181.8765411, 182.3787537
2: -128.0802307, 83.5373688, -131.9106750, 86.0364151, -214.1166382, 215.4480438
3: -136.3782196, 72.4749680, -140.4179230, 74.6174088, -210.9956360, 212.8928528
4: -125.0280228, 96.0387878, -128.7778625, 98.9234467, -223.9514771, 224.8166046
5: -112.4646454, 88.0521622, -115.8332901, 90.6909180, -203.1555634, 203.8854370
6: -107.3558884, 103.0788345, -110.5902023, 106.1421432, -213.4980316, 213.6690369
7: -117.0479889, 98.9223404, -120.5437164, 101.8562393, -218.9042206, 219.4660492
8: -140.1947174, 95.8313446, -144.4513702, 98.7412338, -238.9359436, 240.2827148
9: -106.7692184, 105.3013611, -109.9310684, 108.4529266, -215.2221375, 215.2323914

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0099872, upper bound: 242.0125492
time: 6.52 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0099872, upper bound: 242.0186776
time: 7.17 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -109.7468109, 87.4839935, -112.6171646, 89.7754440, -199.5222321, 200.1011505
1: -91.5857315, 77.5246658, -94.0151520, 79.5290222, -171.1147461, 171.5398254
2: -120.7421722, 78.7073212, -123.8703842, 80.8204727, -201.5626068, 202.5776825
3: -128.6688232, 68.2125854, -131.8877563, 70.0827560, -198.7515717, 200.1003265
4: -117.8866043, 90.4888306, -120.9601974, 92.9099731, -210.7965698, 211.4490204
5: -106.0573044, 82.9739075, -108.8213348, 85.2099457, -191.2672272, 191.7952271
6: -101.2080841, 97.2142487, -103.8833008, 99.7081070, -200.9161987, 201.0975494
7: -110.3572235, 93.3150787, -113.2327805, 95.7327271, -206.0899048, 206.5478516
8: -132.0536804, 90.1369934, -135.6281738, 92.6851654, -224.7388458, 225.7651367
9: -100.6824036, 99.2240601, -103.2885513, 101.8811111, -202.5634918, 202.5126038

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0079949, upper bound: 242.0099088
time: 6.41 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0079949, upper bound: 242.0156868
time: 5.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.06 seconds
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 13.06
Output dim: 7, lower bound: -242.0058584, upper bound: 242.0081104
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.06
Output dim: 7, lower bound: -242.0058584, upper bound: 242.0161018
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 13.06
Output dim: 7, lower bound: -242.0099872, upper bound: 242.0125492
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.06
Output dim: 7, lower bound: -242.0099872, upper bound: 242.0186776
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 13.06
Output dim: 7, lower bound: -242.0079949, upper bound: 242.0099088
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.06
Output dim: 7, lower bound: -242.0079949, upper bound: 242.0156868

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -112.6317139, 89.7925034, -107.5060654, 85.7508545, -198.3825684, 197.2985382
1: -94.0531082, 79.5544891, -89.7403564, 75.8904114, -169.9435120, 169.2948303
2: -123.9218903, 80.8288345, -118.2596817, 77.1554565, -201.0773468, 199.0885162
3: -131.9329987, 70.0943756, -125.8651352, 66.8420029, -198.7749939, 195.9594879
4: -120.9868698, 92.9311295, -115.4971466, 88.7138519, -209.7007141, 208.4282227
5: -108.8516998, 85.2263260, -103.9413300, 81.4007568, -190.2524414, 189.1676178
6: -103.8736191, 99.7379532, -99.1404343, 95.1850281, -199.0586395, 198.8783875
7: -113.2460403, 95.7331543, -108.0864563, 91.3960953, -204.6421356, 203.8195648
8: -135.6550293, 92.7286301, -129.5045929, 88.5489120, -224.2039032, 222.2332001
9: -103.3043289, 101.8814468, -98.5942459, 97.2404938, -200.5448151, 200.4756622

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -241.9986712, upper bound: 242.0102296
time: 7.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0047985, upper bound: 242.0143407
time: 7.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -116.4001999, 92.7732773, -115.3595352, 91.9651337, -208.3653259, 208.1328125
1: -97.2215118, 82.2316818, -96.3469696, 81.4689941, -178.6905060, 178.5786438
2: -128.0802307, 83.5373688, -126.9190979, 82.7938156, -210.8740387, 210.4564667
3: -136.3782196, 72.4749680, -135.1389160, 71.8030090, -208.1812134, 207.6138458
4: -125.0280228, 96.0387878, -123.9117508, 95.1867371, -220.2147522, 219.9505310
5: -112.4646454, 88.0521622, -111.4708710, 87.2975616, -199.7622070, 199.5230408
6: -107.3558884, 103.0788345, -106.3994598, 102.1400833, -209.4959717, 209.4783020
7: -117.0479889, 98.9223404, -116.0046310, 98.0456085, -215.0935669, 214.9269562
8: -140.1947174, 95.8313446, -138.9573975, 95.0015106, -235.1962280, 234.7887421
9: -106.7692184, 105.3013611, -105.8180695, 104.3705444, -211.1397552, 211.1194153

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0077560, upper bound: 242.0170399
time: 7.19 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0079784, upper bound: 242.0107248
time: 7.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.7468109, 87.4839935, -108.0706177, 86.1807632, -195.9275818, 195.5545959
1: -91.5857315, 77.5246658, -90.2190781, 76.3462067, -167.9319458, 167.7437286
2: -120.7421722, 78.7073212, -118.8841858, 77.5800171, -198.3221741, 197.5914612
3: -128.6688232, 68.2125854, -126.6150970, 67.2710190, -195.9398346, 194.8276367
4: -117.8866043, 90.4888306, -116.0982513, 89.1773224, -207.0639343, 206.5870667
5: -106.0573044, 82.9739075, -104.4628983, 81.8200150, -187.8772736, 187.4367828
6: -101.2080841, 97.2142487, -99.6966171, 95.7097321, -196.9178162, 196.9108582
7: -110.3572235, 93.3150787, -108.6983109, 91.9250412, -202.2822571, 202.0133972
8: -132.0536804, 90.1369934, -130.1392212, 88.9483871, -221.0020599, 220.2762146
9: -100.6824036, 99.2240601, -99.1791306, 97.8020706, -198.4844360, 198.4031982

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0057152, upper bound: 242.0139686
time: 8.20 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0060491, upper bound: 242.0139705
time: 7.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.87 seconds
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.87
Output dim: 7, lower bound: -241.9986712, upper bound: 242.0102296
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 7, lower bound: -242.0047985, upper bound: 242.0143407
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 7, lower bound: -242.0077560, upper bound: 242.0170399
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 16.87
Output dim: 7, lower bound: -242.0079784, upper bound: 242.0107248
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 7, lower bound: -242.0057152, upper bound: 242.0139686
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.87
Output dim: 7, lower bound: -242.0060491, upper bound: 242.0139705

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -109.4246674, 87.2390289, -106.7949753, 85.1846161, -194.6092834, 194.0339813
1: -91.3577957, 77.2978516, -89.1429443, 75.3903046, -166.7480774, 166.4407654
2: -120.3998413, 78.5209579, -117.4792404, 76.6440277, -197.0438690, 196.0001984
3: -128.1707611, 68.1226959, -125.0311050, 66.4052200, -194.5759888, 193.1537781
4: -117.5351944, 90.2729492, -114.7315140, 88.1246109, -205.6597900, 205.0044556
5: -105.7436600, 82.7694397, -103.2517929, 80.8555679, -186.5992279, 186.0212097
6: -100.9250870, 96.9086990, -98.4865646, 94.5577240, -195.4827881, 195.3952637
7: -110.0096817, 93.0249710, -107.3696594, 90.7957458, -200.8054199, 200.3946075
8: -131.7859497, 90.0823212, -128.6465607, 87.9619064, -219.7478485, 218.7288666
9: -100.3590698, 98.9815216, -97.9405975, 96.5971527, -196.9562225, 196.9221191

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0134464, upper bound: 242.0143407
time: 6.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0134464, upper bound: 242.0143407
time: 6.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -110.8914566, 88.3855743, -113.4082947, 90.4109879, -201.3024445, 201.7938690
1: -92.6225204, 78.3991241, -94.7179413, 80.1115417, -172.7340698, 173.1170502
2: -122.0536652, 79.6112976, -124.7846069, 81.4035873, -203.4572449, 204.3959045
3: -129.9602661, 69.1200790, -132.8650665, 70.6148758, -200.5751343, 201.9851379
4: -119.1002045, 91.4949570, -121.8121414, 93.5769501, -212.6771545, 213.3070984
5: -107.1336288, 83.9197235, -109.5824738, 85.8333130, -192.9669495, 193.5021820
6: -102.2822418, 98.2421341, -104.6019211, 100.4272079, -202.7094421, 202.8440552
7: -111.5587692, 94.3191376, -114.0605545, 96.4152527, -207.9740143, 208.3796539
8: -133.5525665, 91.2169571, -136.6047516, 93.3671875, -226.9197083, 227.8217163
9: -101.7485657, 100.3306122, -104.0399551, 102.6096573, -204.3582153, 204.3705750

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0161770, upper bound: 242.0170399
time: 6.96 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0161770, upper bound: 242.0170399
time: 6.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -104.1163330, 82.9967804, -106.1006927, 84.6114883, -188.7278137, 189.0974426
1: -86.8811340, 73.6073151, -88.5740891, 74.9760132, -161.8571320, 162.1813812
2: -114.5823669, 74.6949158, -116.7296143, 76.1762619, -190.7586212, 191.4245300
3: -122.1126480, 64.7838058, -124.3205261, 66.0713501, -188.1839600, 189.1043396
4: -111.8268356, 85.8400192, -113.9789047, 87.5519867, -199.3788147, 199.8189240
5: -100.6045837, 78.7447815, -102.5560455, 80.3412781, -180.9458618, 181.3008270
6: -96.0203094, 92.2697449, -97.8822403, 93.9805450, -190.0008545, 190.1519775
7: -104.7506485, 88.6106873, -106.7363129, 90.2795868, -195.0302429, 195.3470001
8: -125.2603302, 85.4128571, -127.7635422, 87.2971039, -212.5574341, 213.1763916
9: -95.5505829, 94.1382599, -97.3848343, 96.0238876, -191.5744171, 191.5231018

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0080907, upper bound: 242.0079940
time: 6.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0119932, upper bound: 242.0121047
time: 6.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -107.3399200, 85.5641556, -105.4149246, 84.0684204, -191.4083405, 190.9790802
1: -89.6200104, 75.9007950, -88.0052338, 74.5039825, -164.1239929, 163.9060364
2: -118.1678085, 77.0085678, -115.9869690, 75.6924438, -193.8602142, 192.9955444
3: -125.9498520, 66.7617111, -123.5260010, 65.6534805, -191.6033173, 190.2876892
4: -115.3044662, 88.4950256, -113.2450943, 86.9893875, -202.2938080, 201.7401123
5: -103.7077103, 81.1474762, -101.8932114, 79.8286591, -183.5363770, 183.0406799
6: -99.0125122, 95.1448441, -97.2550583, 93.3839264, -192.3964386, 192.3998871
7: -108.0146484, 91.3424530, -106.0623550, 89.7125626, -197.7272034, 197.4047546
8: -129.1411438, 88.0428391, -126.9342270, 86.7226944, -215.8638306, 214.9770355
9: -98.5043869, 97.0678558, -96.7651367, 95.4124146, -193.9167938, 193.8329926

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0081648, upper bound: 242.0079975
time: 7.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0121435, upper bound: 242.0121555
time: 5.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.66 seconds
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0134464, upper bound: 242.0143407
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0134464, upper bound: 242.0143407
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0161770, upper bound: 242.0170399
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0161770, upper bound: 242.0170399
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0080907, upper bound: 242.0079940
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0119932, upper bound: 242.0121047
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0081648, upper bound: 242.0079975
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0121435, upper bound: 242.0121555

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -109.4246674, 87.2390289, -101.3747635, 80.8861237, -190.3107910, 188.6137695
1: -91.3577957, 77.2978516, -84.5881424, 71.5784912, -162.9362793, 161.8859711
2: -120.3998413, 78.5209579, -111.5041428, 72.7669678, -193.1668091, 190.0251007
3: -128.1707611, 68.1226959, -118.6880798, 63.0342064, -191.2049713, 186.8107758
4: -117.5351944, 90.2729492, -108.9191971, 83.6531601, -201.1883240, 199.1921387
5: -105.7436600, 82.7694397, -98.0396118, 76.7804337, -182.5240936, 180.8090363
6: -100.9250870, 96.9086990, -93.4963150, 89.7753448, -190.7004395, 190.4050140
7: -110.0096817, 93.0249710, -101.9330215, 86.2427597, -196.2524261, 194.9579926
8: -131.7859497, 90.0823212, -122.0900421, 83.4635696, -215.2495117, 212.1723480
9: -100.3590698, 98.9815216, -93.0006027, 91.7110977, -192.0701599, 191.9821167

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0113192, upper bound: 242.0122294
time: 7.24 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0113224, upper bound: 242.0122182
time: 7.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -109.4246674, 87.2390289, -94.6759415, 75.5497055, -184.9743652, 181.9149628
1: -91.3577957, 77.2978516, -78.8850632, 66.8277969, -158.1855927, 156.1828918
2: -120.3998413, 78.5209579, -104.0986557, 67.9115448, -188.3113861, 182.6196136
3: -128.1707611, 68.1226959, -110.9168015, 58.7372360, -186.9079895, 179.0394745
4: -117.5351944, 90.2729492, -101.7178650, 78.0561447, -195.5913239, 191.9908142
5: -105.7436600, 82.7694397, -91.5827942, 71.6550598, -177.3987122, 174.3522339
6: -100.9250870, 96.9086990, -87.3073044, 83.8580551, -184.7831421, 184.2160034
7: -110.0096817, 93.0249710, -95.1862717, 80.6005402, -190.6102142, 188.2112274
8: -131.7859497, 90.0823212, -113.8844528, 77.7267914, -209.5127258, 203.9667511
9: -100.3590698, 98.9815216, -86.8758545, 85.5746307, -185.9337006, 185.8573761

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0113192, upper bound: 242.0122294
time: 7.19 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0113224, upper bound: 242.0122182
time: 7.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -110.8914566, 88.3855743, -107.9871368, 86.1105804, -197.0020447, 196.3726959
1: -92.6225204, 78.3991241, -90.1598358, 76.2969208, -168.9194336, 168.5589294
2: -122.0536652, 79.6112976, -118.8055420, 77.5227966, -199.5764465, 198.4168243
3: -129.9602661, 69.1200790, -126.5173416, 67.2404022, -197.2006531, 195.6374207
4: -119.1002045, 91.4949570, -115.9962921, 89.1030884, -208.2032928, 207.4912415
5: -107.1336288, 83.9197235, -104.3685837, 81.7555695, -188.8891907, 188.2882843
6: -102.2822418, 98.2421341, -99.6073990, 95.6417542, -197.9239960, 197.8495331
7: -111.5587692, 94.3191376, -108.6208496, 91.8596878, -203.4184418, 202.9399567
8: -133.5525665, 91.2169571, -130.0428925, 88.8650665, -222.4176178, 221.2598572
9: -101.7485657, 100.3306122, -99.0988693, 97.7201691, -199.4687195, 199.4294739

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0104744, upper bound: 242.0116039
time: 7.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0142086, upper bound: 242.0152951
time: 6.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -110.8914566, 88.3855743, -101.3549728, 80.8375320, -191.7289734, 189.7405396
1: -92.6225204, 78.3991241, -84.5403442, 71.6042480, -164.2267761, 162.9394531
2: -122.0536652, 79.6112976, -111.4894638, 72.7069016, -194.7605438, 191.1007233
3: -129.9602661, 69.1200790, -118.8314819, 62.9899368, -192.9501953, 187.9515533
4: -119.1002045, 91.4949570, -108.8749542, 83.5674896, -202.6676941, 200.3699036
5: -107.1336288, 83.9197235, -97.9810638, 76.6901169, -183.8237305, 181.9007874
6: -102.2822418, 98.2421341, -93.4761047, 89.7962723, -192.0785065, 191.7182312
7: -111.5587692, 94.3191376, -101.9510193, 86.2690430, -197.8278046, 196.2701263
8: -133.5525665, 91.2169571, -121.9216690, 83.1846466, -216.7371979, 213.1386261
9: -101.7485657, 100.3306122, -93.0278625, 91.6559448, -193.4045105, 193.3584747

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0104744, upper bound: 242.0116039
time: 8.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0142086, upper bound: 242.0152951
time: 7.45 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 16.62 seconds
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.62
Output dim: 7, lower bound: -242.0113192, upper bound: 242.0122294
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 16.62
Output dim: 7, lower bound: -242.0113224, upper bound: 242.0122182
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.62
Output dim: 7, lower bound: -242.0113192, upper bound: 242.0122294
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 16.62
Output dim: 7, lower bound: -242.0113224, upper bound: 242.0122182
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.62
Output dim: 7, lower bound: -242.0104744, upper bound: 242.0116039
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.62
Output dim: 7, lower bound: -242.0142086, upper bound: 242.0152951
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.62
Output dim: 7, lower bound: -242.0104744, upper bound: 242.0116039
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.62
Output dim: 7, lower bound: -242.0142086, upper bound: 242.0152951

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -107.6993942, 85.8428955, -107.2646637, 85.5350723, -193.2344513, 193.1075439
1: -89.9398422, 76.1530533, -89.5527344, 75.7886047, -165.7284088, 165.7057800
2: -118.5475540, 77.3152618, -118.0120468, 77.0027542, -195.5503082, 195.3272705
3: -126.2157516, 67.1588287, -125.6700974, 66.7962570, -193.0120087, 192.8288879
4: -115.6636963, 88.8485031, -115.2186813, 88.5042877, -204.1679535, 204.0671844
5: -104.0393372, 81.4740372, -103.6683655, 81.2022705, -185.2416077, 185.1423950
6: -99.3477402, 95.4252090, -98.9431686, 95.0045547, -194.3522949, 194.3683624
7: -108.3381729, 91.6239548, -107.8916245, 91.2497025, -199.5878754, 199.5155792
8: -129.7026520, 88.5823746, -129.1716614, 88.2689362, -217.9715881, 217.7540283
9: -98.8167572, 97.4431839, -98.4355392, 97.0668030, -195.8835449, 195.8786774

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0232532, upper bound: 242.0227658
time: 8.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0213703, upper bound: 242.0215607
time: 8.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -107.6993942, 85.8428955, -100.6619949, 80.2856674, -187.9850464, 186.5048676
1: -89.9398422, 76.1530533, -83.9586105, 71.1169891, -161.0567932, 160.1116638
2: -118.5475540, 77.3152618, -110.7284927, 72.2086945, -190.7562408, 188.0437012
3: -126.2157516, 67.1588287, -118.0188828, 62.5644760, -188.7802277, 185.1777039
4: -115.6636963, 88.8485031, -108.1287460, 82.9931412, -198.6568146, 196.9772491
5: -104.0393372, 81.4740372, -97.3092957, 76.1591110, -180.1984558, 178.7833252
6: -99.3477402, 95.4252090, -92.8395691, 89.1848145, -188.5325623, 188.2647705
7: -108.3381729, 91.6239548, -101.2520599, 85.6843414, -194.0225220, 192.8760071
8: -129.7026520, 88.5823746, -121.0860901, 82.6131592, -212.3158112, 209.6684418
9: -98.8167572, 97.4431839, -92.3914642, 91.0292664, -189.8460083, 189.8346252

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0101432, upper bound: 242.0108270
time: 7.51 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0101432, upper bound: 242.0152951
time: 6.35 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 14.99 seconds
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 14.99
Output dim: 7, lower bound: -242.0232532, upper bound: 242.0227658
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 14.99
Output dim: 7, lower bound: -242.0213703, upper bound: 242.0215607
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 14.99
Output dim: 7, lower bound: -242.0101432, upper bound: 242.0108270
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 14.99
Output dim: 7, lower bound: -242.0101432, upper bound: 242.0152951

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -106.9730530, 85.2668304, -105.0923004, 83.8141708, -190.7872314, 190.3591309
1: -89.3310852, 75.6446686, -87.7336121, 74.2685928, -163.5996704, 163.3782654
2: -117.7478333, 76.8011932, -115.6242065, 75.4701843, -193.2180023, 192.4253998
3: -125.3620148, 66.7163849, -123.1210632, 65.4722137, -190.8342285, 189.8374481
4: -114.8797760, 88.2534103, -112.8768082, 86.7252808, -201.6050568, 201.1301727
5: -103.3393478, 80.9337158, -101.5780869, 79.5865402, -182.9258881, 182.5117798
6: -98.6805038, 94.7814636, -96.9511795, 93.0804749, -191.7609863, 191.7326355
7: -107.6084595, 91.0131683, -105.7133179, 89.4248123, -197.0332642, 196.7264862
8: -128.8281250, 87.9896927, -126.5613708, 86.4965210, -215.3246460, 214.5510559
9: -98.1554565, 96.7923355, -96.4575272, 95.1221619, -193.2776184, 193.2498627

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0165693, upper bound: 242.0167478
time: 8.24 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0165693, upper bound: 242.0227658
time: 7.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -106.4190292, 84.8264923, -129.0928802, 102.8030319, -209.2220459, 213.9193726
1: -88.8644867, 75.2562561, -107.9111557, 91.1542053, -180.0186920, 183.1674042
2: -117.1365280, 76.4113998, -142.1179352, 92.4498520, -209.5863800, 218.5293274
3: -124.7116776, 66.3806992, -151.3168488, 80.4600601, -205.1717377, 217.6975403
4: -114.2826996, 87.7987595, -138.6234436, 106.4111557, -220.6938477, 226.4221954
5: -102.8045883, 80.5227432, -124.6154175, 97.4290848, -200.2336731, 205.1381531
6: -98.1737900, 94.2887421, -118.9459229, 114.3369370, -212.5107269, 213.2346497
7: -107.0548553, 90.5473938, -129.6803894, 109.4261780, -216.4810333, 220.2277527
8: -128.1580963, 87.5366440, -155.4717560, 106.3727951, -234.5308838, 243.0083923
9: -97.6506577, 96.2938690, -118.2090683, 116.6316681, -214.2823181, 214.5029297

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0144457, upper bound: 242.0156824
time: 6.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0144457, upper bound: 242.0215607
time: 6.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -107.6993942, 85.8428955, -98.3062363, 78.4096909, -186.1090698, 184.1491089
1: -89.9398422, 76.1530533, -81.9809265, 69.4607849, -159.4006348, 158.1339722
2: -118.5475540, 77.3152618, -108.1419678, 70.5143890, -189.0619507, 185.4572144
3: -126.2157516, 67.1588287, -115.2574692, 61.1179810, -187.3337402, 182.4162903
4: -115.6636963, 88.8485031, -105.5929947, 81.0403748, -196.7040405, 194.4414978
5: -104.0393372, 81.4740372, -95.0255661, 74.3539124, -178.3932495, 176.4996033
6: -99.3477402, 95.4252090, -90.6762619, 87.1060333, -186.4537659, 186.1014404
7: -108.3381729, 91.6239548, -98.8764114, 83.6965485, -192.0347137, 190.5003662
8: -129.7026520, 88.5823746, -118.2449112, 80.6700745, -210.3727264, 206.8272858
9: -98.8167572, 97.4431839, -90.2282791, 88.8983154, -187.7150574, 187.6714325

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0054660, upper bound: 242.0125010
time: 7.23 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0054660, upper bound: 242.0152922
time: 8.48 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 16.85 seconds
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 16.85
Output dim: 7, lower bound: -242.0165693, upper bound: 242.0167478
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 16.85
Output dim: 7, lower bound: -242.0165693, upper bound: 242.0227658
IS_A2_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 16.85
Output dim: 7, lower bound: -242.0144457, upper bound: 242.0156824
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 16.85
Output dim: 7, lower bound: -242.0144457, upper bound: 242.0215607
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 16.85
Output dim: 7, lower bound: -242.0054660, upper bound: 242.0125010
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 16.85
Output dim: 7, lower bound: -242.0054660, upper bound: 242.0152922

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -92.6542664, 73.9449615, -105.0923004, 83.8141708, -176.4684448, 179.0372620
1: -77.2970123, 65.4896393, -87.7336121, 74.2685928, -151.5656128, 153.2232513
2: -101.9489136, 66.5450211, -115.6242065, 75.4701843, -177.4190979, 182.1692200
3: -108.5038605, 57.7151031, -123.1210632, 65.4722137, -173.9760742, 180.8361664
4: -99.5294952, 76.4543762, -112.8768082, 86.7252808, -186.2547760, 189.3311615
5: -89.5998688, 70.2022705, -101.5780869, 79.5865402, -169.1864014, 171.7803650
6: -85.4728622, 82.1000061, -96.9511795, 93.0804749, -178.5533447, 179.0511780
7: -93.2131653, 78.9356995, -105.7133179, 89.4248123, -182.6379700, 184.6490173
8: -111.5750046, 76.2061615, -126.5613708, 86.4965210, -198.0715332, 202.7675171
9: -85.0346680, 83.8402100, -96.4575272, 95.1221619, -180.1568298, 180.2977295

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0144185, upper bound: 242.0143396
time: 6.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0147837, upper bound: 242.0148737
time: 8.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -100.5790863, 80.2123947, -105.0923004, 83.8141708, -184.3932495, 185.3046875
1: -83.9562912, 71.1150131, -87.7336121, 74.2685928, -158.2248840, 158.8486176
2: -110.6830902, 72.2267456, -115.6242065, 75.4701843, -186.1532593, 187.8509521
3: -117.8537598, 62.7124825, -123.1210632, 65.4722137, -183.3259735, 185.8335419
4: -108.0200348, 82.9830627, -112.8768082, 86.7252808, -194.7453156, 195.8598480
5: -97.1972885, 76.1527557, -101.5780869, 79.5865402, -176.7838135, 177.7308350
6: -92.7913513, 89.1170807, -96.9511795, 93.0804749, -185.8718262, 186.0682526
7: -101.1945190, 85.6394958, -105.7133179, 89.4248123, -190.6193237, 191.3528137
8: -121.1113510, 82.7132187, -126.5613708, 86.4965210, -207.6078796, 209.2745972
9: -92.3245926, 91.0320129, -96.4575272, 95.1221619, -187.4467468, 187.4895325

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0144185, upper bound: 242.0194976
time: 8.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0147837, upper bound: 242.0198145
time: 8.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -92.1235428, 73.5232925, -129.0928802, 102.8030319, -194.9265747, 202.6161652
1: -76.8501587, 65.1176224, -107.9111557, 91.1542053, -168.0043488, 173.0287781
2: -101.3632736, 66.1720963, -142.1179352, 92.4498520, -193.8131256, 208.2900238
3: -107.8813477, 57.3940315, -151.3168488, 80.4600601, -188.3414001, 208.7108459
4: -98.9576263, 76.0187454, -138.6234436, 106.4111557, -205.3687592, 214.6421661
5: -89.0876846, 69.8087616, -124.6154175, 97.4290848, -186.5167694, 194.4241791
6: -84.9875717, 81.6281433, -118.9459229, 114.3369370, -199.3244934, 200.5740662
7: -92.6834335, 78.4898529, -129.6803894, 109.4261780, -202.1096191, 208.1701965
8: -110.9328766, 75.7726135, -155.4717560, 106.3727951, -217.3056641, 231.2443695
9: -84.5511322, 83.3628006, -118.2090683, 116.6316681, -201.1828003, 201.5718689

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0119435, upper bound: 242.0131375
time: 7.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0126900, upper bound: 242.0137949
time: 8.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -100.0246353, 79.7717514, -129.0928802, 102.8030319, -202.8276672, 208.8645935
1: -83.4893417, 70.7263641, -107.9111557, 91.1542053, -174.6435547, 178.6375122
2: -110.0713577, 71.8367157, -142.1179352, 92.4498520, -202.5212097, 213.9546509
3: -117.2029648, 62.3766594, -151.3168488, 80.4600601, -197.6630249, 213.6934967
4: -107.4224548, 82.5280914, -138.6234436, 106.4111557, -213.8336029, 221.1515350
5: -96.6620865, 75.7415314, -124.6154175, 97.4290848, -194.0911713, 200.3569336
6: -92.2843399, 88.6239548, -118.9459229, 114.3369370, -206.6212769, 207.5698853
7: -100.6406174, 85.1733932, -129.6803894, 109.4261780, -210.0668030, 214.8537750
8: -120.4407196, 82.2598038, -155.4717560, 106.3727951, -226.8135071, 237.7315521
9: -91.8194580, 90.5332336, -118.2090683, 116.6316681, -208.4511108, 208.7423096

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 184

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0119435, upper bound: 242.0183379
time: 7.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0126900, upper bound: 242.0187180
time: 8.09 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -101.3054962, 80.7884216, -98.3062363, 78.4096909, -179.7151489, 179.0946350
1: -84.5650253, 71.6233368, -81.9809265, 69.4607849, -154.0258179, 153.6042633
2: -111.4828186, 72.7407379, -108.1419678, 70.5143890, -181.9972076, 180.8827057
3: -118.7074661, 63.1548271, -115.2574692, 61.1179810, -179.8254242, 178.4122925
4: -108.8040619, 83.5781555, -105.5929947, 81.0403748, -189.8444366, 189.1711426
5: -97.8973465, 76.6930389, -95.0255661, 74.3539124, -172.2512207, 171.7185974
6: -93.4586334, 89.7608032, -90.6762619, 87.1060333, -180.5646667, 180.4370270
7: -101.9241943, 86.2502975, -98.8764114, 83.6965485, -185.6207275, 185.1267090
8: -121.9860229, 83.3058548, -118.2449112, 80.6700745, -202.6560974, 201.5507660
9: -92.9858856, 91.6827850, -90.2282791, 88.8983154, -181.8841705, 181.9110565

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0026182, upper bound: 242.0084652
time: 7.47 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0103681, upper bound: 242.0149650
time: 6.87 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 15.50 seconds
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0144185, upper bound: 242.0143396
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0147837, upper bound: 242.0148737
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0144185, upper bound: 242.0194976
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0147837, upper bound: 242.0198145
IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0119435, upper bound: 242.0131375
IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0126900, upper bound: 242.0137949
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0119435, upper bound: 242.0183379
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0126900, upper bound: 242.0187180
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0026182, upper bound: 242.0084652
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 15.50
Output dim: 7, lower bound: -242.0103681, upper bound: 242.0149650

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -89.9523010, 71.8074341, -96.6110992, 77.1092224, -167.0614929, 168.4185181
1: -75.0437851, 63.5982094, -80.6569443, 68.3309479, -143.3747253, 144.2551422
2: -98.9907074, 64.6454315, -106.3385849, 69.5027237, -168.4934082, 170.9839935
3: -105.3453827, 56.0757370, -113.1983795, 60.3080215, -165.6534119, 169.2741089
4: -96.5897446, 74.2349396, -103.6825409, 79.7774506, -176.3671875, 177.9174805
5: -87.0103531, 68.1793823, -93.4394608, 73.2515259, -160.2618713, 161.6188354
6: -82.9773254, 79.7260056, -89.1304169, 85.6299133, -168.6072388, 168.8564148
7: -90.5143433, 76.6820984, -97.2431030, 82.3444672, -172.8588104, 173.9251862
8: -108.3167648, 74.0105591, -116.3552704, 79.6169815, -187.9337463, 190.3657990
9: -82.5864868, 81.4185791, -88.7763596, 87.5434570, -170.1299438, 170.1949463

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0136588, upper bound: 242.0128962
time: 8.14 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0150358, upper bound: 242.0140966
time: 8.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -90.5963211, 72.3198929, -100.0277405, 79.8145752, -170.4108582, 172.3476257
1: -75.5777435, 64.0485535, -83.5024414, 70.7236938, -146.3014374, 147.5509949
2: -99.6973038, 65.1008606, -110.0847244, 71.9150391, -171.6123352, 175.1855774
3: -106.1041641, 56.4707336, -117.2182007, 62.4116173, -168.5157776, 173.6889343
4: -97.2949600, 74.7628860, -107.3803101, 82.5639496, -179.8589020, 182.1431885
5: -87.6313019, 68.6686707, -96.7320480, 75.8115692, -163.4428558, 165.4007111
6: -83.5731735, 80.2934723, -92.2774277, 88.6361313, -172.2093048, 172.5708923
7: -91.1589279, 77.2237778, -100.6573563, 85.2076263, -176.3665314, 177.8811340
8: -109.0908737, 74.5336761, -120.4514236, 82.3827362, -191.4735870, 194.9851074
9: -83.1740036, 81.9934692, -91.8792877, 90.5820923, -173.7560425, 173.8727570

Time for backsubstitution: 1.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0140822, upper bound: 242.0134030
time: 9.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0154100, upper bound: 242.0145704
time: 7.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -97.8513336, 78.0552139, -96.6110992, 77.1092224, -174.9605408, 174.6663208
1: -81.6815872, 69.2058334, -80.6569443, 68.3309479, -150.0125122, 149.8627777
2: -107.6971512, 70.3092270, -106.3385849, 69.5027237, -177.1998596, 176.6478119
3: -114.6660385, 61.0582275, -113.1983795, 60.3080215, -174.9740448, 174.2566071
4: -105.0524750, 80.7425690, -103.6825409, 79.7774506, -184.8299255, 184.4251099
5: -94.5834351, 74.1119614, -93.4394608, 73.2515259, -167.8349609, 167.5514221
6: -90.2717896, 86.7206955, -89.1304169, 85.6299133, -175.9017029, 175.8511047
7: -98.4705505, 83.3641052, -97.2431030, 82.3444672, -180.8150177, 180.6072083
8: -117.8212204, 80.4977036, -116.3552704, 79.6169815, -197.4382019, 196.8529663
9: -89.8524094, 88.5870895, -88.7763596, 87.5434570, -177.3958740, 177.3634491

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0189990, upper bound: 242.0186962
time: 8.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0197641, upper bound: 242.0193635
time: 6.31 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -98.5559464, 78.6138916, -100.0277405, 79.8145752, -178.3704681, 178.6415863
1: -82.2660141, 69.6986618, -83.5024414, 70.7236938, -152.9897003, 153.2011108
2: -108.4694748, 70.8063812, -110.0847244, 71.9150391, -180.3845215, 180.8911133
3: -115.4944992, 61.4895020, -117.2182007, 62.4116173, -177.9061127, 178.7077026
4: -105.8227386, 81.3203354, -107.3803101, 82.5639496, -188.3866577, 188.7006531
5: -95.2608414, 74.6435318, -96.7320480, 75.8115692, -171.0724030, 171.3755646
6: -90.9232559, 87.3410492, -92.2774277, 88.6361313, -179.5593719, 179.6184692
7: -99.1744995, 83.9545670, -100.6573563, 85.2076263, -184.3821106, 184.6119232
8: -118.6700134, 81.0693054, -120.4514236, 82.3827362, -201.0527496, 201.5207214
9: -90.4950180, 89.2173462, -91.8792877, 90.5820923, -181.0770721, 181.0966187

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0192163, upper bound: 242.0188914
time: 7.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0200088, upper bound: 242.0196233
time: 7.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -90.0661545, 71.8986816, -123.9146271, 98.7122345, -188.7783661, 195.8133087
1: -75.1312790, 63.6768799, -103.5825577, 87.5273819, -162.6586609, 167.2594299
2: -99.1122437, 64.7283249, -136.4549866, 88.8098373, -187.9220886, 201.1832886
3: -105.4823227, 56.1499710, -145.2823486, 77.3264008, -182.8087158, 201.4323120
4: -96.7236557, 74.3277206, -133.0028076, 102.1557312, -198.8793945, 207.3305359
5: -87.1196823, 68.2755890, -119.6595001, 93.5669327, -180.6865997, 187.9350891
6: -83.0884094, 79.8220749, -114.1650162, 109.7921753, -192.8805847, 193.9870758
7: -90.6297379, 76.7783966, -124.5109024, 105.1092987, -195.7390442, 201.2893066
8: -108.4494247, 74.1006393, -149.2248383, 102.1681671, -210.6175842, 223.3254547
9: -82.6910095, 81.5165634, -113.5281601, 111.9897919, -194.6807861, 195.0447235

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0125308, upper bound: 242.0124111
time: 8.14 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0138912, upper bound: 242.0134749
time: 9.15 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -97.2938538, 77.6121979, -120.6122208, 96.0959473, -193.3898010, 198.2244110
1: -81.2121582, 68.8151169, -100.8301163, 85.2131271, -166.4252625, 169.6452332
2: -107.0820389, 69.9171371, -132.8300781, 86.4736252, -193.5556488, 202.7471924
3: -114.0117569, 60.7206345, -141.3914948, 75.2899780, -189.3017273, 202.1121216
4: -104.4516144, 80.2852020, -129.4390869, 99.4618912, -203.9135132, 209.7242889
5: -94.0454102, 73.6985703, -116.4737625, 91.0919952, -185.1374054, 190.1723328
6: -89.7620850, 86.2248840, -111.1208725, 106.8865509, -196.6486359, 197.3457489
7: -97.9136276, 82.8956985, -121.2092361, 102.3421631, -200.2557831, 204.1049347
8: -117.1469116, 80.0419159, -145.2636108, 99.4877319, -216.6346436, 225.3055267
9: -89.3446198, 88.0856171, -110.5289383, 109.0543518, -198.3989258, 198.6145630

Time for backsubstitution: 1.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0172607, upper bound: 242.0175276
time: 7.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0179754, upper bound: 242.0181724
time: 6.89 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 15.08 seconds
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0136588, upper bound: 242.0128962
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0150358, upper bound: 242.0140966
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0140822, upper bound: 242.0134030
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0154100, upper bound: 242.0145704
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0189990, upper bound: 242.0186962
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0197641, upper bound: 242.0193635
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0192163, upper bound: 242.0188914
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0200088, upper bound: 242.0196233
IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0125308, upper bound: 242.0124111
IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0138912, upper bound: 242.0134749
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0172607, upper bound: 242.0175276
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 15.08
Output dim: 7, lower bound: -242.0179754, upper bound: 242.0181724
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 15.08
Output dim: 7, lower bound: -242.0126900, upper bound: 242.0187180
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 15.08
Output dim: 7, lower bound: -242.0103681, upper bound: 242.0149650
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=243.70181274414062
rel_dist={7: [-242.0414140657913, 242.04141406382126]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0316651, upper bound: 242.0310970
time: 10.10 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0385653, upper bound: 242.0385653
time: 9.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.65
Output dim: 7, lower bound: -242.0316651, upper bound: 242.0310970
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.65
Output dim: 7, lower bound: -242.0385653, upper bound: 242.0385653

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -119.5140991, 95.2666245, -121.4869385, 96.8178482, -216.3319397, 216.7535400
1: -99.9024429, 84.4575806, -101.5364609, 85.8404922, -185.7429352, 185.9940491
2: -131.5668793, 85.7744370, -133.7214355, 87.1900864, -218.7569580, 219.4958649
3: -140.1050415, 74.4418182, -142.3750916, 75.6793365, -215.7843475, 216.8168945
4: -128.4107208, 98.6364822, -130.5062103, 100.2631378, -228.6738586, 229.1426697
5: -115.4722366, 90.4375610, -117.3620682, 91.9045715, -207.3767853, 207.7996063
6: -110.2378006, 105.8749466, -112.0578308, 107.6000824, -217.8378906, 217.9327698
7: -120.2295685, 101.5678558, -122.1876755, 103.2165604, -223.4461365, 223.7555237
8: -143.9831390, 98.4588470, -146.3653259, 100.0880356, -244.0711517, 244.8241730
9: -109.6788559, 108.1721420, -111.4513016, 109.9388428, -219.6177063, 219.6234283

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0246733, upper bound: 242.0243178
time: 7.87 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0285328, upper bound: 242.0279632
time: 8.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.0534821, 99.6287537, -126.8320007, 101.0347672, -226.0882416, 226.4607544
1: -104.4979324, 88.3349228, -105.9834213, 89.5798416, -194.0777740, 194.3182983
2: -137.6306152, 89.7292252, -139.5813751, 90.9968338, -228.6274414, 229.3105927
3: -146.5207214, 77.8730164, -148.5843048, 78.9728699, -225.4935760, 226.4573059
4: -134.3097839, 103.1774673, -136.2120514, 104.6377487, -238.9475098, 239.3895264
5: -120.7826691, 94.5478973, -122.4879456, 95.8739777, -216.6566467, 217.0358429
6: -115.3268051, 110.7257385, -116.9650192, 112.2899170, -227.6167297, 227.6907654
7: -125.7305145, 106.1953125, -127.5044479, 107.6851044, -233.4156189, 233.6997681
8: -150.6563873, 102.9990997, -152.8035736, 104.4611206, -255.1175079, 255.8026733
9: -114.6585846, 113.1035004, -116.2662506, 114.6993561, -229.3579254, 229.3697510

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0315671, upper bound: 242.0316099
time: 7.65 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0346705, upper bound: 242.0346705
time: 7.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.46 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 16.46
Output dim: 7, lower bound: -242.0246733, upper bound: 242.0243178
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 16.46
Output dim: 7, lower bound: -242.0285328, upper bound: 242.0279632
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 16.46
Output dim: 7, lower bound: -242.0315671, upper bound: 242.0316099
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 16.46
Output dim: 7, lower bound: -242.0346705, upper bound: 242.0346705

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -106.4929047, 84.9724045, -103.6279144, 82.6992798, -189.1921844, 188.6002960
1: -88.9564667, 75.2188721, -86.5194931, 73.1768494, -162.1333160, 161.7383728
2: -117.1906738, 76.4404831, -114.0110626, 74.3978043, -191.5884552, 190.4515381
3: -124.7827301, 66.2513199, -121.3603592, 64.4580841, -189.2408142, 187.6116486
4: -114.4491043, 87.8986816, -111.3633194, 85.5427017, -199.9918060, 199.2619781
5: -102.9802780, 80.6840820, -100.2234802, 78.5274582, -181.5077362, 180.9075623
6: -98.2229385, 94.3394775, -95.5841446, 91.7879333, -190.0108643, 189.9236145
7: -107.1319580, 90.5864563, -104.2349319, 88.1596603, -195.2916260, 194.8213806
8: -128.2823639, 87.7304077, -124.8385925, 85.3866959, -213.6690674, 212.5689850
9: -97.7586212, 96.3939209, -95.1086197, 93.7970505, -191.5556641, 191.5025330

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0044945, upper bound: 242.0039922
time: 8.03 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0033136, upper bound: 242.0025335
time: 9.19 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -112.5897675, 89.7947617, -111.9828796, 89.3067017, -201.8964691, 201.7776489
1: -94.0794373, 79.5523911, -93.5430984, 79.1095734, -173.1890106, 173.0954895
2: -123.9137497, 80.8166962, -123.2173309, 80.3873367, -204.3010864, 204.0340271
3: -131.9733887, 70.1065292, -131.2145233, 69.7306900, -201.7040558, 201.3210449
4: -120.9811783, 92.9281540, -120.3086090, 92.4293213, -213.4104767, 213.2367554
5: -108.8219986, 85.2610474, -108.2325439, 84.7996140, -193.6215668, 193.4935608
6: -103.8578415, 99.7410736, -103.3029327, 99.1810150, -203.0388489, 203.0440063
7: -113.2800369, 95.7456741, -112.6515656, 95.2262802, -208.5063019, 208.3972473
8: -135.6247101, 92.7473068, -134.8936005, 92.2481308, -227.8728333, 227.6408997
9: -103.3626709, 101.9319611, -102.7840576, 101.3755264, -204.7381897, 204.7160034

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0086915, upper bound: 242.0082386
time: 8.26 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0075341, upper bound: 242.0066429
time: 8.04 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -112.3167953, 89.5549698, -109.2532120, 87.1327133, -199.4495087, 198.8081665
1: -93.7868195, 79.2962418, -91.1997375, 77.1134872, -170.9002991, 170.4959717
2: -123.5636673, 80.5941391, -120.1765900, 78.4010925, -201.9647522, 200.7707214
3: -131.5243988, 69.8573608, -127.8920746, 67.9227066, -199.4470978, 197.7494202
4: -120.6495667, 92.6736374, -117.3661880, 90.1483383, -210.7978973, 210.0398254
5: -108.5610428, 85.0048370, -105.6165924, 82.7033920, -191.2644196, 190.6214142
6: -103.5730515, 99.4377060, -100.7495117, 96.7222137, -200.2952576, 200.1872253
7: -112.9137573, 95.4484863, -109.8289108, 92.8596115, -205.7733765, 205.2774048
8: -135.2981262, 92.5004349, -131.6146851, 89.9852676, -225.2833862, 224.1151123
9: -102.9928589, 101.5799713, -100.1735916, 98.8082581, -201.8011169, 201.7535553

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0128983, upper bound: 242.0131793
time: 9.24 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0121627, upper bound: 242.0122270
time: 6.74 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -117.9639664, 94.0238113, -117.1689072, 93.3953781, -211.3593140, 211.1927185
1: -98.5368118, 83.3133316, -97.8579178, 82.7356339, -181.2724457, 181.1712494
2: -129.7963715, 84.6573715, -128.9039612, 84.0830002, -213.8793488, 213.5613251
3: -138.1959076, 73.4333649, -137.2377472, 72.9219437, -211.1178589, 210.6711121
4: -126.7058792, 97.3333664, -125.8465576, 96.6724396, -223.3782959, 223.1799316
5: -113.9730225, 89.2447586, -113.2052002, 88.6467667, -202.6197815, 202.4499512
6: -108.7977448, 104.4467621, -108.0653305, 103.7314529, -212.5291901, 212.5120850
7: -118.6174316, 100.2346268, -117.8093643, 99.5607529, -218.1781616, 218.0439758
8: -142.1005859, 97.1500473, -141.1419220, 96.4885635, -238.5891418, 238.2919617
9: -108.1929626, 106.7156219, -107.4533539, 105.9937897, -214.1867523, 214.1689453

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0158037, upper bound: 242.0160495
time: 8.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0149727, upper bound: 242.0149727
time: 6.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 16.54 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 16.54
Output dim: 7, lower bound: -242.0044945, upper bound: 242.0039922
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 16.54
Output dim: 7, lower bound: -242.0033136, upper bound: 242.0025335
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 16.54
Output dim: 7, lower bound: -242.0086915, upper bound: 242.0082386
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 16.54
Output dim: 7, lower bound: -242.0075341, upper bound: 242.0066429
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 16.54
Output dim: 7, lower bound: -242.0128983, upper bound: 242.0131793
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 16.54
Output dim: 7, lower bound: -242.0121627, upper bound: 242.0122270
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.54
Output dim: 7, lower bound: -242.0158037, upper bound: 242.0160495
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.54
Output dim: 7, lower bound: -242.0149727, upper bound: 242.0149727

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -112.5043716, 89.6932983, -114.1618500, 91.0098877, -203.5142517, 203.8550873
1: -93.9468842, 79.4721527, -95.3292389, 80.6200714, -174.5669556, 174.8013763
2: -123.7750854, 80.7500458, -125.5874252, 81.9313126, -205.7063904, 206.3374634
3: -131.8032990, 70.0355988, -133.7168121, 71.0504990, -202.8537903, 203.7524109
4: -120.8487778, 92.8278046, -122.6209564, 94.1903915, -215.0391693, 215.4487610
5: -108.7222672, 85.1380768, -110.3128586, 86.3845596, -195.1068115, 195.4509277
6: -103.7679138, 99.6281052, -105.2949982, 101.0774765, -204.8453827, 204.9230957
7: -113.1395416, 95.6469345, -114.7923660, 97.0341110, -210.1736450, 210.4393005
8: -135.4926605, 92.6167145, -137.5021667, 93.9919052, -229.4845428, 230.1188660
9: -103.2168045, 101.7918243, -104.7130890, 103.2817841, -206.4985657, 206.5049133

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0105583, upper bound: 242.0107481
time: 9.14 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0143546, upper bound: 242.0146008
time: 8.02 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -105.8758316, 84.4239502, -103.3165436, 82.4020386, -188.2778625, 187.7404938
1: -88.3305740, 74.7817230, -86.2100143, 73.0007401, -161.3313141, 160.9917297
2: -116.4636154, 75.9358063, -113.6352997, 74.1771545, -190.6407776, 189.5711060
3: -124.1208725, 65.7868271, -121.0370255, 64.3100662, -188.4309235, 186.8238525
4: -113.7331009, 87.2969589, -110.9967270, 85.2519150, -198.9849854, 198.2936859
5: -102.3390045, 80.0777588, -99.8848648, 78.2383194, -180.5773163, 179.9626160
6: -97.6414719, 93.7856903, -95.3258209, 91.5104294, -189.1519012, 189.1114960
7: -106.4719086, 90.0591278, -103.9262924, 87.9320526, -194.4039612, 193.9854126
8: -127.3786697, 86.9423828, -124.3834076, 84.9831848, -212.3618469, 211.3257599
9: -97.1508255, 95.7344131, -94.8381577, 93.5107117, -190.6615295, 190.5725708

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0133076, upper bound: 242.0132847
time: 9.04 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0133087, upper bound: 242.0133087
time: 7.82 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 17.99 seconds
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 17.99
Output dim: 7, lower bound: -242.0105583, upper bound: 242.0107481
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 17.99
Output dim: 7, lower bound: -242.0143546, upper bound: 242.0146008
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 17.99
Output dim: 7, lower bound: -242.0133076, upper bound: 242.0132847
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 17.99
Output dim: 7, lower bound: -242.0133087, upper bound: 242.0133087

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -110.2877274, 87.9276123, -110.9881058, 88.4816055, -198.7693176, 198.9157104
1: -92.0837555, 77.9128113, -92.6621780, 78.3872681, -170.4710236, 170.5749664
2: -121.3407974, 79.1545639, -122.1017151, 79.6463852, -200.9871826, 201.2562714
3: -129.2035370, 68.6729355, -129.9938202, 69.0995560, -198.3030853, 198.6667480
4: -118.4630051, 90.9906921, -119.2041016, 91.5602646, -210.0232697, 210.1947784
5: -106.5738831, 83.4401703, -107.2364273, 83.9538422, -190.5276947, 190.6765900
6: -101.7300491, 97.6730652, -102.3767319, 98.2777252, -200.0077820, 200.0498047
7: -110.9020233, 93.7754440, -111.5883789, 94.3540039, -205.2560120, 205.3638153
8: -132.8193054, 90.7875900, -133.6742401, 91.3724365, -224.1917267, 224.4618225
9: -101.1814651, 99.7877350, -101.7981262, 100.4118042, -201.5932617, 201.5858612

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0122430, upper bound: 242.0126371
time: 8.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0123377, upper bound: 242.0127087
time: 7.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 17.48 seconds
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 17.48
Output dim: 7, lower bound: -242.0122430, upper bound: 242.0126371
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 17.48
Output dim: 7, lower bound: -242.0123377, upper bound: 242.0127087
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=243.70181274414062
rel_dist={7: [-242.04091628504267, 242.04091628504267]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0327655, upper bound: 242.0317305
time: 7.22 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0388649, upper bound: 242.0388649
time: 8.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.76 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 15.76
Output dim: 7, lower bound: -242.0327655, upper bound: 242.0317305
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.76
Output dim: 7, lower bound: -242.0388649, upper bound: 242.0388649

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -119.5140991, 95.2666245, -123.9901352, 98.7941895, -218.3082886, 219.2567596
1: -99.9024429, 84.4575806, -103.6230469, 87.5920410, -187.4944763, 188.0806274
2: -131.5668793, 85.7744370, -136.4665375, 88.9734268, -220.5403137, 222.2409668
3: -140.1050415, 74.4418182, -145.2816925, 77.2240067, -217.3290405, 219.7235107
4: -128.4107208, 98.6364822, -133.1806946, 102.3147049, -230.7254333, 231.8171387
5: -115.4722366, 90.4375610, -119.7623672, 93.7665024, -209.2387390, 210.1999207
6: -110.2378006, 105.8749466, -114.3592682, 109.7988510, -220.0366516, 220.2341919
7: -120.2295685, 101.5678558, -124.6804352, 105.3106918, -225.5402527, 226.2482910
8: -143.9831390, 98.4588470, -149.3835297, 102.1403503, -246.1234894, 247.8423767
9: -109.6788559, 108.1721420, -113.7094421, 112.1761703, -221.8550262, 221.8815918

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0260469, upper bound: 242.0253965
time: 7.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0294798, upper bound: 242.0285284
time: 8.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -125.0534821, 99.6287537, -128.2992249, 102.1946869, -227.2481384, 227.9279785
1: -104.4979324, 88.3349228, -107.2089386, 90.6068573, -195.1047974, 195.5438232
2: -137.6306152, 89.7292252, -141.1908722, 92.0425873, -229.6732025, 230.9200897
3: -146.5207214, 77.8730164, -150.2870636, 79.8801270, -226.4008484, 228.1600494
4: -134.3097839, 103.1774673, -137.7814484, 105.8423157, -240.1520844, 240.9589233
5: -120.7826691, 94.5478973, -123.8949280, 96.9679947, -217.7506714, 218.4428253
6: -115.3268051, 110.7257385, -118.3164825, 113.5805511, -228.9073486, 229.0422211
7: -125.7305145, 106.1953125, -128.9676666, 108.9141312, -234.6446533, 235.1629791
8: -150.6563873, 102.9990997, -154.5748596, 105.6669846, -256.3233643, 257.5739746
9: -114.6585846, 113.1035004, -117.5921783, 116.0159225, -230.6744537, 230.6956787

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0321160, upper bound: 242.0321420
time: 7.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0349974, upper bound: 242.0349974
time: 6.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.62
Output dim: 7, lower bound: -242.0260469, upper bound: 242.0253965
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.62
Output dim: 7, lower bound: -242.0294798, upper bound: 242.0285284
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.62
Output dim: 7, lower bound: -242.0321160, upper bound: 242.0321420
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.62
Output dim: 7, lower bound: -242.0349974, upper bound: 242.0349974

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -109.7947159, 87.5827560, -106.1791306, 84.7131271, -194.5078430, 193.7618866
1: -91.7316513, 77.5614471, -88.6458893, 74.9622421, -166.6938324, 166.2073059
2: -120.8356934, 78.8078003, -116.8081436, 76.2149124, -197.0505981, 195.6159363
3: -128.6669006, 68.3268127, -124.3209000, 66.0325317, -194.6994324, 192.6477051
4: -117.9885635, 90.6212769, -114.0883179, 87.6332474, -205.6218109, 204.7095947
5: -106.1484222, 83.1566772, -102.6693649, 80.4245605, -186.5729675, 185.8260498
6: -101.2699738, 97.2632141, -97.9292297, 94.0286407, -195.2986145, 195.1924133
7: -110.4528427, 93.3705978, -106.7750626, 90.2929153, -200.7457123, 200.1456604
8: -132.2630768, 90.4495087, -127.9146652, 87.4780579, -219.7411041, 218.3641663
9: -100.7799759, 99.3796082, -97.4095459, 96.0771255, -196.8571014, 196.7891541

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 185

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0070921, upper bound: 242.0063045
time: 6.90 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0050436, upper bound: 242.0035887
time: 7.24 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -114.6814957, 91.4476166, -114.4643478, 91.2652740, -205.9467773, 205.9119110
1: -95.8384933, 81.0332565, -95.6111679, 80.8453827, -176.6838684, 176.6443939
2: -126.2253723, 82.3139877, -125.9381332, 82.1556625, -208.3809662, 208.2521057
3: -134.4291077, 71.4151535, -134.0944672, 71.2607117, -205.6898193, 205.5096130
4: -123.2251587, 94.6522903, -122.9599991, 94.4626770, -217.6878357, 217.6122742
5: -110.8311996, 86.8242950, -110.6111832, 86.6445618, -197.4757385, 197.4354401
6: -105.7845535, 101.5940170, -105.5845718, 101.3605957, -207.1451416, 207.1785889
7: -115.3786545, 97.5039673, -115.1229935, 97.3017197, -212.6803589, 212.6269379
8: -138.1497955, 94.4725266, -137.8861237, 94.2819138, -232.4317017, 232.3586121
9: -105.2700043, 103.8159561, -105.0223846, 103.5927582, -208.8627625, 208.8382874

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0110219, upper bound: 242.0103040
time: 7.93 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0089398, upper bound: 242.0073800
time: 6.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -115.5666122, 92.1255646, -110.6926422, 88.2710876, -203.8376923, 202.8181915
1: -96.5191422, 81.6024780, -92.4020538, 78.1211166, -174.6402435, 174.0045319
2: -127.1526718, 82.9244003, -121.7556610, 79.4274979, -206.5801392, 204.6800385
3: -135.3501129, 71.9019928, -129.5619507, 68.8131714, -204.1632690, 201.4639130
4: -124.1337128, 95.3534012, -118.9058456, 91.3300476, -215.4637299, 214.2592163
5: -111.6795273, 87.4400330, -106.9967728, 83.7764130, -195.4559326, 194.4367676
6: -106.5715256, 102.3173218, -102.0753784, 97.9884033, -204.5599365, 204.3926697
7: -116.1831818, 98.1899567, -111.2644958, 94.0654602, -210.2486267, 209.4544220
8: -139.2160645, 95.1789474, -133.3528900, 91.1687088, -230.3847656, 228.5318298
9: -105.9681854, 104.5192566, -101.4745560, 100.0998917, -206.0680847, 205.9938049

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0141420, upper bound: 242.0146959
time: 7.10 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0126917, upper bound: 242.0128111
time: 7.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -120.1003799, 95.7126617, -118.6602249, 94.5740891, -214.6744537, 214.3728943
1: -100.3326111, 84.8262100, -99.1031418, 83.7796783, -184.1122742, 183.9293518
2: -132.1571960, 86.1856918, -130.5395050, 85.1454086, -217.3026123, 216.7251740
3: -140.7035675, 74.7705841, -138.9675446, 73.8441849, -214.5477600, 213.7381287
4: -128.9977112, 99.0940170, -127.4410400, 97.8968353, -226.8945465, 226.5350647
5: -116.0253983, 90.8423691, -114.6346436, 89.7586212, -205.7840271, 205.4769897
6: -110.7651596, 106.3387375, -109.4387283, 105.0427475, -215.8079071, 215.7774048
7: -120.7608490, 102.0307159, -119.2967224, 100.8093185, -221.5701599, 221.3274231
8: -144.6785431, 98.9124374, -142.9420776, 97.7139359, -242.3924255, 241.8545227
9: -110.1409607, 108.6394653, -108.8010330, 107.3314209, -217.4723816, 217.4404907

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0169662, upper bound: 242.0173941
time: 6.63 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0153518, upper bound: 242.0153518
time: 5.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.58 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 13.58
Output dim: 7, lower bound: -242.0070921, upper bound: 242.0063045
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 13.58
Output dim: 7, lower bound: -242.0050436, upper bound: 242.0035887
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 13.58
Output dim: 7, lower bound: -242.0110219, upper bound: 242.0103040
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 13.58
Output dim: 7, lower bound: -242.0089398, upper bound: 242.0073800
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 7, lower bound: -242.0141420, upper bound: 242.0146959
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 13.58
Output dim: 7, lower bound: -242.0126917, upper bound: 242.0128111
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 7, lower bound: -242.0169662, upper bound: 242.0173941
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 7, lower bound: -242.0153518, upper bound: 242.0153518

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -110.1071930, 87.7956848, -109.7947540, 87.5590439, -197.6662292, 197.5904236
1: -91.9306259, 77.7625885, -91.6474838, 77.4894867, -169.4201050, 169.4100647
2: -121.1340408, 79.0180588, -120.7659073, 78.7850647, -199.9190826, 199.7839661
3: -128.9607391, 68.5057297, -128.5110626, 68.2546463, -197.2153931, 197.0167847
4: -118.2799988, 90.8492126, -117.9429932, 90.5892868, -208.8692932, 208.7922058
5: -106.4292221, 83.3350525, -106.1332321, 83.1013260, -189.5305481, 189.4682922
6: -101.5440674, 97.5007782, -101.2485275, 97.1962051, -198.7402496, 198.7492981
7: -110.7060318, 93.6032562, -110.3637390, 93.3111038, -204.0171356, 203.9669952
8: -132.6111603, 90.6475525, -132.2666779, 90.4236145, -223.0347748, 222.9142303
9: -100.9926300, 99.5978775, -100.6561890, 99.2904892, -200.2831116, 200.2540588

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0079298, upper bound: 242.0081542
time: 7.60 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0123170, upper bound: 242.0129395
time: 8.12 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -114.6229477, 91.3681946, -117.7572632, 93.8578186, -208.4807434, 209.1254578
1: -95.7275848, 80.9727020, -98.3438950, 83.1444931, -178.8720703, 179.3165894
2: -126.1161652, 82.2657166, -129.5438843, 84.4994659, -210.6156311, 211.8096008
3: -134.2909546, 71.3620605, -137.9103394, 73.2822647, -207.5732117, 209.2723999
4: -123.1215286, 94.5737228, -126.4725723, 97.1517029, -220.2732239, 221.0462952
5: -110.7576065, 86.7227325, -113.7661514, 89.0794678, -199.8370667, 200.4888611
6: -105.7192535, 101.5045547, -108.6068497, 104.2459717, -209.9652252, 210.1114044
7: -115.2651672, 97.4279404, -118.3909225, 100.0506821, -215.3158264, 215.8188477
8: -138.0495605, 94.3646927, -141.8494415, 96.9642029, -235.0137634, 236.2141266
9: -105.1484756, 103.6998825, -107.9782333, 106.5172958, -211.6657715, 211.6781158

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0117852, upper bound: 242.0121814
time: 8.38 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0155080, upper bound: 242.0159538
time: 6.66 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -107.9885864, 86.0942078, -108.7654724, 86.7224121, -194.7109985, 194.8596802
1: -90.1070938, 76.2789230, -90.7836380, 76.8255234, -166.9326172, 167.0625610
2: -118.7990189, 77.4482498, -119.6322632, 78.0686188, -196.8676453, 197.0805054
3: -126.6031570, 67.1104584, -127.3952026, 67.6920700, -194.2952118, 194.5056610
4: -116.0001297, 89.0390320, -116.8347855, 89.7384567, -205.7385712, 205.8738098
5: -104.3685226, 81.6585159, -105.1209946, 82.3231201, -186.6916504, 186.7794952
6: -99.5880280, 95.6567688, -100.3391647, 96.3138275, -195.9018555, 195.9959412
7: -108.5922623, 91.8362350, -109.3788147, 92.5014648, -201.0937195, 201.2150574
8: -129.9303131, 88.6861267, -130.9714813, 89.4969635, -219.4272766, 219.6576080
9: -99.0783081, 97.6388855, -99.7892303, 98.4147949, -197.4930725, 197.4281006

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0104419, upper bound: 242.0102855
time: 6.18 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0138900, upper bound: 242.0138900
time: 7.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.44 seconds
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 14.44
Output dim: 7, lower bound: -242.0079298, upper bound: 242.0081542
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 14.44
Output dim: 7, lower bound: -242.0123170, upper bound: 242.0129395
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 14.44
Output dim: 7, lower bound: -242.0117852, upper bound: 242.0121814
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 7, lower bound: -242.0155080, upper bound: 242.0159538
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 14.44
Output dim: 7, lower bound: -242.0104419, upper bound: 242.0102855
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.44
Output dim: 7, lower bound: -242.0138900, upper bound: 242.0138900

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -113.2121124, 90.2445602, -114.5905151, 91.3349915, -204.5470886, 204.8350677
1: -94.5418472, 79.9803085, -95.6821594, 80.9163971, -175.4582520, 175.6624756
2: -124.5666275, 81.2505493, -126.0653534, 82.2195969, -206.7862244, 207.3159027
3: -132.6358948, 70.4949265, -134.1950531, 71.3356476, -203.9715424, 204.6899719
4: -121.6031952, 93.4046326, -123.0630569, 94.5270386, -216.1302338, 216.4676819
5: -109.3901291, 85.6421432, -110.6962051, 86.6537704, -196.0438995, 196.3383484
6: -104.4221420, 100.2602158, -105.6947098, 101.4524536, -205.8746033, 205.9549103
7: -113.8413315, 96.2368240, -115.1936798, 97.3761139, -211.2174377, 211.4304657
8: -136.3482056, 93.2004929, -138.0293579, 94.3506012, -230.6988068, 231.2298279
9: -103.8532257, 102.4245071, -105.0696259, 103.6532669, -207.5065002, 207.4941101

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0132613, upper bound: 242.0140210
time: 7.41 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0133298, upper bound: 242.0140484
time: 7.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -106.6191254, 85.0037460, -105.6300659, 84.2251892, -190.8442535, 190.6338043
1: -88.9569702, 75.3157349, -88.1487350, 74.6195297, -163.5764923, 163.4644470
2: -117.2951355, 76.4632950, -116.1881332, 75.8121872, -193.1073303, 192.6513824
3: -124.9972305, 66.2692947, -123.7160492, 65.7655258, -190.7627563, 189.9853516
4: -114.5258102, 87.9038467, -113.4579163, 87.1405029, -201.6663055, 201.3617554
5: -103.0410995, 80.6091309, -102.0818558, 79.9216232, -182.9626923, 182.6909790
6: -98.3298187, 94.4486465, -97.4559555, 93.5475159, -191.8773193, 191.9046021
7: -107.2106400, 90.6802292, -106.2136154, 89.8543320, -197.0649719, 196.8938446
8: -128.2787781, 87.5568924, -127.1889877, 86.9103851, -215.1891479, 214.7458801
9: -97.8210373, 96.4009857, -96.9096909, 95.5795364, -193.4005737, 193.3106689

Time for backsubstitution: 1.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 66

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0117411, upper bound: 242.0118255
time: 6.05 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0118883, upper bound: 242.0118883
time: 7.38 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.54 seconds
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.54
Output dim: 7, lower bound: -242.0132613, upper bound: 242.0140210
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.54
Output dim: 7, lower bound: -242.0133298, upper bound: 242.0140484
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 14.54
Output dim: 7, lower bound: -242.0117411, upper bound: 242.0118255
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 14.54
Output dim: 7, lower bound: -242.0118883, upper bound: 242.0118883

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -107.7186661, 85.8687668, -111.6576309, 88.9993591, -196.7180176, 197.5263519
1: -89.9554062, 76.1577759, -93.2341537, 78.8760223, -168.8314056, 169.3918915
2: -118.5567780, 77.3352814, -122.8575134, 80.1306534, -198.6874237, 200.1927948
3: -126.2355270, 67.1497192, -130.7783813, 69.5500259, -195.7855530, 197.9280853
4: -115.6916885, 88.8724442, -119.9071350, 92.1073151, -207.7990112, 208.7795715
5: -104.0738297, 81.5212708, -107.8582382, 84.4537582, -188.5275879, 189.3795166
6: -99.3621140, 95.4363174, -102.9933243, 98.8772507, -198.2393646, 198.4296417
7: -108.3675232, 91.6463776, -112.2723846, 94.9263229, -203.2938385, 203.9187622
8: -129.7241974, 88.5984802, -134.4937744, 91.8938751, -221.6180725, 223.0922394
9: -98.8461761, 97.4666901, -102.3965530, 101.0062637, -199.8524170, 199.8632507

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0132613, upper bound: 242.0140210
time: 7.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0132613, upper bound: 242.0140210
time: 7.23 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -110.2666473, 87.8991699, -110.9813690, 88.4648666, -198.7315063, 198.8805389
1: -92.1309967, 77.9868698, -92.6742172, 78.4132919, -170.5442810, 170.6610565
2: -121.4083633, 79.1730347, -122.1283798, 79.6556244, -201.0639801, 201.3014221
3: -129.2965088, 68.7192078, -129.9980011, 69.1377792, -198.4342804, 198.7172089
4: -118.4456482, 90.9690781, -119.1850281, 91.5540695, -209.9997253, 210.1541138
5: -106.5220642, 83.4136353, -107.2040558, 83.9473190, -190.4693909, 190.6176147
6: -101.7365112, 97.7232971, -102.3779297, 98.2915192, -200.0280304, 200.1012268
7: -110.9661026, 93.8195572, -111.6125946, 94.3700180, -205.3361206, 205.4321594
8: -132.7937469, 90.6608734, -133.6738281, 91.3268738, -224.1206055, 224.3347015
9: -101.1900482, 99.7925339, -101.7900848, 100.4076920, -201.5977478, 201.5826111

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 155
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 175
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 62
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0133298, upper bound: 242.0140484
time: 6.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -242.0133298, upper bound: 242.0140484
time: 7.22 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 14.66 seconds
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0132613, upper bound: 242.0140210
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0132613, upper bound: 242.0140210
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0133298, upper bound: 242.0140484
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 14.66
Output dim: 7, lower bound: -242.0133298, upper bound: 242.0140484

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -107.7186661, 85.8687668, -107.1623764, 85.4330902, -193.1517487, 193.0311279
1: -89.9554062, 76.1577759, -89.4549713, 75.7132416, -165.6686401, 165.6127167
2: -118.5567780, 77.3352814, -117.8995132, 76.9142609, -195.4710388, 195.2347870
3: -126.2355270, 67.1497192, -125.5154572, 66.7518158, -192.9873352, 192.6651764
4: -115.6916885, 88.8724442, -115.0843048, 88.3976288, -204.0893250, 203.9567566
5: -104.0738297, 81.5212708, -103.5351486, 81.0719681, -185.1457977, 185.0564270
6: -99.3621140, 95.4363174, -98.8523254, 94.9090576, -194.2711639, 194.2886353
7: -108.3675232, 91.6463776, -107.7626495, 91.1495056, -199.5170288, 199.4090271
8: -129.7241974, 88.5984802, -129.0530548, 88.1611710, -217.8853149, 217.6515198
9: -98.8461761, 97.4666901, -98.2997971, 96.9516678, -195.7978210, 195.7664795

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0078443, upper bound: 242.0088219
time: 7.22 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0071828, upper bound: 242.0078720
time: 6.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -107.7186661, 85.8687668, -100.5844879, 80.2045288, -187.9231873, 186.4532166
1: -89.9554062, 76.1577759, -83.8833771, 71.0601730, -161.0155792, 160.0411530
2: -118.5567780, 77.3352814, -110.6444016, 72.1386414, -190.6953888, 187.9796753
3: -126.2355270, 67.1497192, -117.8948135, 62.5368690, -188.7723846, 185.0445099
4: -115.6916885, 88.8724442, -108.0242844, 82.9068222, -198.5985107, 196.8967285
5: -104.0738297, 81.5212708, -97.1998520, 76.0489655, -180.1228027, 178.7211304
6: -99.3621140, 95.4363174, -92.7745895, 89.1112671, -188.4733582, 188.2109070
7: -108.3675232, 91.6463776, -101.1521072, 85.6063004, -193.9738159, 192.7984772
8: -129.7241974, 88.5984802, -120.9992371, 82.5275116, -212.2516937, 209.5977173
9: -98.8461761, 97.4666901, -92.2807846, 90.9370346, -189.7832031, 189.7474670

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0078443, upper bound: 242.0088219
time: 6.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0071828, upper bound: 242.0078720
time: 6.95 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -110.2666473, 87.8991699, -106.4327316, 84.8567581, -195.1234131, 194.3318634
1: -92.1309967, 77.9868698, -88.8510208, 75.2141724, -167.3451691, 166.8378601
2: -121.4083633, 79.1730347, -117.1139221, 76.4023590, -197.8107147, 196.2869568
3: -129.2965088, 68.7192078, -124.6745300, 66.3076782, -195.6041870, 193.3937378
4: -118.4456482, 90.9690781, -114.3062820, 87.8004227, -206.2460632, 205.2753448
5: -106.5220642, 83.4136353, -102.8291397, 80.5255737, -187.0476379, 186.2427673
6: -101.7365112, 97.7232971, -98.1886444, 94.2773590, -196.0138702, 195.9119263
7: -110.9661026, 93.8195572, -107.0512466, 90.5496521, -201.5157318, 200.8708038
8: -132.7937469, 90.6608734, -128.1701202, 87.5491562, -220.3428955, 218.8309784
9: -101.1900482, 99.7925339, -97.6455994, 96.3060455, -197.4960938, 197.4381256

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0077924, upper bound: 242.0088125
time: 7.52 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0071198, upper bound: 242.0078414
time: 8.59 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -110.2666473, 87.8991699, -100.1783676, 79.8851929, -190.1518402, 188.0775146
1: -92.1309967, 77.9868698, -83.5500793, 70.7824020, -162.9133911, 161.5369263
2: -121.4083633, 79.1730347, -110.2094040, 71.8547516, -193.2631226, 189.3824310
3: -129.2965088, 68.7192078, -117.4266281, 62.2885971, -191.5850830, 186.1457977
4: -118.4456482, 90.9690781, -107.5934906, 82.5780411, -201.0236664, 198.5625610
5: -106.5220642, 83.4136353, -96.8084106, 75.7471008, -182.2691498, 180.2220306
6: -101.7365112, 97.7232971, -92.4066391, 88.7604294, -190.4969330, 190.1299438
7: -110.9661026, 93.8195572, -100.7594986, 85.2737274, -196.2398376, 194.5790558
8: -132.7937469, 90.6608734, -120.5062180, 82.1879807, -214.9817200, 211.1670837
9: -101.1900482, 99.7925339, -91.9192657, 90.5794220, -191.7694397, 191.7117767

Time for backsubstitution: 1.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 155
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 232
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 175
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 7

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 155

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0077924, upper bound: 242.0088125
time: 7.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -242.0071198, upper bound: 242.0078414
time: 7.57 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 15.86 seconds
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 15.86
Output dim: 7, lower bound: -242.0078443, upper bound: 242.0088219
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 15.86
Output dim: 7, lower bound: -242.0071828, upper bound: 242.0078720
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 15.86
Output dim: 7, lower bound: -242.0078443, upper bound: 242.0088219
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 15.86
Output dim: 7, lower bound: -242.0071828, upper bound: 242.0078720
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 15.86
Output dim: 7, lower bound: -242.0077924, upper bound: 242.0088125
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 15.86
Output dim: 7, lower bound: -242.0071198, upper bound: 242.0078414
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 15.86
Output dim: 7, lower bound: -242.0077924, upper bound: 242.0088125
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 15.86
Output dim: 7, lower bound: -242.0071198, upper bound: 242.0078414
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=243.70181274414062
rel_dist={7: [-242.04118532033573, 242.04118532033573]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 1691.93 seconds
