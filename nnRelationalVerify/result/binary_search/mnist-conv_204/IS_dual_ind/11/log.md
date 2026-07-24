## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.41320994132
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.6942806, -5.0616770, -9.6942806, -5.0616770, -4.3366542, 4.3366537)
1: (-15.0952425, -10.8431473, -15.0952425, -10.8431473, -4.2520952, 4.2520952)
2: (-9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.2964392, 3.2964392)
3: (-11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.1194048, 4.1194048)
4: (-5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.5223343, 3.5223343)
5: (-3.5736499, -0.4953117, -3.5736499, -0.4953117, -3.0783381, 3.0783381)
6: (-11.5837259, -6.9704914, -11.5837259, -6.9704914, -4.5754027, 4.5754023)
7: (-2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6390057, 3.6390057)
8: (-5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.6043172, 3.6043172)
9: (0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.6205788, 2.6205788)

## BASE Result
execution time: IAR + LP analysis = 15.24 + 33.85 = 49.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.1425934, upper bound: 2.1425903


# Binary Search by BASE starts (time budget: 3550.92 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.567917585372925
rel_dist={9: [-1.824091324888559, 1.824091281312575]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.44260835647583
rel_dist={9: [-1.592849651897651, 1.5928489343896768]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.3590686321258545
rel_dist={9: [-1.414766908177059, 1.4147664541724545]}

## Binary Search Result
Binary search time: 152.68 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 3398.24 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240692, upper bound: 1.8222150
time: 5.18 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240692, upper bound: 1.8240689
time: 4.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.29 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.29
Output dim: 9, lower bound: -1.8240692, upper bound: 1.8222150
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.29
Output dim: 9, lower bound: -1.8240692, upper bound: 1.8240689

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.6942806, -5.0616770, -3.6486897, 3.6473520
1: -15.0922241, -10.8474169, -15.0952425, -10.8431473, -3.7677727, 3.7662728
2: -9.0594254, -5.7682366, -9.0615978, -5.7651587, -3.0366640, 3.0351608
3: -11.5214615, -7.4080415, -11.5230656, -7.4036608, -4.0464029, 4.0436363
4: -5.4654369, -1.9569887, -5.4777827, -1.9554484, -3.4319100, 3.4441113
5: -3.5687127, -0.4961605, -3.5736499, -0.4953117, -2.9226208, 2.9274914
6: -11.5807753, -6.9714108, -11.5837259, -6.9704914, -3.9172435, 3.9194589
7: -2.8032007, 0.8274651, -2.8098021, 0.8292036, -3.5788045, 3.5842595
8: -5.0741324, -1.4750624, -5.0775828, -1.4732656, -3.1885386, 3.1899610
9: 0.4387226, 3.0539322, 0.4356761, 3.0562549, -2.5647740, 2.5636451

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8222130, upper bound: 1.8222131
time: 5.18 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8222130, upper bound: 1.8222138
time: 5.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7343044, -5.0527325, -9.6942787, -5.0616827, -3.6904697, 3.6768310
1: -15.1312637, -10.8247185, -15.0952377, -10.8431606, -3.8093185, 3.7894056
2: -9.0981321, -5.7384038, -9.0615921, -5.7651634, -3.0826058, 3.0699911
3: -11.5890598, -7.3757129, -11.5230627, -7.4036703, -4.1152487, 4.0705161
4: -5.5177908, -1.8448060, -5.4777594, -1.9554527, -3.5123529, 3.5255461
5: -3.6030333, -0.4554825, -3.5736399, -0.4953117, -2.9759536, 2.9529965
6: -11.5974503, -6.9417963, -11.5837221, -6.9704943, -3.9385333, 3.9483271
7: -2.8284822, 0.8835974, -2.8097863, 0.8291988, -3.6067019, 3.6580834
8: -5.0990133, -1.4518523, -5.0775743, -1.4732704, -3.2176409, 3.2123561
9: 0.3557720, 3.0763309, 0.4356833, 3.0562494, -2.6538160, 2.6179335

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8213785, upper bound: 1.8240561
time: 6.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240559, upper bound: 1.8240560
time: 4.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.28
Output dim: 9, lower bound: -1.8222130, upper bound: 1.8222131
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.28
Output dim: 9, lower bound: -1.8222130, upper bound: 1.8222138
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.28
Output dim: 9, lower bound: -1.8213785, upper bound: 1.8240561
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.28
Output dim: 9, lower bound: -1.8240559, upper bound: 1.8240560

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.6923923, -5.0648394, -3.6439438, 3.6439428
1: -15.0922241, -10.8474169, -15.0922241, -10.8474169, -3.7632570, 3.7632580
2: -9.0594254, -5.7682366, -9.0594254, -5.7682366, -3.0336452, 3.0336452
3: -11.5214615, -7.4080415, -11.5214615, -7.4080415, -4.0416164, 4.0416169
4: -5.4654369, -1.9569887, -5.4654369, -1.9569887, -3.4296107, 3.4296112
5: -3.5687127, -0.4961605, -3.5687127, -0.4961605, -2.9215899, 2.9215903
6: -11.5807753, -6.9714108, -11.5807753, -6.9714108, -3.9161620, 3.9161615
7: -2.8032007, 0.8274651, -2.8032007, 0.8274651, -3.5752068, 3.5752063
8: -5.0741324, -1.4750624, -5.0741324, -1.4750624, -3.1868410, 3.1868410
9: 0.4387226, 3.0539322, 0.4387226, 3.0539322, -2.5605016, 2.5605013

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8222000, upper bound: 1.8195129
time: 4.92 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221998, upper bound: 1.8222010
time: 5.23 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.7343044, -5.0527325, -3.6573806, 3.6857300
1: -15.0922241, -10.8474169, -15.1312637, -10.8247185, -3.7857714, 3.8048167
2: -9.0594254, -5.7682366, -9.0981321, -5.7384038, -3.0684414, 3.0795949
3: -11.5214615, -7.4080415, -11.5890598, -7.3757129, -4.0685005, 4.1096807
4: -5.4654369, -1.9569887, -5.5177908, -1.8448060, -3.5110431, 3.4806767
5: -3.5687127, -0.4961605, -3.6030333, -0.4554825, -2.9471092, 2.9595232
6: -11.5807753, -6.9714108, -11.5974503, -6.9417963, -3.9450359, 3.9341936
7: -2.8032007, 0.8274651, -2.8284822, 0.8835974, -3.6323686, 3.6031132
8: -5.0741324, -1.4750624, -5.0990133, -1.4518523, -3.2092419, 3.2154918
9: 0.4387226, 3.0539322, 0.3557720, 3.0763309, -2.5876617, 2.6495521

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221998, upper bound: 1.8195145
time: 4.97 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221998, upper bound: 1.8222022
time: 5.11 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.7343044, -5.0527325, -9.6934052, -5.0639477, -3.6882715, 3.6759148
1: -15.1312637, -10.8247185, -15.0926180, -10.8434973, -3.8089561, 3.7860479
2: -9.0981321, -5.7384038, -9.0605202, -5.7662716, -3.0814605, 3.0684249
3: -11.5890598, -7.3757129, -11.5224562, -7.4049730, -4.1133695, 4.0698228
4: -5.5177908, -1.8448060, -5.4727859, -1.9561607, -3.5116005, 3.5202951
5: -3.6030333, -0.4554825, -3.5722094, -0.4961987, -2.9745426, 2.9511635
6: -11.5974503, -6.9417963, -11.5826092, -6.9707985, -3.9376469, 3.9467511
7: -2.8284822, 0.8835974, -2.8088903, 0.8269205, -3.6039891, 3.6571674
8: -5.0990133, -1.4518523, -5.0764704, -1.4740038, -3.2167320, 3.2107501
9: 0.3557720, 3.0763309, 0.4370999, 3.0540252, -2.6516154, 2.6166039

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8213784, upper bound: 1.8213772
time: 5.00 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8213784, upper bound: 1.8240559
time: 5.19 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.7343006, -5.0527411, -9.7479515, -5.0552669, -3.7100348, 3.7236123
1: -15.1312599, -10.8247194, -15.1277533, -10.7960215, -3.8572607, 3.8437424
2: -9.0981312, -5.7384062, -9.0764198, -5.7325544, -3.1293926, 3.1070647
3: -11.5890579, -7.3757162, -11.5622654, -7.3845053, -4.1315670, 4.1139312
4: -5.5177851, -1.8448066, -5.5115185, -1.8854260, -3.5542765, 3.5696254
5: -3.6030309, -0.4554825, -3.5987160, -0.4530239, -2.9954376, 2.9731460
6: -11.5974483, -6.9417944, -11.5975637, -6.9488916, -3.9802246, 3.9802599
7: -2.8284802, 0.8835917, -2.8622637, 0.8414440, -3.6432686, 3.6802011
8: -5.0990129, -1.4518523, -5.1077251, -1.4480362, -3.2508373, 3.2702374
9: 0.3557739, 3.0763273, 0.3841162, 3.0669699, -2.6680026, 2.6555812

Time for backsubstitution: 15.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8111601, upper bound: 1.8204164
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240477, upper bound: 1.8240486
time: 6.87 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.04 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.04
Output dim: 9, lower bound: -1.8222000, upper bound: 1.8195129
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.04
Output dim: 9, lower bound: -1.8221998, upper bound: 1.8222010
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.04
Output dim: 9, lower bound: -1.8221998, upper bound: 1.8195145
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.04
Output dim: 9, lower bound: -1.8221998, upper bound: 1.8222022
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.04
Output dim: 9, lower bound: -1.8213784, upper bound: 1.8213772
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.04
Output dim: 9, lower bound: -1.8213784, upper bound: 1.8240559
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.04
Output dim: 9, lower bound: -1.8111601, upper bound: 1.8204164
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.04
Output dim: 9, lower bound: -1.8240477, upper bound: 1.8240486

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.6923923, -5.0648394, -3.6430264, 3.6417480
1: -15.0896091, -10.8477516, -15.0922241, -10.8474169, -3.7599010, 3.7628956
2: -9.0583544, -5.7693434, -9.0594254, -5.7682366, -3.0320859, 3.0325027
3: -11.5208578, -7.4093418, -11.5214615, -7.4080415, -4.0409298, 4.0394702
4: -5.4604731, -1.9576910, -5.4654369, -1.9569887, -3.4240670, 3.4288602
5: -3.5672817, -0.4970436, -3.5687127, -0.4961605, -2.9197569, 2.9201832
6: -11.5796518, -6.9717102, -11.5807753, -6.9714108, -3.9145608, 3.9152780
7: -2.8023047, 0.8251872, -2.8032007, 0.8274651, -3.5742912, 3.5724969
8: -5.0730295, -1.4757953, -5.0741324, -1.4750624, -3.1852398, 3.1859322
9: 0.4401379, 3.0517111, 0.4387226, 3.0539322, -2.5591745, 2.5583045

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195127, upper bound: 1.8195129
time: 5.00 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195127, upper bound: 1.8195143
time: 49.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.7460480, -5.0584192, -9.6923904, -5.0648422, -3.7000670, 3.6635065
1: -15.1247530, -10.8002377, -15.0922222, -10.8474169, -3.8175941, 3.8112426
2: -9.0743618, -5.7355871, -9.0594234, -5.7682385, -3.0706692, 3.0898032
3: -11.5606031, -7.3888259, -11.5214596, -7.4080458, -4.0850067, 4.0653143
4: -5.4992199, -1.8870091, -5.4654288, -1.9569901, -3.4842186, 3.4894896
5: -3.5939014, -0.4538774, -3.5687099, -0.4961615, -2.9453235, 2.9507082
6: -11.5945415, -6.9498158, -11.5807724, -6.9714088, -3.9480734, 3.9578409
7: -2.8556561, 0.8396807, -2.8031998, 0.8274622, -3.6299014, 3.6117382
8: -5.1042891, -1.4498472, -5.0741310, -1.4750643, -3.2447300, 3.2200141
9: 0.3871465, 3.0646892, 0.4387226, 3.0539284, -2.6164455, 2.5746496

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8186726, upper bound: 1.8094301
time: 5.20 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221932, upper bound: 1.8221923
time: 5.22 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.7343044, -5.0527325, -3.6564660, 3.6835356
1: -15.0896091, -10.8477516, -15.1312637, -10.8247185, -3.7824154, 3.8044548
2: -9.0583544, -5.7693434, -9.0981321, -5.7384038, -3.0668831, 3.0784519
3: -11.5208578, -7.4093418, -11.5890598, -7.3757129, -4.0678139, 4.1075330
4: -5.4604731, -1.9576910, -5.5177908, -1.8448060, -3.5057902, 3.4799261
5: -3.5672817, -0.4970436, -3.6030333, -0.4554825, -2.9452772, 2.9581161
6: -11.5796518, -6.9717102, -11.5974503, -6.9417963, -3.9434347, 3.9333100
7: -2.8023047, 0.8251872, -2.8284822, 0.8835974, -3.6314521, 3.6004033
8: -5.0730295, -1.4757953, -5.0990133, -1.4518523, -3.2076416, 3.2145824
9: 0.4401379, 3.0517111, 0.3557720, 3.0763309, -2.5863352, 2.6473553

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8213782, upper bound: 1.8195147
time: 6.18 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8213781, upper bound: 1.8195126
time: 10.35 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.7460480, -5.0584192, -9.7343006, -5.0527411, -3.7135043, 3.7052958
1: -15.1247530, -10.8002377, -15.1312599, -10.8247194, -3.8401089, 3.8528025
2: -9.0743618, -5.7355871, -9.0981312, -5.7384062, -3.1054668, 3.1264298
3: -11.5606031, -7.3888259, -11.5890579, -7.3757162, -4.1118898, 4.1264019
4: -5.4992199, -1.8870091, -5.5177851, -1.8448066, -3.5550385, 3.5401192
5: -3.5939014, -0.4538774, -3.6030309, -0.4554825, -2.9673271, 2.9889450
6: -11.5945415, -6.9498158, -11.5974483, -6.9417944, -3.9769473, 3.9758744
7: -2.8556561, 0.8396807, -2.8284802, 0.8835917, -3.6675286, 3.6396465
8: -5.1042891, -1.4498472, -5.0990129, -1.4518523, -3.2671328, 3.2486649
9: 0.3871465, 3.0646892, 0.3557739, 3.0763273, -2.6436050, 2.6637006

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204162, upper bound: 1.8094300
time: 5.07 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240483, upper bound: 1.8221923
time: 5.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.7334185, -5.0549893, -9.6934052, -5.0639477, -3.6874027, 3.6737213
1: -15.1286526, -10.8250637, -15.0926180, -10.8434973, -3.8055849, 3.7856808
2: -9.0971193, -5.7395234, -9.0605202, -5.7662716, -3.0798984, 3.0672596
3: -11.5884438, -7.3770781, -11.5224562, -7.4049730, -4.1126924, 4.0676723
4: -5.5128527, -1.8455058, -5.4727859, -1.9561607, -3.5060501, 3.5195808
5: -3.6015487, -0.4563608, -3.5722094, -0.4961987, -2.9728107, 2.9497442
6: -11.5963335, -6.9421077, -11.5826092, -6.9707985, -3.9365215, 3.9458637
7: -2.8275809, 0.8813424, -2.8088903, 0.8269205, -3.6030660, 3.6544662
8: -5.0979142, -1.4525871, -5.0764704, -1.4740038, -3.2151356, 3.2098327
9: 0.3571882, 3.0741472, 0.4370999, 3.0540252, -2.6502678, 2.6144586

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195127, upper bound: 1.8213771
time: 5.04 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195128, upper bound: 1.8213771
time: 5.08 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7876139, -5.0466785, -9.6934052, -5.0639477, -3.7274189, 3.6839533
1: -15.1617699, -10.7778406, -15.0926180, -10.8434973, -3.8411875, 3.8336596
2: -9.1145000, -5.7063303, -9.0605202, -5.7662716, -3.1069779, 3.1225717
3: -11.6285610, -7.3574991, -11.5224562, -7.4049730, -4.1334558, 4.0873599
4: -5.5502605, -1.7750721, -5.4727859, -1.9561607, -3.5467358, 3.5407858
5: -3.6278422, -0.4160423, -3.5722094, -0.4961987, -3.0000238, 2.9611027
6: -11.6107731, -6.9201183, -11.5826092, -6.9707985, -3.9676962, 3.9808602
7: -2.8808923, 0.8953681, -2.8088903, 0.8269205, -3.6585383, 3.6697068
8: -5.1272721, -1.4269066, -5.0764704, -1.4740038, -3.2606487, 3.2433596
9: 0.3055973, 3.0870466, 0.4370999, 3.0540252, -2.6757233, 2.6281486

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195128, upper bound: 1.8240559
time: 5.38 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195128, upper bound: 1.8240561
time: 5.08 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7082729, -5.1544166, -9.7457428, -5.0663366, -3.6294661, 3.6188049
1: -15.0922861, -10.8529520, -15.1235247, -10.7990866, -3.7950559, 3.8029833
2: -9.0486259, -5.7698059, -9.0710316, -5.7356319, -3.0725431, 3.0636735
3: -11.5572805, -7.4114013, -11.5587711, -7.3879242, -4.0923133, 4.0691710
4: -5.4759641, -1.8787701, -5.5073738, -1.8890769, -3.5005178, 3.5323451
5: -3.5723045, -0.4659157, -3.5957999, -0.4541206, -2.9637480, 2.9596996
6: -11.5663691, -6.9903345, -11.5946064, -6.9542608, -3.9223838, 3.9296479
7: -2.7901053, 0.8567533, -2.8585496, 0.8386235, -3.5906954, 3.6480055
8: -5.0355582, -1.4757357, -5.1008067, -1.4501157, -3.1845222, 3.2140059
9: 0.4640474, 3.0631402, 0.3958206, 3.0658062, -2.5501575, 2.6061230

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8094294, upper bound: 1.8204163
time: 6.10 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8094294, upper bound: 1.8204161
time: 5.28 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7342987, -5.0527439, -9.7479515, -5.0552669, -3.7100339, 3.6720757
1: -15.1312571, -10.8247204, -15.1277533, -10.7960215, -3.8572569, 3.8457379
2: -9.0981293, -5.7384076, -9.0764198, -5.7325544, -3.1123896, 3.1066184
3: -11.5890579, -7.3757181, -11.5622654, -7.3845053, -4.1469798, 4.1132846
4: -5.5177822, -1.8448077, -5.5115185, -1.8854260, -3.5531836, 3.5638576
5: -3.6030309, -0.4554844, -3.5987160, -0.4530239, -2.9943900, 2.9749272
6: -11.5974483, -6.9417968, -11.5975637, -6.9488916, -3.9802227, 3.9561710
7: -2.8284788, 0.8835917, -2.8622637, 0.8414440, -3.6428699, 3.6831870
8: -5.0990105, -1.4518547, -5.1077251, -1.4480362, -3.2188835, 3.2700839
9: 0.3557787, 3.0763273, 0.3841162, 3.0669699, -2.6679983, 2.6624625

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204161, upper bound: 1.8111603
time: 5.12 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204162, upper bound: 1.8240481
time: 4.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8195127, upper bound: 1.8195129
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8195127, upper bound: 1.8195143
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8186726, upper bound: 1.8094301
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8221932, upper bound: 1.8221923
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8213782, upper bound: 1.8195147
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8213781, upper bound: 1.8195126
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8204162, upper bound: 1.8094300
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8240483, upper bound: 1.8221923
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8195127, upper bound: 1.8213771
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8195128, upper bound: 1.8213771
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8195128, upper bound: 1.8240559
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8195128, upper bound: 1.8240561
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8094294, upper bound: 1.8204163
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8094294, upper bound: 1.8204161
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8204161, upper bound: 1.8111603
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.33
Output dim: 9, lower bound: -1.8204162, upper bound: 1.8240481

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.6915264, -5.0670991, -3.6408319, 3.6408324
1: -15.0896091, -10.8477516, -15.0896091, -10.8477516, -3.7595396, 3.7595394
2: -9.0583544, -5.7693434, -9.0583544, -5.7693434, -3.0309439, 3.0309439
3: -11.5208578, -7.4093418, -11.5208578, -7.4093418, -4.0387836, 4.0387826
4: -5.4604731, -1.9576910, -5.4604731, -1.9576910, -3.4233160, 3.4233155
5: -3.5672817, -0.4970436, -3.5672817, -0.4970436, -2.9183483, 2.9183488
6: -11.5796518, -6.9717102, -11.5796518, -6.9717102, -3.9136763, 3.9136772
7: -2.8023047, 0.8251872, -2.8023047, 0.8251872, -3.5715809, 3.5715809
8: -5.0730295, -1.4757953, -5.0730295, -1.4757953, -3.1843309, 3.1843309
9: 0.4401379, 3.0517111, 0.4401379, 3.0517111, -2.5569780, 2.5569777

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8066408, upper bound: 1.8159046
time: 5.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195057, upper bound: 1.8195061
time: 5.16 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.7457018, -5.0585723, -3.6509852, 3.6975007
1: -15.0896091, -10.8477516, -15.1223698, -10.8002434, -3.8078842, 3.7949193
2: -9.0583544, -5.7693434, -9.0741844, -5.7359200, -3.0864654, 3.0578880
3: -11.5208578, -7.4093418, -11.5605087, -7.3892784, -4.0585098, 4.0825043
4: -5.4604731, -1.9576910, -5.4986024, -1.8870873, -3.4841709, 3.4634581
5: -3.5672817, -0.4970436, -3.5936368, -0.4564457, -2.9464722, 2.9437621
6: -11.5796518, -6.9717102, -11.5938129, -6.9499998, -3.9527521, 3.9452367
7: -2.8023047, 0.8251872, -2.8553586, 0.8392324, -3.5868821, 3.6269035
8: -5.0730295, -1.4757953, -5.1025443, -1.4501300, -3.2176952, 3.2300177
9: 0.4401379, 3.0517111, 0.3883085, 3.0646622, -2.5710213, 2.6130075

Time for backsubstitution: 15.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8066408, upper bound: 1.8159062
time: 6.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195057, upper bound: 1.8195082
time: 5.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7438335, -5.0694909, -9.6663446, -5.1665711, -3.5951433, 3.6008794
1: -15.1205254, -10.8033028, -15.0539284, -10.8756962, -3.7768145, 3.7539549
2: -9.0689754, -5.7386665, -9.0098896, -5.7996836, -3.0269861, 3.0322409
3: -11.5571041, -7.3922467, -11.4896059, -7.4439020, -4.0402861, 4.0259895
4: -5.4950790, -1.8906591, -5.4243698, -1.9902948, -3.4477134, 3.4360712
5: -3.5909855, -0.4549723, -3.5385904, -0.5064659, -2.9317350, 2.9191699
6: -11.5915833, -6.9551854, -11.5495987, -7.0200167, -3.8973894, 3.9018099
7: -2.8519430, 0.8368635, -2.7651596, 0.8011947, -3.5976100, 3.5593524
8: -5.0973692, -1.4519229, -5.0106921, -1.4988379, -3.1974344, 3.1539092
9: 0.3988481, 3.0635276, 0.5460753, 3.0406203, -2.5694997, 2.4591854

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8098899, upper bound: 1.8094241
time: 5.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8186664, upper bound: 1.8094243
time: 5.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7460480, -5.0584192, -9.6923895, -5.0648460, -3.6484795, 3.6635058
1: -15.1247530, -10.8002377, -15.0922213, -10.8474197, -3.8195896, 3.8112390
2: -9.0743618, -5.7355871, -9.0594215, -5.7682400, -3.0702224, 3.0729198
3: -11.5606031, -7.3888259, -11.5214577, -7.4080462, -4.0843563, 4.0820398
4: -5.4992199, -1.8870091, -5.4654274, -1.9569907, -3.4797869, 3.4883966
5: -3.5939014, -0.4538774, -3.5687096, -0.4961615, -2.9475989, 2.9497001
6: -11.5945415, -6.9498158, -11.5807705, -6.9714117, -3.9238234, 3.9578395
7: -2.8556561, 0.8396807, -2.8031969, 0.8274608, -3.6341462, 3.6113319
8: -5.1042891, -1.4498472, -5.0741291, -1.4750643, -3.2445803, 3.1885049
9: 0.3871465, 3.0646892, 0.4387264, 3.0539279, -2.6245310, 2.5746450

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8094305, upper bound: 1.8186723
time: 5.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8094305, upper bound: 1.8221924
time: 6.90 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.7334185, -5.0549893, -3.6542745, 3.6826668
1: -15.0896091, -10.8477516, -15.1286526, -10.8250637, -3.7820477, 3.8010831
2: -9.0583544, -5.7693434, -9.0971193, -5.7395234, -3.0657167, 3.0768900
3: -11.5208578, -7.4093418, -11.5884438, -7.3770781, -4.0656638, 4.1068220
4: -5.4604731, -1.9576910, -5.5128527, -1.8455058, -3.5050764, 3.4743342
5: -3.5672817, -0.4970436, -3.6015487, -0.4563608, -2.9438577, 2.9563639
6: -11.5796518, -6.9717102, -11.5963335, -6.9421077, -3.9425464, 3.9322209
7: -2.8023047, 0.8251872, -2.8275809, 0.8813424, -3.6287508, 3.5994802
8: -5.0730295, -1.4757953, -5.0979142, -1.4525871, -3.2067242, 3.2129860
9: 0.4401379, 3.0517111, 0.3571882, 3.0741472, -2.5841627, 2.6460085

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8083740, upper bound: 1.8159046
time: 4.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8213694, upper bound: 1.8195069
time: 4.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.7876139, -5.0466785, -3.6644082, 3.7226601
1: -15.0896091, -10.8477516, -15.1617699, -10.7778406, -3.8300257, 3.8366849
2: -9.0583544, -5.7693434, -9.1145000, -5.7063303, -3.1210303, 3.1039691
3: -11.5208578, -7.4093418, -11.6285610, -7.3574991, -4.0853500, 4.1282825
4: -5.4604731, -1.9576910, -5.5502605, -1.7750721, -3.5262804, 3.5140843
5: -3.5672817, -0.4970436, -3.6278422, -0.4160423, -2.9552164, 2.9825163
6: -11.5796518, -6.9717102, -11.6107731, -6.9201183, -3.9775190, 3.9638257
7: -2.8023047, 0.8251872, -2.8808923, 0.8953681, -3.6439905, 3.6549535
8: -5.0730295, -1.4757953, -5.1272721, -1.4269066, -3.2402506, 3.2585001
9: 0.4401379, 3.0517111, 0.3055973, 3.0870466, -2.5972266, 2.6714487

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8083740, upper bound: 1.8159039
time: 7.11 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8213694, upper bound: 1.8195057
time: 6.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7438335, -5.0694909, -9.7082729, -5.1544166, -3.6086149, 3.6247015
1: -15.1205254, -10.8033028, -15.0922861, -10.8529520, -3.7993464, 3.7906725
2: -9.0689754, -5.7386665, -9.0486259, -5.7698059, -3.0620775, 3.0695817
3: -11.5571041, -7.3922467, -11.5572805, -7.4114013, -4.0671277, 4.0871453
4: -5.4950790, -1.8906591, -5.4759641, -1.8787701, -3.5177522, 3.4862585
5: -3.5909855, -0.4549723, -3.5723045, -0.4659157, -2.9538741, 2.9572086
6: -11.5915833, -6.9551854, -11.5663691, -6.9903345, -3.9263372, 3.9197428
7: -2.8519430, 0.8368635, -2.7901053, 0.8567533, -3.6353350, 3.5870800
8: -5.0973692, -1.4519229, -5.0355582, -1.4757357, -3.2109332, 3.1823521
9: 0.3988481, 3.0635276, 0.4640474, 3.0631402, -2.5962269, 2.5458539

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8116357, upper bound: 1.8094245
time: 5.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204100, upper bound: 1.8094242
time: 5.31 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7460480, -5.0584192, -9.7342987, -5.0527439, -3.6619167, 3.7052941
1: -15.1247530, -10.8002377, -15.1312571, -10.8247204, -3.8421011, 3.8527985
2: -9.0743618, -5.7355871, -9.0981293, -5.7384076, -3.1050205, 3.1094186
3: -11.5606031, -7.3888259, -11.5890579, -7.3757181, -4.1112442, 4.1418033
4: -5.4992199, -1.8870091, -5.5177822, -1.8448077, -3.5492549, 3.5390263
5: -3.5939014, -0.4538774, -3.6030309, -0.4554844, -2.9691081, 2.9878972
6: -11.5945415, -6.9498158, -11.5974483, -6.9417968, -3.9528599, 3.9758735
7: -2.8556561, 0.8396807, -2.8284788, 0.8835917, -3.6705289, 3.6392488
8: -5.1042891, -1.4498472, -5.0990105, -1.4518547, -3.2669783, 3.2167463
9: 0.3871465, 3.0646892, 0.3557787, 3.0763273, -2.6516910, 2.6636956

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8111603, upper bound: 1.8186725
time: 6.46 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8111604, upper bound: 1.8221928
time: 5.15 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.7334185, -5.0549893, -9.6915264, -5.0670991, -3.6826668, 3.6542742
1: -15.1286526, -10.8250637, -15.0896091, -10.8477516, -3.8010836, 3.7820470
2: -9.0971193, -5.7395234, -9.0583544, -5.7693434, -3.0768890, 3.0657172
3: -11.5884438, -7.3770781, -11.5208578, -7.4093418, -4.1068225, 4.0656638
4: -5.5128527, -1.8455058, -5.4604731, -1.9576910, -3.4743338, 3.5050755
5: -3.6015487, -0.4563608, -3.5672817, -0.4970436, -2.9563637, 2.9438579
6: -11.5963335, -6.9421077, -11.5796518, -6.9717102, -3.9322214, 3.9425473
7: -2.8275809, 0.8813424, -2.8023047, 0.8251872, -3.5994797, 3.6287503
8: -5.0979142, -1.4525871, -5.0730295, -1.4757953, -3.2129860, 3.2067242
9: 0.3571882, 3.0741472, 0.4401379, 3.0517111, -2.6460083, 2.5841632

Time for backsubstitution: 15.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8066408, upper bound: 1.8176495
time: 4.87 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195057, upper bound: 1.8213696
time: 4.91 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.7334185, -5.0549893, -9.7334185, -5.0549893, -3.6953945, 3.6953940
1: -15.1286526, -10.8250637, -15.1286526, -10.8250637, -3.8200197, 3.8200192
2: -9.0971193, -5.7395234, -9.0971193, -5.7395234, -3.1114254, 3.1114254
3: -11.5884438, -7.3770781, -11.5884438, -7.3770781, -4.1344380, 4.1344385
4: -5.5128527, -1.8455058, -5.5128527, -1.8455058, -3.5495577, 3.5495572
5: -3.6015487, -0.4563608, -3.6015487, -0.4563608, -2.9835052, 2.9835055
6: -11.5963335, -6.9421077, -11.5963335, -6.9421077, -3.9553370, 3.9553370
7: -2.8275809, 0.8813424, -2.8275809, 0.8813424, -3.6737452, 3.6737447
8: -5.0979142, -1.4525871, -5.0979142, -1.4525871, -3.2328300, 3.2328300
9: 0.3571882, 3.0741472, 0.3571882, 3.0741472, -2.6741147, 2.6741147

Time for backsubstitution: 15.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8066408, upper bound: 1.8176492
time: 5.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195057, upper bound: 1.8213694
time: 4.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7876139, -5.0466785, -9.6915264, -5.0670991, -3.7226605, 3.6644084
1: -15.1617699, -10.7778406, -15.0896091, -10.8477516, -3.8366852, 3.8300254
2: -9.1145000, -5.7063303, -9.0583544, -5.7693434, -3.1039696, 3.1210299
3: -11.6285610, -7.3574991, -11.5208578, -7.4093418, -4.1282825, 4.0853505
4: -5.5502605, -1.7750721, -5.4604731, -1.9576910, -3.5140848, 3.5262809
5: -3.6278422, -0.4160423, -3.5672817, -0.4970436, -2.9825153, 2.9552164
6: -11.6107731, -6.9201183, -11.5796518, -6.9717102, -3.9638262, 3.9775186
7: -2.8808923, 0.8953681, -2.8023047, 0.8251872, -3.6549540, 3.6439905
8: -5.1272721, -1.4269066, -5.0730295, -1.4757953, -3.2585001, 3.2402511
9: 0.3055973, 3.0870466, 0.4401379, 3.0517111, -2.6714492, 2.5972271

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8066408, upper bound: 1.8204160
time: 4.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195057, upper bound: 1.8240480
time: 4.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7876139, -5.0466785, -9.7334185, -5.0549893, -3.7385578, 3.7056262
1: -15.1617699, -10.7778406, -15.1286526, -10.8250637, -3.8556223, 3.8679976
2: -9.1145000, -5.7063303, -9.0971193, -5.7395234, -3.1385050, 3.1580710
3: -11.6285610, -7.3574991, -11.5884438, -7.3770781, -4.1552033, 4.1508527
4: -5.5502605, -1.7750721, -5.5128527, -1.8455058, -3.5902433, 3.5792077
5: -3.6278422, -0.4160423, -3.6015487, -0.4563608, -3.0053482, 2.9948642
6: -11.6107731, -6.9201183, -11.5963335, -6.9421077, -3.9865127, 3.9945145
7: -2.8808923, 0.8953681, -2.8275809, 0.8813424, -3.6957626, 3.6889405
8: -5.1272721, -1.4269066, -5.0979142, -1.4525871, -3.2783432, 3.2663569
9: 0.3055973, 3.0870466, 0.3571882, 3.0741472, -2.7012212, 2.6878047

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8066408, upper bound: 1.8204155
time: 4.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8195057, upper bound: 1.8240479
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.7082729, -5.1544166, -9.7438335, -5.0694909, -3.6247015, 3.6086154
1: -15.0922861, -10.8529520, -15.1205254, -10.8033028, -3.7906718, 3.7993474
2: -9.0486259, -5.7698059, -9.0689754, -5.7386665, -3.0695815, 3.0620780
3: -11.5572805, -7.4114013, -11.5571041, -7.3922467, -4.0871453, 4.0671268
4: -5.4759641, -1.8787701, -5.4950790, -1.8906591, -3.4862585, 3.5177522
5: -3.5723045, -0.4659157, -3.5909855, -0.4549723, -2.9572082, 2.9538741
6: -11.5663691, -6.9903345, -11.5915833, -6.9551854, -3.9197426, 3.9263372
7: -2.7901053, 0.8567533, -2.8519430, 0.8368635, -3.5870795, 3.6353350
8: -5.0355582, -1.4757357, -5.0973692, -1.4519229, -3.1823521, 3.2109334
9: 0.4640474, 3.0631402, 0.3988481, 3.0635276, -2.5458536, 2.5962272

Time for backsubstitution: 15.19 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.567917585372925
rel_dist={9: [-1.824091324888559, 1.824091281312575]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928307, upper bound: 1.5911394
time: 5.48 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928307, upper bound: 1.5928290
time: 5.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.90 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.90
Output dim: 9, lower bound: -1.5928307, upper bound: 1.5911394
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.90
Output dim: 9, lower bound: -1.5928307, upper bound: 1.5928290

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.6942806, -5.0616770, -3.3064122, 3.3050745
1: -15.0922241, -10.8474169, -15.0952425, -10.8431473, -3.4361458, 3.4346473
2: -9.0594254, -5.7682366, -9.0615978, -5.7651587, -2.8471918, 2.8456895
3: -11.5214615, -7.4080415, -11.5230656, -7.4036608, -3.7644768, 3.7617102
4: -5.4654369, -1.9569887, -5.4777827, -1.9554484, -3.2660613, 3.2782621
5: -3.5687127, -0.4961605, -3.5736499, -0.4953117, -2.6985550, 2.7034261
6: -11.5807753, -6.9714108, -11.5837259, -6.9704914, -3.5898128, 3.5920281
7: -2.8032007, 0.8274651, -2.8098021, 0.8292036, -3.4126401, 3.4180951
8: -5.0741324, -1.4750624, -5.0775828, -1.4732656, -2.9134007, 2.9148233
9: 0.4387226, 3.0539322, 0.4356761, 3.0562549, -2.4394650, 2.4383357

Time for backsubstitution: 15.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911407, upper bound: 1.5911393
time: 5.41 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911407, upper bound: 1.5911396
time: 5.26 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7338886, -5.0528698, -9.6942778, -5.0616894, -3.3477120, 3.3307507
1: -15.1308260, -10.8254185, -15.0952330, -10.8431702, -3.4771128, 3.4568005
2: -9.0964851, -5.7389593, -9.0615883, -5.7651682, -2.8908453, 2.8799231
3: -11.5864315, -7.3764930, -11.5230608, -7.4036779, -3.8263712, 3.7879610
4: -5.5169420, -1.8470776, -5.4777412, -1.9554546, -3.3366270, 3.3523190
5: -3.6025052, -0.4556732, -3.5736322, -0.4953117, -2.7473593, 2.7250516
6: -11.5967216, -6.9421320, -11.5837164, -6.9704962, -3.6086035, 3.6205893
7: -2.8274064, 0.8817344, -2.8097758, 0.8291955, -3.4393210, 3.4847605
8: -5.0984640, -1.4521961, -5.0775695, -1.4732714, -2.9407253, 2.9368477
9: 0.3572431, 3.0751038, 0.4356866, 3.0562463, -2.5230179, 2.4868004

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902827, upper bound: 1.5928151
time: 5.60 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928136, upper bound: 1.5928125
time: 5.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.17 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.17
Output dim: 9, lower bound: -1.5911407, upper bound: 1.5911393
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.17
Output dim: 9, lower bound: -1.5911407, upper bound: 1.5911396
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.17
Output dim: 9, lower bound: -1.5902827, upper bound: 1.5928151
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.17
Output dim: 9, lower bound: -1.5928136, upper bound: 1.5928125

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.6923923, -5.0648394, -3.3016663, 3.3016653
1: -15.0922241, -10.8474169, -15.0922241, -10.8474169, -3.4316320, 3.4316325
2: -9.0594254, -5.7682366, -9.0594254, -5.7682366, -2.8441739, 2.8441744
3: -11.5214615, -7.4080415, -11.5214615, -7.4080415, -3.7596903, 3.7596917
4: -5.4654369, -1.9569887, -5.4654369, -1.9569887, -3.2637620, 3.2637620
5: -3.5687127, -0.4961605, -3.5687127, -0.4961605, -2.6975241, 2.6975250
6: -11.5807753, -6.9714108, -11.5807753, -6.9714108, -3.5887313, 3.5887308
7: -2.8032007, 0.8274651, -2.8032007, 0.8274651, -3.4090424, 3.4090419
8: -5.0741324, -1.4750624, -5.0741324, -1.4750624, -2.9117031, 2.9117033
9: 0.4387226, 3.0539322, 0.4387226, 3.0539322, -2.4351926, 2.4351919

Time for backsubstitution: 15.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911238, upper bound: 1.5885940
time: 10.18 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911238, upper bound: 1.5911246
time: 10.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.7332840, -5.0529642, -3.3147278, 3.3423822
1: -15.0922241, -10.8474169, -15.1304636, -10.8259583, -3.4533443, 3.4721665
2: -9.0594254, -5.7682366, -9.0954428, -5.7394567, -2.8778582, 2.8863890
3: -11.5214615, -7.4080415, -11.5844545, -7.3770170, -3.7855330, 3.8199224
4: -5.4654369, -1.9569887, -5.5163679, -1.8490163, -3.3364868, 3.3135676
5: -3.5687127, -0.4961605, -3.6021209, -0.4560194, -2.7188377, 2.7344565
6: -11.5807753, -6.9714108, -11.5960646, -6.9423647, -3.6169419, 3.6049795
7: -2.8032007, 0.8274651, -2.8266850, 0.8804860, -3.4629068, 3.4349174
8: -5.0741324, -1.4750624, -5.0980043, -1.4524732, -2.9334469, 2.9379547
9: 0.4387226, 3.0539322, 0.3583713, 3.0743384, -2.4596918, 2.5174484

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911238, upper bound: 1.5885917
time: 6.83 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911238, upper bound: 1.5911222
time: 6.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.7337856, -5.0528741, -9.6934032, -5.0639515, -3.3454213, 3.3298042
1: -15.1307964, -10.8254528, -15.0926113, -10.8435097, -3.4767175, 3.4533901
2: -9.0964699, -5.7390060, -9.0605173, -5.7662768, -2.8896799, 2.8782952
3: -11.5863228, -7.3765073, -11.5224552, -7.4049797, -3.8244171, 3.7872586
4: -5.5169268, -1.8472395, -5.4727659, -1.9561635, -3.3358579, 3.3469617
5: -3.6024871, -0.4557390, -3.5722027, -0.4962025, -2.7459240, 2.7231569
6: -11.5966587, -6.9421396, -11.5826035, -6.9708014, -3.6076112, 3.6189628
7: -2.8273873, 0.8816991, -2.8088799, 0.8269172, -3.4365826, 3.4837952
8: -5.0984268, -1.4522176, -5.0764651, -1.4740076, -2.9397507, 2.9352195
9: 0.3573103, 3.0750971, 0.4371061, 3.0540218, -2.5207419, 2.4854631

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902827, upper bound: 1.5902837
time: 5.39 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902827, upper bound: 1.5928152
time: 5.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.7338829, -5.0528774, -9.7476940, -5.0553837, -3.3639522, 3.3715491
1: -15.1308165, -10.8254204, -15.1259909, -10.7960339, -3.5168633, 3.5070591
2: -9.0964813, -5.7389631, -9.0762844, -5.7328062, -2.9366522, 2.9150929
3: -11.5864296, -7.3764968, -11.5621958, -7.3848467, -3.8417897, 3.8311095
4: -5.5169301, -1.8470814, -5.5110426, -1.8854862, -3.3729630, 3.3907461
5: -3.6024990, -0.4556751, -3.5985138, -0.4549227, -2.7617331, 2.7450900
6: -11.5967159, -6.9421349, -11.5970240, -6.9490318, -3.6474085, 3.6510801
7: -2.8274040, 0.8817248, -2.8620307, 0.8411093, -3.4715910, 3.5056841
8: -5.0984607, -1.4522014, -5.1064324, -1.4482503, -2.9733887, 2.9860647
9: 0.3572497, 3.0750978, 0.3849778, 3.0669465, -2.5353441, 2.5189445

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863623, upper bound: 1.5913326
time: 5.15 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928075, upper bound: 1.5928069
time: 5.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.85 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 9, lower bound: -1.5911238, upper bound: 1.5885940
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 9, lower bound: -1.5911238, upper bound: 1.5911246
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 9, lower bound: -1.5911238, upper bound: 1.5885917
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 9, lower bound: -1.5911238, upper bound: 1.5911222
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 9, lower bound: -1.5902827, upper bound: 1.5902837
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 9, lower bound: -1.5902827, upper bound: 1.5928152
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 9, lower bound: -1.5863623, upper bound: 1.5913326
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 25.85
Output dim: 9, lower bound: -1.5928075, upper bound: 1.5928069

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.6923923, -5.0648394, -3.3007488, 3.2994704
1: -15.0896091, -10.8477516, -15.0922241, -10.8474169, -3.4282761, 3.4312701
2: -9.0583544, -5.7693434, -9.0594254, -5.7682366, -2.8426156, 2.8430314
3: -11.5208578, -7.4093418, -11.5214615, -7.4080415, -3.7590036, 3.7575440
4: -5.4604731, -1.9576910, -5.4654369, -1.9569887, -3.2582183, 3.2630110
5: -3.5672817, -0.4970436, -3.5687127, -0.4961605, -2.6956911, 2.6961179
6: -11.5796518, -6.9717102, -11.5807753, -6.9714108, -3.5871301, 3.5878472
7: -2.8023047, 0.8251872, -2.8032007, 0.8274651, -3.4081268, 3.4063325
8: -5.0730295, -1.4757953, -5.0741324, -1.4750624, -2.9101028, 2.9107945
9: 0.4401379, 3.0517111, 0.4387226, 3.0539322, -2.4338651, 2.4329953

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5885926, upper bound: 1.5885937
time: 5.64 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5885925, upper bound: 1.5885945
time: 5.13 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.7457924, -5.0585303, -9.6923895, -5.0648446, -3.3532743, 3.3179100
1: -15.1229973, -10.8002377, -15.0922174, -10.8474159, -3.4818912, 3.4796069
2: -9.0742321, -5.7358327, -9.0594244, -5.7682409, -2.8792953, 2.8990159
3: -11.5605345, -7.3891582, -11.5214586, -7.4080462, -3.8028135, 3.7820945
4: -5.4987645, -1.8870655, -5.4654212, -1.9569918, -3.3122711, 3.3180306
5: -3.5937066, -0.4557724, -3.5687077, -0.4961624, -2.7211409, 2.7215302
6: -11.5940046, -6.9499516, -11.5807705, -6.9714079, -3.6192026, 3.6290686
7: -2.8554366, 0.8393497, -2.8031988, 0.8274589, -3.4635191, 3.4412785
8: -5.1030016, -1.4500561, -5.0741305, -1.4750628, -2.9663768, 2.9443438
9: 0.3880038, 3.0646696, 0.4387264, 3.0539250, -2.4875319, 2.4476759

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5897590, upper bound: 1.5847387
time: 4.92 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911181, upper bound: 1.5911186
time: 8.22 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.7331839, -5.0529709, -3.3137865, 3.3400958
1: -15.0896091, -10.8477516, -15.1304359, -10.8259935, -3.4499369, 3.4717708
2: -9.0583544, -5.7693434, -9.0954285, -5.7395048, -2.8762331, 2.8852272
3: -11.5208578, -7.4093418, -11.5843391, -7.3770308, -3.7848358, 3.8179650
4: -5.4604731, -1.9576910, -5.5163507, -1.8491858, -3.3311229, 3.3127990
5: -3.5672817, -0.4970436, -3.6021037, -0.4560900, -2.7169409, 2.7330289
6: -11.5796518, -6.9717102, -11.5959969, -6.9423714, -3.6152902, 3.6040249
7: -2.8023047, 0.8251872, -2.8266630, 0.8804507, -3.4619389, 3.4321823
8: -5.0730295, -1.4757953, -5.0979671, -1.4524932, -2.9318242, 2.9369764
9: 0.4401379, 3.0517111, 0.3584399, 3.0743318, -2.4583559, 2.5151737

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902825, upper bound: 1.5885938
time: 5.83 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902825, upper bound: 1.5885924
time: 8.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.7457924, -5.0585303, -9.7332802, -5.0529714, -3.3663683, 3.3586261
1: -15.1229973, -10.8002377, -15.1304560, -10.8259611, -3.5036039, 3.5120595
2: -9.0742321, -5.7358327, -9.0954390, -5.7394619, -2.9129782, 2.9325454
3: -11.5605345, -7.3891582, -11.5844536, -7.3770213, -3.8286552, 3.8353496
4: -5.4987645, -1.8870655, -5.5163555, -1.8490200, -3.3748317, 3.3674209
5: -3.5937066, -0.4557724, -3.6021175, -0.4560223, -2.7389450, 2.7587581
6: -11.5940046, -6.9499516, -11.5960598, -6.9423623, -3.6474142, 3.6452949
7: -2.8554366, 0.8393497, -2.8266816, 0.8804774, -3.4938564, 3.4671550
8: -5.1030016, -1.4500561, -5.0980010, -1.4524760, -2.9827170, 2.9705944
9: 0.3880038, 3.0646696, 0.3583760, 3.0743334, -2.5118847, 2.5297356

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5913332, upper bound: 1.5847387
time: 4.91 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928083, upper bound: 1.5911183
time: 5.24 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.7329006, -5.0551271, -9.6934032, -5.0639515, -3.3445544, 3.3276088
1: -15.1281881, -10.8257942, -15.0926113, -10.8435097, -3.4733453, 3.4530225
2: -9.0954571, -5.7401257, -9.0605173, -5.7662768, -2.8881197, 2.8771288
3: -11.5857029, -7.3778706, -11.5224552, -7.4049797, -3.8237371, 3.7851071
4: -5.5119886, -1.8479432, -5.4727659, -1.9561635, -3.3303113, 3.3462446
5: -3.6010041, -0.4566202, -3.5722027, -0.4962025, -2.7441921, 2.7217357
6: -11.5955429, -6.9424500, -11.5826035, -6.9708014, -3.6064830, 3.6180744
7: -2.8264871, 0.8794427, -2.8088799, 0.8269172, -3.4356594, 3.4810607
8: -5.0973272, -1.4529490, -5.0764651, -1.4740076, -2.9381523, 2.9343066
9: 0.3587279, 3.0729144, 0.4371061, 3.0540218, -2.5194225, 2.4833162

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5888759, upper bound: 1.5838346
time: 5.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902770, upper bound: 1.5902776
time: 5.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7867823, -5.0469470, -9.6934032, -5.0639515, -3.3786211, 3.3375957
1: -15.1591663, -10.7785854, -15.0926113, -10.8435097, -3.5065265, 3.5009885
2: -9.1126804, -5.7072487, -9.0605173, -5.7662768, -2.9142580, 2.9308386
3: -11.6257267, -7.3586783, -11.5224552, -7.4049797, -3.8443136, 3.8038831
4: -5.5488253, -1.7775922, -5.4727659, -1.9561635, -3.3703475, 3.3673694
5: -3.6270323, -0.4185982, -3.5722027, -0.4962025, -2.7692027, 2.7322772
6: -11.6092148, -6.9206352, -11.5826035, -6.9708014, -3.6359029, 3.6461551
7: -2.8795097, 0.8930650, -2.8088799, 0.8269172, -3.4908695, 3.4956884
8: -5.1251278, -1.4275270, -5.0764651, -1.4740076, -2.9811907, 2.9671850
9: 0.3081713, 3.0857964, 0.4371061, 3.0540218, -2.5425458, 2.4969883

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5888759, upper bound: 1.5863620
time: 5.39 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902770, upper bound: 1.5928091
time: 6.24 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7078571, -5.1545525, -9.7431364, -5.0780787, -3.2654123, 3.2642493
1: -15.0918465, -10.8536530, -15.1173744, -10.8023100, -3.4505520, 3.4592698
2: -9.0469723, -5.7703719, -9.0652580, -5.7391386, -2.8759365, 2.8653643
3: -11.5546598, -7.4121914, -11.5550404, -7.3918891, -3.7987709, 3.7821503
4: -5.4751177, -1.8810186, -5.5025311, -1.8929678, -3.3154087, 3.3495092
5: -3.5717769, -0.4661045, -3.5924997, -0.4571657, -2.7290068, 2.7285633
6: -11.5656395, -6.9906740, -11.5909252, -6.9599953, -3.5800114, 3.5968785
7: -2.7890291, 0.8549008, -2.8543570, 0.8353348, -3.4140182, 3.4686592
8: -5.0350127, -1.4760828, -5.0922484, -1.4525480, -2.9046245, 2.9172652
9: 0.4655428, 3.0619116, 0.4089417, 3.0645401, -2.4175129, 2.4561052

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863593, upper bound: 1.5868782
time: 5.26 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863593, upper bound: 1.5913293
time: 5.29 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7338829, -5.0528831, -9.7476940, -5.0553837, -3.3627262, 3.3114090
1: -15.1308184, -10.8254223, -15.1259909, -10.7960339, -3.5168614, 3.5080523
2: -9.0964794, -5.7389660, -9.0762844, -5.7328062, -2.9150839, 2.9146471
3: -11.5864286, -7.3764992, -11.5621958, -7.3848467, -3.8518429, 3.8304629
4: -5.5169282, -1.8470823, -5.5110426, -1.8854862, -3.3718700, 3.3818617
5: -3.6024985, -0.4556751, -3.5985138, -0.4549227, -2.7606859, 2.7454891
6: -11.5967150, -6.9421368, -11.5970240, -6.9490318, -3.6455021, 3.6214879
7: -2.8274035, 0.8817225, -2.8620307, 0.8411093, -3.4711943, 3.5051177
8: -5.0984573, -1.4522009, -5.1064324, -1.4482503, -2.9350200, 2.9838295
9: 0.3572540, 3.0750973, 0.3849778, 3.0669465, -2.5353413, 2.5249863

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5913328, upper bound: 1.5863638
time: 5.02 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5913328, upper bound: 1.5863619
time: 5.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.45 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5885926, upper bound: 1.5885937
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5885925, upper bound: 1.5885945
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5897590, upper bound: 1.5847387
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5911181, upper bound: 1.5911186
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5902825, upper bound: 1.5885938
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5902825, upper bound: 1.5885924
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5913332, upper bound: 1.5847387
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5928083, upper bound: 1.5911183
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5888759, upper bound: 1.5838346
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5902770, upper bound: 1.5902776
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5888759, upper bound: 1.5863620
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5902770, upper bound: 1.5928091
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5863593, upper bound: 1.5868782
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5863593, upper bound: 1.5913293
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5913328, upper bound: 1.5863638
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.45
Output dim: 9, lower bound: -1.5913328, upper bound: 1.5863619

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.6915264, -5.0670991, -3.2985544, 3.2985549
1: -15.0896091, -10.8477516, -15.0896091, -10.8477516, -3.4279137, 3.4279139
2: -9.0583544, -5.7693434, -9.0583544, -5.7693434, -2.8414726, 2.8414731
3: -11.5208578, -7.4093418, -11.5208578, -7.4093418, -3.7568564, 3.7568574
4: -5.4604731, -1.9576910, -5.4604731, -1.9576910, -3.2574673, 3.2574663
5: -3.5672817, -0.4970436, -3.5672817, -0.4970436, -2.6942835, 2.6942835
6: -11.5796518, -6.9717102, -11.5796518, -6.9717102, -3.5862455, 3.5862465
7: -2.8023047, 0.8251872, -2.8023047, 0.8251872, -3.4054165, 3.4054165
8: -5.0730295, -1.4757953, -5.0730295, -1.4757953, -2.9091930, 2.9091933
9: 0.4401379, 3.0517111, 0.4401379, 3.0517111, -2.4316685, 2.4316685

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5822117, upper bound: 1.5873012
time: 5.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5885866, upper bound: 1.5885882
time: 5.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.7453823, -5.0587049, -3.3084641, 3.3506248
1: -15.0896091, -10.8477516, -15.1202345, -10.8003330, -3.4761553, 3.4608707
2: -9.0583544, -5.7693434, -9.0739965, -5.7362223, -2.8953876, 2.8674464
3: -11.5208578, -7.4093418, -11.5604286, -7.3896976, -3.7756562, 3.8002510
4: -5.4604731, -1.9576910, -5.4980412, -1.8871609, -3.3127003, 3.2969408
5: -3.5672817, -0.4970436, -3.5933952, -0.4587517, -2.7178979, 2.7195513
6: -11.5796518, -6.9717102, -11.5931530, -6.9501691, -3.6241465, 3.6160364
7: -2.8023047, 0.8251872, -2.8550863, 0.8388319, -3.4201436, 3.4604750
8: -5.0730295, -1.4757953, -5.1009674, -1.4503856, -2.9419093, 2.9523735
9: 0.4401379, 3.0517111, 0.3893466, 3.0646329, -2.4456873, 2.4844513

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5822117, upper bound: 1.5872989
time: 5.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5885866, upper bound: 1.5885860
time: 5.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7412291, -5.0812283, -9.6663437, -5.1665764, -3.2459383, 3.2372155
1: -15.1143837, -10.8065138, -15.0539227, -10.8756933, -3.4340734, 3.4185061
2: -9.0632076, -5.7421684, -9.0098896, -5.7996864, -2.8292866, 2.8375118
3: -11.5533762, -7.3962059, -11.4896049, -7.4439011, -3.7538958, 3.7390728
4: -5.4902592, -1.8945446, -5.4243641, -1.9902965, -3.2713985, 3.2608116
5: -3.5876906, -0.4580164, -3.5385878, -0.5064697, -2.7043886, 2.6889548
6: -11.5879087, -6.9609184, -11.5495987, -7.0200157, -3.5649290, 3.5616260
7: -2.8477650, 0.8335786, -2.7651587, 0.8011904, -3.4263172, 3.3839054
8: -5.0888147, -1.4543471, -5.0106912, -1.4988384, -2.9010496, 2.8757777
9: 0.4119601, 3.0622656, 0.5460796, 3.0406165, -2.4249055, 2.3308144

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5853060, upper bound: 1.5847372
time: 5.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5897558, upper bound: 1.5847359
time: 5.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7457924, -5.0585303, -9.6923904, -5.0648508, -3.2931342, 3.3179083
1: -15.1229973, -10.8002377, -15.0922184, -10.8474178, -3.4828849, 3.4796042
2: -9.0742321, -5.7358327, -9.0594206, -5.7682414, -2.8788471, 2.8775790
3: -11.5605345, -7.3891582, -11.5214596, -7.4080477, -3.8021622, 3.7934895
4: -5.4987645, -1.8870655, -5.4654217, -1.9569917, -3.3047028, 3.3169382
5: -3.5937066, -0.4557724, -3.5687065, -0.4961615, -2.7220397, 2.7205226
6: -11.5940046, -6.9499516, -11.5807686, -6.9714117, -3.5894756, 3.6271591
7: -2.8554366, 0.8393497, -2.8031960, 0.8274574, -3.4641991, 3.4408722
8: -5.1030016, -1.4500561, -5.0741272, -1.4750652, -2.9662261, 2.9064007
9: 0.3880038, 3.0646696, 0.4387298, 3.0539248, -2.4935899, 2.4476717

Time for backsubstitution: 15.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847398, upper bound: 1.5897579
time: 5.28 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847397, upper bound: 1.5911177
time: 6.29 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.7322922, -5.0552263, -3.3115940, 3.3392251
1: -15.0896091, -10.8477516, -15.1278248, -10.8263378, -3.4495673, 3.4683974
2: -9.0583544, -5.7693434, -9.0944166, -5.7406273, -2.8750658, 2.8836646
3: -11.5208578, -7.4093418, -11.5837164, -7.3783956, -3.7826858, 3.8172817
4: -5.4604731, -1.9576910, -5.5114136, -1.8498976, -3.3303814, 3.3072128
5: -3.5672817, -0.4970436, -3.6006236, -0.4569731, -2.7155190, 2.7312753
6: -11.5796518, -6.9717102, -11.5948830, -6.9426842, -3.6143994, 3.6028857
7: -2.8023047, 0.8251872, -2.8257647, 0.8781943, -3.4592328, 3.4312611
8: -5.0730295, -1.4757953, -5.0968666, -1.4532237, -2.9309120, 2.9353757
9: 0.4401379, 3.0517111, 0.3598580, 3.0721493, -2.4561844, 2.5138528

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838351, upper bound: 1.5873011
time: 5.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902764, upper bound: 1.5885883
time: 5.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.7861557, -5.0470428, -3.3214726, 3.3733394
1: -15.0896091, -10.8477516, -15.1587896, -10.7792225, -3.4974270, 3.5015588
2: -9.0583544, -5.7693434, -9.1116085, -5.7077708, -2.9287472, 2.9097705
3: -11.5208578, -7.4093418, -11.6237221, -7.3592129, -3.8014359, 3.8378105
4: -5.4604731, -1.9576910, -5.5482440, -1.7795750, -3.3513184, 3.3463120
5: -3.5672817, -0.4970436, -3.6266298, -0.4189644, -2.7260737, 2.7572770
6: -11.5796518, -6.9717102, -11.6085281, -6.9208732, -3.6424627, 3.6327095
7: -2.8023047, 0.8251872, -2.8787699, 0.8917923, -3.4738665, 3.4864612
8: -5.0730295, -1.4757953, -5.1246400, -1.4278097, -2.9637804, 2.9783721
9: 0.4401379, 3.0517111, 0.3093262, 3.0850301, -2.4692278, 2.5369456

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838351, upper bound: 1.5872989
time: 5.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5902764, upper bound: 1.5885890
time: 6.46 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7412291, -5.0812283, -9.7072573, -5.1546507, -3.2590628, 3.2601748
1: -15.1143837, -10.8065138, -15.0914927, -10.8541956, -3.4558077, 3.4457574
2: -9.0632076, -5.7421684, -9.0459280, -5.7708731, -2.8632488, 2.8718264
3: -11.5533762, -7.3962059, -11.5526848, -7.4127202, -3.7796869, 3.7923255
4: -5.4902592, -1.8945446, -5.4745507, -1.8829449, -3.3337097, 3.3097610
5: -3.5876906, -0.4580164, -3.5713997, -0.4664497, -2.7224064, 2.7259915
6: -11.5879087, -6.9609184, -11.5649815, -6.9909077, -3.5932117, 3.5779083
7: -2.8477650, 0.8335786, -2.7883062, 0.8536634, -3.4568462, 3.4095922
8: -5.0888147, -1.4543471, -5.0345569, -1.4763579, -2.9139090, 2.9018431
9: 0.4119601, 3.0622656, 0.4666820, 3.0611482, -2.4489841, 2.4119363

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5868789, upper bound: 1.5847373
time: 5.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5913300, upper bound: 1.5847356
time: 5.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7457924, -5.0585303, -9.7332821, -5.0529776, -3.3062277, 3.3574967
1: -15.1229973, -10.8002377, -15.1304550, -10.8259602, -3.5045977, 3.5120578
2: -9.0742321, -5.7358327, -9.0954380, -5.7394624, -2.9125328, 2.9109671
3: -11.5605345, -7.3891582, -11.5844536, -7.3770227, -3.8280096, 3.8453941
4: -5.4987645, -1.8870655, -5.5163546, -1.8490229, -3.3659520, 3.3663278
5: -3.5937066, -0.4557724, -3.6021156, -0.4560242, -2.7393446, 2.7577109
6: -11.5940046, -6.9499516, -11.5960588, -6.9423676, -3.6177926, 3.6433878
7: -2.8554366, 0.8393497, -2.8266802, 0.8804765, -3.4933186, 3.4667597
8: -5.1030016, -1.4500561, -5.0979996, -1.4524760, -2.9804811, 2.9322703
9: 0.3880038, 3.0646696, 0.3583813, 3.0743327, -2.5179260, 2.5297329

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863628, upper bound: 1.5897579
time: 5.08 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863627, upper bound: 1.5911176
time: 7.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.7283897, -5.0778275, -9.6673851, -5.1656833, -3.2371960, 3.2428372
1: -15.1194391, -10.8320732, -15.0543098, -10.8717976, -3.4254127, 3.3919580
2: -9.0844212, -5.7464266, -9.0109177, -5.7977295, -2.8380785, 2.8157325
3: -11.5785551, -7.3848791, -11.4906082, -7.4409084, -3.7743578, 3.7421079
4: -5.5034132, -1.8555316, -5.4316673, -1.9894820, -3.2894549, 3.2885022
5: -3.5948973, -0.4588985, -3.5420740, -0.5065069, -2.7274389, 2.6891799
6: -11.5894909, -6.9534039, -11.5514278, -7.0194006, -3.5521884, 3.5661259
7: -2.8187776, 0.8735580, -2.7708359, 0.8006363, -3.3984232, 3.4204574
8: -5.0831747, -1.4572415, -5.0130515, -1.4978175, -2.8973570, 2.8658004
9: 0.3828635, 3.0705600, 0.5445151, 3.0407114, -2.4554884, 2.3665099

Time for backsubstitution: 15.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5844235, upper bound: 1.5838316
time: 4.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5888728, upper bound: 1.5838320
time: 4.89 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.7329006, -5.0551271, -9.6934023, -5.0639563, -3.2844028, 3.3276083
1: -15.1281881, -10.8257942, -15.0926113, -10.8435078, -3.4743400, 3.4530199
2: -9.0954571, -5.7401257, -9.0605173, -5.7662787, -2.8876724, 2.8559577
3: -11.5857029, -7.3778706, -11.5224543, -7.4049821, -3.8218756, 3.7965374
4: -5.5119886, -1.8479432, -5.4727659, -1.9561645, -3.3222213, 3.3451514
5: -3.6010041, -0.4566202, -3.5722013, -0.4962015, -2.7450895, 2.7207296
6: -11.5955429, -6.9424500, -11.5826006, -6.9708014, -3.5767612, 3.6180725
7: -2.8264871, 0.8794427, -2.8088789, 0.8269162, -3.4362497, 3.4798195
8: -5.0973272, -1.4529490, -5.0764627, -1.4740071, -2.9380040, 2.8967514
9: 0.3587279, 3.0729144, 0.4371090, 3.0540214, -2.5254812, 2.4833121

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838354, upper bound: 1.5888769
time: 5.12 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838353, upper bound: 1.5888778
time: 6.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7822304, -5.0696397, -9.6673851, -5.1656833, -3.2712235, 3.2527883
1: -15.1504230, -10.7848587, -15.0543098, -10.8717976, -3.4586244, 3.4399264
2: -9.1016531, -5.7135940, -9.0109177, -5.7977295, -2.8609791, 2.8694558
3: -11.6185856, -7.3657036, -11.4906082, -7.4409084, -3.7949905, 3.7608495
4: -5.5401883, -1.7851996, -5.4316673, -1.9894820, -3.3294997, 3.3094788
5: -3.6209238, -0.4208679, -3.5420740, -0.5065069, -2.7524571, 2.6997306
6: -11.6031380, -6.9315853, -11.5514278, -7.0194006, -3.5815892, 3.5788021
7: -2.8718045, 0.8871799, -2.7708359, 0.8006363, -3.4536104, 3.4350219
8: -5.1109495, -1.4318438, -5.0130515, -1.4978175, -2.9262643, 2.8986216
9: 0.3323002, 3.0834157, 0.5445151, 3.0407114, -2.4784448, 2.3801510

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5844235, upper bound: 1.5863589
time: 5.23 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5888728, upper bound: 1.5863588
time: 4.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7867823, -5.0469470, -9.6934023, -5.0639563, -3.3184805, 3.3375952
1: -15.1591663, -10.7785854, -15.0926113, -10.8435078, -3.5075231, 3.5009854
2: -9.1126804, -5.7072487, -9.0605173, -5.7662787, -2.9138107, 2.9094133
3: -11.6257267, -7.3586783, -11.5224543, -7.4049821, -3.8424511, 3.8151960
4: -5.5488253, -1.7775922, -5.4727659, -1.9561645, -3.3617234, 3.3662772
5: -3.6270323, -0.4185982, -3.5722013, -0.4962015, -2.7695942, 2.7312710
6: -11.6092148, -6.9206352, -11.5826006, -6.9708014, -3.6061802, 3.6442444
7: -2.8795097, 0.8930650, -2.8088789, 0.8269162, -3.4913883, 3.4944470
8: -5.1251278, -1.4275270, -5.0764627, -1.4740071, -2.9810424, 2.9293275
9: 0.3081713, 3.0857964, 0.4371090, 3.0540214, -2.5486045, 2.4969842

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838354, upper bound: 1.5913337
time: 5.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838353, upper bound: 1.5928071
time: 8.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.7003975, -5.2003579, -9.7028465, -5.2601786, -3.0686021, 3.0729694
1: -15.0828114, -10.8602409, -15.0816813, -10.8308601, -3.4135647, 3.4104760
2: -9.0403175, -5.8112001, -9.0313587, -5.9012280, -2.7013903, 2.7069697
3: -11.5521202, -7.4514279, -11.5386772, -7.5476518, -3.6431866, 3.6756258
4: -5.4495025, -1.8858517, -5.3990054, -1.9178880, -3.2340016, 3.2361512
5: -3.5580597, -0.4684343, -3.5391946, -0.4690895, -2.6884665, 2.6680019
6: -11.5591574, -7.0494576, -11.5509481, -7.1943474, -3.3378401, 3.3605452
7: -2.7597318, 0.8497138, -2.7369881, 0.8084874, -3.3311319, 3.3447909
8: -5.0273166, -1.5125833, -5.0527353, -1.5983057, -2.7517128, 2.7717006
9: 0.4831409, 3.0583057, 0.4802632, 3.0470324, -2.3580272, 2.3748078

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838322, upper bound: 1.5844231
time: 5.27 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838322, upper bound: 1.5844224
time: 5.31 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.7078571, -5.1545525, -9.7431335, -5.0780897, -3.1232605, 3.2551548
1: -15.0918465, -10.8536530, -15.1173725, -10.8023129, -3.4505506, 3.4678473
2: -9.0469723, -5.7703719, -9.0652571, -5.7391443, -2.7802329, 2.8601062
3: -11.5546598, -7.4121914, -11.5550404, -7.3918953, -3.7652950, 3.7812610
4: -5.4751177, -1.8810186, -5.5025272, -1.8929687, -3.3277965, 3.3495076
5: -3.5717769, -0.4661045, -3.5924969, -0.4571667, -2.7271924, 2.7231116
6: -11.5656395, -6.9906740, -11.5909271, -6.9600029, -3.4191217, 3.5847461
7: -2.7890291, 0.8549008, -2.8543530, 0.8353329, -3.4230022, 3.4686577
8: -5.0350127, -1.4760828, -5.0922470, -1.4525523, -2.8719168, 2.9110603
9: 0.4655428, 3.0619116, 0.4089437, 3.0645401, -2.4147539, 2.4554236

Time for backsubstitution: 15.16 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.44260835647583
rel_dist={9: [-1.592849651897651, 1.5928489343896768]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147661, upper bound: 1.4142321
time: 4.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147661, upper bound: 1.4147652
time: 5.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.52 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.52
Output dim: 9, lower bound: -1.4147661, upper bound: 1.4142321
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.52
Output dim: 9, lower bound: -1.4147661, upper bound: 1.4147652

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.6931152, -5.0636296, -3.0752974, 3.0747745
1: -15.0922241, -10.8474169, -15.0933781, -10.8457947, -3.2122726, 3.2117021
2: -9.0594254, -5.7682366, -9.0602570, -5.7670708, -2.7190056, 2.7184401
3: -11.5214615, -7.4080415, -11.5220776, -7.4063749, -3.5735645, 3.5725155
4: -5.4654369, -1.9569887, -5.4701414, -1.9563961, -3.1540675, 3.1587424
5: -3.5687127, -0.4961605, -3.5705919, -0.4958344, -2.5485411, 2.5504055
6: -11.5807753, -6.9714108, -11.5818977, -6.9710603, -3.3708568, 3.3716760
7: -2.8032007, 0.8274651, -2.8057141, 0.8281331, -3.2996464, 3.3017201
8: -5.0741324, -1.4750624, -5.0754433, -1.4743752, -2.7289257, 2.7294571
9: 0.4387226, 3.0539322, 0.4375563, 3.0548153, -2.3532906, 2.3528562

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135328, upper bound: 1.4142312
time: 5.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147654, upper bound: 1.4142326
time: 4.87 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7316389, -5.0531878, -9.6942730, -5.0616975, -3.1173224, 3.0993121
1: -15.1295862, -10.8272877, -15.0952234, -10.8431835, -3.2544727, 3.2338016
2: -9.0930634, -5.7406940, -9.0615826, -5.7651758, -2.7597523, 2.7516012
3: -11.5797920, -7.3782063, -11.5230589, -7.4036922, -3.6302009, 3.5986395
4: -5.5150671, -1.8537853, -5.4777136, -1.9554579, -3.2182350, 3.2336786
5: -3.6012430, -0.4570122, -3.5736222, -0.4953146, -2.5938673, 2.5719490
6: -11.5943975, -6.9428940, -11.5837116, -6.9704976, -3.3858166, 3.4009767
7: -2.8250456, 0.8776355, -2.8097610, 0.8291945, -3.3258500, 3.3661535
8: -5.0968332, -1.4531322, -5.0775614, -1.4732752, -2.7534304, 2.7524140
9: 0.3610535, 3.0728159, 0.4356918, 3.0562398, -2.4315882, 2.3966188

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135328, upper bound: 1.4147641
time: 8.73 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147654, upper bound: 1.4147677
time: 6.11 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 30.17 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.17
Output dim: 9, lower bound: -1.4135328, upper bound: 1.4142312
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.17
Output dim: 9, lower bound: -1.4147654, upper bound: 1.4142326
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.17
Output dim: 9, lower bound: -1.4135328, upper bound: 1.4147641
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.17
Output dim: 9, lower bound: -1.4147654, upper bound: 1.4147677

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.6920090, -5.0658493, -9.6922436, -5.0658951, -3.0726986, 3.0728981
1: -15.0910730, -10.8475676, -15.0907621, -10.8461285, -3.2104654, 3.2081828
2: -9.0589504, -5.7687283, -9.0591869, -5.7681789, -2.7171822, 2.7163727
3: -11.5211935, -7.4086242, -11.5214739, -7.4076748, -3.5711107, 3.5708985
4: -5.4632382, -1.9573005, -5.4651737, -1.9571003, -3.1508899, 3.1528659
5: -3.5680788, -0.4965277, -3.5691614, -0.4967194, -2.5463161, 2.5479634
6: -11.5802650, -6.9715419, -11.5807800, -6.9713593, -3.3692913, 3.3697205
7: -2.8028021, 0.8264341, -2.8048177, 0.8258553, -3.2965274, 3.2996116
8: -5.0736690, -1.4753838, -5.0743380, -1.4751105, -2.7273126, 2.7274597
9: 0.4393415, 3.0529428, 0.4389744, 3.0525930, -2.3505120, 2.3505504

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4097622, upper bound: 1.4122214
time: 5.43 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135308, upper bound: 1.4142294
time: 4.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.6923876, -5.0648513, -9.7457066, -5.0574589, -3.0891237, 3.1216249
1: -15.0922127, -10.8474197, -15.1223621, -10.8024502, -3.2560477, 3.2585223
2: -9.0594215, -5.7682443, -9.0735693, -5.7351518, -2.7721872, 2.7507567
3: -11.5214567, -7.4080505, -11.5608873, -7.3884726, -3.5942221, 3.6148624
4: -5.4654131, -1.9569916, -5.5027199, -1.8870562, -3.2039471, 3.2027025
5: -3.5687034, -0.4961643, -3.5951595, -0.4572792, -2.5686407, 2.5736129
6: -11.5807657, -6.9714117, -11.5938940, -6.9498248, -3.4061050, 3.3996568
7: -2.8031955, 0.8274522, -2.8575377, 0.8395181, -3.3283615, 3.3556719
8: -5.0741282, -1.4750671, -5.1021986, -1.4497042, -2.7608194, 2.7772026
9: 0.4387307, 3.0539188, 0.3878641, 3.0653071, -2.3644197, 2.4007905

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4109922, upper bound: 1.4122239
time: 4.75 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147634, upper bound: 1.4142310
time: 4.67 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.7311153, -5.0542016, -9.6933975, -5.0639620, -3.1146202, 3.0973890
1: -15.1284065, -10.8275490, -15.0926065, -10.8435230, -3.2526131, 3.2301729
2: -9.0924530, -5.7412505, -9.0605116, -5.7662840, -2.7576981, 2.7494314
3: -11.5793772, -7.3788524, -11.5224504, -7.4049926, -3.6279011, 3.5969648
4: -5.5128517, -1.8543191, -5.4727359, -1.9561654, -3.2150249, 3.2279522
5: -3.6005578, -0.4574785, -3.5721910, -0.4962034, -2.5916500, 2.5694096
6: -11.5937595, -6.9430428, -11.5825958, -6.9708004, -3.3840556, 3.3989396
7: -2.8246207, 0.8765140, -2.8088622, 0.8269138, -3.3226919, 3.3639138
8: -5.0962315, -1.4534807, -5.0764570, -1.4740119, -2.7516861, 2.7503774
9: 0.3617620, 3.0718336, 0.4371119, 3.0540156, -2.4287167, 2.3943148

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4097622, upper bound: 1.4127451
time: 4.80 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135308, upper bound: 1.4147628
time: 7.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.7316332, -5.0532002, -9.7472610, -5.0554972, -3.1307454, 3.1359673
1: -15.1295738, -10.8272858, -15.1243658, -10.7991047, -3.2856789, 3.2808366
2: -9.0930576, -5.7407026, -9.0751104, -5.7330904, -2.8051372, 2.7842774
3: -11.5797892, -7.3782148, -11.5621243, -7.3856573, -3.6444554, 3.6413221
4: -5.5150471, -1.8537903, -5.5104351, -1.8857274, -3.2507253, 3.2680686
5: -3.6012321, -0.4570160, -3.5982008, -0.4565678, -2.6046400, 2.5917344
6: -11.5943918, -6.9428973, -11.5962000, -6.9492130, -3.4205694, 3.4267240
7: -2.8250408, 0.8776202, -2.8617263, 0.8407497, -3.3548174, 3.3867879
8: -5.0968275, -1.4531355, -5.1048512, -1.4484897, -2.7854795, 2.7917097
9: 0.3610620, 3.0728059, 0.3857021, 3.0667651, -2.4426165, 2.4254615

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4109922, upper bound: 1.4127473
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147634, upper bound: 1.4147632
time: 9.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.54 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 29.54
Output dim: 9, lower bound: -1.4097622, upper bound: 1.4122214
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.54
Output dim: 9, lower bound: -1.4135308, upper bound: 1.4142294
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 29.54
Output dim: 9, lower bound: -1.4109922, upper bound: 1.4122239
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.54
Output dim: 9, lower bound: -1.4147634, upper bound: 1.4142310
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 29.54
Output dim: 9, lower bound: -1.4097622, upper bound: 1.4127451
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.54
Output dim: 9, lower bound: -1.4135308, upper bound: 1.4147628
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 29.54
Output dim: 9, lower bound: -1.4109922, upper bound: 1.4127473
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.54
Output dim: 9, lower bound: -1.4147634, upper bound: 1.4147632

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.6920090, -5.0658536, -9.6922455, -5.0658922, -3.0726976, 3.0070381
1: -15.0910683, -10.8475685, -15.0907612, -10.8461285, -3.2104635, 3.2085090
2: -9.0589485, -5.7687278, -9.0591869, -5.7681766, -2.6929817, 2.7159259
3: -11.5211926, -7.4086266, -11.5214739, -7.4076748, -3.5790796, 3.5702467
4: -5.4632378, -1.9573015, -5.4651737, -1.9571007, -3.1507559, 3.1436768
5: -3.5680778, -0.4965296, -3.5691624, -0.4967203, -2.5462971, 2.5479426
6: -11.5802650, -6.9715433, -11.5807781, -6.9713621, -3.3692899, 3.3363466
7: -2.8028016, 0.8264308, -2.8048158, 0.8258548, -3.2961202, 3.2979817
8: -5.0736675, -1.4753833, -5.0743380, -1.4751115, -2.6853676, 2.7273109
9: 0.4393444, 3.0529423, 0.4389744, 3.0525939, -2.3504972, 2.3572416

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4115141, upper bound: 1.4104705
time: 4.50 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4115140, upper bound: 1.4104681
time: 5.27 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.6923847, -5.0648541, -9.7457075, -5.0574589, -3.0891242, 3.0557230
1: -15.0922117, -10.8474197, -15.1223612, -10.8024502, -3.2560434, 3.2588487
2: -9.0594187, -5.7682447, -9.0735683, -5.7351522, -2.7477212, 2.7503097
3: -11.5214567, -7.4080524, -11.5608864, -7.3884740, -3.6020689, 3.6142120
4: -5.4654121, -1.9569935, -5.5027199, -1.8870579, -3.2028542, 3.1930614
5: -3.5687039, -0.4961643, -3.5951598, -0.4572773, -2.5676126, 2.5735924
6: -11.5807657, -6.9714122, -11.5938931, -6.9498243, -3.4041977, 3.3662844
7: -2.8031936, 0.8274498, -2.8575368, 0.8395185, -3.3279524, 3.3539720
8: -5.0741262, -1.4750657, -5.1021986, -1.4497046, -2.7185802, 2.7749674
9: 0.4387326, 3.0539169, 0.3878632, 3.0653057, -2.3644035, 2.4062886

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4127455, upper bound: 1.4104675
time: 4.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4127455, upper bound: 1.4104676
time: 5.12 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7311153, -5.0542064, -9.6934004, -5.0639610, -3.1146193, 3.0315282
1: -15.1284065, -10.8275509, -15.0926056, -10.8435202, -3.2526112, 3.2305012
2: -9.0924511, -5.7412534, -9.0605125, -5.7662854, -2.7334604, 2.7489882
3: -11.5793762, -7.3788524, -11.5224504, -7.4049940, -3.6345086, 3.5963197
4: -5.5128498, -1.8543202, -5.4727364, -1.9561663, -3.2149353, 3.2174926
5: -3.6005583, -0.4574766, -3.5721922, -0.4962025, -2.5914760, 2.5688672
6: -11.5937595, -6.9430456, -11.5825977, -6.9708014, -3.3840551, 3.3657117
7: -2.8246179, 0.8765116, -2.8088632, 0.8269134, -3.3222957, 3.3610802
8: -5.0962305, -1.4534798, -5.0764561, -1.4740119, -2.7093601, 2.7502220
9: 0.3617663, 3.0718327, 0.4371128, 3.0540147, -2.4286680, 2.4010055

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4115141, upper bound: 1.4109940
time: 4.82 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4115140, upper bound: 1.4109916
time: 5.10 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7316332, -5.0532060, -9.7472630, -5.0554972, -3.1270533, 3.0700641
1: -15.1295691, -10.8272896, -15.1243687, -10.7991047, -3.2856760, 3.2811632
2: -9.0930557, -5.7407036, -9.0751095, -5.7330899, -2.7805204, 2.7838335
3: -11.5797882, -7.3782163, -11.5621243, -7.3856583, -3.6509438, 3.6406779
4: -5.5150442, -1.8537920, -5.5104342, -1.8857284, -3.2496328, 3.2571576
5: -3.6012316, -0.4570150, -3.5981998, -0.4565678, -2.6035728, 2.5911915
6: -11.5943890, -6.9428997, -11.5961962, -6.9492140, -3.4186630, 3.3930449
7: -2.8250384, 0.8776193, -2.8617272, 0.8407502, -3.3544197, 3.3838830
8: -5.0968261, -1.4531384, -5.1048512, -1.4484897, -2.7428555, 2.7894745
9: 0.3610668, 3.0728049, 0.3857026, 3.0667646, -2.4425664, 2.4309428

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4127455, upper bound: 1.4109909
time: 5.09 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4127455, upper bound: 1.4109914
time: 5.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.14 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.14
Output dim: 9, lower bound: -1.4115141, upper bound: 1.4104705
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.14
Output dim: 9, lower bound: -1.4115140, upper bound: 1.4104681
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.14
Output dim: 9, lower bound: -1.4127455, upper bound: 1.4104675
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.14
Output dim: 9, lower bound: -1.4127455, upper bound: 1.4104676
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.14
Output dim: 9, lower bound: -1.4115141, upper bound: 1.4109940
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.14
Output dim: 9, lower bound: -1.4115140, upper bound: 1.4109916
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 25.14
Output dim: 9, lower bound: -1.4127455, upper bound: 1.4109909
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 25.14
Output dim: 9, lower bound: -1.4127455, upper bound: 1.4109914
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.3590686321258545
rel_dist={9: [-1.414766908177059, 1.4147664541724545]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5058189, upper bound: 1.5046053
time: 4.46 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5058189, upper bound: 1.5058177
time: 4.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.42
Output dim: 9, lower bound: -1.5058189, upper bound: 1.5046053
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.42
Output dim: 9, lower bound: -1.5058189, upper bound: 1.5058177

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.6937771, -5.0625219, -3.1910524, 3.1900649
1: -15.0922241, -10.8474169, -15.0944366, -10.8442984, -3.3243961, 3.3233006
2: -9.0594254, -5.7682366, -9.0610199, -5.7659874, -2.7832241, 2.7821288
3: -11.5214615, -7.4080415, -11.5226402, -7.4048352, -3.6692200, 3.6672001
4: -5.4654369, -1.9569887, -5.4744735, -1.9558575, -3.2101569, 3.2191062
5: -3.5687127, -0.4961605, -3.5723267, -0.4955359, -2.6235919, 2.6271605
6: -11.5807753, -6.9714108, -11.5829325, -6.9707384, -3.4803805, 3.4819889
7: -2.8032007, 0.8274651, -2.8080320, 0.8287425, -3.3562927, 3.3602834
8: -5.0741324, -1.4750624, -5.0766544, -1.4737434, -2.8212337, 2.8222692
9: 0.4387226, 3.0539322, 0.4364882, 3.0556307, -2.3965542, 2.3957276

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034602, upper bound: 1.5045925
time: 4.06 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5058037, upper bound: 1.5045902
time: 4.43 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.7328081, -5.0530224, -9.6942730, -5.0616941, -3.2325640, 3.2150517
1: -15.1302290, -10.8262949, -15.0952282, -10.8431759, -3.3658257, 3.3453443
2: -9.0948830, -5.7397919, -9.0615864, -5.7651715, -2.8254542, 2.8158038
3: -11.5832500, -7.3773108, -11.5230618, -7.4036827, -3.7283940, 3.6933398
4: -5.5160413, -1.8502890, -5.4777298, -1.9554555, -3.2774663, 3.2930326
5: -3.6018953, -0.4563103, -3.5736279, -0.4953136, -2.6706395, 2.6485293
6: -11.5956211, -6.9424963, -11.5837135, -6.9704971, -3.4972677, 3.5108137
7: -2.8262763, 0.8797851, -2.8097696, 0.8291960, -3.3826447, 3.4255393
8: -5.0977087, -1.4526463, -5.0775671, -1.4732757, -2.8471661, 2.8446529
9: 0.3590660, 3.0739331, 0.4356890, 3.0562444, -2.4773960, 2.4417968

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034602, upper bound: 1.5058055
time: 5.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5058037, upper bound: 1.5058030
time: 4.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.63 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.63
Output dim: 9, lower bound: -1.5034602, upper bound: 1.5045925
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.63
Output dim: 9, lower bound: -1.5058037, upper bound: 1.5045902
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.63
Output dim: 9, lower bound: -1.5034602, upper bound: 1.5058055
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.63
Output dim: 9, lower bound: -1.5058037, upper bound: 1.5058030

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648394, -9.6929045, -5.0647831, -3.1888542, 3.1891484
1: -15.0922241, -10.8474169, -15.0918198, -10.8446312, -3.3240323, 3.3199434
2: -9.0594254, -5.7682366, -9.0599499, -5.7670946, -2.7820787, 2.7805653
3: -11.5214615, -7.4080415, -11.5220356, -7.4061360, -3.6670709, 3.6665072
4: -5.4654369, -1.9569887, -5.4695044, -1.9565641, -3.2094059, 3.2135625
5: -3.5687127, -0.4961605, -3.5708957, -0.4964237, -2.6221814, 2.6253257
6: -11.5807753, -6.9714108, -11.5818176, -6.9710431, -3.4794950, 3.4804139
7: -2.8032007, 0.8274651, -2.8071351, 0.8264623, -3.3535810, 3.3593659
8: -5.0741324, -1.4750624, -5.0755501, -1.4744802, -2.8203254, 2.8206637
9: 0.4387226, 3.0539322, 0.4379072, 3.0534067, -2.3943548, 2.3943987

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034602, upper bound: 1.5022468
time: 4.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034601, upper bound: 1.5045925
time: 4.48 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.6923904, -5.0648460, -9.7469788, -5.0562649, -3.2061424, 3.2394571
1: -15.0922184, -10.8474188, -15.1244078, -10.7986736, -3.3706679, 3.3719664
2: -9.0594215, -5.7682405, -9.0751591, -5.7337480, -2.8373232, 2.8160462
3: -11.5214596, -7.4080491, -11.5617247, -7.3863864, -3.6908393, 3.7101097
4: -5.4654193, -1.9569913, -5.5074978, -1.8860211, -3.2623730, 3.2654238
5: -3.5687065, -0.4961624, -3.5970871, -0.4559517, -2.6457467, 2.6505640
6: -11.5807667, -6.9714098, -11.5958166, -6.9493642, -3.5182586, 3.5115409
7: -2.8031964, 0.8274555, -2.8601403, 0.8404717, -3.3869200, 3.4146256
8: -5.0741291, -1.4750657, -5.1047459, -1.4488440, -2.8535881, 2.8752034
9: 0.4387274, 3.0539229, 0.3861289, 3.0662529, -2.4084146, 2.4460256

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5009465, upper bound: 1.5031936
time: 4.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5058005, upper bound: 1.5045891
time: 4.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.7327023, -5.0530281, -9.6934023, -5.0639567, -3.2302685, 3.2141037
1: -15.1301994, -10.8263302, -15.0926075, -10.8435116, -3.3654280, 3.3419299
2: -9.0948715, -5.7398415, -9.0605145, -5.7662807, -2.8242888, 2.8141720
3: -11.5831289, -7.3773298, -11.5224524, -7.4049854, -3.7264338, 3.6926355
4: -5.5160246, -1.8504642, -5.4727550, -1.9561651, -3.2766948, 3.2876639
5: -3.6018767, -0.4563818, -3.5721979, -0.4962044, -2.6692052, 2.6466296
6: -11.5955524, -6.9425049, -11.5826025, -6.9708004, -3.4962821, 3.5091825
7: -2.8262548, 0.8797493, -2.8088722, 0.8269153, -3.3799019, 3.4245715
8: -5.0976696, -1.4526663, -5.0764623, -1.4740081, -2.8461838, 2.8430250
9: 0.3591375, 3.0739267, 0.4371071, 3.0540187, -2.4751163, 2.4404597

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034601, upper bound: 1.5034616
time: 4.84 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034601, upper bound: 1.5058055
time: 5.21 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.7328043, -5.0530319, -9.7475204, -5.0554333, -3.2476516, 3.2538240
1: -15.1302195, -10.8262959, -15.1252489, -10.7970219, -3.4018278, 3.3940697
2: -9.0948792, -5.7397962, -9.0758896, -5.7329278, -2.8710957, 2.8499420
3: -11.5832462, -7.3773179, -11.5621624, -7.3851500, -3.7433295, 3.7363024
4: -5.5160265, -1.8502930, -5.5107889, -1.8855720, -3.3119106, 3.3294845
5: -3.6018903, -0.4563131, -3.5983877, -0.4556875, -2.6832552, 2.6684735
6: -11.5956144, -6.9424973, -11.5966911, -6.9491072, -3.5340843, 3.5392509
7: -2.8262725, 0.8797731, -2.8619046, 0.8409519, -3.4133320, 3.4463418
8: -5.0977049, -1.4526510, -5.1057634, -1.4483528, -2.8795652, 2.8890901
9: 0.3590732, 3.0739253, 0.3853197, 3.0668850, -2.4891057, 2.4723001

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5009465, upper bound: 1.5043768
time: 4.93 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5058005, upper bound: 1.5058021
time: 4.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.10 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 9, lower bound: -1.5034602, upper bound: 1.5022468
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 9, lower bound: -1.5034601, upper bound: 1.5045925
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 9, lower bound: -1.5009465, upper bound: 1.5031936
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 9, lower bound: -1.5058005, upper bound: 1.5045891
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 9, lower bound: -1.5034601, upper bound: 1.5034616
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 9, lower bound: -1.5034601, upper bound: 1.5058055
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 9, lower bound: -1.5009465, upper bound: 1.5043768
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.10
Output dim: 9, lower bound: -1.5058005, upper bound: 1.5058021

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.6929045, -5.0647831, -3.1879387, 3.1869538
1: -15.0896091, -10.8477516, -15.0918198, -10.8446312, -3.3206763, 3.3195810
2: -9.0583544, -5.7693434, -9.0599499, -5.7670946, -2.7805195, 2.7794230
3: -11.5208578, -7.4093418, -11.5220356, -7.4061360, -3.6663842, 3.6643596
4: -5.4604731, -1.9576910, -5.4695044, -1.9565641, -3.2038612, 3.2128124
5: -3.5672817, -0.4970436, -3.5708957, -0.4964237, -2.6203465, 2.6239185
6: -11.5796518, -6.9717102, -11.5818176, -6.9710431, -3.4778948, 3.4795299
7: -2.8023047, 0.8251872, -2.8071351, 0.8264623, -3.3526645, 3.3566556
8: -5.0730295, -1.4757953, -5.0755501, -1.4744802, -2.8187242, 2.8197548
9: 0.4401379, 3.0517111, 0.4379072, 3.0534067, -2.3930278, 2.3922017

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5019371, upper bound: 1.4973362
time: 4.58 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034568, upper bound: 1.5022428
time: 4.30 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.7451410, -5.0587645, -9.6929045, -5.0647831, -3.2377672, 3.1967533
1: -15.1192884, -10.8019457, -15.0918198, -10.8446312, -3.3525615, 3.3660622
2: -9.0733662, -5.7363815, -9.0599499, -5.7670946, -2.8055286, 2.8325186
3: -11.5603905, -7.3901405, -11.5220356, -7.4061360, -3.7095208, 3.6824851
4: -5.4977083, -1.8872907, -5.4695044, -1.9565641, -3.2429781, 3.2658710
5: -3.5932226, -0.4597197, -3.5708957, -0.4964237, -2.6454701, 2.6457934
6: -11.5926962, -6.9502735, -11.5818176, -6.9710431, -3.5066032, 3.5168877
7: -2.8549216, 0.8386259, -2.8071351, 0.8264623, -3.4075551, 3.3710618
8: -5.1000733, -1.4505224, -5.0755501, -1.4744802, -2.8603625, 2.8521197
9: 0.3897710, 3.0645382, 0.4379072, 3.0534067, -2.4436293, 2.4061098

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5019371, upper bound: 1.4997670
time: 5.10 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034568, upper bound: 1.5045885
time: 4.27 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.6663399, -5.1665778, -9.7405777, -5.0879841, -3.1181507, 3.1301706
1: -15.0539236, -10.8756943, -15.1123800, -10.8074141, -3.3061781, 3.3194401
2: -9.0098877, -5.7996879, -9.0597982, -5.7426286, -2.7727370, 2.7611268
3: -11.4896030, -7.4439049, -11.5515795, -7.3962536, -3.6449375, 3.6587677
4: -5.4243598, -1.9902974, -5.4955883, -1.8965447, -3.2028923, 3.2211690
5: -3.5385876, -0.5064697, -3.5885816, -0.4590836, -2.6123486, 2.6313062
6: -11.5495968, -7.0200186, -11.5872784, -6.9646149, -3.4483480, 3.4544616
7: -2.7651567, 0.8011880, -2.8493609, 0.8323445, -3.3255696, 3.3735681
8: -5.0106888, -1.4988389, -5.0848312, -1.4548836, -2.7830896, 2.8030298
9: 0.5460806, 3.0406148, 0.4195390, 3.0628402, -2.2904391, 2.3750436

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5009446, upper bound: 1.5002602
time: 4.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5009446, upper bound: 1.5031917
time: 4.54 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.6923866, -5.0648527, -9.7469788, -5.0562649, -3.2061405, 3.1764495
1: -15.0922155, -10.8474216, -15.1244078, -10.7986736, -3.3706670, 3.3726263
2: -9.0594215, -5.7682428, -9.0751591, -5.7337480, -2.8143773, 2.8155994
3: -11.5214567, -7.4080486, -11.5617247, -7.3863864, -3.7004642, 3.7094612
4: -5.4654179, -1.9569925, -5.5074978, -1.8860211, -3.2612791, 3.2568264
5: -3.5687056, -0.4961634, -3.5970871, -0.4559517, -2.6447382, 2.6510034
6: -11.5807667, -6.9714112, -11.5958166, -6.9493642, -3.5163484, 3.4799914
7: -2.8031955, 0.8274527, -2.8601403, 0.8404717, -3.3865128, 3.4141078
8: -5.0741262, -1.4750652, -5.1047459, -1.4488440, -2.8134747, 2.8731923
9: 0.4387317, 3.0539227, 0.3861289, 3.0662529, -2.4084098, 2.4518042

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5043747, upper bound: 1.4997649
time: 4.91 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5043747, upper bound: 1.5045877
time: 6.42 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.7318096, -5.0552831, -9.6934023, -5.0639567, -3.2293949, 3.2119055
1: -15.1275902, -10.8266764, -15.0926075, -10.8435116, -3.3620539, 3.3415599
2: -9.0938568, -5.7409630, -9.0605145, -5.7662807, -2.8227258, 2.8130033
3: -11.5825024, -7.3786945, -11.5224524, -7.4049854, -3.7257471, 3.6904840
4: -5.5110874, -1.8511829, -5.4727550, -1.9561651, -3.2711511, 3.2869191
5: -3.6003981, -0.4572687, -3.5721979, -0.4962044, -2.6674714, 2.6452041
6: -11.5944386, -6.9428177, -11.5826025, -6.9708004, -3.4950290, 3.5082898
7: -2.8253570, 0.8774910, -2.8088722, 0.8269153, -3.3789806, 3.4218345
8: -5.0965652, -1.4533973, -5.0764623, -1.4740081, -2.8445816, 2.8421121
9: 0.3605566, 3.0717435, 0.4371071, 3.0540187, -2.4737945, 2.4383116

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5019371, upper bound: 1.4985191
time: 4.43 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034568, upper bound: 1.5034576
time: 4.86 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7854223, -5.0471625, -9.6934023, -5.0639567, -3.2614379, 3.2217565
1: -15.1575975, -10.7812939, -15.0926075, -10.8435116, -3.3941336, 3.3875413
2: -9.1104336, -5.7082748, -9.0605145, -5.7662807, -2.8478808, 2.8658454
3: -11.6224537, -7.3599548, -11.5224524, -7.4049854, -3.7460451, 3.7085333
4: -5.5475931, -1.7810051, -5.4727550, -1.9561651, -3.3108201, 3.3075442
5: -3.6262250, -0.4202309, -3.5721979, -0.4962044, -2.6911592, 2.6553569
6: -11.6075859, -6.9211149, -11.5826025, -6.9708004, -3.5234456, 3.5339658
7: -2.8781822, 0.8908691, -2.8088722, 0.8269153, -3.4340000, 3.4360597
8: -5.1233883, -1.4281268, -5.0764623, -1.4740081, -2.8859806, 2.8746235
9: 0.3104758, 3.0845823, 0.4371071, 3.0540187, -2.4965434, 2.4518714

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5019371, upper bound: 1.5009484
time: 4.60 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5034568, upper bound: 1.5058017
time: 4.95 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7067795, -5.1547079, -9.7411242, -5.0871525, -3.1421127, 3.1445735
1: -15.0912609, -10.8545303, -15.1132240, -10.8057594, -3.3331118, 3.3415711
2: -9.0453682, -5.7712092, -9.0605268, -5.7418094, -2.8073530, 2.7952988
3: -11.5514803, -7.4130201, -11.5520191, -7.3950181, -3.6973743, 3.6849098
4: -5.4742241, -1.8842086, -5.4988747, -1.8960965, -3.2520738, 3.2849622
5: -3.5711746, -0.4667406, -3.5898836, -0.4588203, -2.6497169, 2.6494937
6: -11.5645370, -6.9910421, -11.5881529, -6.9643598, -3.4642115, 3.4821587
7: -2.7878966, 0.8529663, -2.8511224, 0.8328233, -3.3517876, 3.4055307
8: -5.0342617, -1.4765344, -5.0858483, -1.4543953, -2.8088837, 2.8167338
9: 0.4673834, 3.0607414, 0.4187322, 3.0634720, -2.3702350, 2.4010775

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5009446, upper bound: 1.5014410
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5009446, upper bound: 1.5043749
time: 4.61 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.7328033, -5.0530362, -9.7475204, -5.0554333, -3.2449350, 3.1908169
1: -15.1302204, -10.8262978, -15.1252489, -10.7970219, -3.4018259, 3.3947315
2: -9.0948792, -5.7397976, -9.0758896, -5.7329278, -2.8480039, 2.8494976
3: -11.5832462, -7.3773198, -11.5621624, -7.3851500, -3.7515993, 3.7356558
4: -5.5160251, -1.8502936, -5.5107889, -1.8855720, -3.3108177, 3.3195853
5: -3.6018882, -0.4563141, -3.5983877, -0.4556875, -2.6822081, 2.6684124
6: -11.5956125, -6.9424982, -11.5966911, -6.9491072, -3.5321779, 3.5074084
7: -2.8262715, 0.8797736, -2.8619046, 0.8409519, -3.4129372, 3.4446058
8: -5.0977011, -1.4526491, -5.1057634, -1.4483528, -2.8390684, 2.8868544
9: 0.3590765, 3.0739255, 0.3853197, 3.0668850, -2.4891043, 2.4780612

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5043747, upper bound: 1.5009463
time: 4.76 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5043747, upper bound: 1.5009470
time: 8.95 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.51 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5019371, upper bound: 1.4973362
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5034568, upper bound: 1.5022428
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5019371, upper bound: 1.4997670
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5034568, upper bound: 1.5045885
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5009446, upper bound: 1.5002602
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5009446, upper bound: 1.5031917
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5043747, upper bound: 1.4997649
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5043747, upper bound: 1.5045877
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5019371, upper bound: 1.4985191
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5034568, upper bound: 1.5034576
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5019371, upper bound: 1.5009484
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5034568, upper bound: 1.5058017
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5009446, upper bound: 1.5014410
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5009446, upper bound: 1.5043749
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5043747, upper bound: 1.5009463
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.51
Output dim: 9, lower bound: -1.5043747, upper bound: 1.5009470

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.6851816, -5.0988283, -9.6668797, -5.1665139, -3.0786810, 3.1054611
1: -15.0775948, -10.8564939, -15.0535192, -10.8729172, -3.2680678, 3.2555428
2: -9.0429134, -5.7781734, -9.0103645, -5.7985492, -2.7255831, 2.7148905
3: -11.5106983, -7.4192719, -11.4901848, -7.4420662, -3.6149917, 3.6185074
4: -5.4486961, -1.9681814, -5.4284139, -1.9898791, -3.1595898, 3.1572070
5: -3.5588379, -0.5001907, -3.5407662, -0.5067310, -2.6009531, 2.5896540
6: -11.5711508, -6.9869652, -11.5506449, -7.0196447, -3.4208155, 3.4262176
7: -2.7915583, 0.8170652, -2.7690926, 0.8001857, -3.3116241, 3.2952757
8: -5.0531421, -1.4817910, -5.0121322, -1.4982829, -2.7717390, 2.7493656
9: 0.4735427, 3.0483375, 0.5453043, 3.0400987, -2.3399978, 2.2742932

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4990085, upper bound: 1.4973322
time: 4.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5019352, upper bound: 1.4973345
time: 4.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6915264, -5.0670991, -9.6929045, -5.0647898, -3.1249332, 3.1869535
1: -15.0896091, -10.8477516, -15.0918169, -10.8446341, -3.3213367, 3.3195775
2: -9.0583544, -5.7693434, -9.0599480, -5.7670956, -2.7800727, 2.7567577
3: -11.5208578, -7.4093418, -11.5220318, -7.4061375, -3.6657324, 3.6741066
4: -5.4604731, -1.9576910, -5.4695024, -1.9565649, -3.1957321, 3.2126775
5: -3.5672817, -0.4970436, -3.5708942, -0.4964247, -2.6207848, 2.6239183
6: -11.5796518, -6.9717102, -11.5818176, -6.9710436, -3.4463463, 3.4795284
7: -2.8023047, 0.8251872, -2.8071327, 0.8264623, -3.3522687, 3.3562503
8: -5.0730295, -1.4757953, -5.0755463, -1.4744797, -2.8185749, 2.7799740
9: 0.4401379, 3.0517111, 0.4379110, 3.0534067, -2.3999991, 2.3921974

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5007558
time: 5.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5007532
time: 4.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7387390, -5.0904870, -9.6668797, -5.1665139, -3.1284828, 3.1152976
1: -15.1072683, -10.8106842, -15.0535192, -10.8729172, -3.3000026, 3.3020263
2: -9.0580044, -5.7452626, -9.0103645, -5.7985492, -2.7505770, 2.7679381
3: -11.5502453, -7.4000120, -11.4901848, -7.4420662, -3.6581807, 3.6365862
4: -5.4858069, -1.8978130, -5.4284139, -1.9898791, -3.1987171, 3.2064018
5: -3.5847187, -0.4628534, -3.5407662, -0.5067310, -2.6261892, 2.6124091
6: -11.5841579, -6.9655266, -11.5506449, -7.0196447, -3.4495268, 3.4480729
7: -2.8441484, 0.8305020, -2.7690926, 0.8001857, -3.3664985, 3.3097086
8: -5.0801582, -1.4565549, -5.0121322, -1.4982829, -2.7992744, 2.7816479
9: 0.4231720, 3.0611267, 0.5453043, 3.0400987, -2.3726454, 2.2881584

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4990085, upper bound: 1.4997621
time: 4.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5019352, upper bound: 1.4997645
time: 4.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7451410, -5.0587645, -9.6929045, -5.0647898, -3.1747599, 3.1967533
1: -15.1192884, -10.8019457, -15.0918169, -10.8446341, -3.3532219, 3.3660586
2: -9.0733662, -5.7363815, -9.0599480, -5.7670956, -2.8050818, 2.8095884
3: -11.5603905, -7.3901405, -11.5220318, -7.4061375, -3.7088690, 3.6921120
4: -5.4977083, -1.8872907, -5.4695024, -1.9565649, -3.2343864, 3.2647784
5: -3.5932226, -0.4597197, -3.5708942, -0.4964247, -2.6459103, 2.6447868
6: -11.5926962, -6.9502735, -11.5818176, -6.9710436, -3.4750547, 3.5160649
7: -2.8549216, 0.8386259, -2.8071327, 0.8264623, -3.4070897, 3.3706560
8: -5.1000733, -1.4505224, -5.0755463, -1.4744797, -2.8602142, 2.8120391
9: 0.3897710, 3.0645382, 0.4379110, 3.0534067, -2.4494085, 2.4061050

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5031931
time: 4.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4985177, upper bound: 1.5031906
time: 5.49 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.6558361, -5.2295833, -9.7002506, -5.2700939, -2.9176607, 2.9291654
1: -15.0415325, -10.8847713, -15.0767870, -10.8359566, -3.2655783, 3.2686355
2: -9.0008259, -5.8559079, -9.0258055, -5.9047232, -2.5961037, 2.5956376
3: -11.4860697, -7.4980593, -11.5352249, -7.5520258, -3.4882898, 3.5396938
4: -5.3890328, -1.9970634, -5.3920493, -1.9214675, -3.1126585, 3.1066446
5: -3.5198221, -0.5097294, -3.5353389, -0.4710083, -2.5665998, 2.5696075
6: -11.5405741, -7.1010289, -11.5472631, -7.1989713, -3.2027016, 3.2171934
7: -2.7246351, 0.7939901, -2.7319622, 0.8054967, -3.2354932, 3.2482004
8: -4.9999638, -1.5491562, -5.0453134, -1.6006436, -2.6274834, 2.6495650
9: 0.5703721, 3.0357385, 0.4909058, 3.0453050, -2.2449000, 2.2923684

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5002602
time: 4.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5002606
time: 4.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.6663399, -5.1665778, -9.7405758, -5.0879941, -2.9692197, 3.1210723
1: -15.0539236, -10.8756943, -15.1123829, -10.8074141, -3.3061767, 3.3276129
2: -9.0098877, -5.7996879, -9.0597973, -5.7426348, -2.6725426, 2.7579236
3: -11.4896030, -7.4439049, -11.5515804, -7.3962612, -3.6098900, 3.6566663
4: -5.4243598, -1.9902974, -5.4955850, -1.8965453, -3.2146120, 3.2211652
5: -3.5385876, -0.5064697, -3.5885799, -0.4590855, -2.6105347, 2.6256003
6: -11.5495968, -7.0200186, -11.5872774, -6.9646215, -3.2797861, 3.4530923
7: -2.7651567, 0.8011880, -2.8493576, 0.8323421, -3.3342056, 3.3735633
8: -5.0106888, -1.4988389, -5.0848303, -1.4548888, -2.7488298, 2.7968225
9: 0.5460806, 3.0406148, 0.4195409, 3.0628395, -2.2904391, 2.3743291

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5031917
time: 5.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5031895
time: 6.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.6920815, -5.0656910, -9.7207336, -5.1579580, -3.1032658, 3.1392806
1: -15.0912294, -10.8479223, -15.0860853, -10.8269339, -3.3354321, 3.3177958
2: -9.0589066, -5.7686296, -9.0257053, -5.7653642, -2.7738338, 2.7613280
3: -11.5193996, -7.4081864, -11.5299206, -7.4222741, -3.6517143, 3.6734381
4: -5.4647560, -1.9574398, -5.4661112, -1.9194402, -3.2282681, 3.2234559
5: -3.5677309, -0.4963169, -3.5668125, -0.4662142, -2.6332960, 2.6185107
6: -11.5806980, -6.9728279, -11.5645227, -6.9979601, -3.4689665, 3.4555399
7: -2.8020697, 0.8269968, -2.8219078, 0.8141756, -3.3575048, 3.3607023
8: -5.0733857, -1.4755898, -5.0412159, -1.4727731, -2.8023772, 2.8091075
9: 0.4395952, 3.0535245, 0.4935846, 3.0528264, -2.3880630, 2.3297570

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5014385, upper bound: 1.4997651
time: 5.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5043722, upper bound: 1.4997630
time: 4.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.6923866, -5.0648527, -9.7469769, -5.0562696, -3.1431351, 3.1757331
1: -15.0922155, -10.8474216, -15.1244068, -10.7986736, -3.3713274, 3.3726239
2: -9.0594215, -5.7682428, -9.0751572, -5.7337503, -2.8143735, 2.7934325
3: -11.5214567, -7.4080486, -11.5617218, -7.3863883, -3.7004623, 3.7198954
4: -5.4654179, -1.9569925, -5.5074954, -1.8860224, -3.2535052, 3.2568254
5: -3.5687056, -0.4961634, -3.5970869, -0.4559507, -2.6451468, 2.6510015
6: -11.5807667, -6.9714112, -11.5958157, -6.9493647, -3.4863148, 3.4799900
7: -2.8031955, 0.8274527, -2.8601379, 0.8404713, -3.3864202, 3.4141040
8: -5.0741262, -1.4750652, -5.1047435, -1.4488440, -2.8134747, 2.8353000
9: 0.4387317, 3.0539227, 0.3861318, 3.0662534, -2.4153807, 2.4518018

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5014392, upper bound: 1.4997624
time: 4.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5043727, upper bound: 1.4997629
time: 4.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.7254715, -5.0870037, -9.6673861, -5.1656852, -3.1201162, 3.1197798
1: -15.1153793, -10.8354187, -15.0543070, -10.8718014, -3.3088984, 3.2775311
2: -9.0784760, -5.7498007, -9.0109158, -5.7977333, -2.7677870, 2.7485790
3: -11.5723658, -7.3885126, -11.4906063, -7.4409142, -3.6739488, 3.6446090
4: -5.4990869, -1.8618524, -5.4316540, -1.9894836, -3.2268796, 3.2270629
5: -3.5917645, -0.4604502, -3.5420694, -0.5065079, -2.6482115, 2.6118376
6: -11.5859604, -6.9580507, -11.5514240, -7.0194011, -3.4379144, 3.4517949
7: -2.8145223, 0.8692117, -2.7708306, 0.8006344, -3.3378811, 3.3593836
8: -5.0766869, -1.4594302, -5.0130486, -1.4978189, -2.7976589, 2.7716920
9: 0.3942165, 3.0684032, 0.5445175, 3.0407095, -2.4016151, 2.3203959

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4990085, upper bound: 1.4985145
time: 4.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5019352, upper bound: 1.4985173
time: 4.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.7318096, -5.0552831, -9.6934013, -5.0639615, -3.1663885, 3.2119045
1: -15.1275902, -10.8266764, -15.0926075, -10.8435154, -3.3627138, 3.3415575
2: -9.0938568, -5.7409630, -9.0605145, -5.7662821, -2.8222790, 2.7902980
3: -11.5825024, -7.3786945, -11.5224514, -7.4049864, -3.7238851, 3.7001295
4: -5.5110874, -1.8511829, -5.4727530, -1.9561657, -3.2620211, 3.2858267
5: -3.6003981, -0.4572687, -3.5721965, -0.4962015, -2.6679115, 2.6441972
6: -11.5944386, -6.9428177, -11.5825968, -6.9708033, -3.4634809, 3.5082889
7: -2.8253570, 0.8774910, -2.8088722, 0.8269162, -3.3783836, 3.4205909
8: -5.0965652, -1.4533973, -5.0764589, -1.4740100, -2.8444333, 2.8024127
9: 0.3605566, 3.0717435, 0.4371114, 3.0540185, -2.4795740, 2.4383068

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5019385
time: 4.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4985177, upper bound: 1.5034556
time: 4.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7790279, -5.0788746, -9.6673861, -5.1656852, -3.1520662, 3.1295922
1: -15.1453915, -10.7900305, -15.0543070, -10.8718014, -3.3410439, 3.3235159
2: -9.0950680, -5.7171726, -9.0109158, -5.7977333, -2.7919989, 2.8014240
3: -11.6123333, -7.3697948, -11.4906063, -7.4409142, -3.6943107, 3.6626096
4: -5.5354981, -1.7917001, -5.4316540, -1.9894836, -3.2665677, 3.2475054
5: -3.6175914, -0.4233999, -3.5420694, -0.5065079, -2.6719093, 2.6220033
6: -11.5990791, -6.9363484, -11.5514240, -7.0194011, -3.4663053, 3.4641426
7: -2.8673575, 0.8825879, -2.7708306, 0.8006344, -3.3928690, 3.3735385
8: -5.1034770, -1.4341955, -5.0130486, -1.4978189, -2.8243256, 2.8041241
9: 0.3441248, 3.0811591, 0.5445175, 3.0407095, -2.4242287, 2.3339140

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4990085, upper bound: 1.5009431
time: 5.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5019352, upper bound: 1.5009459
time: 4.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7854223, -5.0471625, -9.6934013, -5.0639615, -3.1984301, 3.2217555
1: -15.1575975, -10.7812939, -15.0926075, -10.8435154, -3.3947945, 3.3875389
2: -9.1104336, -5.7082748, -9.0605145, -5.7662821, -2.8474340, 2.8428869
3: -11.6224537, -7.3599548, -11.5224514, -7.4049864, -3.7441831, 3.7180605
4: -5.5475931, -1.7810051, -5.4727530, -1.9561657, -3.3011608, 3.3064528
5: -3.6262250, -0.4202309, -3.5721965, -0.4962015, -2.6910906, 2.6543500
6: -11.6075859, -6.9211149, -11.5825968, -6.9708033, -3.4918985, 3.5320551
7: -2.8781822, 0.8908691, -2.8088722, 0.8269162, -3.4333324, 3.4348154
8: -5.1233883, -1.4281268, -5.0764589, -1.4740100, -2.8858323, 2.8346214
9: 0.3104758, 3.0845823, 0.4371114, 3.0540185, -2.5023229, 2.4518666

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5043763
time: 4.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4985177, upper bound: 1.5057992
time: 4.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.6962729, -5.2177052, -9.7007990, -5.2692618, -2.9416709, 2.9437151
1: -15.0788803, -10.8636456, -15.0776043, -10.8343067, -3.2925391, 3.2908516
2: -9.0361195, -5.8273954, -9.0265503, -5.9039068, -2.6305366, 2.6299345
3: -11.5479708, -7.4670539, -11.5356617, -7.5507956, -3.5404387, 3.5670342
4: -5.4389863, -1.8908851, -5.3953362, -1.9210224, -3.1630826, 3.1699250
5: -3.5523810, -0.4700003, -3.5366178, -0.4707432, -2.6046300, 2.5873647
6: -11.5554972, -7.0719328, -11.5481367, -7.1987123, -3.2185693, 3.2342365
7: -2.7475586, 0.8457923, -2.7337317, 0.8059764, -3.2623868, 3.2797086
8: -5.0235558, -1.5267544, -5.0463324, -1.6001544, -2.6532784, 2.6635609
9: 0.4915948, 3.0557477, 0.4900923, 3.0459378, -2.3053396, 2.3183112

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5014410
time: 4.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5014415
time: 4.87 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.7067795, -5.1547079, -9.7411213, -5.0871611, -2.9931812, 3.1354749
1: -15.0912609, -10.8545303, -15.1132221, -10.8057613, -3.3331094, 3.3497372
2: -9.0453682, -5.7712092, -9.0605278, -5.7418151, -2.7070866, 2.7918866
3: -11.5514803, -7.4130201, -11.5520191, -7.3950214, -3.6623025, 3.6825156
4: -5.4742241, -1.8842086, -5.4988713, -1.8960981, -3.2637916, 3.2849596
5: -3.5711746, -0.4667406, -3.5898781, -0.4588213, -2.6479020, 2.6437814
6: -11.5645370, -6.9910421, -11.5881538, -6.9643669, -3.2956505, 3.4696722
7: -2.7878966, 0.8529663, -2.8511198, 0.8328214, -3.3604202, 3.4055295
8: -5.0342617, -1.4765344, -5.0858479, -1.4544001, -2.7746229, 2.8105264
9: 0.4673834, 3.0607414, 0.4187336, 3.0634718, -2.3674741, 2.4003632

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5043749
time: 5.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5031897
time: 5.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7324953, -5.0538774, -9.7212839, -5.1571240, -3.1421366, 3.1536498
1: -15.1292315, -10.8267956, -15.0869198, -10.8252831, -3.3626013, 3.3399014
2: -9.0943470, -5.7401915, -9.0264349, -5.7645426, -2.8025646, 2.7952220
3: -11.5811853, -7.3774605, -11.5303593, -7.4210300, -3.6972198, 3.6996355
4: -5.5153742, -1.8507395, -5.4693918, -1.9189939, -3.2778339, 3.2750816
5: -3.6009114, -0.4564676, -3.5681019, -0.4659529, -2.6707287, 2.6371372
6: -11.5955448, -6.9439230, -11.5653973, -6.9977040, -3.4847937, 3.4722180
7: -2.8251410, 0.8793159, -2.8236704, 0.8146515, -3.3839293, 3.3880792
8: -5.0969582, -1.4531770, -5.0422368, -1.4722919, -2.8279672, 2.8227746
9: 0.3599410, 3.0735610, 0.4927845, 3.0534582, -2.4394107, 2.3560112

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5014385, upper bound: 1.5009466
time: 5.25 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5043722, upper bound: 1.5009444
time: 5.03 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7328033, -5.0530362, -9.7475224, -5.0554380, -3.1846447, 3.1901002
1: -15.1302204, -10.8262978, -15.1252499, -10.7970238, -3.4024954, 3.3947277
2: -9.0948792, -5.7397976, -9.0758886, -5.7329288, -2.8480024, 2.8272872
3: -11.5832462, -7.3773198, -11.5621624, -7.3851504, -3.7515984, 3.7459850
4: -5.5160251, -1.8502936, -5.5107880, -1.8855728, -3.3020439, 3.3195841
5: -3.6018882, -0.4563141, -3.5983868, -0.4556885, -2.6826172, 2.6678596
6: -11.5956125, -6.9424982, -11.5966883, -6.9491096, -3.5021458, 3.5073891
7: -2.8262715, 0.8797736, -2.8619032, 0.8409505, -3.4126358, 3.4446039
8: -5.0977011, -1.4526491, -5.1057615, -1.4483523, -2.8390660, 2.8490436
9: 0.3590765, 3.0739255, 0.3853211, 3.0668848, -2.4948730, 2.4780593

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5014392, upper bound: 1.5009441
time: 4.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5043728, upper bound: 1.5009438
time: 4.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.31 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4990085, upper bound: 1.4973322
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5019352, upper bound: 1.4973345
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5007558
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5007532
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4990085, upper bound: 1.4997621
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5019352, upper bound: 1.4997645
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5031931
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4985177, upper bound: 1.5031906
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5002602
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5002606
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5031917
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5031895
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5014385, upper bound: 1.4997651
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5043722, upper bound: 1.4997630
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5014392, upper bound: 1.4997624
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5043727, upper bound: 1.4997629
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4990085, upper bound: 1.4985145
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5019352, upper bound: 1.4985173
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5019385
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4985177, upper bound: 1.5034556
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4990085, upper bound: 1.5009431
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5019352, upper bound: 1.5009459
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4985178, upper bound: 1.5043763
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4985177, upper bound: 1.5057992
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5014410
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5014415
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5043749
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.4979896, upper bound: 1.5031897
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5014385, upper bound: 1.5009466
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5043722, upper bound: 1.5009444
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5014392, upper bound: 1.5009441
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.31
Output dim: 9, lower bound: -1.5043728, upper bound: 1.5009438

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.6448536, -5.2809391, -9.6563797, -5.2295179, -2.9019556, 2.9049113
1: -15.0426483, -10.8850346, -15.0411272, -10.8819981, -3.2177186, 3.2147131
2: -9.0089436, -5.9407063, -9.0012999, -5.8547754, -2.5748425, 2.5378106
3: -11.4943399, -7.5751948, -11.4866467, -7.4962330, -3.5182962, 3.4621911
4: -5.3450928, -1.9931837, -5.3930931, -1.9966493, -3.0444994, 3.0906262
5: -3.5045786, -0.5121078, -3.5220132, -0.5099916, -2.5385156, 2.5522828
6: -11.5313234, -7.2214360, -11.5416231, -7.1006489, -3.1985569, 3.1807895
7: -2.6739349, 0.7902918, -2.7285814, 0.7929859, -3.1859927, 3.2217553
8: -5.0138502, -1.6276164, -5.0014124, -1.5485983, -2.6270447, 2.5937498
9: 0.5447631, 3.0308151, 0.5695925, 3.0352199, -2.2571383, 2.2286928

Time for backsubstitution: 14.59 seconds
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.4008383750915527
rel_dist={9: [-1.505820462064309, 1.5058195856423922]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.00390625
execution time: 2748.38 seconds
