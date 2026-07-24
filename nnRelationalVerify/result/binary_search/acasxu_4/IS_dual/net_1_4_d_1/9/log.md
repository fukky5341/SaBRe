## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 51.042030738


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.1557789, 40.7848358, -22.1557789, 40.7848358, -62.9406128, 62.9406128)
1: (-24.9570370, 38.1305771, -24.9570370, 38.1305771, -63.0876160, 63.0876160)
2: (-25.5280037, 37.2813034, -25.5280037, 37.2813034, -62.8093071, 62.8093071)
3: (-30.7027779, 44.2187729, -30.7027779, 44.2187729, -74.9215546, 74.9215546)
4: (-28.9112663, 41.7009773, -28.9112663, 41.7009773, -70.6122437, 70.6122437)

## BASE Result
execution time: IAR + LP analysis = 2.56 + 1.93 = 4.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -54.3001214, upper bound: 54.3001214


# Binary Search by BASE starts (time budget: 1195.51 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=62.94061279296875
rel_dist={0: [-54.30012139088531, 54.300121390885295]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=62.94061279296875
rel_dist={0: [-54.300068832219395, 54.30006883221938]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=62.94061279296875
rel_dist={0: [-54.29988917735716, 54.29988917735716]}

## Binary search (step 3) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=62.94061279296875
rel_dist={0: [-54.29940933754574, 54.29940933754574]}

## Binary search (step 4) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=62.94061279296875
rel_dist={0: [-54.29912131073952, 54.29912131073952]}

## Binary search (step 5) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=62.94061279296875
rel_dist={0: [-54.29896729898809, 54.29896729898809]}

## Binary search (step 6) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=62.94061279296875
rel_dist={0: [-54.2988899607592, 54.2988899607592]}

## Binary search (step 7) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=62.94061279296875
rel_dist={0: [-54.29885077378792, 54.29885077378792]}

## Binary search (step 8) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=62.94061279296875
rel_dist={0: [-54.29882823002279, 54.29882823002279]}

## Binary search (step 9) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=62.94061279296875
rel_dist={0: [-54.298816167618625, 54.29881616761861]}

## Binary search (step 10) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=62.94061279296875
rel_dist={0: [-54.29881013483843, 54.29881013483843]}

## Binary search (step 11) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=62.94061279296875
rel_dist={0: [-54.29880711845274, 54.29880711845274]}

## Binary search (step 12) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=62.94061279296875
rel_dist={0: [-54.29880561026867, 54.29880561026867]}

## Binary search (step 13) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=62.94061279296875
rel_dist={0: [-54.298804856131724, 54.298804856131724]}

## Binary search (step 14) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=62.94061279296875
rel_dist={0: [-54.29880447922184, 54.29880447922184]}

## Binary search (step 15) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=62.94061279296875
rel_dist={0: [-54.29880429076999, 54.29880429076999]}

## Binary search (step 16) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=62.94061279296875
rel_dist={0: [-54.2988042812305, 54.298804196664946]}

## Binary search (step 17) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=62.94061279296875
rel_dist={0: [-54.298804159499696, 54.29880423627368]}

## Binary search (step 18) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=62.94061279296875
rel_dist={0: [-54.298804174574364, 54.298804367142566]}

## Binary Search Result
Binary search time: 89.88 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1105.63 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2914959, upper bound: 54.1871416
time: 0.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1790725, upper bound: 54.1790725
time: 0.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -54.2914959, upper bound: 54.1871416
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -54.1790725, upper bound: 54.1790725

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -21.4620304, 39.5157166, -22.1557789, 40.7848358, -62.2468643, 61.6714935
1: -24.1841469, 36.9097633, -24.9570370, 38.1305771, -62.3147202, 61.8667984
2: -24.7331676, 36.1095581, -25.5280037, 37.2813034, -62.0144730, 61.6375618
3: -29.7504082, 42.7644386, -30.7027779, 44.2187729, -73.9691696, 73.4672165
4: -27.9875393, 40.3854408, -28.9112663, 41.7009773, -69.6885147, 69.2967072

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2527251, upper bound: 54.1823729
time: 0.64 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2575290, upper bound: 54.1821137
time: 0.98 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -26.7091293, 48.1378326, -21.9749126, 40.4432640, -67.1523895, 70.1127396
1: -30.1031876, 45.2596588, -24.7504501, 37.8254662, -67.9286499, 70.0101089
2: -30.7034187, 44.1854095, -25.3195629, 36.9863930, -67.6897888, 69.5049591
3: -36.9264183, 52.4159775, -30.4443893, 43.8622894, -80.7886887, 82.8603668
4: -34.6874695, 49.6881943, -28.6792946, 41.3561096, -76.0435791, 78.3674927

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9926693, upper bound: 54.1465253
time: 0.74 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1748440, upper bound: 54.1748441
time: 0.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.05 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 0, lower bound: -54.2527251, upper bound: 54.1823729
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 0, lower bound: -54.2575290, upper bound: 54.1821137
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 0, lower bound: -53.9926693, upper bound: 54.1465253
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.05
Output dim: 0, lower bound: -54.1748440, upper bound: 54.1748441

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -17.7770462, 33.3612900, -22.1557789, 40.7848358, -58.5618820, 55.5170670
1: -20.0228424, 30.7962456, -24.9570370, 38.1305771, -58.1534119, 55.7532806
2: -20.5344810, 30.2653313, -25.5280037, 37.2813034, -57.8157845, 55.7933311
3: -24.5992012, 35.4297714, -30.7027779, 44.2187729, -68.8179626, 66.1325531
4: -23.1935501, 33.5928764, -28.9112663, 41.7009773, -64.8945312, 62.5041428

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2016667, upper bound: 54.0650355
time: 0.64 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2016667, upper bound: 54.0652229
time: 0.85 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -22.4064369, 41.7706070, -22.1557789, 40.7848358, -63.1912613, 63.9263840
1: -25.2263889, 38.4714890, -24.9570370, 38.1305771, -63.3569641, 63.4285278
2: -25.8467789, 37.6974525, -25.5280037, 37.2813034, -63.1280823, 63.2254562
3: -31.0495148, 44.5623512, -30.7027779, 44.2187729, -75.2682724, 75.2651291
4: -29.1854439, 42.1343079, -28.9112663, 41.7009773, -70.8864212, 71.0455780

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1984376, upper bound: 54.0649025
time: 0.93 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1444543, upper bound: 54.0599679
time: 0.61 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -21.9749126, 40.4432640, -63.6948929, 64.1023788
1: -26.1851673, 39.4338531, -24.7504501, 37.8254662, -64.0106354, 64.1843033
2: -26.7555904, 38.6069260, -25.3195629, 36.9863930, -63.7419624, 63.9264755
3: -32.0657310, 45.5586777, -30.4443893, 43.8622894, -75.9280243, 76.0030670
4: -30.2180138, 43.1763458, -28.6792946, 41.3561096, -71.5741272, 71.8556366

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9926693, upper bound: 54.1465253
time: 0.72 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9820428, upper bound: 54.0426481
time: 0.64 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -27.9166622, 50.7506638, -21.9749126, 40.4432640, -68.3599167, 72.7255783
1: -31.3626385, 47.0429802, -24.7504501, 37.8254662, -69.1881027, 71.7934265
2: -32.1079407, 45.9946899, -25.3195629, 36.9863930, -69.0942993, 71.3142395
3: -38.4340134, 54.6292343, -30.4443893, 43.8622894, -82.2962875, 85.0736237
4: -36.1575050, 51.6862259, -28.6792946, 41.3561096, -77.5136108, 80.3655090

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1619239, upper bound: 54.0606097
time: 0.63 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
time: 0.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.20 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -54.2016667, upper bound: 54.0650355
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -54.2016667, upper bound: 54.0652229
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -54.1984376, upper bound: 54.0649025
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -54.1444543, upper bound: 54.0599679
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -53.9926693, upper bound: 54.1465253
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -53.9820428, upper bound: 54.0426481
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -54.1619239, upper bound: 54.0606097
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -16.5308781, 31.2338371, -22.1557789, 40.7848358, -57.3157120, 53.3896179
1: -18.6381226, 28.8172264, -24.9570370, 38.1305771, -56.7686996, 53.7742615
2: -19.1109467, 28.3388729, -25.5280037, 37.2813034, -56.3922424, 53.8668747
3: -22.8944168, 33.0924683, -30.7027779, 44.2187729, -67.1131744, 63.7952461
4: -21.6167774, 31.3948212, -28.9112663, 41.7009773, -63.3177567, 60.3060875

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1609253, upper bound: 53.9959512
time: 0.74 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1609253, upper bound: 54.0650355
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -21.9719696, 40.4561348, -57.7771912, 54.6418228
1: -19.5089912, 30.2682056, -24.7536144, 37.8298950, -57.3388863, 55.0218124
2: -20.0418243, 29.7094154, -25.3177185, 36.9890594, -57.0308838, 55.0271301
3: -23.9326496, 34.9102211, -30.4559536, 43.8661957, -67.7988434, 65.3661728
4: -22.7689781, 32.8528252, -28.6795635, 41.3692627, -64.1382446, 61.5323868

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1609253, upper bound: 53.9962945
time: 0.66 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1609253, upper bound: 54.0652229
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -22.1557789, 40.7848358, -61.7915192, 61.4297485
1: -23.6711845, 36.2145233, -24.9570370, 38.1305771, -61.8017616, 61.1715622
2: -24.2448483, 35.5027695, -25.5280037, 37.2813034, -61.5261536, 61.0307732
3: -29.1427917, 41.9068718, -30.7027779, 44.2187729, -73.3615646, 72.6096497
4: -27.4242115, 39.6093559, -28.9112663, 41.7009773, -69.1251907, 68.5206223

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1706459, upper bound: 53.9958797
time: 0.93 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1706459, upper bound: 54.0647810
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -22.4484406, 41.5291443, -21.9719696, 40.4561348, -62.9045753, 63.5010948
1: -25.2864914, 38.5519638, -24.7536144, 37.8298950, -63.1163864, 63.3055649
2: -25.9020805, 37.7519112, -25.3177185, 36.9890594, -62.8911400, 63.0696259
3: -31.0561619, 44.6263237, -30.4559536, 43.8661957, -74.9223480, 75.0822678
4: -29.3174286, 42.1048203, -28.6795635, 41.3692627, -70.6866913, 70.7843781

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1389988, upper bound: 53.9939070
time: 0.89 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1389988, upper bound: 54.0481037
time: 0.60 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -20.7481594, 38.2981796, -61.5498199, 62.8756180
1: -26.1851673, 39.4338531, -23.3815594, 35.8556557, -62.0408249, 62.8154068
2: -26.7555904, 38.6069260, -23.9160175, 35.0785332, -61.8341064, 62.5229416
3: -32.0657310, 45.5586777, -28.7637291, 41.5402298, -73.6059570, 74.3224030
4: -30.2180138, 43.1763458, -27.1237297, 39.1522064, -69.3702164, 70.3000793

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9920750, upper bound: 54.1390968
time: 1.08 seconds

## Relational analysis of IS_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9829485, upper bound: 54.1439708
time: 0.80 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9791379, upper bound: 54.1079156
time: 0.75 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -23.0131550, 41.6745110, -22.0372124, 40.4683151, -63.4814682, 63.7117233
1: -25.9205322, 39.0511322, -24.8191833, 38.1223183, -64.0428467, 63.8703117
2: -26.4819469, 38.2338181, -25.4267216, 37.2558136, -63.7377625, 63.6605377
3: -31.7458191, 45.1090736, -30.4748821, 44.1751518, -75.9209595, 75.5839539
4: -29.9184723, 42.7505302, -28.9127579, 41.4936104, -71.4120789, 71.6632843

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9765872, upper bound: 53.9765872
time: 0.80 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9765872, upper bound: 54.0426481
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -26.3305855, 47.8209915, -21.9749126, 40.4432640, -66.7738495, 69.7959061
1: -29.5782242, 44.4302101, -24.7504501, 37.8254662, -67.4036865, 69.1806564
2: -30.2837696, 43.4649162, -25.3195629, 36.9863930, -67.2701416, 68.7844467
3: -36.2265930, 51.5584373, -30.4443893, 43.8622894, -80.0888672, 82.0028229
4: -34.1130981, 48.7578659, -28.6792946, 41.3561096, -75.4691925, 77.4371490

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1465253, upper bound: 53.9926693
time: 0.62 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1465253, upper bound: 54.0606097
time: 0.66 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -25.3870430, 45.4857788, -21.7962761, 40.1227722, -65.5097961, 67.2820435
1: -28.5911274, 42.9807968, -24.5532036, 37.5333977, -66.1245270, 67.5339737
2: -29.2279434, 42.0015869, -25.1151505, 36.7023239, -65.9302597, 67.1167374
3: -35.0345459, 49.8726997, -30.2055187, 43.5198669, -78.5544052, 80.0782166
4: -33.0906830, 47.0013428, -28.4543819, 41.0344467, -74.1251221, 75.4557266

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
time: 0.62 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
time: 0.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.88 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1609253, upper bound: 53.9959512
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1609253, upper bound: 54.0650355
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1609253, upper bound: 53.9962945
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1609253, upper bound: 54.0652229
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1706459, upper bound: 53.9958797
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1706459, upper bound: 54.0647810
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1389988, upper bound: 53.9939070
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1389988, upper bound: 54.0481037
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -53.9829485, upper bound: 54.1439708
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -53.9791379, upper bound: 54.1079156
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -53.9765872, upper bound: 53.9765872
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -53.9765872, upper bound: 54.0426481
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1465253, upper bound: 53.9926693
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.1465253, upper bound: 54.0606097
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.88
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -16.5308781, 31.2338371, -18.4868755, 34.6143112, -51.1451797, 49.7207108
1: -18.6381226, 28.8172264, -20.8144035, 32.0120316, -50.6501541, 49.6316299
2: -19.1109467, 28.3388729, -21.3472996, 31.4293518, -50.5402985, 49.6861725
3: -22.8944168, 33.0924683, -25.5772781, 36.8826523, -59.7770538, 58.6697464
4: -21.6167774, 31.3948212, -24.1322937, 34.9039764, -56.5207520, 55.5271072

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1601727, upper bound: 53.9957568
time: 0.72 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1559255, upper bound: 53.9956103
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -16.5308781, 31.2338371, -23.0712452, 42.9994736, -59.5303497, 54.3050842
1: -18.6381226, 28.8172264, -25.9661083, 39.6324463, -58.2705688, 54.7833328
2: -19.1109467, 28.3388729, -26.6078491, 38.8119164, -57.9228554, 54.9467239
3: -22.8944168, 33.0924683, -31.9601364, 45.9458580, -68.8402634, 65.0526047
4: -21.6167774, 31.3948212, -30.0656071, 43.3875542, -65.0043335, 61.4604263

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1601727, upper bound: 54.0414928
time: 0.74 seconds

## Relational analysis of IS_A1_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1559255, upper bound: 54.0415125
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -18.3061295, 34.2973289, -51.6183815, 50.9759827
1: -19.5089912, 30.2682056, -20.6146069, 31.7220993, -51.2310905, 50.8828125
2: -20.0418243, 29.7094154, -21.1406250, 31.1471024, -51.1889267, 50.8500404
3: -23.9326496, 34.9102211, -25.3340645, 36.5417137, -60.4743614, 60.2442818
4: -22.7689781, 32.8528252, -23.9055099, 34.5824242, -57.3514023, 56.7583351

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1543383, upper bound: 53.9943762
time: 0.90 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1538079, upper bound: 53.9943279
time: 0.95 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -22.9181595, 42.7247810, -60.0458374, 55.5880127
1: -19.5089912, 30.2682056, -25.7969837, 39.3840027, -58.8929825, 56.0651855
2: -20.0418243, 29.7094154, -26.4328575, 38.5703392, -58.6121635, 56.1422729
3: -23.9326496, 34.9102211, -31.7542496, 45.6534767, -69.5861206, 66.6644745
4: -22.7689781, 32.8528252, -29.8733959, 43.1125488, -65.8815155, 62.7262192

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1539124, upper bound: 54.0562866
time: 0.75 seconds

## Relational analysis of IS_A1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1537501, upper bound: 54.0560538
time: 0.80 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -18.4868755, 34.6143112, -55.6209946, 57.7608490
1: -23.6711845, 36.2145233, -20.8144035, 32.0120316, -55.6832161, 57.0289230
2: -24.2448483, 35.5027695, -21.3472996, 31.4293518, -55.6742020, 56.8500671
3: -29.1427917, 41.9068718, -25.5772781, 36.8826523, -66.0254440, 67.4841385
4: -27.4242115, 39.6093559, -24.1322937, 34.9039764, -62.3281746, 63.7416420

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1533565, upper bound: 53.9939614
time: 1.09 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2

### Relational analysis result of IS_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1611590, upper bound: 53.9939597
time: 0.70 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -23.0712452, 42.9994736, -64.0061493, 62.3452148
1: -23.6711845, 36.2145233, -25.9661083, 39.6324463, -63.3036270, 62.1806259
2: -24.2448483, 35.5027695, -26.6078491, 38.8119164, -63.0567627, 62.1106186
3: -29.1427917, 41.9068718, -31.9601364, 45.9458580, -75.0886536, 73.8670044
4: -27.4242115, 39.6093559, -30.0656071, 43.3875542, -70.8117599, 69.6749496

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B2_A1

### Relational analysis result of IS_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1533565, upper bound: 54.0558447
time: 0.76 seconds

## Relational analysis of IS_A1_A2_A1_B2_A2

### Relational analysis result of IS_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1611591, upper bound: 54.0556636
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -22.4484406, 41.5291443, -18.3061295, 34.2973289, -56.7457695, 59.8352737
1: -25.2864914, 38.5519638, -20.6146069, 31.7220993, -57.0085907, 59.1665688
2: -25.9020805, 37.7519112, -21.1406250, 31.1471024, -57.0491829, 58.8925362
3: -31.0561619, 44.6263237, -25.3340645, 36.5417137, -67.5978775, 69.9603729
4: -29.3174286, 42.1048203, -23.9055099, 34.5824242, -63.8998528, 66.0103302

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B1_A1

### Relational analysis result of IS_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1382461, upper bound: 53.9936125
time: 0.89 seconds

## Relational analysis of IS_A1_A2_A2_B1_A2

### Relational analysis result of IS_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1211881, upper bound: 53.9920034
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -22.4484406, 41.5291443, -22.9181595, 42.7247810, -65.1732178, 64.4472809
1: -25.2864914, 38.5519638, -25.7969837, 39.3840027, -64.6704941, 64.3489304
2: -25.9020805, 37.7519112, -26.4328575, 38.5703392, -64.4724197, 64.1847687
3: -31.0561619, 44.6263237, -31.7542496, 45.6534767, -76.7096329, 76.3805695
4: -29.3174286, 42.1048203, -29.8733959, 43.1125488, -72.4299698, 71.9782028

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B2_A1

### Relational analysis result of IS_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1382461, upper bound: 54.0301794
time: 0.76 seconds

## Relational analysis of IS_A1_A2_A2_B2_A2

### Relational analysis result of IS_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1211881, upper bound: 54.0228657
time: 0.69 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -23.1439342, 41.9426193, -18.8565254, 35.2581787, -58.4021149, 60.7991409
1: -26.0632668, 39.2541008, -21.2323189, 32.8468399, -58.9101067, 60.4864197
2: -26.6321430, 38.4330444, -21.7648010, 32.1878815, -58.8200226, 60.1978455
3: -31.9142723, 45.3474846, -26.0929852, 37.9828491, -69.8970947, 71.4404678
4: -30.0768166, 42.9752731, -24.6841431, 35.7008400, -65.7776566, 67.6594086

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9733032, upper bound: 54.1370199
time: 0.58 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9829184, upper bound: 54.1380535
time: 0.70 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9829485, upper bound: 54.1439708
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9829485, upper bound: 54.1439708
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -19.8570061, 36.7411232, -59.9927444, 61.9844780
1: -26.1851673, 39.4338531, -22.3912964, 34.3691978, -60.5543518, 61.8251419
2: -26.7555904, 38.6069260, -22.8991966, 33.6290970, -60.3846779, 61.5061226
3: -32.0657310, 45.5586777, -27.5594521, 39.8093071, -71.8750381, 73.1181335
4: -30.2180138, 43.1763458, -25.9880447, 37.5196838, -67.7377014, 69.1643906

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9690634, upper bound: 54.0985561
time: 0.83 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9790395, upper bound: 54.1079156
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9791379, upper bound: 54.1079156
time: 0.76 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9791379, upper bound: 54.1079156
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -23.0131550, 41.6745110, -17.9301167, 33.6796761, -56.6928329, 59.6046295
1: -25.9205322, 39.0511322, -20.1899490, 31.2837181, -57.2042465, 59.2410812
2: -26.4819469, 38.2338181, -20.7305031, 30.6835175, -57.1654510, 58.9643059
3: -31.7458191, 45.1090736, -24.7750206, 36.1267471, -67.8725510, 69.8840942
4: -29.9184723, 42.7505302, -23.5687275, 33.9583397, -63.8768044, 66.3192596

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9765872, upper bound: 53.9765872
time: 0.83 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9765872, upper bound: 53.9765872
time: 0.77 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -23.0131550, 41.6745110, -22.9330025, 42.3720627, -65.3852081, 64.6075134
1: -25.9205322, 39.0511322, -25.8211708, 39.3839493, -65.3044815, 64.8722992
2: -26.4819469, 38.2338181, -26.4544449, 38.5499840, -65.0319290, 64.6882401
3: -31.7458191, 45.1090736, -31.7128220, 45.6295700, -77.3753586, 76.8218994
4: -29.9184723, 42.7505302, -29.9670410, 42.9926949, -72.9111633, 72.7175674

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9670616, upper bound: 54.0356702
time: 0.74 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9613336, upper bound: 54.0352413
time: 0.72 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -26.3305855, 47.8209915, -18.3034840, 34.2739182, -60.6045036, 66.1244736
1: -29.5782242, 44.4302101, -20.6071434, 31.7062950, -61.2845154, 65.0373383
2: -30.2837696, 43.4649162, -21.1361504, 31.1318893, -61.4156532, 64.6010513
3: -36.2265930, 51.5584373, -25.3187904, 36.5232048, -72.7497940, 76.8772278
4: -34.1130981, 48.7578659, -23.8958721, 34.5622978, -68.6753998, 72.6537323

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1390968, upper bound: 53.9920750
time: 0.72 seconds

## Relational analysis of IS_A2_A2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1439708, upper bound: 53.9829485
time: 0.77 seconds

## Relational analysis of IS_A2_A2_A1_B1_A2

### Relational analysis result of IS_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1079155, upper bound: 53.9791379
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -26.3305855, 47.8209915, -22.8873901, 42.6426125, -68.9731903, 70.7083740
1: -29.5782242, 44.4302101, -25.7547188, 39.3206635, -68.8988876, 70.1849136
2: -30.2837696, 43.4649162, -26.3955593, 38.5092926, -68.7930603, 69.8604736
3: -36.2265930, 51.5584373, -31.6952229, 45.5803528, -81.8069458, 83.2536621
4: -34.1130981, 48.7578659, -29.8270912, 43.0329742, -77.1460724, 78.5849609

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A1_B2_A1

### Relational analysis result of IS_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1390968, upper bound: 54.0373703
time: 0.93 seconds

## Relational analysis of IS_A2_A2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_A1_B2_A1

### Relational analysis result of IS_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1439708, upper bound: 54.0535790
time: 1.14 seconds

## Relational analysis of IS_A2_A2_A1_B2_A2

### Relational analysis result of IS_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1079156, upper bound: 54.0479578
time: 0.90 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -25.3870430, 45.4857788, -20.7481594, 38.2981796, -63.6852188, 66.2339172
1: -28.5911274, 42.9807968, -23.3815594, 35.8556557, -64.4467850, 66.3623352
2: -29.2279434, 42.0015869, -23.9160175, 35.0785332, -64.3064728, 65.9175949
3: -35.0345459, 49.8726997, -28.7637291, 41.5402298, -76.5747604, 78.6364288
4: -33.0906830, 47.0013428, -27.1237297, 39.1522064, -72.2428818, 74.1250763

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0247501, upper bound: 54.0431287
time: 0.84 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9374045, upper bound: 54.0389105
time: 0.99 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
time: 0.76 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
time: 0.70 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -25.3870430, 45.4857788, -22.0372124, 40.4683151, -65.8553467, 67.5229797
1: -28.5911274, 42.9807968, -24.8191833, 38.1223183, -66.7134476, 67.7999573
2: -29.2279434, 42.0015869, -25.4267216, 37.2558136, -66.4837570, 67.4283066
3: -35.0345459, 49.8726997, -30.4748821, 44.1751518, -79.2096863, 80.3475800
4: -33.0906830, 47.0013428, -28.9127579, 41.4936104, -74.5842896, 75.9141006

Time for backsubstitution: 2.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
time: 0.88 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
time: 0.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 12.64 seconds
IS_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1601727, upper bound: 53.9957568
IS_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1559255, upper bound: 53.9956103
IS_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1601727, upper bound: 54.0414928
IS_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1559255, upper bound: 54.0415125
IS_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1543383, upper bound: 53.9943762
IS_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1538079, upper bound: 53.9943279
IS_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1539124, upper bound: 54.0562866
IS_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1537501, upper bound: 54.0560538
IS_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1533565, upper bound: 53.9939614
IS_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1611590, upper bound: 53.9939597
IS_A1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1533565, upper bound: 54.0558447
IS_A1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1611591, upper bound: 54.0556636
IS_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1382461, upper bound: 53.9936125
IS_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1211881, upper bound: 53.9920034
IS_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1382461, upper bound: 54.0301794
IS_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1211881, upper bound: 54.0228657
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -53.9829485, upper bound: 54.1439708
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -53.9829485, upper bound: 54.1439708
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -53.9791379, upper bound: 54.1079156
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -53.9791379, upper bound: 54.1079156
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -53.9765872, upper bound: 53.9765872
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -53.9765872, upper bound: 53.9765872
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -53.9670616, upper bound: 54.0356702
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -53.9613336, upper bound: 54.0352413
IS_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1439708, upper bound: 53.9829485
IS_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1079155, upper bound: 53.9791379
IS_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1439708, upper bound: 54.0535790
IS_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.1079156, upper bound: 54.0479578
IS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
IS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
IS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037
IS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 12.64
Output dim: 0, lower bound: -54.0481037, upper bound: 54.0481037

## BFS IS instance: IS_A1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.7575665, 28.1929665, -18.4868755, 34.6143112, -49.3718719, 46.6798401
1: -16.6229954, 25.9320927, -20.8144035, 32.0120316, -48.6350250, 46.7464981
2: -17.0983772, 25.5411491, -21.3472996, 31.4293518, -48.5277290, 46.8884506
3: -20.3865776, 29.7530460, -25.5772781, 36.8826523, -57.2692299, 55.3303223
4: -19.3397903, 28.1963501, -24.1322937, 34.9039764, -54.2437515, 52.3286400

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1531597, upper bound: 53.9938982
time: 0.88 seconds

## Relational analysis of IS_A1_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1529975, upper bound: 53.9938945
time: 1.28 seconds

## BFS IS instance: IS_A1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18.9096489, 35.2107239, -18.4868755, 34.6143112, -53.5239525, 53.6976013
1: -21.2144661, 32.0417938, -20.8144035, 32.0120316, -53.2264938, 52.8561935
2: -21.8204556, 31.5343571, -21.3472996, 31.4293518, -53.2498093, 52.8816566
3: -25.9214249, 36.9682693, -25.5772781, 36.8826523, -62.8040771, 62.5455475
4: -24.3476276, 35.2109489, -24.1322937, 34.9039764, -59.2515869, 59.3432236

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1488490, upper bound: 53.9937485
time: 1.08 seconds

## Relational analysis of IS_A1_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1409303, upper bound: 53.9934880
time: 1.47 seconds

## BFS IS instance: IS_A1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14.7575665, 28.1929665, -23.0712452, 42.9994736, -57.7570419, 51.2642136
1: -16.6229954, 25.9320927, -25.9661083, 39.6324463, -56.2554398, 51.8981972
2: -17.0983772, 25.5411491, -26.6078491, 38.8119164, -55.9102936, 52.1489983
3: -20.3865776, 29.7530460, -31.9601364, 45.9458580, -66.3324280, 61.7131805
4: -19.3397903, 28.1963501, -30.0656071, 43.3875542, -62.7273369, 58.2619476

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6549401, upper bound: 52.7405839
time: 0.87 seconds

## Relational analysis of IS_A1_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1682970, upper bound: 54.0185911
time: 0.99 seconds

## BFS IS instance: IS_A1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.9096489, 35.2107239, -23.0712452, 42.9994736, -61.9091034, 58.2819672
1: -21.2144661, 32.0417938, -25.9661083, 39.6324463, -60.8469009, 58.0078926
2: -21.8204556, 31.5343571, -26.6078491, 38.8119164, -60.6323700, 58.1422043
3: -25.9214249, 36.9682693, -31.9601364, 45.9458580, -71.8672791, 68.9284058
4: -24.3476276, 35.2109489, -30.0656071, 43.3875542, -67.7351837, 65.2765503

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8049611, upper bound: 53.1098148
time: 0.85 seconds

## Relational analysis of IS_A1_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1685824, upper bound: 54.0186070
time: 1.35 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15.8795328, 30.3089066, -18.3061295, 34.2973289, -50.1768608, 48.6150360
1: -17.9076004, 27.9379883, -20.6146069, 31.7220993, -49.6296997, 48.5525932
2: -18.4050026, 27.4526844, -21.1406250, 31.1471024, -49.5521049, 48.5932961
3: -21.9679794, 32.1441765, -25.3340645, 36.5417137, -58.5096855, 57.4782333
4: -20.9160748, 30.2952805, -23.9055099, 34.5824242, -55.4984970, 54.2007904

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1543383, upper bound: 53.9943762
time: 1.19 seconds

## Relational analysis of IS_A1_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1543383, upper bound: 53.9943762
time: 1.22 seconds

## BFS IS instance: IS_A1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16.9116573, 31.9373970, -18.2710571, 34.2380142, -51.1496658, 50.2084541
1: -19.0364380, 29.5442429, -20.5751457, 31.6649628, -50.7014008, 50.1193886
2: -19.5863628, 29.0132828, -21.1005993, 31.0919151, -50.6782761, 50.1138763
3: -23.2963791, 34.0250092, -25.2851257, 36.4739189, -59.7702827, 59.3101311
4: -22.2521667, 32.0214119, -23.8604622, 34.5188675, -56.7710266, 55.8818703

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1538079, upper bound: 53.9943279
time: 1.11 seconds

## Relational analysis of IS_A1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1538079, upper bound: 53.9943279
time: 1.37 seconds

## BFS IS instance: IS_A1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15.8795328, 30.3089066, -22.9181595, 42.7247810, -58.6043129, 53.2270584
1: -17.9076004, 27.9379883, -25.7969837, 39.3840027, -57.2916031, 53.7349701
2: -18.4050026, 27.4526844, -26.4328575, 38.5703392, -56.9753418, 53.8855286
3: -21.9679794, 32.1441765, -31.7542496, 45.6534767, -67.6214523, 63.8984261
4: -20.9160748, 30.2952805, -29.8733959, 43.1125488, -64.0286102, 60.1686783

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8540959, upper bound: 53.2536435
time: 0.80 seconds

## Relational analysis of IS_A1_A1_A2_B2_A1_A2

### Relational analysis result of IS_A1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1766734, upper bound: 54.0346881
time: 1.29 seconds

## BFS IS instance: IS_A1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16.9116573, 31.9373970, -22.8854332, 42.6675415, -59.5792007, 54.8228302
1: -19.0364380, 29.5442429, -25.7601776, 39.3298607, -58.3662834, 55.3044205
2: -19.5863628, 29.0132828, -26.3954201, 38.5180473, -58.1044083, 55.4087029
3: -23.2963791, 34.0250092, -31.7086620, 45.5893707, -68.8857498, 65.7336731
4: -22.2521667, 32.0214119, -29.8312740, 43.0521545, -65.3043213, 61.8526726

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1858028, upper bound: 54.0560537
time: 0.84 seconds

## Relational analysis of IS_A1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1858028, upper bound: 54.0560537
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -19.5626640, 36.8832169, -18.4868755, 34.6143112, -54.1769753, 55.3700943
1: -22.0713921, 33.8672447, -20.8144035, 32.0120316, -54.0834198, 54.6816406
2: -22.6047363, 33.2341690, -21.3472996, 31.4293518, -54.0340805, 54.5814590
3: -27.1787014, 39.1302681, -25.5772781, 36.8826523, -64.0613403, 64.7075348
4: -25.5730801, 37.0336227, -24.1322937, 34.9039764, -60.4770432, 61.1658974

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B1_A1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1526039, upper bound: 53.9938090
time: 0.79 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_A1_A1

### Relational analysis result of IS_A1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1457133, upper bound: 53.9837029
time: 0.74 seconds

## Relational analysis of IS_A1_A2_A1_B1_A1_A2

### Relational analysis result of IS_A1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1476398, upper bound: 53.9845714
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -20.2488823, 37.9664078, -18.4517918, 34.5549431, -54.8038254, 56.4181976
1: -22.8299332, 34.8501892, -20.7749252, 31.9548569, -54.7847900, 55.6251144
2: -23.3819160, 34.1935081, -21.3072605, 31.3741302, -54.7560463, 55.5007706
3: -28.0467186, 40.2492828, -25.5283241, 36.8148041, -64.8615112, 65.7776031
4: -26.4318771, 38.1400070, -24.0872231, 34.8403816, -61.2722435, 62.2272301

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_A2_A1

### Relational analysis result of IS_A1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1562137, upper bound: 53.9840130
time: 0.66 seconds

## Relational analysis of IS_A1_A2_A1_B1_A2_A2

### Relational analysis result of IS_A1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1551413, upper bound: 53.9845698
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -19.5626640, 36.8832169, -23.0712452, 42.9994736, -62.5621262, 59.9544525
1: -22.0713921, 33.8672447, -25.9661083, 39.6324463, -61.7038269, 59.8333511
2: -22.6047363, 33.2341690, -26.6078491, 38.8119164, -61.4166451, 59.8420029
3: -27.1787014, 39.1302681, -31.9601364, 45.9458580, -73.1245575, 71.0904083
4: -25.5730801, 37.0336227, -30.0656071, 43.3875542, -68.9606323, 67.0992126

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B2_A1_A1

### Relational analysis result of IS_A1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.4300157, upper bound: 52.3306178
time: 1.03 seconds

## Relational analysis of IS_A1_A2_A1_B2_A1_A2

### Relational analysis result of IS_A1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1853158, upper bound: 54.0343514
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -20.2488823, 37.9664078, -23.0385208, 42.9422188, -63.1911011, 61.0049286
1: -22.8299332, 34.8501892, -25.9293022, 39.5783081, -62.4082413, 60.7794914
2: -23.3819160, 34.1935081, -26.5704041, 38.7596283, -62.1415443, 60.7639122
3: -28.0467186, 40.2492828, -31.9145546, 45.8817673, -73.9284668, 72.1638184
4: -26.4318771, 38.1400070, -30.0234680, 43.3272018, -69.7590790, 68.1634598

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1736285, upper bound: 54.0532333
time: 0.90 seconds

## Relational analysis of IS_A1_A2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1963535, upper bound: 54.0556636
time: 0.98 seconds

## Relational analysis of IS_A1_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1963535, upper bound: 54.0556636
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -20.8273449, 38.6348724, -18.3061295, 34.2973289, -55.1246719, 56.9410019
1: -23.4697704, 35.9373055, -20.6146069, 31.7220993, -55.1918716, 56.5519066
2: -24.0478859, 35.2222443, -21.1406250, 31.1471024, -55.1949883, 56.3628693
3: -28.8204193, 41.5316734, -25.3340645, 36.5417137, -65.3621368, 66.8657150
4: -27.2456913, 39.2179260, -23.9055099, 34.5824242, -61.8281174, 63.1234245

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1382461, upper bound: 53.9936125
time: 0.91 seconds

## Relational analysis of IS_A1_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1382461, upper bound: 53.9936125
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.2482128, 40.9994583, -18.3061295, 34.2973289, -56.5455284, 59.3055801
1: -25.0029221, 37.9370193, -20.6146069, 31.7220993, -56.7250214, 58.5516281
2: -25.6542149, 37.1875839, -21.1406250, 31.1471024, -56.8013153, 58.3282089
3: -30.6625767, 43.9393959, -25.3340645, 36.5417137, -67.2042923, 69.2734451
4: -28.8929577, 41.4821053, -23.9055099, 34.5824242, -63.4753723, 65.3876190

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1211881, upper bound: 53.9920034
time: 1.04 seconds

## Relational analysis of IS_A1_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1211881, upper bound: 53.9920034
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -20.8273449, 38.6348724, -22.9181595, 42.7247810, -63.5521240, 61.5530243
1: -23.4697704, 35.9373055, -25.7969837, 39.3840027, -62.8537560, 61.7342911
2: -24.0478859, 35.2222443, -26.4328575, 38.5703392, -62.6182251, 61.6551018
3: -28.8204193, 41.5316734, -31.7542496, 45.6534767, -74.4738922, 73.2859192
4: -27.2456913, 39.2179260, -29.8733959, 43.1125488, -70.3582382, 69.0913086

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1078682, upper bound: 54.0228657
time: 1.22 seconds

## Relational analysis of IS_A1_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1078682, upper bound: 54.0228657
time: 1.28 seconds

## BFS IS instance: IS_A1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.2482128, 40.9994583, -22.9181595, 42.7247810, -64.9729919, 63.9176140
1: -25.0029221, 37.9370193, -25.7969837, 39.3840027, -64.3869247, 63.7339973
2: -25.6542149, 37.1875839, -26.4328575, 38.5703392, -64.2245407, 63.6204414
3: -30.6625767, 43.9393959, -31.7542496, 45.6534767, -76.3160553, 75.6936417
4: -28.8929577, 41.4821053, -29.8733959, 43.1125488, -72.0054932, 71.3554993

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1078682, upper bound: 54.0228657
time: 1.41 seconds

## Relational analysis of IS_A1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1078682, upper bound: 54.0228657
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -23.1439342, 41.9426193, -18.3821220, 34.4183960, -57.5623322, 60.3247185
1: -26.0632668, 39.2541008, -20.7137203, 32.0132790, -58.0765457, 59.9678116
2: -26.6321430, 38.4330444, -21.2223015, 31.3921108, -58.0242538, 59.6553383
3: -31.9142723, 45.3474846, -25.4603615, 36.9794197, -68.8936768, 70.8078461
4: -30.0768166, 42.9752731, -24.0456543, 34.8268623, -64.9036789, 67.0209198

Time for backsubstitution: 2.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9733032, upper bound: 54.1370199
time: 1.10 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9829184, upper bound: 54.1380535
time: 1.04 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9743905, upper bound: 54.1422117
time: 0.96 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9685748, upper bound: 54.1397469
time: 0.99 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -23.1439342, 41.9426193, -23.3192635, 42.2748337, -65.4187698, 65.2618637
1: -26.0632668, 39.2541008, -26.2810936, 39.6397324, -65.7030029, 65.5351944
2: -26.6321430, 38.4330444, -26.8154907, 38.7507286, -65.3828659, 65.2485352
3: -31.9142723, 45.3474846, -32.2024269, 45.9048309, -77.8190918, 77.5499115
4: -30.0768166, 42.9752731, -30.2900677, 43.4219742, -73.4987946, 73.2653351

Time for backsubstitution: 2.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9733032, upper bound: 54.1370199
time: 0.81 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9829184, upper bound: 54.1380535
time: 0.63 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=62.94061279296875
rel_dist={0: [-54.30012139088531, 54.300121390885295]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2872604, upper bound: 54.1866363
time: 0.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1790725, upper bound: 54.1790725
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -54.2872604, upper bound: 54.1866363
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -54.1790725, upper bound: 54.1790725

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -21.4620304, 39.5157166, -22.1557789, 40.7848358, -62.2468643, 61.6714935
1: -24.1841469, 36.9097633, -24.9570370, 38.1305771, -62.3147202, 61.8667984
2: -24.7331676, 36.1095581, -25.5280037, 37.2813034, -62.0144730, 61.6375618
3: -29.7504082, 42.7644386, -30.7027779, 44.2187729, -73.9691696, 73.4672165
4: -27.9875393, 40.3854408, -28.9112663, 41.7009773, -69.6885147, 69.2967072

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2388120, upper bound: 54.1761228
time: 0.63 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2566549, upper bound: 54.1819576
time: 0.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -26.7091293, 48.1378326, -21.5461693, 39.6128311, -66.3219376, 69.6839981
1: -30.1031876, 45.2596588, -24.2659225, 37.1016273, -67.2048187, 69.5255661
2: -30.7034187, 44.1854095, -24.8243217, 36.2833900, -66.9868088, 69.0097198
3: -36.9264183, 52.4159775, -29.8423767, 43.0157852, -79.9421768, 82.2583542
4: -34.6874695, 49.6881943, -28.1310196, 40.5432968, -75.2307663, 77.8192139

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9859275, upper bound: 54.0812001
time: 0.72 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1748440, upper bound: 54.1748441
time: 0.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.37 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -54.2388120, upper bound: 54.1761228
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -54.2566549, upper bound: 54.1819576
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -53.9859275, upper bound: 54.0812001
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -54.1748440, upper bound: 54.1748441

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -17.7770462, 33.3612900, -21.5081902, 39.6732483, -57.4502945, 54.8694801
1: -20.0228424, 30.7962456, -24.2265549, 37.0184097, -57.0412521, 55.0228004
2: -20.5344810, 30.2653313, -24.7906876, 36.2220840, -56.7565651, 55.0560150
3: -24.5992012, 35.4297714, -29.8022003, 42.9040718, -67.5032730, 65.2319717
4: -23.1935501, 33.5928764, -28.0556107, 40.4867325, -63.6802826, 61.6484833

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9877268
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 54.1761228
time: 0.88 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -22.4064369, 41.7706070, -21.3992023, 39.5110817, -61.9175072, 63.1697922
1: -25.2263889, 38.4714890, -24.0871258, 36.8941040, -62.1204910, 62.5586052
2: -25.8467789, 37.6974525, -24.6748428, 36.1065140, -61.9532928, 62.3722954
3: -31.0495148, 44.5623512, -29.6082268, 42.7722549, -73.8217468, 74.1705704
4: -29.1854439, 42.1343079, -27.9484043, 40.2693329, -69.4547729, 70.0826950

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0896865, upper bound: 53.9877268
time: 2.50 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0896865, upper bound: 54.1819576
time: 0.88 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -20.8932667, 38.4961853, -61.7478180, 63.0207329
1: -26.1851673, 39.4338531, -23.5303307, 35.9838333, -62.1689987, 62.9641838
2: -26.7555904, 38.6069260, -24.0820923, 35.2039948, -61.9595642, 62.6890182
3: -32.0657310, 45.5586777, -28.9355888, 41.6932869, -73.7590179, 74.4942627
4: -30.2180138, 43.1763458, -27.2704716, 39.3201599, -69.5381775, 70.4468155

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764952, upper bound: 53.9764952
time: 0.71 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764952, upper bound: 54.0812001
time: 0.81 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -27.9166622, 50.7506638, -20.8003788, 38.3541794, -66.2708435, 71.5510406
1: -31.3626385, 47.0429802, -23.4021950, 35.8803101, -67.2429428, 70.4451752
2: -32.1079407, 45.9946899, -23.9846268, 35.1214218, -67.2293625, 69.9793015
3: -38.4340134, 54.6292343, -28.7536221, 41.5829735, -80.0169754, 83.3828583
4: -36.1575050, 51.6862259, -27.1774101, 39.1221008, -75.2796021, 78.8636322

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0812001, upper bound: 53.9859275
time: 0.68 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0812001, upper bound: 54.1748441
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.16 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9877268
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -54.0787000, upper bound: 54.1761228
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -54.0896865, upper bound: 53.9877268
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -54.0896865, upper bound: 54.1819576
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -53.9764952, upper bound: 53.9764952
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -53.9764952, upper bound: 54.0812001
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -54.0812001, upper bound: 53.9859275
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 0, lower bound: -54.0812001, upper bound: 54.1748441

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -17.7770462, 33.3612900, -18.4868755, 34.6143112, -52.3913498, 51.8481674
1: -20.0228424, 30.7962456, -20.8144035, 32.0120316, -52.0348740, 51.6106491
2: -20.5344810, 30.2653313, -21.3472996, 31.4293518, -51.9638329, 51.6126213
3: -24.5992012, 35.4297714, -25.5772781, 36.8826523, -61.4818497, 61.0070496
4: -23.1935501, 33.5928764, -24.1322937, 34.9039764, -58.0975037, 57.7251587

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0321903, upper bound: 53.9858221
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9884491
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -17.7770462, 33.3612900, -23.0712452, 42.9994736, -60.7765198, 56.4325333
1: -20.0228424, 30.7962456, -25.9661083, 39.6324463, -59.6552734, 56.7623520
2: -20.5344810, 30.2653313, -26.6078491, 38.8119164, -59.3463974, 56.8731804
3: -24.5992012, 35.4297714, -31.9601364, 45.9458580, -70.5450592, 67.3899078
4: -23.1935501, 33.5928764, -30.0656071, 43.3875542, -66.5811005, 63.6584778

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0321903, upper bound: 54.0577150
time: 0.82 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 54.0584632
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -22.4064369, 41.7706070, -18.4868755, 34.6143112, -57.0207443, 60.2574806
1: -25.2263889, 38.4714890, -20.8144035, 32.0120316, -57.2384186, 59.2858887
2: -25.8467789, 37.6974525, -21.3472996, 31.4293518, -57.2761307, 59.0447540
3: -31.0495148, 44.5623512, -25.5772781, 36.8826523, -67.9321671, 70.1396332
4: -29.1854439, 42.1343079, -24.1322937, 34.9039764, -64.0894165, 66.2665863

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0896864, upper bound: 53.9876450
time: 0.79 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9877268
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -22.4064369, 41.7706070, -23.0712452, 42.9994736, -65.4058838, 64.8418503
1: -25.2263889, 38.4714890, -25.9661083, 39.6324463, -64.8588333, 64.4375916
2: -25.8467789, 37.6974525, -26.6078491, 38.8119164, -64.6586914, 64.3052979
3: -31.0495148, 44.5623512, -31.9601364, 45.9458580, -76.9953690, 76.5224915
4: -29.1854439, 42.1343079, -30.0656071, 43.3875542, -72.5729980, 72.1999054

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0896865, upper bound: 54.0580013
time: 0.95 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 54.0480272
time: 1.00 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -17.8692932, 33.4467926, -56.6984329, 59.9967575
1: -26.1851673, 39.4338531, -20.1190281, 30.9808598, -57.1660271, 59.5528755
2: -26.7555904, 38.6069260, -20.6351967, 30.4245644, -57.1801529, 59.2421227
3: -32.0657310, 45.5586777, -24.7122555, 35.6704750, -67.7362061, 70.2709351
4: -30.2180138, 43.1763458, -23.3373508, 33.7536583, -63.9716721, 66.5136948

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764952, upper bound: 53.9764952
time: 1.00 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764952, upper bound: 53.9764952
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -22.3790874, 41.6610069, -64.9126129, 64.5065384
1: -26.1851673, 39.4338531, -25.1683559, 38.4540787, -64.6392441, 64.6022034
2: -26.7555904, 38.6069260, -25.8082199, 37.6694984, -64.4250870, 64.4151459
3: -32.0657310, 45.5586777, -30.9587479, 44.5637016, -76.6294327, 76.5174255
4: -30.2180138, 43.1763458, -29.1638145, 42.0469780, -72.2649918, 72.3401642

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764952, upper bound: 54.0812001
time: 0.82 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764952, upper bound: 54.0812001
time: 0.68 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -27.9166622, 50.7506638, -17.8692932, 33.4467926, -61.3634415, 68.6199493
1: -31.3626385, 47.0429802, -20.1190281, 30.9808598, -62.3434868, 67.1620102
2: -32.1079407, 45.9946899, -20.6351967, 30.4245644, -62.5325012, 66.6298828
3: -38.4340134, 54.6292343, -24.7122555, 35.6704750, -74.1044846, 79.3414917
4: -36.1575050, 51.6862259, -23.3373508, 33.7536583, -69.9111633, 75.0235748

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9003545, upper bound: 53.9642019
time: 0.86 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0773762, upper bound: 53.9764758
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -27.9166622, 50.7506638, -22.3491688, 41.5318413, -69.4484940, 73.0998306
1: -31.3626385, 47.0429802, -25.1342163, 38.3536491, -69.7162857, 72.1771774
2: -32.1079407, 45.9946899, -25.7738743, 37.5732002, -69.6811371, 71.7685547
3: -38.4340134, 54.6292343, -30.9188213, 44.4575539, -82.8915710, 85.5480576
4: -36.1575050, 51.6862259, -29.1282406, 41.9428787, -78.1003876, 80.8144531

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0812001, upper bound: 54.0557383
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0425912, upper bound: 54.0480272
time: 0.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 6.47 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0321903, upper bound: 53.9858221
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9884491
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0321903, upper bound: 54.0577150
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0787000, upper bound: 54.0584632
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0896864, upper bound: 53.9876450
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9877268
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0896865, upper bound: 54.0580013
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0787000, upper bound: 54.0480272
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -53.9764952, upper bound: 53.9764952
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -53.9764952, upper bound: 53.9764952
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -53.9764952, upper bound: 54.0812001
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -53.9764952, upper bound: 54.0812001
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -53.9003545, upper bound: 53.9642019
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0773762, upper bound: 53.9764758
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0812001, upper bound: 54.0557383
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.47
Output dim: 0, lower bound: -54.0425912, upper bound: 54.0480272

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.5308781, 31.2338371, -18.4868755, 34.6143112, -51.1451797, 49.7207108
1: -18.6381226, 28.8172264, -20.8144035, 32.0120316, -50.6501541, 49.6316299
2: -19.1109467, 28.3388729, -21.3472996, 31.4293518, -50.5402985, 49.6861725
3: -22.8944168, 33.0924683, -25.5772781, 36.8826523, -59.7770538, 58.6697464
4: -21.6167774, 31.3948212, -24.1322937, 34.9039764, -56.5207520, 55.5271072

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0316175, upper bound: 53.9856055
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0173796, upper bound: 53.9845558
time: 0.94 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -17.5432320, 32.9866867, -50.3077393, 50.2130852
1: -19.5089912, 30.2682056, -19.7717991, 30.5212173, -50.0302086, 50.0400009
2: -20.0418243, 29.7094154, -20.2705116, 29.9775467, -50.0193710, 49.9799271
3: -23.9326496, 34.9102211, -24.3076992, 35.1291504, -59.0617981, 59.2179184
4: -22.7689781, 32.8528252, -22.9556389, 33.2461967, -56.0151749, 55.8084641

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9884491
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9884491
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.5308781, 31.2338371, -23.0712452, 42.9994736, -59.5303497, 54.3050842
1: -18.6381226, 28.8172264, -25.9661083, 39.6324463, -58.2705688, 54.7833328
2: -19.1109467, 28.3388729, -26.6078491, 38.8119164, -57.9228554, 54.9467239
3: -22.8944168, 33.0924683, -31.9601364, 45.9458580, -68.8402634, 65.0526047
4: -21.6167774, 31.3948212, -30.0656071, 43.3875542, -65.0043335, 61.4604263

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1142591, upper bound: 54.0577150
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1142591, upper bound: 54.0577150
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -22.2001801, 41.4381142, -58.7591705, 54.8700333
1: -19.5089912, 30.2682056, -25.0036697, 38.2131805, -57.7221718, 55.2718735
2: -20.0418243, 29.7094154, -25.6123142, 37.4335480, -57.4753723, 55.3217316
3: -23.9326496, 34.9102211, -30.7872925, 44.2754898, -68.2081375, 65.6975098
4: -22.7689781, 32.8528252, -28.9705982, 41.8185196, -64.5874939, 61.8234215

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1250130, upper bound: 54.0584632
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1250130, upper bound: 54.0584632
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -18.4868755, 34.6143112, -55.6209946, 57.7608490
1: -23.6711845, 36.2145233, -20.8144035, 32.0120316, -55.6832161, 57.0289230
2: -24.2448483, 35.5027695, -21.3472996, 31.4293518, -55.6742020, 56.8500671
3: -29.1427917, 41.9068718, -25.5772781, 36.8826523, -66.0254440, 67.4841385
4: -27.4242115, 39.6093559, -24.1322937, 34.9039764, -62.3281746, 63.7416420

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0321903, upper bound: 53.9876450
time: 0.81 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0321903, upper bound: 53.9876450
time: 1.04 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.4484406, 41.5291443, -17.5432320, 32.9866867, -55.4351273, 59.0723648
1: -25.2864914, 38.5519638, -19.7717991, 30.5212173, -55.8077087, 58.3237610
2: -25.9020805, 37.7519112, -20.2705116, 29.9775467, -55.8796272, 58.0224190
3: -31.0561619, 44.6263237, -24.3076992, 35.1291504, -66.1853027, 68.9340210
4: -29.3174286, 42.1048203, -22.9556389, 33.2461967, -62.5636253, 65.0604553

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9877268
time: 0.92 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9877268
time: 1.74 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -23.0712452, 42.9994736, -64.0061493, 62.3452148
1: -23.6711845, 36.2145233, -25.9661083, 39.6324463, -63.3036270, 62.1806259
2: -24.2448483, 35.5027695, -26.6078491, 38.8119164, -63.0567627, 62.1106186
3: -29.1427917, 41.9068718, -31.9601364, 45.9458580, -75.0886536, 73.8670044
4: -27.4242115, 39.6093559, -30.0656071, 43.3875542, -70.8117599, 69.6749496

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0567199, upper bound: 54.0480272
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0567199, upper bound: 54.0480272
time: 1.37 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.4484406, 41.5291443, -22.2001801, 41.4381142, -63.8865547, 63.7292976
1: -25.2864914, 38.5519638, -25.0036697, 38.2131805, -63.4996719, 63.5556335
2: -25.9020805, 37.7519112, -25.6123142, 37.4335480, -63.3356285, 63.3642235
3: -31.0561619, 44.6263237, -30.7872925, 44.2754898, -75.3316498, 75.4136200
4: -29.3174286, 42.1048203, -28.9705982, 41.8185196, -71.1359482, 71.0754166

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0796370, upper bound: 54.0232445
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0906418, upper bound: 54.0228657
time: 0.84 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -17.7770462, 33.3612900, -56.6129227, 59.9045181
1: -26.1851673, 39.4338531, -20.0228424, 30.7962456, -56.9814148, 59.4566879
2: -26.7555904, 38.6069260, -20.5344810, 30.2653313, -57.0209122, 59.1414070
3: -32.0657310, 45.5586777, -24.5992012, 35.4297714, -67.4954987, 70.1578827
4: -30.2180138, 43.1763458, -23.1935501, 33.5928764, -63.8108902, 66.3698959

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9670007, upper bound: 53.9616827
time: 0.85 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9612526, upper bound: 53.9612526
time: 0.72 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -23.1562748, 41.9909935, -65.2426071, 65.2837372
1: -26.1851673, 39.4338531, -26.0854168, 39.3220367, -65.5072021, 65.5192719
2: -26.7555904, 38.6069260, -26.6512947, 38.4988022, -65.2543793, 65.2582245
3: -32.0657310, 45.5586777, -31.9547825, 45.4292755, -77.4950104, 77.5134583
4: -30.2180138, 43.1763458, -30.1156654, 43.0629501, -73.2809601, 73.2920074

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9616827, upper bound: 53.9670007
time: 1.16 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9612526, upper bound: 53.9612526
time: 0.98 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -22.4064369, 41.7706070, -65.0222244, 64.5338821
1: -26.1851673, 39.4338531, -25.2263889, 38.4714890, -64.6566544, 64.6602402
2: -26.7555904, 38.6069260, -25.8467789, 37.6974525, -64.4530334, 64.4537048
3: -32.0657310, 45.5586777, -31.0495148, 44.5623512, -76.6280823, 76.6081924
4: -30.2180138, 43.1763458, -29.1854439, 42.1343079, -72.3523178, 72.3617859

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9852098, upper bound: 54.0784895
time: 0.84 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9790614, upper bound: 54.0195758
time: 1.51 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -23.2516384, 42.1274719, -27.9166622, 50.7506638, -74.0022736, 70.0441132
1: -26.1851673, 39.4338531, -31.3626385, 47.0429802, -73.2281494, 70.7964935
2: -26.7555904, 38.6069260, -32.1079407, 45.9946899, -72.7502747, 70.7148666
3: -32.0657310, 45.5586777, -38.4340134, 54.6292343, -86.6949615, 83.9926910
4: -30.2180138, 43.1763458, -36.1575050, 51.6862259, -81.9042358, 79.3338470

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9606317, upper bound: 53.9003547
time: 1.02 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764758, upper bound: 54.0773764
time: 1.00 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -24.9989815, 45.6509018, -17.1405716, 32.2228241, -57.2218056, 62.7914734
1: -28.0598755, 42.1997566, -19.2937546, 29.7738991, -57.8337746, 61.4935112
2: -28.7609997, 41.3211327, -19.7993946, 29.2545338, -58.0155334, 61.1205254
3: -34.3129692, 48.9402542, -23.6857548, 34.2639389, -68.5768967, 72.6259842
4: -32.3584633, 46.2080040, -22.3846855, 32.4048386, -64.7633057, 68.5926743

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9655547, upper bound: 53.9642019
time: 0.71 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9401938, upper bound: 53.9529307
time: 0.75 seconds

## Relational analysis of IS_A2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9655547, upper bound: 53.9642019
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9655547, upper bound: 53.9642019
time: 0.83 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -27.1733932, 49.4113731, -17.7389507, 33.2224159, -60.3958092, 67.1503220
1: -30.5476780, 45.8119125, -19.9737701, 30.7707958, -61.3184738, 65.7856750
2: -31.2554703, 44.8115463, -20.4867859, 30.2209587, -61.4764252, 65.2983322
3: -37.4488449, 53.1881409, -24.5352821, 35.4246635, -72.8735046, 77.7234192
4: -35.2109261, 50.3438759, -23.1739292, 33.5195122, -68.7304306, 73.5177994

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0773762, upper bound: 53.9764758
time: 1.13 seconds

## Relational analysis of IS_A2_A2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0353902, upper bound: 53.9721289
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -26.3305855, 47.8209915, -22.3491688, 41.5318413, -67.8624268, 70.1701584
1: -29.5782242, 44.4302101, -25.1342163, 38.3536491, -67.9318695, 69.5643997
2: -30.2837696, 43.4649162, -25.7738743, 37.5732002, -67.8569717, 69.2387695
3: -36.2265930, 51.5584373, -30.9188213, 44.4575539, -80.6841431, 82.4772568
4: -34.1130981, 48.7578659, -29.1282406, 41.9428787, -76.0559769, 77.8860931

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0480272, upper bound: 54.0480272
time: 0.79 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0480272, upper bound: 54.0480272
time: 0.87 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -25.3870430, 45.4857788, -21.4798164, 39.9875755, -65.3745956, 66.9655762
1: -28.5911274, 42.9807968, -24.1730194, 36.9501610, -65.5412903, 67.1538086
2: -29.2279434, 42.0015869, -24.7804985, 36.2099991, -65.4379425, 66.7820816
3: -35.0345459, 49.8726997, -29.7458286, 42.8041458, -77.8386917, 79.6185303
4: -33.0906830, 47.0013428, -28.0345459, 40.3887253, -73.4793930, 75.0358887

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0480272, upper bound: 54.0480272
time: 1.05 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0480272, upper bound: 54.0480272
time: 1.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.92 seconds
IS_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0316175, upper bound: 53.9856055
IS_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0173796, upper bound: 53.9845558
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9884491
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9884491
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.1142591, upper bound: 54.0577150
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.1142591, upper bound: 54.0577150
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.1250130, upper bound: 54.0584632
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.1250130, upper bound: 54.0584632
IS_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0321903, upper bound: 53.9876450
IS_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0321903, upper bound: 53.9876450
IS_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9877268
IS_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0787000, upper bound: 53.9877268
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0567199, upper bound: 54.0480272
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0567199, upper bound: 54.0480272
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0796370, upper bound: 54.0232445
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0906418, upper bound: 54.0228657
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9670007, upper bound: 53.9616827
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9612526, upper bound: 53.9612526
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9616827, upper bound: 53.9670007
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9612526, upper bound: 53.9612526
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9852098, upper bound: 54.0784895
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9790614, upper bound: 54.0195758
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9606317, upper bound: 53.9003547
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9764758, upper bound: 54.0773764
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9655547, upper bound: 53.9642019
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -53.9655547, upper bound: 53.9642019
IS_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0773762, upper bound: 53.9764758
IS_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0353902, upper bound: 53.9721289
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0480272, upper bound: 54.0480272
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0480272, upper bound: 54.0480272
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0480272, upper bound: 54.0480272
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.92
Output dim: 0, lower bound: -54.0480272, upper bound: 54.0480272

## BFS IS instance: IS_A1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -14.7575665, 28.1929665, -18.1947308, 34.1073837, -48.8649406, 46.3876953
1: -16.6229954, 25.9320927, -20.4817753, 31.5364971, -48.1594925, 46.4138680
2: -17.0983772, 25.5411491, -21.0145321, 30.9701366, -48.0685120, 46.5556793
3: -20.3865776, 29.7530460, -25.1647320, 36.3235207, -56.7100983, 54.9177780
4: -19.3397903, 28.1963501, -23.7577686, 34.3713989, -53.7111778, 51.9541168

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9486123, upper bound: 53.9800214
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9486123, upper bound: 53.9837503
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -18.9096489, 35.2107239, -17.7471294, 33.3181534, -52.2278023, 52.9578552
1: -21.2144661, 32.0417938, -19.9770184, 30.7361431, -51.9506035, 52.0188026
2: -21.8204556, 31.5343571, -20.5014935, 30.2064705, -52.0269241, 52.0358505
3: -25.9214249, 36.9682693, -24.5366287, 35.3787918, -61.3002167, 61.5048981
4: -24.3476276, 35.2109489, -23.1613789, 33.5057831, -57.8534088, 58.3723221

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0173796, upper bound: 53.9845558
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0173796, upper bound: 53.9845558
time: 1.00 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -16.8506260, 31.7674160, -49.0884705, 49.5204773
1: -19.5089912, 30.2682056, -18.9991264, 29.3381729, -48.8471642, 49.2673340
2: -20.0418243, 29.7094154, -19.4781857, 28.8419895, -48.8838120, 49.1875992
3: -23.9326496, 34.9102211, -23.3514500, 33.7177162, -57.6503601, 58.2616730
4: -22.7689781, 32.8528252, -22.0431385, 31.9697781, -54.7387543, 54.8959618

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9638582, upper bound: 53.9824327
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0730101, upper bound: 53.9865308
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -21.8094082, 39.4364548, -56.7575073, 54.4792633
1: -19.5089912, 30.2682056, -24.5873260, 37.1600266, -56.6690178, 54.8555298
2: -20.0418243, 29.7094154, -25.1059303, 36.3899307, -56.4317551, 54.8153458
3: -23.9326496, 34.9102211, -30.1401520, 42.8923607, -66.8250122, 65.0503693
4: -22.7689781, 32.8528252, -28.4200897, 40.6501579, -63.4191360, 61.2729149

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9521996, upper bound: 53.9824327
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0730101, upper bound: 53.9865308
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -16.5308781, 31.2338371, -21.6836166, 40.5148277, -57.0457039, 52.9174538
1: -18.6381226, 28.8172264, -24.4234161, 37.3932343, -56.0313568, 53.2406425
2: -19.1109467, 28.3388729, -25.0189934, 36.6350670, -55.7460136, 53.3578644
3: -22.8944168, 33.0924683, -30.0688705, 43.3086166, -66.2030258, 63.1613388
4: -21.6167774, 31.3948212, -28.3149128, 40.8809204, -62.4976959, 59.7097321

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1142591, upper bound: 54.0577150
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1142591, upper bound: 54.0577150
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -16.5308781, 31.2338371, -23.1183300, 42.7579613, -59.2888412, 54.3521652
1: -18.6381226, 28.8172264, -26.0306053, 39.7178230, -58.3559418, 54.8478317
2: -19.1109467, 28.3388729, -26.6682301, 38.8740158, -57.9849586, 55.0070992
3: -22.8944168, 33.0924683, -31.9716911, 46.0147934, -68.9091949, 65.0641632
4: -21.6167774, 31.3948212, -30.2052059, 43.3611488, -64.9779282, 61.6000290

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1142591, upper bound: 54.0577150
time: 1.36 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1142591, upper bound: 54.0577150
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -21.6836166, 40.5148277, -57.8358803, 54.3534660
1: -19.5089912, 30.2682056, -24.4234161, 37.3932343, -56.9022255, 54.6916199
2: -20.0418243, 29.7094154, -25.0189934, 36.6350670, -56.6768913, 54.7284088
3: -23.9326496, 34.9102211, -30.0688705, 43.3086166, -67.2412643, 64.9790955
4: -22.7689781, 32.8528252, -28.3149128, 40.8809204, -63.6498985, 61.1677361

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1242244, upper bound: 54.0563852
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1250130, upper bound: 54.0584632
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1250130, upper bound: 54.0584632
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.3210545, 32.6698532, -23.1183300, 42.7579613, -60.0790176, 55.7881851
1: -19.5089912, 30.2682056, -26.0306053, 39.7178230, -59.2268066, 56.2988091
2: -20.0418243, 29.7094154, -26.6682301, 38.8740158, -58.9158401, 56.3776436
3: -23.9326496, 34.9102211, -31.9716911, 46.0147934, -69.9474258, 66.8819122
4: -22.7689781, 32.8528252, -30.2052059, 43.3611488, -66.1301270, 63.0580254

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1250130, upper bound: 54.0584632
time: 1.25 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1250130, upper bound: 54.0584632
time: 1.02 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -17.7770462, 33.3612900, -54.3679733, 57.0510178
1: -23.6711845, 36.2145233, -20.0228424, 30.7962456, -54.4674301, 56.2373619
2: -24.2448483, 35.5027695, -20.5344810, 30.2653313, -54.5101700, 56.0372505
3: -29.1427917, 41.9068718, -24.5992012, 35.4297714, -64.5725632, 66.5060654
4: -27.4242115, 39.6093559, -23.1935501, 33.5928764, -61.0170708, 62.8029022

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0887903, upper bound: 53.9870680
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0201302, upper bound: 53.9846536
time: 1.29 seconds

## BFS IS instance: IS_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -23.1562748, 41.9909935, -62.9976768, 62.4302444
1: -23.6711845, 36.2145233, -26.0854168, 39.3220367, -62.9932213, 62.2999420
2: -24.2448483, 35.5027695, -26.6512947, 38.4988022, -62.7436523, 62.1540604
3: -29.1427917, 41.9068718, -31.9547825, 45.4292755, -74.5720673, 73.8616486
4: -27.4242115, 39.6093559, -30.1156654, 43.0629501, -70.4871597, 69.7250137

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9460820, upper bound: 53.9802404
time: 1.22 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0826413, upper bound: 53.9856941
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.4484406, 41.5291443, -16.8506260, 31.7674160, -54.2158585, 58.3797569
1: -25.2864914, 38.5519638, -18.9991264, 29.3381729, -54.6246567, 57.5510750
2: -25.9020805, 37.7519112, -19.4781857, 28.8419895, -54.7440720, 57.2300873
3: -31.0561619, 44.6263237, -23.3514500, 33.7177162, -64.7738800, 67.9777756
4: -29.3174286, 42.1048203, -22.0431385, 31.9697781, -61.2872086, 64.1479416

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0849851, upper bound: 53.9871693
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0814194, upper bound: 53.9872791
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.4484406, 41.5291443, -21.8094082, 39.4364548, -61.8848953, 63.3385353
1: -25.2864914, 38.5519638, -24.5873260, 37.1600266, -62.4465179, 63.1392708
2: -25.9020805, 37.7519112, -25.1059303, 36.3899307, -62.2920113, 62.8578339
3: -31.0561619, 44.6263237, -30.1401520, 42.8923607, -73.9485245, 74.7664795
4: -29.3174286, 42.1048203, -28.4200897, 40.6501579, -69.9675903, 70.5249100

Time for backsubstitution: 2.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0849851, upper bound: 53.9871693
time: 1.01 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0814194, upper bound: 53.9872791
time: 0.98 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -21.6836166, 40.5148277, -61.5215149, 60.9575882
1: -23.6711845, 36.2145233, -24.4234161, 37.3932343, -61.0644188, 60.6379280
2: -24.2448483, 35.5027695, -25.0189934, 36.6350670, -60.8799057, 60.5217552
3: -29.1427917, 41.9068718, -30.0688705, 43.3086166, -72.4514084, 71.9757385
4: -27.4242115, 39.6093559, -28.3149128, 40.8809204, -68.3051224, 67.9242630

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1355304, upper bound: 54.0580012
time: 1.29 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1355304, upper bound: 54.0580012
time: 1.09 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.0066853, 39.2739716, -23.1183300, 42.7579613, -63.7646484, 62.3923035
1: -23.6711845, 36.2145233, -26.0306053, 39.7178230, -63.3889999, 62.2451286
2: -24.2448483, 35.5027695, -26.6682301, 38.8740158, -63.1188660, 62.1709976
3: -29.1427917, 41.9068718, -31.9716911, 46.0147934, -75.1575775, 73.8785553
4: -27.4242115, 39.6093559, -30.2052059, 43.3611488, -70.7853546, 69.8145447

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1355304, upper bound: 54.0580012
time: 1.07 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1355304, upper bound: 54.0580012
time: 0.97 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.2257233, 41.1281204, -20.2308922, 37.8341560, -60.0598602, 61.3590126
1: -25.0369930, 38.1925125, -22.7456207, 34.9712601, -60.0082550, 60.9381256
2: -25.6471863, 37.4042206, -23.3625546, 34.3011627, -59.9483299, 60.7667770
3: -30.7498035, 44.2006836, -27.9783401, 40.4751472, -71.2249527, 72.1790085
4: -29.0326462, 41.7082787, -26.4481869, 38.1560440, -67.1886902, 68.1564636

Time for backsubstitution: 2.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0796370, upper bound: 54.0232445
time: 0.97 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0796369, upper bound: 54.0232444
time: 2.26 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.7295628, 40.2910728, -21.4611282, 39.6866493, -61.4162064, 61.7522011
1: -24.4627953, 37.3277931, -24.1077957, 36.5149002, -60.9776878, 61.4355736
2: -25.0836639, 36.5738754, -24.7333221, 35.8324585, -60.9161186, 61.3071976
3: -30.0252247, 43.1889801, -29.6412792, 42.2921944, -72.3174210, 72.8302612
4: -28.3707962, 40.7442093, -27.8022099, 39.9918289, -68.3626251, 68.5463867

Time for backsubstitution: 2.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0906417, upper bound: 54.0228657
time: 1.36 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0906417, upper bound: 54.0228657
time: 0.88 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -22.6879253, 41.1678200, -17.5932045, 33.0357285, -55.7236443, 58.7610245
1: -25.5380325, 38.5620499, -19.8127041, 30.4866123, -56.0246429, 58.3747559
2: -26.1186962, 37.7568893, -20.3245621, 29.9664879, -56.0851822, 58.0814514
3: -31.2614918, 44.5497284, -24.3367157, 35.0669289, -66.3283997, 68.8864288
4: -29.5164909, 42.1602592, -22.9560833, 33.2461395, -62.7626305, 65.1163330

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9780008, upper bound: 54.0653872
time: 0.91 seconds

## Relational analysis of IS_A2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9780008, upper bound: 54.0714341
time: 0.78 seconds

## BFS IS instance: IS_A2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -22.8221550, 41.3871040, -17.7112522, 33.2447548, -56.0669022, 59.0983582
1: -25.6904716, 38.7174873, -19.9470749, 30.6846199, -56.3750916, 58.6645622
2: -26.2676792, 37.9167290, -20.4596767, 30.1575909, -56.4252625, 58.3764000
3: -31.4501839, 44.7205887, -24.5044899, 35.2995071, -66.7496948, 69.2250824
4: -29.6572666, 42.3763924, -23.1077518, 33.4680367, -63.1253052, 65.4841461

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9721212, upper bound: 54.0620306
time: 1.03 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9730714, upper bound: 54.0684700
time: 0.80 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -23.0630093, 41.7915726, -22.6010666, 41.0435104, -64.1065140, 64.3926315
1: -25.9702396, 39.1207695, -25.4470558, 38.4601746, -64.4304123, 64.5678253
2: -26.5411682, 38.3045883, -26.0237408, 37.6582336, -64.1993942, 64.3283234
3: -31.7985649, 45.1924591, -31.1604176, 44.4320755, -76.2306290, 76.3528671
4: -29.9754677, 42.8236237, -29.4234562, 42.0567856, -72.0322571, 72.2470779

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9612526, upper bound: 53.9612526
time: 1.18 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9612526, upper bound: 53.9612526
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -23.1857567, 42.0140305, -22.7344017, 41.2614822, -64.4472351, 64.7484283
1: -26.1092834, 39.3240700, -25.5986557, 38.6146011, -64.7238846, 64.9227219
2: -26.6807404, 38.5011978, -26.1717892, 37.8171082, -64.4978409, 64.6729813
3: -31.9714127, 45.4301872, -31.3480244, 44.6017151, -76.5731277, 76.7782135
4: -30.1321602, 43.0537643, -29.5632267, 42.2719345, -72.4040909, 72.6169891

Time for backsubstitution: 2.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9612526, upper bound: 53.9612526
time: 0.85 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9612526, upper bound: 53.9612526
time: 0.76 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -22.9435520, 41.5987473, -20.3654213, 38.0456085, -60.9891586, 61.9641685
1: -25.8370190, 38.9264107, -22.8911438, 35.1187973, -60.9558182, 61.8175545
2: -26.4044189, 38.1171761, -23.5145607, 34.4564972, -60.8609123, 61.6317368
3: -31.6334839, 44.9616394, -28.1487122, 40.6303978, -72.2638702, 73.1103363
4: -29.8191299, 42.6128998, -26.5712662, 38.3529472, -68.1720734, 69.1841660

Time for backsubstitution: 2.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9870680, upper bound: 54.0887904
time: 0.84 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9871693, upper bound: 54.0849851
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -22.4729671, 40.7321548, -23.5930691, 43.1950836, -65.6680527, 64.3252258
1: -25.3061218, 38.0592728, -26.4277229, 39.5947609, -64.9008789, 64.4869995
2: -25.8632736, 37.2906151, -27.1425877, 38.8779221, -64.7411957, 64.4331818
3: -30.9690876, 43.9331017, -32.4202347, 45.8534431, -76.8225327, 76.3533325
4: -29.1727448, 41.6818390, -30.2695675, 43.5349312, -72.7076721, 71.9514084

Time for backsubstitution: 2.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9846536, upper bound: 54.0201302
time: 0.91 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9872791, upper bound: 54.0814194
time: 0.92 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -22.4622440, 40.7807999, -24.9989815, 45.6509018, -68.1131439, 65.7797852
1: -25.2914982, 38.1213570, -28.0598755, 42.1997566, -67.4912491, 66.1812286
2: -25.8515587, 37.3374405, -28.7609997, 41.3211327, -67.1726761, 66.0984421
3: -30.9547825, 44.0184402, -34.3129692, 48.9402542, -79.8950348, 78.3314056
4: -29.1847477, 41.7058144, -32.3584633, 46.2080040, -75.3927460, 74.0642776

Time for backsubstitution: 2.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9491959, upper bound: 53.8772804
time: 0.87 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9606317, upper bound: 53.9003545
time: 0.66 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9606317, upper bound: 53.9003545
time: 0.80 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9516767, upper bound: 53.8937781
time: 0.75 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9411353, upper bound: 53.8450141
time: 0.64 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -23.1156273, 41.8867722, -27.1733932, 49.4113731, -72.5270004, 69.0601578
1: -26.0338917, 39.2133369, -30.5476780, 45.8119125, -71.8457947, 69.7610168
2: -26.6003227, 38.3927765, -31.2554703, 44.8115463, -71.4118652, 69.6482468
3: -31.8824482, 45.3003731, -37.4488449, 53.1881409, -85.0705872, 82.7492065
4: -30.0450230, 42.9321632, -35.2109261, 50.3438759, -80.3888855, 78.1430817

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=62.94061279296875
rel_dist={0: [-54.300068832219395, 54.30006883221938]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.0177475, upper bound: 53.1363703
time: 0.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2886130, upper bound: 54.2886133
time: 1.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.37
Output dim: 0, lower bound: -53.0177475, upper bound: 53.1363703
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.37
Output dim: 0, lower bound: -54.2886130, upper bound: 54.2886133

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -16.5254841, 30.8425541, -18.1124954, 33.7820473, -50.3075333, 48.9550438
1: -18.5339508, 27.8673267, -20.3776798, 31.0368958, -49.5708389, 48.2450066
2: -19.0491295, 27.4293900, -20.9011612, 30.4646339, -49.5137596, 48.3305511
3: -22.6070614, 32.1743011, -25.0021877, 35.8536530, -58.4607162, 57.1764908
4: -21.2885952, 30.4737797, -23.5352306, 33.9214745, -55.2100639, 54.0090065

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9750837, upper bound: 53.0440854
time: 0.66 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.0171968, upper bound: 53.1363703
time: 1.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -21.2966690, 39.2646828, -21.9582253, 40.4331512, -61.7298088, 61.2229080
1: -23.9793358, 36.6251450, -24.7321815, 37.7829094, -61.7622299, 61.3573189
2: -24.5428982, 35.8322372, -25.3013039, 36.9466248, -61.4895096, 61.1335373
3: -29.4833717, 42.4556999, -30.4222565, 43.8112946, -73.2946548, 72.8779526
4: -27.7718010, 40.0325012, -28.6487274, 41.3156776, -69.0874786, 68.6812286

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2527724, upper bound: 54.2742867
time: 0.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2523226, upper bound: 54.2523230
time: 0.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.38 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.38
Output dim: 0, lower bound: -52.9750837, upper bound: 53.0440854
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.38
Output dim: 0, lower bound: -53.0171968, upper bound: 53.1363703
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.38
Output dim: 0, lower bound: -54.2527724, upper bound: 54.2742867
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.38
Output dim: 0, lower bound: -54.2523226, upper bound: 54.2523230

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -14.9923897, 28.1005573, -17.1869335, 32.1613197, -47.1537018, 45.2874908
1: -16.7889214, 25.3870392, -19.3232193, 29.5292606, -46.3181839, 44.7102585
2: -17.3098106, 25.0030460, -19.8454933, 29.0101852, -46.3199959, 44.8485336
3: -20.4368458, 29.2671432, -23.6915855, 34.0789986, -54.5158463, 52.9587288
4: -19.3326645, 27.6816902, -22.3471050, 32.2373886, -51.5700531, 50.0287933

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0522213, upper bound: 51.6125982
time: 0.68 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9690448, upper bound: 53.0267506
time: 0.75 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -20.0556755, 36.9833221, -16.7151642, 31.3043118, -51.3599854, 53.6984863
1: -22.4143677, 33.3178368, -18.7914391, 28.6141968, -51.0285568, 52.1092682
2: -23.0606251, 32.7815018, -19.3066311, 28.1399212, -51.2005386, 52.0881233
3: -27.2647629, 38.5627785, -23.0276566, 33.0055466, -60.2703094, 61.5904350
4: -25.6043415, 36.5415535, -21.6931190, 31.2897110, -56.8940506, 58.2346725

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5162115, upper bound: 52.3940630
time: 0.77 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.4628732, upper bound: 52.2429347
time: 0.69 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -20.1166878, 37.1267357, -18.2696972, 34.1689835, -54.2856712, 55.3964272
1: -22.6481361, 34.7948074, -20.6024342, 32.0359268, -54.6840630, 55.3972397
2: -23.1930676, 34.0588837, -21.0932274, 31.3867416, -54.5798111, 55.1521111
3: -27.8605499, 40.3221550, -25.3939705, 37.0507622, -64.9113007, 65.7161102
4: -26.3023472, 37.9474106, -23.9559937, 34.8649101, -61.1672592, 61.9033966

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2526867, upper bound: 54.2742433
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2527541, upper bound: 54.2738376
time: 1.09 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -20.8220463, 38.4329910, -20.8906746, 38.5417480, -59.3637886, 59.3236656
1: -23.4426136, 35.8197784, -23.5217857, 35.9573746, -59.3999863, 59.3415604
2: -23.9984264, 35.0554428, -24.0766964, 35.1869125, -59.1853333, 59.1321259
3: -28.8200550, 41.5037079, -28.9240761, 41.6561279, -70.4761658, 70.4277802
4: -27.1391563, 39.1404419, -27.2251472, 39.2918053, -66.4309616, 66.3655853

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2521393, upper bound: 54.2523230
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2523226, upper bound: 54.2523230
time: 0.83 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.42 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 0, lower bound: -52.0522213, upper bound: 51.6125982
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 0, lower bound: -52.9690448, upper bound: 53.0267506
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 0, lower bound: -52.5162115, upper bound: 52.3940630
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 0, lower bound: -52.4628732, upper bound: 52.2429347
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 0, lower bound: -54.2526867, upper bound: 54.2742433
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 0, lower bound: -54.2527541, upper bound: 54.2738376
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 0, lower bound: -54.2521393, upper bound: 54.2523230
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.42
Output dim: 0, lower bound: -54.2523226, upper bound: 54.2523230

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -13.3025894, 25.3313179, -16.3058205, 30.6964340, -43.9990158, 41.6371384
1: -14.8735695, 22.6367455, -18.3212357, 28.1263485, -42.9999161, 40.9579811
2: -15.3751059, 22.3426399, -18.8388157, 27.6596642, -43.0347710, 41.1814461
3: -18.0677719, 26.0422325, -22.4403381, 32.4319267, -50.4996948, 48.4825592
4: -17.1277199, 24.5757027, -21.2144585, 30.6312141, -47.7589340, 45.7901611

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9295969, upper bound: 51.6125982
time: 0.75 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9295969, upper bound: 51.6125982
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -14.2863503, 26.8515530, -16.7874107, 31.4685154, -45.7548637, 43.6389618
1: -16.0010967, 24.2250328, -18.8792057, 28.8768234, -44.8779221, 43.1042366
2: -16.5066109, 23.8736973, -19.3920212, 28.3721504, -44.8787537, 43.2657166
3: -19.4713306, 27.9206352, -23.1519871, 33.3211212, -52.7924423, 51.0726242
4: -18.4412441, 26.3941021, -21.8459435, 31.5184536, -49.9596901, 48.2400436

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.1029694, upper bound: 50.5253773
time: 0.62 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.0556825, upper bound: 49.8656930
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -17.0212803, 31.8636723, -15.0058947, 28.4550552, -45.4763336, 46.8695602
1: -19.0156555, 28.3688354, -16.8927765, 25.8453274, -44.8609772, 45.2616119
2: -19.5960712, 27.9861813, -17.3570652, 25.4758320, -45.0719032, 45.3432388
3: -23.1335812, 32.7591476, -20.7033615, 29.7293148, -52.8628883, 53.4625092
4: -21.7402630, 31.1473503, -19.5197506, 28.2570267, -49.9972839, 50.6670990

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9995719, upper bound: 52.1272146
time: 0.75 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9995719, upper bound: 52.3940630
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -22.5890446, 40.6863098, -15.9525709, 29.9060593, -52.4951019, 56.6388817
1: -25.2664318, 36.8469315, -17.9439049, 27.2769947, -52.5434265, 54.7908363
2: -25.9408741, 36.1606827, -18.4347019, 26.8335228, -52.7743874, 54.5953827
3: -30.8294144, 42.8328133, -21.9907646, 31.4472733, -62.2766876, 64.8235779
4: -28.6975880, 40.6355171, -20.6903915, 29.8496780, -58.5472641, 61.3259010

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9229366, upper bound: 51.9229366
time: 0.82 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9229366, upper bound: 52.2429347
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -19.1262684, 35.4154243, -16.4986382, 31.1948605, -50.3211212, 51.9140511
1: -21.5221024, 33.2144623, -18.5913639, 29.2335453, -50.7556419, 51.8058243
2: -22.0657043, 32.5385857, -19.0786762, 28.6920090, -50.7577057, 51.6172600
3: -26.4638042, 38.4643364, -22.9049873, 33.7548256, -60.2186279, 61.3693237
4: -25.0444221, 36.1622391, -21.7105160, 31.7184830, -56.7629013, 57.8727570

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8760442, upper bound: 52.7585404
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2512451, upper bound: 54.2711909
time: 1.23 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -18.6288567, 34.5022392, -21.2013111, 38.9973755, -57.6262321, 55.7035522
1: -20.9675426, 32.1965256, -23.7604370, 36.0897293, -57.0572739, 55.9569626
2: -21.4956932, 31.5625763, -24.4172459, 35.4656029, -56.9612961, 55.9798203
3: -25.7730007, 37.2465096, -29.1079578, 41.7508698, -67.5238495, 66.3544693
4: -24.3313599, 35.1164589, -27.3307571, 39.5552711, -63.8866196, 62.4472160

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8763524, upper bound: 52.7646229
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8361472, upper bound: 52.6452181
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -19.8773823, 36.7983475, -19.1332855, 35.5225143, -55.3998947, 55.9316330
1: -22.3663864, 34.3025780, -21.5216751, 33.1684303, -55.5348091, 55.8242531
2: -22.9225082, 33.5969734, -22.0777931, 32.5036392, -55.4261475, 55.6747589
3: -27.4843121, 39.7186928, -26.4423389, 38.3840637, -65.8683777, 66.1610336
4: -25.9340858, 37.4282722, -24.9969387, 36.1431999, -62.0772820, 62.4252090

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6704962, upper bound: 52.6192757
time: 1.36 seconds

## Relational analysis of IS_A2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2507000, upper bound: 54.2509091
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -19.2779064, 35.7086220, -23.3822346, 42.7314224, -62.0093155, 59.0908432
1: -21.6993141, 33.1441574, -26.2149143, 39.5178719, -61.2171860, 59.3590698
2: -22.2368736, 32.4836121, -26.9117889, 38.7612038, -60.9980698, 59.3954010
3: -26.6571407, 38.3395576, -32.0890121, 45.7939377, -72.4510803, 70.4285736
4: -25.1016293, 36.2185135, -30.1322136, 43.3616829, -68.4633026, 66.3507233

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1363699, upper bound: 53.0171967
time: 0.98 seconds

## Relational analysis of IS_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1363703, upper bound: 54.2523230
time: 0.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.94 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -51.9295969, upper bound: 51.6125982
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -51.9295969, upper bound: 51.6125982
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -51.1029694, upper bound: 50.5253773
IS_A1_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.94
Output dim: 0, lower bound: -50.0556825, upper bound: 49.8656930
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -51.9995719, upper bound: 52.1272146
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -51.9995719, upper bound: 52.3940630
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -51.9229366, upper bound: 51.9229366
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -51.9229366, upper bound: 52.2429347
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -52.8760442, upper bound: 52.7585404
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -54.2512451, upper bound: 54.2711909
IS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -52.8763524, upper bound: 52.7646229
IS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -52.8361472, upper bound: 52.6452181
IS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -52.6704962, upper bound: 52.6192757
IS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -54.2507000, upper bound: 54.2509091
IS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -53.1363699, upper bound: 53.0171967
IS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 0, lower bound: -53.1363703, upper bound: 54.2523230

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -13.3025894, 25.3313179, -14.9106464, 28.0017853, -41.3043747, 40.2419586
1: -14.8735695, 22.6367455, -16.6943645, 25.2083817, -40.0819511, 39.3311081
2: -15.3751059, 22.3426399, -17.2073860, 24.8377190, -40.2128181, 39.5500259
3: -18.0677719, 26.0422325, -20.3130341, 29.0483589, -47.1161308, 46.3552551
4: -17.1277199, 24.5757027, -19.1945705, 27.4606056, -44.5883255, 43.7702713

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9255461, upper bound: 51.6120919
time: 0.95 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.5744001, upper bound: 51.2522272
time: 1.40 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -13.3025894, 25.3313179, -19.3228664, 35.9179840, -49.2205734, 44.6541824
1: -14.8735695, 22.6367455, -21.7342033, 33.4551582, -48.3287277, 44.3709488
2: -15.3751059, 22.3426399, -22.2891464, 32.7867317, -48.1618309, 44.6317863
3: -18.0677719, 26.0422325, -26.6880608, 38.7114525, -56.7792206, 52.7302704
4: -17.1277199, 24.5757027, -25.2392139, 36.4140205, -53.5417404, 49.8149185

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9255461, upper bound: 51.6120919
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.5744001, upper bound: 51.2939103
time: 1.09 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -14.2863503, 26.8515530, -16.5140839, 30.9938164, -45.2801666, 43.3656349
1: -16.0010967, 24.2250328, -18.5750580, 28.4108067, -44.4119034, 42.8000908
2: -16.5066109, 23.8736973, -19.0789604, 27.9225578, -44.4291649, 42.9526520
3: -19.4713306, 27.9206352, -22.7793446, 32.7705994, -52.2419281, 50.6999817
4: -18.4412441, 26.3941021, -21.4856567, 31.0161591, -49.4573975, 47.8797569

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.6819042, upper bound: 50.3039600
time: 0.68 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.6819042, upper bound: 50.5253773
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -17.0212803, 31.8636723, -14.1425638, 26.7744713, -43.7957535, 46.0062370
1: -19.0156555, 28.3688354, -15.8589430, 23.9562874, -42.9719391, 44.2277794
2: -19.5960712, 27.9861813, -16.3317528, 23.6585617, -43.2546310, 44.3179283
3: -23.1335812, 32.7591476, -19.3152924, 27.5514069, -50.6849785, 52.0744400
4: -21.7402630, 31.1473503, -18.2219524, 26.1601868, -47.9004440, 49.3693008

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9995719, upper bound: 52.1272146
time: 0.90 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9995719, upper bound: 52.1272146
time: 1.48 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -17.0212803, 31.8636723, -17.5401173, 32.8165817, -49.8378601, 49.4037895
1: -19.0156555, 28.3688354, -19.7946110, 30.2548790, -49.2705307, 48.1634445
2: -19.5960712, 27.9861813, -20.2520580, 29.7290707, -49.3251419, 48.2382355
3: -23.1335812, 32.7591476, -24.3517399, 34.8692169, -58.0027924, 57.1108818
4: -21.7402630, 31.1473503, -22.8755436, 33.1264420, -54.8666992, 54.0228958

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9995719, upper bound: 52.3940629
time: 1.09 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9995719, upper bound: 52.3940630
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -22.5890446, 40.6863098, -15.0150127, 28.0836754, -50.6727142, 55.7013130
1: -25.2664318, 36.8469315, -16.8335724, 25.2481480, -50.5145721, 53.6804962
2: -25.9408741, 36.1606827, -17.3233738, 24.8777046, -50.8185692, 53.4840546
3: -30.8294144, 42.8328133, -20.5131969, 29.1057281, -59.9351387, 63.3460083
4: -28.6975880, 40.6355171, -19.2963333, 27.6075859, -56.3051605, 59.9318428

Time for backsubstitution: 2.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9229366, upper bound: 51.9229366
time: 0.96 seconds

## Relational analysis of IS_A1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9229366, upper bound: 51.9229366
time: 0.83 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -22.5890446, 40.6863098, -18.8339729, 34.8921661, -57.4812088, 59.5202827
1: -25.2664318, 36.8469315, -21.2161179, 32.3570900, -57.6235199, 58.0630455
2: -25.9408741, 36.1606827, -21.7307091, 31.7270050, -57.6678772, 57.8913841
3: -30.8294144, 42.8328133, -26.0758228, 37.4147034, -68.2441177, 68.9086380
4: -28.6975880, 40.6355171, -24.5280190, 35.3800201, -64.0776062, 65.1635361

Time for backsubstitution: 2.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9229366, upper bound: 52.2429347
time: 0.68 seconds

## Relational analysis of IS_A1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9229366, upper bound: 52.2429347
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -18.2310848, 33.9503860, -14.7332125, 28.3919678, -46.6230507, 48.6835976
1: -20.5044823, 31.8045692, -16.5834026, 26.4521065, -46.9565811, 48.3879700
2: -21.0502243, 31.1796799, -17.0837955, 26.0044918, -47.0547142, 48.2634697
3: -25.1922798, 36.7929230, -20.4109383, 30.4876060, -55.6798859, 57.2038574
4: -23.8991356, 34.5217133, -19.4562569, 28.5432358, -52.4423714, 53.9779663

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8760442, upper bound: 52.7569066
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6737809, upper bound: 52.5026768
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -18.7009888, 34.6752472, -15.6689186, 29.7273483, -48.4283371, 50.3441658
1: -21.0497284, 32.5150604, -17.6663170, 27.8720512, -48.9217720, 50.1813736
2: -21.5816612, 31.8544064, -18.1343174, 27.3629360, -48.9445915, 49.9887199
3: -25.8905659, 37.6491508, -21.7807789, 32.1748543, -58.0654221, 59.4299088
4: -24.5050278, 35.3917503, -20.6655617, 30.2167015, -54.7217293, 56.0573120

Time for backsubstitution: 2.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1952863, upper bound: 54.2711909
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2512451, upper bound: 54.2711868
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -16.8257236, 31.5356750, -18.1446762, 33.8838730, -50.7095947, 49.6803513
1: -18.9636841, 29.2734814, -20.3556995, 31.1314220, -50.0951080, 49.6291809
2: -19.4439697, 28.7533360, -20.9250641, 30.6491432, -50.0931129, 49.6783943
3: -23.3323975, 33.7887764, -24.9768238, 35.9235039, -59.2558975, 58.7655869
4: -22.0437508, 31.9178658, -23.4537621, 34.1143913, -56.1581421, 55.3716202

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8724842, upper bound: 52.7646229
time: 1.03 seconds

## Relational analysis of IS_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8724842, upper bound: 52.7585373
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -17.8182583, 33.0527916, -23.0611305, 41.5925980, -59.4108582, 56.1139221
1: -20.0694771, 30.8107395, -25.9010601, 38.5678520, -58.6373291, 56.7117996
2: -20.5729294, 30.2216721, -26.5084343, 37.8360138, -58.4089394, 56.7301064
3: -24.6772804, 35.6196327, -31.7901516, 44.7587700, -69.4360428, 67.4097824
4: -23.2764454, 33.6033440, -29.5958557, 42.5036011, -65.7800369, 63.1991997

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8361472, upper bound: 52.6452181
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8361472, upper bound: 52.6452181
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -18.9968853, 35.3506699, -17.4653702, 32.9105988, -51.9074860, 52.8160400
1: -21.3646069, 32.9151459, -19.6239166, 30.5751266, -51.9397278, 52.5390625
2: -21.9190845, 32.2632828, -20.1878395, 29.9951439, -51.9142303, 52.4511223
3: -26.2322807, 38.0775261, -24.0806465, 35.3234634, -61.5557442, 62.1581726
4: -24.8073502, 35.8162270, -22.8687439, 33.1449509, -57.9523010, 58.6849709

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_B1_B1_B1

### Relational analysis result of IS_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.6125982, upper bound: 52.0522209
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B1_B1_B2

### Relational analysis result of IS_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.6125982, upper bound: 52.2950902
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -19.4494019, 36.0555534, -18.2987614, 34.0816116, -53.5310020, 54.3543129
1: -21.8912983, 33.5984955, -20.5930519, 31.8079185, -53.6992149, 54.1915436
2: -22.4350071, 32.9094582, -21.1291828, 31.1732140, -53.6082230, 54.0386353
3: -26.9079285, 38.8984108, -25.3140850, 36.8022499, -63.7101746, 64.2124939
4: -25.3906231, 36.6538544, -23.9470024, 34.6395874, -60.0302086, 60.6008453

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_B1_B2_B1

### Relational analysis result of IS_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8431412, upper bound: 52.6717570
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B1_B2_B2

### Relational analysis result of IS_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8228264, upper bound: 52.6304408
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -19.2779064, 35.7086220, -18.9709873, 35.2253380, -54.5032425, 54.6796112
1: -21.6993141, 33.1441574, -21.2177734, 31.6253281, -53.3246422, 54.3619270
2: -22.2368736, 32.4836121, -21.8248272, 31.1474152, -53.3842888, 54.3084412
3: -26.6571407, 38.3395576, -25.8299618, 36.5135880, -63.1707306, 64.1695175
4: -25.1016293, 36.2185135, -24.1779175, 34.6967239, -59.7983475, 60.3964310

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_B2_B1_B1

### Relational analysis result of IS_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.2606269, upper bound: 52.2501992
time: 0.81 seconds

## Relational analysis of IS_A2_B2_B2_B1_B2

### Relational analysis result of IS_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0792045, upper bound: 52.1806549
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -19.2779064, 35.7086220, -22.7466888, 41.5940132, -60.8719139, 58.4552917
1: -21.6993141, 33.1441574, -25.4922924, 38.3836517, -60.0829544, 58.6364441
2: -22.2368736, 32.4836121, -26.1793785, 37.6753731, -59.9122429, 58.6629906
3: -26.6571407, 38.3395576, -31.1963139, 44.4578629, -71.1150055, 69.5358734
4: -25.1016293, 36.2185135, -29.2672024, 42.1353760, -67.2369995, 65.4857178

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_B2_B2_B1

### Relational analysis result of IS_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.2606272, upper bound: 52.4287672
time: 1.09 seconds

## Relational analysis of IS_A2_B2_B2_B2_B2

### Relational analysis result of IS_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0792048, upper bound: 52.3370163
time: 0.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.89 seconds
IS_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9255461, upper bound: 51.6120919
IS_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.5744001, upper bound: 51.2522272
IS_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9255461, upper bound: 51.6120919
IS_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.5744001, upper bound: 51.2939103
IS_A1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 5.89
Output dim: 0, lower bound: -50.6819042, upper bound: 50.3039600
IS_A1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 5.89
Output dim: 0, lower bound: -50.6819042, upper bound: 50.5253773
IS_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9995719, upper bound: 52.1272146
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9995719, upper bound: 52.1272146
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9995719, upper bound: 52.3940629
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9995719, upper bound: 52.3940630
IS_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9229366, upper bound: 51.9229366
IS_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9229366, upper bound: 51.9229366
IS_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9229366, upper bound: 52.2429347
IS_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.9229366, upper bound: 52.2429347
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.8760442, upper bound: 52.7569066
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.6737809, upper bound: 52.5026768
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -54.1952863, upper bound: 54.2711909
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -54.2512451, upper bound: 54.2711868
IS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.8724842, upper bound: 52.7646229
IS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.8724842, upper bound: 52.7585373
IS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.8361472, upper bound: 52.6452181
IS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.8361472, upper bound: 52.6452181
IS_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.6125982, upper bound: 52.0522209
IS_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -51.6125982, upper bound: 52.2950902
IS_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.8431412, upper bound: 52.6717570
IS_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.8228264, upper bound: 52.6304408
IS_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.2606269, upper bound: 52.2501992
IS_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.0792045, upper bound: 52.1806549
IS_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.2606272, upper bound: 52.4287672
IS_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.89
Output dim: 0, lower bound: -52.0792048, upper bound: 52.3370163

## BFS IS instance: IS_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -11.6521578, 22.5730209, -12.0364532, 23.2270241, -34.8791809, 34.6094704
1: -13.0325680, 19.9286842, -13.4799318, 20.5058632, -33.5384293, 33.4086151
2: -13.4975100, 19.7060280, -13.9348164, 20.2897701, -33.7872696, 33.6408463
3: -15.8189354, 22.8791580, -16.3703575, 23.5293179, -39.3482513, 39.2495155
4: -15.0380220, 21.6170158, -15.5355625, 22.3116207, -37.3496437, 37.1525803

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9255461, upper bound: 51.6432929
time: 0.69 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9255461, upper bound: 51.6432929
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -12.6309538, 24.0975151, -17.8347893, 32.5331573, -45.1641121, 41.9323044
1: -14.1216068, 21.4484119, -19.9663010, 29.4454803, -43.5670853, 41.4146996
2: -14.6127138, 21.1842766, -20.5132351, 28.9752865, -43.5880013, 41.6975021
3: -17.1369896, 24.6736031, -24.3356991, 34.0616035, -51.1985931, 49.0092926
4: -16.2555046, 23.2685852, -22.7941532, 32.2685585, -48.5240631, 46.0627365

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.5744001, upper bound: 51.2522272
time: 0.85 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.5744001, upper bound: 51.2522272
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -11.6521578, 22.5730209, -15.8427868, 30.1382065, -41.7903595, 38.4158096
1: -13.0325680, 19.9286842, -17.8633614, 27.7250919, -40.7576599, 37.7920456
2: -13.4975100, 19.7060280, -18.3228683, 27.2563934, -40.7539024, 38.0288963
3: -15.8189354, 22.8791580, -21.9799461, 31.9471664, -47.7661018, 44.8591042
4: -15.0380220, 21.6170158, -20.7917442, 30.1983261, -45.2363472, 42.4087563

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0486476, upper bound: 51.6120919
time: 0.88 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0486476, upper bound: 51.6120919
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -12.6309538, 24.0975151, -21.9980202, 39.9113388, -52.5422859, 46.0955353
1: -14.1216068, 21.4484119, -24.7292957, 37.2145271, -51.3361320, 46.1777039
2: -14.6127138, 21.1842766, -25.3082428, 36.3979530, -51.0106659, 46.4925194
3: -17.1369896, 24.6736031, -30.3613701, 43.1933670, -60.3303566, 55.0349693
4: -16.2555046, 23.2685852, -28.4945126, 40.7080116, -56.9635162, 51.7630959

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.8698255, upper bound: 51.2939103
time: 1.08 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.8698259, upper bound: 51.2939103
time: 1.48 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -17.0212803, 31.8636723, -13.3956299, 25.4394798, -42.4607620, 45.2592964
1: -19.0156555, 28.3688354, -15.0039501, 22.7628326, -41.7784805, 43.3727875
2: -19.5960712, 27.9861813, -15.4915314, 22.4869041, -42.0829773, 43.4777031
3: -23.1335812, 32.7591476, -18.2437553, 26.1828747, -49.3164482, 51.0029030
4: -21.7402630, 31.1473503, -17.2964096, 24.8145237, -46.5547676, 48.4437561

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9995719, upper bound: 52.1272146
time: 0.70 seconds

## Relational analysis of IS_A1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9995719, upper bound: 52.1272146
time: 1.01 seconds

## BFS IS instance: IS_A1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -17.0212803, 31.8636723, -18.2529373, 34.0002289, -51.0215073, 50.1166077
1: -19.0156555, 28.3688354, -20.4021015, 30.4082851, -49.4239388, 48.7709351
2: -19.5960712, 27.9861813, -21.0066833, 29.9716339, -49.5677032, 48.9928589
3: -23.1335812, 32.7591476, -24.8276405, 35.1273308, -58.2609100, 57.5867844
4: -21.7402630, 31.1473503, -23.3165874, 33.3703308, -55.1105881, 54.4639359

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9746332, upper bound: 52.1272147
time: 0.76 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9746332, upper bound: 52.1272146
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -17.0212803, 31.8636723, -16.8699837, 31.6736107, -48.6948929, 48.7336540
1: -19.0156555, 28.3688354, -19.0340347, 29.2566261, -48.2722816, 47.4028702
2: -19.5960712, 27.9861813, -19.4906197, 28.7545319, -48.3506012, 47.4767952
3: -23.1335812, 32.7591476, -23.4209785, 33.7138062, -56.8473854, 56.1801262
4: -21.7402630, 31.1473503, -22.0604324, 31.9950752, -53.7353325, 53.2077827

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.5162109, upper bound: 52.3940625
time: 1.35 seconds

## Relational analysis of IS_A1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.2501995, upper bound: 52.2606271
time: 1.43 seconds

## BFS IS instance: IS_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -17.0212803, 31.8636723, -21.6649590, 39.7856064, -56.8068848, 53.5286331
1: -19.0156555, 28.3688354, -24.3191013, 36.5207901, -55.5364380, 52.6879349
2: -19.5960712, 27.9861813, -24.9412041, 35.8857117, -55.4817772, 52.9273682
3: -23.1335812, 32.7591476, -29.7923584, 42.2377815, -65.3713608, 62.5514946
4: -21.7402630, 31.1473503, -27.8897667, 40.1846466, -61.9249115, 59.0371170

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=62.94061279296875
rel_dist={0: [-54.29988917735716, 54.29988917735716]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1106.16 seconds
