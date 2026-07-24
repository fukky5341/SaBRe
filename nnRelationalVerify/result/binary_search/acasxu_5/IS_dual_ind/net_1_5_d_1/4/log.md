## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 60.201135133499996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014)
1: (-17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989)
2: (-14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656)
3: (-16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100)
4: (-13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183)

## BASE Result
execution time: IAR + LP analysis = 1.96 + 2.13 = 4.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -60.2373092, upper bound: 60.2373092


# Binary Search by BASE starts (time budget: 1195.91 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=65.54161834716797
rel_dist={4: [-60.23730919021928, 60.23730919021929]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=65.54161834716797
rel_dist={4: [-60.236541379106946, 60.23654137910691]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=65.54161834716797
rel_dist={4: [-60.234725823217936, 60.23472582321793]}

## Binary search (step 3) starts
Candidate diff: 0.0208333


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0208333, mid=0.0208333, abs_max=65.54161834716797
rel_dist={4: [-60.23221675720202, 60.23221675720205]}

## Binary search (step 4) starts
Candidate diff: 0.0104167


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0104167, mid=0.0104167, abs_max=65.54161834716797
rel_dist={4: [-60.23007248428422, 60.230072484284236]}

## Binary search (step 5) starts
Candidate diff: 0.0052083


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0052083, mid=0.0052083, abs_max=65.54161834716797
rel_dist={4: [-60.22892424106238, 60.2289242410624]}

## Binary search (step 6) starts
Candidate diff: 0.0026042


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0026042, mid=0.0026042, abs_max=65.54161834716797
rel_dist={4: [-60.2283423421207, 60.2283423421207]}

## Binary search (step 7) starts
Candidate diff: 0.0013021


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0013021, mid=0.0013021, abs_max=65.54161834716797
rel_dist={4: [-60.22798780668879, 60.22798780668879]}

## Binary search (step 8) starts
Candidate diff: 0.0006510


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0006510, mid=0.0006510, abs_max=65.54161834716797
rel_dist={4: [-60.2278026105641, 60.2278026105641]}

## Binary search (step 9) starts
Candidate diff: 0.0003255


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0003255, mid=0.0003255, abs_max=65.54161834716797
rel_dist={4: [-60.227707013532644, 60.22770701353265]}

## Binary search (step 10) starts
Candidate diff: 0.0001628


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0001628, mid=0.0001628, abs_max=65.54161834716797
rel_dist={4: [-60.227659215314496, 60.2276592153145]}

## Binary search (step 11) starts
Candidate diff: 0.0000814


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000814, mid=0.0000814, abs_max=65.54161834716797
rel_dist={4: [-60.22763531679102, 60.22763531679104]}

## Binary search (step 12) starts
Candidate diff: 0.0000407


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000407, mid=0.0000407, abs_max=65.54161834716797
rel_dist={4: [-60.227623368664055, 60.22762336866404]}

## Binary search (step 13) starts
Candidate diff: 0.0000203


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000203, mid=0.0000203, abs_max=65.54161834716797
rel_dist={4: [-60.227617379735584, 60.227617396736946]}

## Binary search (step 14) starts
Candidate diff: 0.0000102


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000102, mid=0.0000102, abs_max=65.54161834716797
rel_dist={4: [-60.22761441459569, 60.22761441459569]}

## Binary search (step 15) starts
Candidate diff: 0.0000051


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000051, mid=0.0000051, abs_max=65.54161834716797
rel_dist={4: [-60.227612956083206, 60.22761293402213]}

## Binary search (step 16) starts
Candidate diff: 0.0000025


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000025, mid=0.0000025, abs_max=65.54161834716797
rel_dist={4: [-60.227612161216065, 60.22761227015873]}

## Binary search (step 17) starts
Candidate diff: 0.0000013


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000013, mid=0.0000013, abs_max=65.54161834716797
rel_dist={4: [-60.227611924177594, 60.22761215221777]}

## Binary search (step 18) starts
Candidate diff: 0.0000006


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000006, mid=0.0000006, abs_max=65.54161834716797
rel_dist={4: [-60.22761173046293, 60.22761171975189]}

## Binary Search Result
Binary search time: 76.04 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1119.87 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2343734
time: 0.82 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 0.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2343734
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.4466705, 33.0043793, -12.1294031, 37.7388992, -48.1855583, 45.1337776
1: -14.8643341, 34.2352486, -17.1822987, 39.1265984, -53.9909325, 51.4175491
2: -12.7760410, 38.1106606, -14.7555904, 43.5125732, -56.2886124, 52.8662491
3: -13.9698524, 49.0998573, -16.1523533, 55.9294815, -69.8993073, 65.2521973
4: -12.0348778, 45.3377380, -13.7831745, 51.7584686, -63.7933464, 59.1209106

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 0.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 1.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.5646820, 36.3494148, -12.0505228, 37.5201912, -49.0848732, 48.3999367
1: -16.3725929, 37.6687965, -17.0745277, 38.9002113, -55.2728043, 54.7433243
2: -14.0860319, 41.9790382, -14.6632338, 43.2624550, -57.3484802, 56.6422691
3: -15.4223356, 53.9925766, -16.0516663, 55.6117439, -71.0340652, 70.0442429
4: -13.2208614, 49.8709221, -13.7009554, 51.4595413, -64.6804047, 63.5718765

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 0.90 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 0.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.4466705, 33.0043793, -10.4466705, 33.0043793, -43.4510307, 43.4510345
1: -14.8643341, 34.2352486, -14.8643341, 34.2352486, -49.0995827, 49.0995827
2: -12.7760410, 38.1106606, -12.7760410, 38.1106606, -50.8867035, 50.8866997
3: -13.9698524, 49.0998573, -13.9698524, 49.0998573, -63.0697060, 63.0697060
4: -12.0348778, 45.3377380, -12.0348778, 45.3377380, -57.3726120, 57.3726120

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2325729
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
time: 0.92 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.4466705, 33.0043793, -11.5646820, 36.3494148, -46.7960739, 44.5690536
1: -14.8643341, 34.2352486, -16.3725929, 37.6687965, -52.5331306, 50.6078415
2: -12.7760410, 38.1106606, -14.0860319, 41.9790382, -54.7550774, 52.1966858
3: -13.9698524, 49.0998573, -15.4223356, 53.9925766, -67.9624252, 64.5221786
4: -12.0348778, 45.3377380, -13.2208614, 49.8709221, -61.9057999, 58.5585938

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2325729
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
time: 1.16 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.5646820, 36.3494148, -10.4466705, 33.0043793, -44.5690536, 46.7960739
1: -16.3725929, 37.6687965, -14.8643341, 34.2352486, -50.6078415, 52.5331268
2: -14.0860319, 41.9790382, -12.7760410, 38.1106606, -52.1966858, 54.7550774
3: -15.4223356, 53.9925766, -13.9698524, 49.0998573, -64.5221863, 67.9624176
4: -13.2208614, 49.8709221, -12.0348778, 45.3377380, -58.5585976, 61.9057999

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2213024
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 1.19 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.5646820, 36.3494148, -11.5646820, 36.3494148, -47.9140968, 47.9140968
1: -16.3725929, 37.6687965, -16.3725929, 37.6687965, -54.0413857, 54.0413895
2: -14.0860319, 41.9790382, -14.0860319, 41.9790382, -56.0650673, 56.0650635
3: -15.4223356, 53.9925766, -15.4223356, 53.9925766, -69.4149094, 69.4149094
4: -13.2208614, 49.8709221, -13.2208614, 49.8709221, -63.0917816, 63.0917816

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2213024
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 1.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.03 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2325729
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2325729
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2213024
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2213024
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.03
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -10.4466705, 33.0043793, -41.7035484, 38.1833496
1: -12.4092007, 28.8249302, -14.8643341, 34.2352486, -46.6444473, 43.6892624
2: -10.6860304, 32.1388817, -12.7760410, 38.1106606, -48.7966919, 44.9149246
3: -11.6377096, 41.3525887, -13.9698524, 49.0998573, -60.7375641, 55.3224411
4: -10.1306763, 38.2451363, -12.0348778, 45.3377380, -55.4684143, 50.2800064

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2353505
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2353505
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -10.3593292, 32.7462234, -49.3114891, 59.7052193
1: -23.0251637, 51.0942726, -14.7422695, 33.9663849, -56.9915428, 65.8365402
2: -19.7440033, 56.8110428, -12.6706734, 37.8118439, -57.5558472, 69.4817200
3: -21.7395668, 72.8528137, -13.8555994, 48.7171173, -70.4566803, 86.7084122
4: -18.1693535, 67.7313232, -11.9381847, 44.9827614, -63.1521149, 79.6695099

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2371510
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2371510
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -11.5646820, 36.3494148, -45.0485916, 39.3013611
1: -12.4092007, 28.8249302, -16.3725929, 37.6687965, -50.0779953, 45.1975250
2: -10.6860304, 32.1388817, -14.0860319, 41.9790382, -52.6650696, 46.2249069
3: -11.6377096, 41.3525887, -15.4223356, 53.9925766, -65.6302872, 56.7749252
4: -10.1306763, 38.2451363, -13.2208614, 49.8709221, -60.0015984, 51.4659882

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2325729
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2325729
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -11.4375248, 35.9618225, -52.5270882, 60.7834129
1: -23.0251637, 51.0942726, -16.1967449, 37.2681465, -60.2933121, 67.2910156
2: -19.7440033, 56.8110428, -13.9348888, 41.5315285, -61.2755318, 70.7459335
3: -21.7395668, 72.8528137, -15.2548981, 53.4180527, -75.1576157, 88.1077118
4: -18.1693535, 67.7313232, -13.0838718, 49.3414345, -67.5107880, 80.8151932

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2343734
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2343734
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -10.4466705, 33.0043793, -42.9555626, 41.9948578
1: -14.1082268, 32.7253380, -14.8643341, 34.2352486, -48.3434715, 47.5896721
2: -12.1443062, 36.5262375, -12.7760410, 38.1106606, -50.2549667, 49.3022766
3: -13.2896729, 46.9455948, -13.9698524, 49.0998573, -62.3895302, 60.9154472
4: -11.4511833, 43.4067154, -12.0348778, 45.3377380, -56.7889137, 55.4415894

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2240800
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2240800
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -10.3593292, 32.7462234, -48.8655205, 59.0984802
1: -22.4673023, 50.3988075, -14.7422695, 33.9663849, -56.4336853, 65.1410751
2: -19.2585125, 56.0775299, -12.6706734, 37.8118439, -57.0703545, 68.7481842
3: -21.2393341, 71.9838791, -13.8555994, 48.7171173, -69.9564514, 85.8394775
4: -17.8179779, 66.7789383, -11.9381847, 44.9827614, -62.8007393, 78.7171173

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2335933
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2335933
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -11.5646820, 36.3494148, -46.3006058, 43.1128769
1: -14.1082268, 32.7253380, -16.3725929, 37.6687965, -51.7770195, 49.0979271
2: -12.1443062, 36.5262375, -14.0860319, 41.9790382, -54.1233444, 50.6122665
3: -13.2896729, 46.9455948, -15.4223356, 53.9925766, -67.2822418, 62.3679314
4: -11.4511833, 43.4067154, -13.2208614, 49.8709221, -61.3221016, 56.6275711

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -11.4375248, 35.9618225, -52.0811195, 60.1766739
1: -22.4673023, 50.3988075, -16.1967449, 37.2681465, -59.7354507, 66.5955505
2: -19.2585125, 56.0775299, -13.9348888, 41.5315285, -60.7900391, 70.0124130
3: -21.2393341, 71.9838791, -15.2548981, 53.4180527, -74.6573868, 87.2387772
4: -17.8179779, 66.7789383, -13.0838718, 49.3414345, -67.1594086, 79.8628006

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2308157
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2308157
time: 0.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.01 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2353505
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2353505
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2371510
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2371510
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2325729
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2325729
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2343734
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2343734
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2240800
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2240800
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2335933
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2335933
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2308157
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.01
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2308157

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -8.6991749, 27.7366791, -36.4358521, 36.4358521
1: -12.4092007, 28.8249302, -12.4092007, 28.8249302, -41.2341309, 41.2341309
2: -10.6860304, 32.1388817, -10.6860304, 32.1388817, -42.8249130, 42.8249130
3: -11.6377096, 41.3525887, -11.6377096, 41.3525887, -52.9902954, 52.9902954
4: -10.1306763, 38.2451363, -10.1306763, 38.2451363, -48.3758125, 48.3758125

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2350919
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2330936
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -16.5652657, 49.3458900, -58.0450668, 44.3019447
1: -12.4092007, 28.8249302, -23.0251637, 51.0942726, -63.5034714, 51.8500900
2: -10.6860304, 32.1388817, -19.7440033, 56.8110428, -67.4970703, 51.8828850
3: -11.6377096, 41.3525887, -21.7395668, 72.8528137, -84.4905243, 63.0921516
4: -10.1306763, 38.2451363, -18.1693535, 67.7313232, -77.8619995, 56.4144897

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2350919
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334250, upper bound: 60.2330936
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -8.6602230, 27.6217041, -44.1869659, 58.0061111
1: -23.0251637, 51.0942726, -12.3543396, 28.7050247, -51.7301865, 63.4486122
2: -19.7440033, 56.8110428, -10.6386271, 32.0058823, -51.7498856, 67.4496689
3: -21.7395668, 72.8528137, -11.5859261, 41.1813660, -62.9209328, 84.4387360
4: -18.1693535, 67.7313232, -10.0872440, 38.0864449, -56.2557983, 77.8185654

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2276546, upper bound: 60.2186772
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319546, upper bound: 60.2355284
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -16.4778557, 49.0316277, -65.5968933, 65.8237457
1: -23.0251637, 51.0942726, -22.8805523, 50.7702751, -73.7954407, 73.9748230
2: -19.7440033, 56.8110428, -19.6148682, 56.4666824, -76.2106857, 76.4259109
3: -21.7395668, 72.8528137, -21.6091194, 72.3687057, -94.1082687, 94.4619217
4: -18.1693535, 67.7313232, -18.0539436, 67.3143234, -85.4836731, 85.7852631

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2276546, upper bound: 60.2186772
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2276546, upper bound: 60.2355284
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -9.9511909, 31.5481949, -40.2473679, 37.6878700
1: -12.4092007, 28.8249302, -14.1082268, 32.7253380, -45.1345367, 42.9331551
2: -10.6860304, 32.1388817, -12.1443062, 36.5262375, -47.2122688, 44.2831879
3: -11.6377096, 41.3525887, -13.2896729, 46.9455948, -58.5833054, 54.6422615
4: -10.1306763, 38.2451363, -11.4511833, 43.4067154, -53.5373917, 49.6963081

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226999, upper bound: 60.2322787
time: 1.15 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2241527, upper bound: 60.2302805
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -16.1192989, 48.7391510, -57.4383240, 43.8559799
1: -12.4092007, 28.8249302, -22.4673023, 50.3988075, -62.8080025, 51.2922325
2: -10.6860304, 32.1388817, -19.2585125, 56.0775299, -66.7635498, 51.3973885
3: -11.6377096, 41.3525887, -21.2393341, 71.9838791, -83.6215897, 62.5919228
4: -10.1306763, 38.2451363, -17.8179779, 66.7789383, -76.9096069, 56.0631142

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226999, upper bound: 60.2322787
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2241527, upper bound: 60.2302805
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -9.9511909, 31.5481949, -48.1134567, 59.2970810
1: -23.0251637, 51.0942726, -14.1082268, 32.7253380, -55.7504959, 65.2024994
2: -19.7440033, 56.8110428, -12.1443062, 36.5262375, -56.2702408, 68.9553528
3: -21.7395668, 72.8528137, -13.2896729, 46.9455948, -68.6851654, 86.1424866
4: -18.1693535, 67.7313232, -11.4511833, 43.4067154, -61.5760689, 79.1825104

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226753, upper bound: 60.2340350
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2238214, upper bound: 60.2302805
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -16.1192989, 48.7391510, -65.3044128, 65.4651871
1: -23.0251637, 51.0942726, -22.4673023, 50.3988075, -73.4239731, 73.5615768
2: -19.7440033, 56.8110428, -19.2585125, 56.0775299, -75.8215332, 76.0695572
3: -21.7395668, 72.8528137, -21.2393341, 71.9838791, -93.7234421, 94.0921478
4: -18.1693535, 67.7313232, -17.8179779, 66.7789383, -84.9482880, 85.5493011

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226753, upper bound: 60.2340350
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2238214, upper bound: 60.2302805
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -8.6991749, 27.7366791, -37.6878700, 40.2473679
1: -14.1082268, 32.7253380, -12.4092007, 28.8249302, -42.9331551, 45.1345367
2: -12.1443062, 36.5262375, -10.6860304, 32.1388817, -44.2831879, 47.2122688
3: -13.2896729, 46.9455948, -11.6377096, 41.3525887, -54.6422615, 58.5833054
4: -11.4511833, 43.4067154, -10.1306763, 38.2451363, -49.6963081, 53.5373917

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2044859, upper bound: 60.1864108
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2238652
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -16.5652657, 49.3458900, -59.2970810, 48.1134567
1: -14.1082268, 32.7253380, -23.0251637, 51.0942726, -65.2024994, 55.7504959
2: -12.1443062, 36.5262375, -19.7440033, 56.8110428, -68.9553528, 56.2702408
3: -13.2896729, 46.9455948, -21.7395668, 72.8528137, -86.1424789, 68.6851654
4: -11.4511833, 43.4067154, -18.1693535, 67.7313232, -79.1825104, 61.5760689

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2044859, upper bound: 60.1864108
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2044859, upper bound: 60.2238652
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -8.6602230, 27.6217041, -43.7409973, 57.3993759
1: -22.4673023, 50.3988075, -12.3543396, 28.7050247, -51.1723251, 62.7531471
2: -19.2585125, 56.0775299, -10.6386271, 32.0058823, -51.2643890, 66.7161560
3: -21.2393341, 71.9838791, -11.5859261, 41.1813660, -62.4207001, 83.5698090
4: -17.8179779, 66.7789383, -10.0872440, 38.0864449, -55.9044228, 76.8661804

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193896, upper bound: 60.2332179
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2302805, upper bound: 60.2300209
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.4778557, 49.0316277, -65.1509247, 65.2170105
1: -22.4673023, 50.3988075, -22.8805523, 50.7702751, -73.2375793, 73.2793579
2: -19.2585125, 56.0775299, -19.6148682, 56.4666824, -75.7251968, 75.6923981
3: -21.2393341, 71.9838791, -21.6091194, 72.3687057, -93.6080399, 93.5929947
4: -17.8179779, 66.7789383, -18.0539436, 67.3143234, -85.1323013, 84.8328857

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193896, upper bound: 60.2332179
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2302805, upper bound: 60.2300209
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -9.9511909, 31.5481949, -41.4993858, 41.4993858
1: -14.1082268, 32.7253380, -14.1082268, 32.7253380, -46.8335609, 46.8335571
2: -12.1443062, 36.5262375, -12.1443062, 36.5262375, -48.6705399, 48.6705399
3: -13.2896729, 46.9455948, -13.2896729, 46.9455948, -60.2352676, 60.2352676
4: -11.4511833, 43.4067154, -11.4511833, 43.4067154, -54.8578911, 54.8578911

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1992214, upper bound: 60.1836546
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -16.1192989, 48.7391510, -58.6903419, 47.6674919
1: -14.1082268, 32.7253380, -22.4673023, 50.3988075, -64.5070267, 55.1926422
2: -12.1443062, 36.5262375, -19.2585125, 56.0775299, -68.2218170, 55.7847481
3: -13.2896729, 46.9455948, -21.2393341, 71.9838791, -85.2735519, 68.1849289
4: -11.4511833, 43.4067154, -17.8179779, 66.7789383, -78.2301178, 61.2246933

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1992214, upper bound: 60.1836546
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -9.9511909, 31.5481949, -47.6674919, 58.6903419
1: -22.4673023, 50.3988075, -14.1082268, 32.7253380, -55.1926422, 64.5070190
2: -19.2585125, 56.0775299, -12.1443062, 36.5262375, -55.7847481, 68.2218170
3: -21.2393341, 71.9838791, -13.2896729, 46.9455948, -68.1849289, 85.2735443
4: -17.8179779, 66.7789383, -11.4511833, 43.4067154, -61.2246933, 78.2301178

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2304047
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210082, upper bound: 60.2272077
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.1192989, 48.7391510, -64.8584518, 64.8584518
1: -22.4673023, 50.3988075, -22.4673023, 50.3988075, -72.8661118, 72.8661118
2: -19.2585125, 56.0775299, -19.2585125, 56.0775299, -75.3360443, 75.3360443
3: -21.2393341, 71.9838791, -21.2393341, 71.9838791, -93.2232132, 93.2232132
4: -17.8179779, 66.7789383, -17.8179779, 66.7789383, -84.5969162, 84.5969162

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2304047
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210082, upper bound: 60.2272077
time: 1.28 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.45 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2350919
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2330936
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2319721, upper bound: 60.2350919
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2334250, upper bound: 60.2330936
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2276546, upper bound: 60.2186772
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2319546, upper bound: 60.2355284
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2276546, upper bound: 60.2186772
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2276546, upper bound: 60.2355284
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2226999, upper bound: 60.2322787
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2241527, upper bound: 60.2302805
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2226999, upper bound: 60.2322787
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2241527, upper bound: 60.2302805
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2226753, upper bound: 60.2340350
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2238214, upper bound: 60.2302805
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2226753, upper bound: 60.2340350
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2238214, upper bound: 60.2302805
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2044859, upper bound: 60.1864108
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2238652
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2044859, upper bound: 60.1864108
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2044859, upper bound: 60.2238652
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2193896, upper bound: 60.2332179
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2302805, upper bound: 60.2300209
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2193896, upper bound: 60.2332179
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2302805, upper bound: 60.2300209
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.1992214, upper bound: 60.1836546
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.1992214, upper bound: 60.1836546
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2304047
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2210082, upper bound: 60.2272077
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2304047
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.45
Output dim: 4, lower bound: -60.2210082, upper bound: 60.2272077

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.4733706, 24.1488380, -8.5692940, 27.3730202, -34.8463898, 32.7181320
1: -10.6540947, 25.1270370, -12.2254000, 28.4491196, -39.1032143, 37.3524361
2: -9.2093201, 28.1137600, -10.5293579, 31.7247047, -40.9340248, 38.6431198
3: -9.9912634, 36.2262306, -11.4691372, 40.8322220, -50.8234863, 47.6953659
4: -8.8301640, 33.4773788, -9.9939022, 37.7549515, -46.5851135, 43.4712830

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2295843, upper bound: 60.2315919
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2292037, upper bound: 60.2321526
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.4589252, 27.0919533, -8.6991749, 27.7366791, -36.1956024, 35.7911263
1: -12.0820923, 28.1615906, -12.4092007, 28.8249302, -40.9070206, 40.5707932
2: -10.4120913, 31.3883038, -10.6860304, 32.1388817, -42.5509720, 42.0743332
3: -11.3392544, 40.4026985, -11.6377096, 41.3525887, -52.6918373, 52.0404091
4: -9.8867702, 37.3420067, -10.1306763, 38.2451363, -48.1319008, 47.4726830

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2320932, upper bound: 60.2311520
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2317126, upper bound: 60.2317126
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.4733706, 24.1488380, -16.3946838, 48.8715248, -56.3448944, 40.5435219
1: -10.6540947, 25.1270370, -22.7716980, 50.6007195, -61.2548141, 47.8987350
2: -9.2093201, 28.1137600, -19.5337372, 56.2694702, -65.4787903, 47.6474991
3: -9.9912634, 36.2262306, -21.5179214, 72.1748123, -82.1660690, 57.7441483
4: -8.8301640, 33.4773788, -17.9921703, 67.0800781, -75.9102402, 51.4695511

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2295843, upper bound: 60.2315374
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2271145
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2317797
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.4589252, 27.0919533, -16.5652657, 49.3458900, -57.8048172, 43.6572151
1: -12.0820923, 28.1615906, -23.0251637, 51.0942726, -63.1763649, 51.1867523
2: -10.4120913, 31.3883038, -19.7440033, 56.8110428, -67.2231369, 51.1323090
3: -11.3392544, 40.4026985, -21.7395668, 72.8528137, -84.1920700, 62.1422653
4: -9.8867702, 37.3420067, -18.1693535, 67.7313232, -77.6180954, 55.5113602

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2320932, upper bound: 60.2310975
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2234892
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2281544
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.1109037, 41.8675880, -8.6602230, 27.6217041, -41.7326050, 50.5278091
1: -19.5679359, 43.4133492, -12.3543396, 28.7050247, -48.2729607, 55.7676888
2: -16.7519665, 48.3994026, -10.6386271, 32.0058823, -48.7578430, 59.0380287
3: -18.5002823, 61.5973701, -11.5859261, 41.1813660, -59.6816483, 73.1832962
4: -15.4016666, 57.6641388, -10.0872440, 38.0864449, -53.4881134, 67.7513809

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2271145, upper bound: 60.2185265
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2234892, upper bound: 60.2185023
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.9450722, 53.3562126, -8.6124983, 27.4947071, -45.4397736, 61.9687004
1: -24.9718819, 55.2171478, -12.2877073, 28.5734634, -53.5453453, 67.5048523
2: -21.3408375, 61.3745003, -10.5802507, 31.8603249, -53.2011642, 71.9547501
3: -23.5634995, 78.4572372, -11.5269823, 40.9957809, -64.5592804, 89.9842224
4: -19.5413036, 73.1064529, -10.0360193, 37.9122162, -57.4535103, 83.1424637

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2317797, upper bound: 60.2281338
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2234892, upper bound: 60.2281097
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.1109037, 41.8675880, -16.4778557, 49.0316277, -63.1425247, 58.3454437
1: -19.5679359, 43.4133492, -22.8805523, 50.7702751, -70.3382111, 66.2938995
2: -16.7519665, 48.3994026, -19.6148682, 56.4666824, -73.2186432, 68.0142670
3: -18.5002823, 61.5973701, -21.6091194, 72.3687057, -90.8689880, 83.2064743
4: -15.4016666, 57.6641388, -18.0539436, 67.3143234, -82.7159882, 75.7180786

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2143773, upper bound: 60.2143773
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2143773, upper bound: 60.2186772
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.9450722, 53.3562126, -16.4223900, 48.8830070, -66.8280792, 69.7785950
1: -24.9718819, 55.2171478, -22.8005943, 50.6156540, -75.5875397, 78.0177383
2: -21.3408375, 61.3745003, -19.5453930, 56.2956467, -77.6364822, 80.9198914
3: -23.5634995, 78.4572372, -21.5388699, 72.1523514, -95.7158508, 99.9961090
4: -19.5413036, 73.1064529, -17.9959869, 67.1071548, -86.6484604, 91.1024323

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2312285
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2355284
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.4733706, 24.1488380, -9.8296013, 31.1956444, -38.6690140, 33.9784393
1: -10.6540947, 25.1270370, -13.9361629, 32.3614120, -43.0155067, 39.0632019
2: -9.2093201, 28.1137600, -11.9998665, 36.1235733, -45.3328934, 40.1136246
3: -9.9912634, 36.2262306, -13.1277885, 46.4352493, -56.4265137, 49.3540192
4: -8.8301640, 33.4773788, -11.3237314, 42.9283142, -51.7584686, 44.8011093

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201348, upper bound: 60.2288473
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170657, upper bound: 60.2306090
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2200494, upper bound: 60.2315662
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.4589252, 27.0919533, -9.9511909, 31.5481949, -40.0071182, 37.0431404
1: -12.0820923, 28.1615906, -14.1082268, 32.7253380, -44.8074265, 42.2698174
2: -10.4120913, 31.3883038, -12.1443062, 36.5262375, -46.9383278, 43.5326080
3: -11.3392544, 40.4026985, -13.2896729, 46.9455948, -58.2848434, 53.6923714
4: -9.8867702, 37.3420067, -11.4511833, 43.4067154, -53.2934837, 48.7931862

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201348, upper bound: 60.2284074
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196810, upper bound: 60.2288680
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226647, upper bound: 60.2298252
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.4733706, 24.1488380, -15.9651785, 48.3056831, -55.7790527, 40.1140137
1: -10.6540947, 25.1270370, -22.2454796, 49.9502563, -60.6043510, 47.3725166
2: -9.2093201, 28.1137600, -19.0712318, 55.5847397, -64.7940598, 47.1849861
3: -9.9912634, 36.2262306, -21.0392017, 71.3628387, -81.3541031, 57.2654305
4: -8.8301640, 33.4773788, -17.6571064, 66.1880646, -75.0182266, 51.1344833

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201348, upper bound: 60.2288473
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155889, upper bound: 60.2245252
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2241436, upper bound: 60.2237210
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.4589252, 27.0919533, -16.1192989, 48.7391510, -57.1980743, 43.2112465
1: -12.0820923, 28.1615906, -22.4673023, 50.3988075, -62.4808998, 50.6288910
2: -10.4120913, 31.3883038, -19.2585125, 56.0775299, -66.4896240, 50.6468124
3: -11.3392544, 40.4026985, -21.2393341, 71.9838791, -83.3231354, 61.6420326
4: -9.8867702, 37.3420067, -17.8179779, 66.7789383, -76.6657028, 55.1599846

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2284941, upper bound: 60.2284074
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2303522, upper bound: 60.2193896
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2303522, upper bound: 60.2302805
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.8627434, 44.4193230, -9.8296013, 31.1956444, -46.0583839, 54.2489204
1: -20.4770241, 46.0121498, -13.9361629, 32.3614120, -52.8384361, 59.9483109
2: -17.6456852, 51.2711029, -11.9998665, 36.1235733, -53.7692566, 63.2709503
3: -19.5267849, 65.8024368, -13.1277885, 46.4352493, -65.9620285, 78.9302216
4: -16.3723183, 61.1303329, -11.3237314, 42.9283142, -59.3006248, 72.4540558

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170657, upper bound: 60.2306090
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201904, upper bound: 60.2336065
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16.3341827, 48.6779442, -9.9511909, 31.5481949, -47.8823776, 58.6291351
1: -22.6960735, 50.4090309, -14.1082268, 32.7253380, -55.4214058, 64.5172501
2: -19.4665813, 56.0498695, -12.1443062, 36.5262375, -55.9928207, 68.1941528
3: -21.4443798, 71.8867035, -13.2896729, 46.9455948, -68.3899689, 85.1763687
4: -17.9188328, 66.8347778, -11.4511833, 43.4067154, -61.3255463, 78.2859650

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196810, upper bound: 60.2288680
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226647, upper bound: 60.2298252
time: 1.18 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.8627434, 44.4193230, -15.9651785, 48.3056831, -63.1684189, 60.3844910
1: -20.4770241, 46.0121498, -22.2454796, 49.9502563, -70.4272766, 68.2576141
2: -17.6456852, 51.2711029, -19.0712318, 55.5847397, -73.2304230, 70.3423309
3: -19.5267849, 65.8024368, -21.0392017, 71.3628387, -90.8896179, 86.8416367
4: -16.3723183, 61.1303329, -17.6571064, 66.1880646, -82.5603790, 78.7874374

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156423, upper bound: 60.2280683
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154447, upper bound: 60.2272640
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16.3341827, 48.6779442, -16.1192989, 48.7391510, -65.0733337, 64.7972412
1: -22.6960735, 50.4090309, -22.4673023, 50.3988075, -73.0948715, 72.8763351
2: -19.4665813, 56.0498695, -19.2585125, 56.0775299, -75.5440979, 75.3083801
3: -21.4443798, 71.8867035, -21.2393341, 71.9838791, -93.4282455, 93.1260300
4: -17.9188328, 66.8347778, -17.8179779, 66.7789383, -84.6977539, 84.6527557

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156127, upper bound: 60.2208379
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154964, upper bound: 60.2189932
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0024462, 31.1266365, -8.5272961, 27.1984768, -37.2009239, 39.6539307
1: -14.2259636, 32.3695755, -12.1715784, 28.2770271, -42.5029793, 44.5411530
2: -12.3026562, 36.0869064, -10.4850693, 31.5316677, -43.8343239, 46.5719757
3: -13.2756996, 46.2974777, -11.4081163, 40.5598831, -53.8355789, 57.7055931
4: -11.5681705, 42.9345131, -9.9473915, 37.5223465, -49.0905037, 52.8819046

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1933787, upper bound: 60.1837300
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1933787, upper bound: 60.1864485
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.7370825, 30.8880539, -8.6991749, 27.7366791, -37.4737625, 39.5872231
1: -13.8120041, 32.0481262, -12.4092007, 28.8249302, -42.6369324, 44.4573288
2: -11.8939190, 35.7687950, -10.6860304, 32.1388817, -44.0327988, 46.4548264
3: -13.0009775, 45.9720306, -11.6377096, 41.3525887, -54.3535614, 57.6097412
4: -11.2234163, 42.5065231, -10.1306763, 38.2451363, -49.4685516, 52.6371994

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2233001, upper bound: 60.2236716
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2233001, upper bound: 60.2240234
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0024462, 31.1266365, -16.4101009, 48.8510437, -58.8534889, 47.5367355
1: -14.2259636, 32.3695755, -22.8117599, 50.5889626, -64.8149185, 55.1813240
2: -12.3026562, 36.0869064, -19.5602798, 56.2571030, -68.5597610, 55.6471863
3: -13.2756996, 46.2974777, -21.5363808, 72.1210098, -85.3967133, 67.8338547
4: -11.5681705, 42.9345131, -17.9972343, 67.0797653, -78.6479340, 60.9317398

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2042816, upper bound: 60.1859050
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 30

Time for candidate selection: 8.62 seconds

### Candidate
type: B, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2049207, upper bound: 60.1863768
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1936602, upper bound: 60.1775867
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.7370825, 30.8880539, -16.5652657, 49.3458900, -59.0829697, 47.4533081
1: -13.8120041, 32.0481262, -23.0251637, 51.0942726, -64.9062805, 55.0732880
2: -11.8939190, 35.7687950, -19.7440033, 56.8110428, -68.7049637, 55.5127983
3: -13.0009775, 45.9720306, -21.7395668, 72.8528137, -85.8537903, 67.7115784
4: -11.2234163, 42.5065231, -18.1693535, 67.7313232, -78.9547424, 60.6758766

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2340301, upper bound: 60.2224604
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2302568, upper bound: 60.2236065
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.6242218, 44.3043480, -8.5302973, 27.2577515, -41.8819695, 52.8346443
1: -20.2457829, 45.8350677, -12.1705046, 28.3290157, -48.5747948, 58.0055695
2: -17.4124851, 51.0891838, -10.4818907, 31.5915146, -49.0039978, 61.5710754
3: -19.2638702, 65.6291351, -11.4172821, 40.6608162, -59.9246864, 77.0464172
4: -16.2089767, 60.8570938, -9.9504223, 37.5961151, -53.8050880, 70.8075180

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193896, upper bound: 60.2288994
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193896, upper bound: 60.2303522
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8565083, 47.9964638, -8.6602230, 27.6217041, -43.4782104, 56.6566849
1: -22.0979061, 49.6354141, -12.3543396, 28.7050247, -50.8029327, 61.9897537
2: -18.9476433, 55.2243576, -10.6386271, 32.0058823, -50.9535141, 65.8629837
3: -20.9038620, 70.9059448, -11.5859261, 41.1813660, -62.0852242, 82.4918671
4: -17.5394630, 65.7744293, -10.0872440, 38.0864449, -55.6259003, 75.8616714

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2302805, upper bound: 60.2288994
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2302805, upper bound: 60.2303522
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.6242218, 44.3043480, -16.3102360, 48.5678406, -63.1920624, 60.6145859
1: -20.2457829, 45.8350677, -22.6321030, 50.2877235, -70.5334930, 68.4671707
2: -17.4124851, 51.0891838, -19.4089298, 55.9369812, -73.3494644, 70.4981155
3: -19.2638702, 65.6291351, -21.3918381, 71.7070923, -90.9709625, 87.0209732
4: -16.2089767, 60.8570938, -17.8806934, 66.6766815, -82.8856583, 78.7377777

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2059931, upper bound: 60.2262364
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2059931, upper bound: 60.2309016
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8565083, 47.9964638, -16.4778557, 49.0316277, -64.8881226, 64.4743195
1: -22.0979061, 49.6354141, -22.8805523, 50.7702751, -72.8681793, 72.5159683
2: -18.9476433, 55.2243576, -19.6148682, 56.4666824, -75.4143219, 74.8392258
3: -20.9038620, 70.9059448, -21.6091194, 72.3687057, -93.2725601, 92.5150452
4: -17.5394630, 65.7744293, -18.0539436, 67.3143234, -84.8537903, 83.8283691

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104263, upper bound: 60.2194369
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2200336, upper bound: 60.2241022
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.7370825, 30.8880539, -9.9511909, 31.5481949, -41.2852783, 40.8392372
1: -13.8120041, 32.0481262, -14.1082268, 32.7253380, -46.5373421, 46.1563530
2: -11.8939190, 35.7687950, -12.1443062, 36.5262375, -48.4201546, 47.9130936
3: -13.0009775, 45.9720306, -13.2896729, 46.9455948, -59.9465714, 59.2617035
4: -11.2234163, 42.5065231, -11.4511833, 43.4067154, -54.6301308, 53.9576988

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1836546, upper bound: 60.1992214
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1836546, upper bound: 60.2210875
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.7370825, 30.8880539, -16.1192989, 48.7391510, -58.4762306, 47.0073395
1: -13.8120041, 32.0481262, -22.4673023, 50.3988075, -64.2108154, 54.5154266
2: -11.8939190, 35.7687950, -19.2585125, 56.0775299, -67.9714432, 55.0273018
3: -13.0009775, 45.9720306, -21.2393341, 71.9838791, -84.9848557, 67.2113571
4: -11.2234163, 42.5065231, -17.8179779, 66.7789383, -78.0023499, 60.3245010

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304047, upper bound: 60.2099024
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2271449, upper bound: 60.2207933
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.6242218, 44.3043480, -9.8296013, 31.1956444, -45.8198662, 54.1339417
1: -20.2457829, 45.8350677, -13.9361629, 32.3614120, -52.6071892, 59.7712326
2: -17.4124851, 51.0891838, -11.9998665, 36.1235733, -53.5360565, 63.0890427
3: -19.2638702, 65.6291351, -13.1277885, 46.4352493, -65.6991196, 78.7569122
4: -16.2089767, 60.8570938, -11.3237314, 42.9283142, -59.1372871, 72.1808167

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2071336, upper bound: 60.2290190
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2299762
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8565083, 47.9964638, -9.9511909, 31.5481949, -47.4047012, 57.9476509
1: -22.0979061, 49.6354141, -14.1082268, 32.7253380, -54.8232384, 63.7436371
2: -18.9476433, 55.2243576, -12.1443062, 36.5262375, -55.4738731, 67.3686523
3: -20.9038620, 70.9059448, -13.2896729, 46.9455948, -67.8494568, 84.1955948
4: -17.5394630, 65.7744293, -11.4511833, 43.4067154, -60.9461632, 77.2256165

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2175960, upper bound: 60.2258220
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2205797, upper bound: 60.2267792
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.6242218, 44.3043480, -15.9651785, 48.3056831, -62.9299049, 60.2695160
1: -20.2457829, 45.8350677, -22.2454796, 49.9502563, -70.1960297, 68.0805511
2: -17.4124851, 51.0891838, -19.0712318, 55.5847397, -72.9972229, 70.1604156
3: -19.2638702, 65.6291351, -21.0392017, 71.3628387, -90.6267090, 86.6683350
4: -16.2089767, 60.8570938, -17.6571064, 66.1880646, -82.3970413, 78.5141983

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2049972, upper bound: 60.2236471
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2116103, upper bound: 60.2228429
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8565083, 47.9964638, -16.1192989, 48.7391510, -64.5956573, 64.1157608
1: -22.0979061, 49.6354141, -22.4673023, 50.3988075, -72.4967117, 72.1027145
2: -18.9476433, 55.2243576, -19.2585125, 56.0775299, -75.0251770, 74.4828720
3: -20.9038620, 70.9059448, -21.2393341, 71.9838791, -92.8877411, 92.1452637
4: -17.5394630, 65.7744293, -17.8179779, 66.7789383, -84.3183975, 83.5924072

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163904, upper bound: 60.2163169
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163905, upper bound: 60.2272077
time: 1.03 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.80 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2295843, upper bound: 60.2315919
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2292037, upper bound: 60.2321526
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2320932, upper bound: 60.2311520
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2317126, upper bound: 60.2317126
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2271145
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2317797
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2234892
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2185023, upper bound: 60.2281544
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2271145, upper bound: 60.2185265
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2234892, upper bound: 60.2185023
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2317797, upper bound: 60.2281338
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2234892, upper bound: 60.2281097
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2143773, upper bound: 60.2143773
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2143773, upper bound: 60.2186772
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2312285
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2355284
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2170657, upper bound: 60.2306090
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2200494, upper bound: 60.2315662
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2196810, upper bound: 60.2288680
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2226647, upper bound: 60.2298252
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2155889, upper bound: 60.2245252
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2241436, upper bound: 60.2237210
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2303522, upper bound: 60.2193896
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2303522, upper bound: 60.2302805
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2170657, upper bound: 60.2306090
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2201904, upper bound: 60.2336065
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2196810, upper bound: 60.2288680
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2226647, upper bound: 60.2298252
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2156423, upper bound: 60.2280683
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2154447, upper bound: 60.2272640
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2156127, upper bound: 60.2208379
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2154964, upper bound: 60.2189932
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.1933787, upper bound: 60.1837300
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.1933787, upper bound: 60.1864485
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2233001, upper bound: 60.2236716
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2233001, upper bound: 60.2240234
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2049207, upper bound: 60.1863768
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.1936602, upper bound: 60.1775867
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2340301, upper bound: 60.2224604
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2302568, upper bound: 60.2236065
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2193896, upper bound: 60.2288994
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2193896, upper bound: 60.2303522
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2302805, upper bound: 60.2288994
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2302805, upper bound: 60.2303522
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2059931, upper bound: 60.2262364
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2059931, upper bound: 60.2309016
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2104263, upper bound: 60.2194369
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2200336, upper bound: 60.2241022
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.1836546, upper bound: 60.1992214
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.1836546, upper bound: 60.2210875
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2304047, upper bound: 60.2099024
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2271449, upper bound: 60.2207933
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2071336, upper bound: 60.2290190
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2299762
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2175960, upper bound: 60.2258220
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2205797, upper bound: 60.2267792
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2049972, upper bound: 60.2236471
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2116103, upper bound: 60.2228429
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2163904, upper bound: 60.2163169
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.80
Output dim: 4, lower bound: -60.2163905, upper bound: 60.2272077

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.4733706, 24.1488380, -7.8953781, 25.5634499, -33.0368118, 32.0442162
1: -10.6540947, 25.1270370, -11.3114214, 26.5612755, -37.2153702, 36.4384575
2: -9.2093201, 28.1137600, -9.7361364, 29.6353054, -38.8446274, 37.8498955
3: -9.9912634, 36.2262306, -10.6232224, 38.1942787, -48.1855431, 46.8494492
4: -8.8301640, 33.4773788, -9.2876635, 35.2739029, -44.1040649, 42.7650414

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2251909, upper bound: 60.2312096
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201247, upper bound: 60.2275221
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291133, upper bound: 60.2314100
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291133, upper bound: 60.2314100
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.4196405, 24.0078678, -8.8967762, 28.4747906, -35.8944321, 32.9046402
1: -10.5789766, 24.9779778, -12.6721573, 29.5220795, -40.1010551, 37.6501312
2: -9.1422520, 27.9490528, -10.8967209, 32.9070091, -42.0492630, 38.8457718
3: -9.9242487, 36.0182304, -11.9140720, 42.4778252, -52.4020729, 47.9322968
4: -8.7711334, 33.2835312, -10.3470421, 39.2232437, -47.9943771, 43.6305733

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291133, upper bound: 60.2321526
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291133, upper bound: 60.2321526
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4589252, 27.0919533, -8.0124798, 25.8991966, -34.3581238, 35.1044273
1: -12.0820923, 28.1615906, -11.4759483, 26.9079971, -38.9900894, 39.6375389
2: -10.4120913, 31.3883038, -9.8741512, 30.0176525, -40.4297447, 41.2624550
3: -11.3392544, 40.4026985, -10.7776766, 38.6751404, -50.0143890, 51.1803741
4: -9.8867702, 37.3420067, -9.4093781, 35.7256699, -45.6124382, 46.7513847

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2309701, upper bound: 60.2309701
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2309701, upper bound: 60.2309701
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4094877, 26.9525166, -9.0249634, 28.8327293, -37.2422180, 35.9774742
1: -12.0130062, 28.0146523, -12.8521852, 29.8926868, -41.9056931, 40.8668365
2: -10.3520050, 31.2265968, -11.0500860, 33.3156586, -43.6676636, 42.2766838
3: -11.2743301, 40.1990051, -12.0788307, 42.9909897, -54.2653122, 52.2778358
4: -9.8336296, 37.1533051, -10.4818535, 39.7062187, -49.5398483, 47.6351585

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291133, upper bound: 60.2317127
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291133, upper bound: 60.2317127
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.4733706, 24.1488380, -13.9227715, 41.3003120, -48.7736816, 38.0716095
1: -10.6540947, 25.1270370, -19.2871590, 42.8253708, -53.4794655, 44.4141960
2: -9.2093201, 28.1137600, -16.5162201, 47.7624969, -56.9718132, 44.6299744
3: -9.9912634, 36.2262306, -18.2561073, 60.8009834, -70.7922440, 54.4823380
4: -8.8301640, 33.4773788, -15.1996059, 56.9115524, -65.7417145, 48.6769867

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2267815
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154153, upper bound: 60.2228348
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167955, upper bound: 60.2266163
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164180, upper bound: 60.2254374
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154736, upper bound: 60.2240419
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152922, upper bound: 60.2232275
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.4344382, 24.0414143, -17.7698860, 52.8597679, -60.2942009, 41.8113022
1: -10.6012220, 25.0160770, -24.7114143, 54.6965866, -65.2978058, 49.7274933
2: -9.1632729, 27.9899826, -21.1234741, 60.7999763, -69.9632416, 49.1134529
3: -9.9428587, 36.0663719, -23.3379917, 77.7605591, -87.7034149, 59.4043617
4: -8.7887726, 33.3289146, -19.3600006, 72.4372864, -81.2260590, 52.6889153

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185265, upper bound: 60.2296050
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226735, upper bound: 60.2228712
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167955, upper bound: 60.2292124
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164180, upper bound: 60.2289916
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2250757, upper bound: 60.2288231
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2248943, upper bound: 60.2280087
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4589252, 27.0919533, -14.1109037, 41.8675880, -50.3265152, 41.2028580
1: -12.0820923, 28.1615906, -19.5679359, 43.4133492, -55.4954414, 47.7295265
2: -10.4120913, 31.3883038, -16.7519665, 48.3994026, -58.8114929, 48.1402664
3: -11.3392544, 40.4026985, -18.5002823, 61.5973701, -72.9366226, 58.9029808
4: -9.8867702, 37.3420067, -15.4016666, 57.6641388, -67.5509109, 52.7436752

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2158267, upper bound: 60.2132620
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183932, upper bound: 60.2234835
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4063702, 26.9487934, -17.9450722, 53.3562126, -61.7625809, 44.8938484
1: -12.0087662, 28.0136032, -24.9718819, 55.2171478, -67.2259140, 52.9854851
2: -10.3484259, 31.2240810, -21.3408375, 61.3745003, -71.7229233, 52.5649185
3: -11.2734461, 40.1931801, -23.5634995, 78.4572372, -89.7306824, 63.7566795
4: -9.8305883, 37.1457024, -19.5413036, 73.1064529, -82.9370422, 56.6869965

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2254984, upper bound: 60.2179272
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2280711, upper bound: 60.2281487
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -13.9227715, 41.3003120, -7.4071975, 23.9491329, -37.8719025, 48.7075081
1: -19.2871590, 42.8253708, -10.5605774, 24.9188671, -44.2060242, 53.3859482
2: -16.5162201, 47.7624969, -9.1286802, 27.8826504, -44.3988571, 56.8911743
3: -18.2561073, 60.8009834, -9.9030905, 35.9287338, -54.1848373, 70.7040710
4: -15.1996059, 56.9115524, -8.7557793, 33.2018623, -48.4014664, 65.6673279

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212568, upper bound: 60.2161875
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 27

Time for candidate selection: 8.20 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2187801, upper bound: 60.2146082
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134007, upper bound: 60.2082098
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.1109037, 41.8675880, -8.4208174, 26.9780807, -41.0889854, 50.2884064
1: -19.5679359, 43.4133492, -12.0280809, 28.0432243, -47.6111603, 55.4414291
2: -16.7519665, 48.3994026, -10.3656874, 31.2567635, -48.0087204, 58.7650909
3: -18.5002823, 61.5973701, -11.2880955, 40.2329979, -58.7332802, 72.8854599
4: -15.4016666, 57.6641388, -9.8443117, 37.1850586, -52.5867233, 67.5084534

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2206163, upper bound: 60.2167045
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 27

Time for candidate selection: 8.11 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2190868, upper bound: 60.2179902
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2137075, upper bound: 60.2115917
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17.7698860, 52.8597679, -7.3683577, 23.8421078, -41.6119881, 60.2281265
1: -24.7114143, 54.6965866, -10.5078516, 24.8083344, -49.5197487, 65.2044373
2: -21.1234741, 60.7999763, -9.0827789, 27.7593117, -48.8827858, 69.8827515
3: -23.3379917, 77.7605591, -9.8548803, 35.7694893, -59.1074829, 87.6154404
4: -19.3600006, 72.4372864, -8.7145424, 33.0538597, -52.4138565, 81.1518250

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212568, upper bound: 60.2256206
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -17.9450722, 53.3562126, -8.3685389, 26.8362865, -44.7813568, 61.7247467
1: -24.9718819, 55.2171478, -11.9551392, 27.8965378, -52.8684196, 67.1722870
2: -21.3408375, 61.3745003, -10.3023033, 31.0940971, -52.4349327, 71.6768036
3: -23.5634995, 78.4572372, -11.2226753, 40.0256844, -63.5891838, 89.6799088
4: -19.5413036, 73.1064529, -9.7883987, 36.9906616, -56.5319557, 82.8948517

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2206163, upper bound: 60.2261376
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281544, upper bound: 60.2281097
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2221711, upper bound: 60.2250329
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2211984, upper bound: 60.2244982
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.1109037, 41.8675880, -14.0264816, 41.5607224, -55.6716232, 55.8940697
1: -19.5679359, 43.4133492, -19.4275627, 43.0964813, -62.6644173, 62.8409119
2: -16.7519665, 48.3994026, -16.6255379, 48.0640945, -64.8160400, 65.0249405
3: -18.5002823, 61.5973701, -18.3753910, 61.1247749, -79.6250534, 79.9727554
4: -15.4016666, 57.6641388, -15.2870178, 57.2594147, -72.6610794, 72.9511566

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.1109037, 41.8675880, -17.7583504, 52.7161827, -66.8270798, 59.6259308
1: -19.5679359, 43.4133492, -24.6639824, 54.5493889, -74.1173248, 68.0773163
2: -16.7519665, 48.3994026, -21.0661926, 60.6673622, -77.4193115, 69.4655914
3: -18.5002823, 61.5973701, -23.2926273, 77.4969101, -95.9971848, 84.8899994
4: -15.4016666, 57.6641388, -19.3064880, 72.2635345, -87.6651993, 76.9706268

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096506
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096506, upper bound: 60.2096874
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.9450722, 53.3562126, -14.0264816, 41.5607224, -59.5057945, 67.3826904
1: -24.9718819, 55.2171478, -19.4275627, 43.0964813, -68.0683517, 74.6447144
2: -21.3408375, 61.3745003, -16.6255379, 48.0640945, -69.4049301, 78.0000381
3: -23.5634995, 78.4572372, -18.3753910, 61.1247749, -84.6882782, 96.8326263
4: -19.5413036, 73.1064529, -15.2870178, 57.2594147, -76.8007202, 88.3934708

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2185223, upper bound: 60.2306575
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184850, upper bound: 60.2234271
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.9450722, 53.3562126, -17.7583504, 52.7161827, -70.6612549, 71.1145554
1: -24.9718819, 55.2171478, -24.6639824, 54.5493889, -79.5212708, 79.8811264
2: -21.3408375, 61.3745003, -21.0661926, 60.6673622, -82.0082016, 82.4406891
3: -23.5634995, 78.4572372, -23.2926273, 77.4969101, -101.0604095, 101.7498627
4: -19.5413036, 73.1064529, -19.3064880, 72.2635345, -91.8048325, 92.4129410

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172576, upper bound: 60.2332730
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164218, upper bound: 60.2194570
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.4733706, 24.1488380, -8.9668245, 28.6942005, -36.1675720, 33.1156616
1: -10.6540947, 25.1270370, -12.6839676, 29.7587776, -40.4128723, 37.8110046
2: -9.2093201, 28.1137600, -10.9171219, 33.2355728, -42.4448929, 39.0308838
3: -9.9912634, 36.2262306, -12.0246906, 42.7058754, -52.6971397, 48.2509232
4: -8.8301640, 33.4773788, -10.3390379, 39.4993629, -48.3295212, 43.8164177

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170657, upper bound: 60.2299804
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2150243, upper bound: 60.2282229
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2151629, upper bound: 60.2300138
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=65.54161834716797
rel_dist={4: [-60.23730919021928, 60.23730919021929]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319216, upper bound: 60.2306415
time: 0.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464
time: 0.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.06 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.06
Output dim: 4, lower bound: -60.2319216, upper bound: 60.2306415
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.06
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.4466705, 33.0043793, -11.7410507, 36.6477890, -47.0944557, 44.7454185
1: -14.8643341, 34.2352486, -16.6493034, 37.9966087, -52.8609428, 50.8845520
2: -12.7760410, 38.1106606, -14.2999506, 42.2655869, -55.0416260, 52.4106102
3: -13.9698524, 49.0998573, -15.6488228, 54.3559875, -68.3258057, 64.7486801
4: -12.0348778, 45.3377380, -13.3779221, 50.2773666, -62.3122444, 58.7156601

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2306415
time: 0.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319216, upper bound: 60.2306415
time: 0.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.5646820, 36.3494148, -11.3334484, 35.5176582, -47.0823402, 47.6828613
1: -16.3725929, 37.6687965, -16.0957565, 36.8292351, -53.2018280, 53.7645531
2: -14.0860319, 41.9790382, -13.8299198, 40.9751244, -55.0611534, 55.8089561
3: -15.4223356, 53.9925766, -15.1328430, 52.7092705, -68.1316071, 69.1254196
4: -13.2208614, 49.8709221, -12.9602242, 48.7316437, -61.9525070, 62.8311462

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2290221
time: 0.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464
time: 0.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.69 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2306415
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 4, lower bound: -60.2319216, upper bound: 60.2306415
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2290221
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.69
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.0631580, 31.8493309, -9.9006662, 31.1688557, -41.2320137, 41.7499924
1: -14.3246431, 33.0462608, -14.0700378, 32.3676949, -46.6923370, 47.1162910
2: -12.3158627, 36.7989120, -12.1031590, 36.0536575, -48.3695221, 48.9020691
3: -13.4566593, 47.4042664, -13.2101288, 46.3044205, -59.7610779, 60.6143913
4: -11.6134672, 43.7827873, -11.3959522, 42.8965683, -54.5100365, 55.1787415

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2306415
time: 0.91 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.2615223, 32.4407578, -18.3114948, 54.3105965, -64.5721207, 50.7522507
1: -14.6010656, 33.6536789, -25.5333328, 56.2338333, -70.8348923, 59.1870117
2: -12.5514145, 37.4663925, -21.8558426, 62.5287628, -75.0801773, 59.3222275
3: -13.7237110, 48.2649994, -24.0195408, 79.9852448, -93.7089539, 72.2845383
4: -11.8317356, 44.5688171, -19.9994240, 74.5350037, -86.3667374, 64.5682373

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319147, upper bound: 60.2281785
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319147, upper bound: 60.2281785
time: 1.06 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.2191038, 35.3188286, -9.4664173, 29.9630108, -41.1821136, 44.7852440
1: -15.8826342, 36.6059456, -13.4767380, 31.1234074, -47.0060425, 50.0826836
2: -13.6658754, 40.8085709, -11.5961876, 34.6756439, -48.3415146, 52.4047585
3: -14.9619980, 52.4795532, -12.6572742, 44.5472107, -59.5092087, 65.1368256
4: -12.8383369, 48.4834862, -10.9415665, 41.2441177, -54.0824547, 59.4250488

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2290221
time: 0.93 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.0889120, 34.8989944, -17.4058952, 51.8680573, -62.9569702, 52.3048897
1: -15.7143784, 36.1689301, -24.2758865, 53.6957932, -69.4101715, 60.4448051
2: -13.5199480, 40.3033752, -20.7874203, 59.7051582, -73.2251053, 61.0907936
3: -14.7961807, 51.8413086, -22.8697548, 76.4501877, -91.2463684, 74.7110596
4: -12.7073727, 47.8885651, -19.0702267, 71.1639862, -83.8713531, 66.9587936

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528
time: 0.93 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2306415
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 4, lower bound: -60.2319147, upper bound: 60.2281785
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 4, lower bound: -60.2319147, upper bound: 60.2281785
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2290221
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -9.9006662, 31.1688557, -39.8680305, 37.6373444
1: -12.4092007, 28.8249302, -14.0700378, 32.3676949, -44.7768936, 42.8949661
2: -10.6860304, 32.1388817, -12.1031590, 36.0536575, -46.7396889, 44.2420387
3: -11.6377096, 41.3525887, -13.2101288, 46.3044205, -57.9421272, 54.5627174
4: -10.1306763, 38.2451363, -11.3959522, 42.8965683, -53.0272446, 49.6410904

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2180946
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16.4778175, 49.0314636, -9.9006662, 31.1688557, -47.6466751, 58.9321251
1: -22.8804989, 50.7700958, -14.0700378, 32.3676949, -55.2481918, 64.8401337
2: -19.6148262, 56.4664841, -12.1031590, 36.0536575, -55.6684799, 68.5696411
3: -21.6090603, 72.3684540, -13.2101288, 46.3044205, -67.9134598, 85.5785828
4: -18.0539055, 67.3141098, -11.3959522, 42.8965683, -60.9504738, 78.7100601

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2223030
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2306415
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2306415
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.6602230, 27.6217041, -18.3114948, 54.3105965, -62.9708099, 45.9331970
1: -12.3543396, 28.7050247, -25.5333328, 56.2338333, -68.5881729, 54.2383575
2: -10.6386271, 32.0058823, -21.8558426, 62.5287628, -73.1673889, 53.8617096
3: -11.5859261, 41.1813660, -24.0195408, 79.9852448, -91.5711670, 65.2009048
4: -10.0872440, 38.0864449, -19.9994240, 74.5350037, -84.6222458, 58.0858688

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2186583
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -16.4778557, 49.0316277, -18.3114948, 54.3105965, -70.7884445, 67.3431091
1: -22.8805523, 50.7702751, -25.5333328, 56.2338333, -79.1143875, 76.3035965
2: -19.6148682, 56.4666824, -21.8558426, 62.5287628, -82.1436310, 78.3225250
3: -21.6091194, 72.3687057, -24.0195408, 79.9852448, -101.5943604, 96.3882446
4: -18.0539436, 67.3143234, -19.9994240, 74.5350037, -92.5889435, 87.3137512

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2180946
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -9.4664173, 29.9630108, -39.9141998, 41.0146103
1: -14.1082268, 32.7253380, -13.4767380, 31.1234074, -45.2316322, 46.2020760
2: -12.1443062, 36.5262375, -11.5961876, 34.6756439, -46.8199501, 48.1224251
3: -13.2896729, 46.9455948, -12.6572742, 44.5472107, -57.8368835, 59.6028671
4: -11.4511833, 43.4067154, -10.9415665, 41.2441177, -52.6952934, 54.3482742

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2164250
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2210528
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16.0714569, 48.5650826, -9.4664173, 29.9630108, -46.0344696, 58.0315018
1: -22.3887920, 50.2195358, -13.4767380, 31.1234074, -53.5121994, 63.6962738
2: -19.1879387, 55.8870087, -11.5961876, 34.6756439, -53.8635826, 67.4831924
3: -21.1674252, 71.7135315, -12.6572742, 44.5472107, -65.7146378, 84.3708038
4: -17.7540932, 66.5464325, -10.9415665, 41.2441177, -58.9982071, 77.4879990

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2176066
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2290138
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7923231, 31.0757713, -17.4058952, 51.8680573, -61.6603813, 48.4816666
1: -13.8838530, 32.2330780, -24.2758865, 53.6957932, -67.5796432, 56.5089645
2: -11.9508610, 35.9805870, -20.7874203, 59.7051582, -71.6560135, 56.7680054
3: -13.0776491, 46.2406044, -22.8697548, 76.4501877, -89.5278244, 69.1103592
4: -11.2742805, 42.7561111, -19.0702267, 71.1639862, -82.4382629, 61.8263359

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -17.4058952, 51.8680573, -67.9873581, 66.1450424
1: -22.4673023, 50.3988075, -24.2758865, 53.6957932, -76.1630936, 74.6746902
2: -19.2585125, 56.0775299, -20.7874203, 59.7051582, -78.9636688, 76.8649521
3: -21.2393341, 71.9838791, -22.8697548, 76.4501877, -97.6895218, 94.8536377
4: -17.8179779, 66.7789383, -19.0702267, 71.1639862, -88.9819641, 85.8491592

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2290464
time: 0.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.38 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2306415
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2306415
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2237269, upper bound: 60.2281785
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2164250
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2210528
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2176066
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2290138
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.38
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2290464

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -8.6991749, 27.7366791, -36.4358521, 36.4358521
1: -12.4092007, 28.8249302, -12.4092007, 28.8249302, -41.2341309, 41.2341309
2: -10.6860304, 32.1388817, -10.6860304, 32.1388817, -42.8249130, 42.8249130
3: -11.6377096, 41.3525887, -11.6377096, 41.3525887, -52.9902954, 52.9902954
4: -10.1306763, 38.2451363, -10.1306763, 38.2451363, -48.3758125, 48.3758125

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113389, upper bound: 60.2103733
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2235483, upper bound: 60.2280849
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -9.9511909, 31.5481949, -40.2473679, 37.6878700
1: -12.4092007, 28.8249302, -14.1082268, 32.7253380, -45.1345367, 42.9331551
2: -10.6860304, 32.1388817, -12.1443062, 36.5262375, -47.2122688, 44.2831879
3: -11.6377096, 41.3525887, -13.2896729, 46.9455948, -58.5833054, 54.6422615
4: -10.1306763, 38.2451363, -11.4511833, 43.4067154, -53.5373917, 49.6963081

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113389, upper bound: 60.2103733
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2235483, upper bound: 60.2280849
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16.4778175, 49.0314636, -8.6991749, 27.7366791, -44.2144966, 57.7306366
1: -22.8804989, 50.7700958, -12.4092007, 28.8249302, -51.7054291, 63.1792946
2: -19.6148262, 56.4664841, -10.6860304, 32.1388817, -51.7537041, 67.1525116
3: -21.6090603, 72.3684540, -11.6377096, 41.3525887, -62.9616470, 84.0061646
4: -18.0539055, 67.3141098, -10.1306763, 38.2451363, -56.2990417, 77.4447861

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2189744, upper bound: 60.2303298
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2189744, upper bound: 60.2289682
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16.4778175, 49.0314636, -9.9511909, 31.5481949, -48.0260124, 58.9826546
1: -22.8804989, 50.7700958, -14.1082268, 32.7253380, -55.6058350, 64.8783112
2: -19.6148262, 56.4664841, -12.1443062, 36.5262375, -56.1410637, 68.6107941
3: -21.6090603, 72.3684540, -13.2896729, 46.9455948, -68.5546570, 85.6581268
4: -18.0539055, 67.3141098, -11.4511833, 43.4067154, -61.4606209, 78.7652893

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2189744, upper bound: 60.2303298
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2235392, upper bound: 60.2289682
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.6602230, 27.6217041, -16.8046818, 50.0466156, -58.7068405, 44.4263840
1: -12.3543396, 28.7050247, -23.3835297, 51.8265610, -64.1809006, 52.0885544
2: -10.6386271, 32.0058823, -20.0460148, 57.6273384, -68.2659683, 52.0518951
3: -11.5859261, 41.1813660, -22.0566101, 73.8997879, -85.4857178, 63.2379723
4: -10.0872440, 38.0864449, -18.4532261, 68.7191925, -78.8064346, 56.5396729

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286164, upper bound: 60.2253749
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319142, upper bound: 60.2280775
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2318660, upper bound: 60.2255980
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.6602230, 27.6217041, -16.3950844, 49.4747887, -58.1350098, 44.0167885
1: -12.3543396, 28.7050247, -22.8638802, 51.1643333, -63.5186691, 51.5689049
2: -10.6386271, 32.0058823, -19.5991325, 56.9245262, -67.5631561, 51.6050148
3: -11.5859261, 41.1813660, -21.5911999, 73.0606918, -84.6466217, 62.7725677
4: -10.0872440, 38.0864449, -18.1113377, 67.8160400, -77.9032822, 56.1977844

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286164, upper bound: 60.2253749
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2298859, upper bound: 60.2279945
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16.4778557, 49.0316277, -16.8046818, 50.0466156, -66.5244751, 65.8362885
1: -22.8805523, 50.7702751, -23.3835297, 51.8265610, -74.7071152, 74.1538010
2: -19.6148682, 56.4666824, -20.0460148, 57.6273384, -77.2422028, 76.5126953
3: -21.6091194, 72.3687057, -22.0566101, 73.8997879, -95.5088959, 94.4253159
4: -18.0539436, 67.3143234, -18.4532261, 68.7191925, -86.7731323, 85.7675476

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2094261, upper bound: 60.2094261
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161799, upper bound: 60.2249570
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16.4778557, 49.0316277, -16.3950844, 49.4747887, -65.9526443, 65.4267120
1: -22.8805523, 50.7702751, -22.8638802, 51.1643333, -74.0448837, 73.6341553
2: -19.6148682, 56.4666824, -19.5991325, 56.9245262, -76.5393906, 76.0658112
3: -21.6091194, 72.3687057, -21.5911999, 73.0606918, -94.6697998, 93.9599075
4: -18.0539436, 67.3143234, -18.1113377, 67.8160400, -85.8699799, 85.4256592

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2094261, upper bound: 60.2094261
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161799, upper bound: 60.2249570
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.7470121, 30.9525509, -8.5719185, 27.3454933, -37.0925064, 39.5244675
1: -13.8253012, 32.1093864, -12.2496243, 28.4244652, -42.2497673, 44.3590088
2: -11.9031410, 35.8449860, -10.5459433, 31.6978016, -43.6009407, 46.3909302
3: -13.0212603, 46.0756302, -11.4941177, 40.7280464, -53.7493057, 57.5697479
4: -11.2350693, 42.6007767, -10.0002117, 37.7046280, -48.9396935, 52.6009903

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2011495, upper bound: 60.2033701
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2011495, upper bound: 60.2151982
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.7768440, 31.0364990, -10.0309877, 31.3629513, -41.1397934, 41.0674858
1: -13.8651476, 32.1935692, -14.1997900, 32.5445366, -46.4096832, 46.3933601
2: -11.9377117, 35.9426842, -12.2250910, 36.3165894, -48.2542992, 48.1677742
3: -13.0608997, 46.1977425, -13.3469543, 46.6344604, -59.6953583, 59.5446968
4: -11.2678738, 42.7165642, -11.4994755, 43.2459297, -54.5138016, 54.2160416

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207325, upper bound: 60.2207325
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15.8737831, 47.9716759, -8.5719185, 27.3454933, -43.2192764, 56.5435944
1: -22.1125431, 49.6083794, -12.2496243, 28.4244652, -50.5370102, 61.8579979
2: -18.9528809, 55.2067261, -10.5459433, 31.6978016, -50.6506805, 65.7526627
3: -20.9076366, 70.8447647, -11.4941177, 40.7280464, -61.6356812, 82.3388672
4: -17.5407963, 65.7497940, -10.0002117, 37.7046280, -55.2454224, 75.7499847

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 2

Time for candidate selection: 7.13 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1985861, upper bound: 60.2035991
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2039826, upper bound: 60.2144356
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.8518715, 47.9288902, -10.0309877, 31.3629513, -47.2148170, 57.9598770
1: -22.0795727, 49.5626221, -14.1997900, 32.5445366, -54.6241074, 63.7624130
2: -18.9236507, 55.1650734, -12.2250910, 36.3165894, -55.2402382, 67.3901672
3: -20.8850937, 70.7933197, -13.3469543, 46.6344604, -67.5195541, 84.1402740
4: -17.5249672, 65.6878738, -11.4994755, 43.2459297, -60.7708969, 77.1873398

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2098436, upper bound: 60.2287065
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207499, upper bound: 60.2269631
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.7923231, 31.0757713, -16.7040405, 49.7543106, -59.5466347, 47.7798119
1: -13.8838530, 32.2330780, -23.2350788, 51.5204048, -65.4042587, 55.4681511
2: -11.9508610, 35.9805870, -19.9190445, 57.2917175, -69.2425613, 55.8996277
3: -13.0776491, 46.2406044, -21.9273682, 73.4507294, -86.5283737, 68.1679688
4: -11.2742805, 42.7561111, -18.3297539, 68.3172836, -79.5915680, 61.0858612

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286509, upper bound: 60.2207325
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.7923231, 31.0757713, -16.3657169, 49.3871498, -59.1794739, 47.4414902
1: -13.8838530, 32.2330780, -22.8206711, 51.0731087, -64.9569626, 55.0537491
2: -11.9508610, 35.9805870, -19.5613575, 56.8234863, -68.7743378, 55.5419464
3: -13.0776491, 46.2406044, -21.5540047, 72.9262695, -86.0038986, 67.7946091
4: -11.2742805, 42.7561111, -18.0741940, 67.6964035, -78.9706879, 60.8302956

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2257717, upper bound: 60.2207325
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.7040405, 49.7543106, -65.8736115, 65.4431915
1: -22.4673023, 50.3988075, -23.2350788, 51.5204048, -73.9877090, 73.6338806
2: -19.2585125, 56.0775299, -19.9190445, 57.2917175, -76.5502243, 75.9965744
3: -21.2393341, 71.9838791, -21.9273682, 73.4507294, -94.6900635, 93.9112473
4: -17.8179779, 66.7789383, -18.3297539, 68.3172836, -86.1352615, 85.1086884

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2160044, upper bound: 60.2287205
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210380, upper bound: 60.2269635
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.3657169, 49.3871498, -65.5064468, 65.1048660
1: -22.4673023, 50.3988075, -22.8206711, 51.0731087, -73.5404129, 73.2194672
2: -19.2585125, 56.0775299, -19.5613575, 56.8234863, -76.0819931, 75.6388855
3: -21.2393341, 71.9838791, -21.5540047, 72.9262695, -94.1655884, 93.5378876
4: -17.8179779, 66.7789383, -18.0741940, 67.6964035, -85.5143814, 84.8531189

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2160044, upper bound: 60.2287205
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210380, upper bound: 60.2269635
time: 1.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.66 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2113389, upper bound: 60.2103733
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2235483, upper bound: 60.2280849
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2113389, upper bound: 60.2103733
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2235483, upper bound: 60.2280849
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2189744, upper bound: 60.2303298
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2189744, upper bound: 60.2289682
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2189744, upper bound: 60.2303298
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2235392, upper bound: 60.2289682
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2319142, upper bound: 60.2280775
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2318660, upper bound: 60.2255980
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2286164, upper bound: 60.2253749
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2298859, upper bound: 60.2279945
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2094261, upper bound: 60.2094261
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2161799, upper bound: 60.2249570
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2094261, upper bound: 60.2094261
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2161799, upper bound: 60.2249570
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2011495, upper bound: 60.2033701
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2011495, upper bound: 60.2151982
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2207325, upper bound: 60.2207325
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.1985861, upper bound: 60.2035991
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2039826, upper bound: 60.2144356
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2098436, upper bound: 60.2287065
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2207499, upper bound: 60.2269631
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2286509, upper bound: 60.2207325
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2257717, upper bound: 60.2207325
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2160044, upper bound: 60.2287205
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2210380, upper bound: 60.2269635
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2160044, upper bound: 60.2287205
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 4, lower bound: -60.2210380, upper bound: 60.2269635

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.8506246, 24.4222717, -8.2079105, 26.2058640, -34.0564880, 32.6301804
1: -11.2154808, 25.5163822, -11.7297058, 27.2632141, -38.4786873, 37.2460861
2: -9.7223167, 28.4403000, -10.1111317, 30.4109688, -40.1332855, 38.5514297
3: -10.3808765, 36.4236946, -10.9799747, 39.1010284, -49.4819031, 47.4036636
4: -9.2233162, 33.8441010, -9.6065502, 36.1914673, -45.4147720, 43.4506493

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304300, upper bound: 60.2244532
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304300, upper bound: 60.2250084
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.5392914, 27.2805252, -8.6991749, 27.7366791, -36.2759705, 35.9796982
1: -12.1856346, 28.3512688, -12.4092007, 28.8249302, -41.0105667, 40.7604675
2: -10.4923859, 31.6131878, -10.6860304, 32.1388817, -42.6312675, 42.2992172
3: -11.4304800, 40.6814766, -11.6377096, 41.3525887, -52.7830696, 52.3191872
4: -9.9563656, 37.6183701, -10.1306763, 38.2451363, -48.2014923, 47.7490463

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2304300, upper bound: 60.2315973
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2320861, upper bound: 60.2320861
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.8506246, 24.4222717, -9.5460930, 30.3203907, -38.1710167, 33.9683609
1: -11.2154808, 25.5163822, -13.5489969, 31.4726715, -42.6881523, 39.0653725
2: -9.7223167, 28.4403000, -11.6645622, 35.1449509, -44.8672676, 40.1048622
3: -10.3808765, 36.4236946, -12.7538462, 45.1537933, -55.5346680, 49.1775322
4: -9.2233162, 33.8441010, -11.0155716, 41.7678146, -50.9911270, 44.8596687

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2088741, upper bound: 60.2081283
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1970448, upper bound: 60.1928184
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1866351, upper bound: 60.1853469
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.5392914, 27.2805252, -9.9511909, 31.5481949, -40.0874825, 37.2317162
1: -12.1856346, 28.3512688, -14.1082268, 32.7253380, -44.9109726, 42.4594955
2: -10.4923859, 31.6131878, -12.1443062, 36.5262375, -47.0186234, 43.7574921
3: -11.4304800, 40.6814766, -13.2896729, 46.9455948, -58.3760757, 53.9711494
4: -9.9563656, 37.6183701, -11.4511833, 43.4067154, -53.3630714, 49.0695496

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2158604, upper bound: 60.2157527
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1813694, upper bound: 60.1911341
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1813694, upper bound: 60.2280849
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.7739029, 44.0974045, -8.2041168, 26.3572369, -41.1311417, 52.3015213
1: -20.3295860, 45.6811409, -11.7093601, 27.3956833, -47.7252693, 57.3905029
2: -17.5125771, 50.9196777, -10.0925407, 30.5641708, -48.0767479, 61.0122147
3: -19.3952198, 65.3102570, -10.9961452, 39.3727684, -58.7679901, 76.3063965
4: -16.2564678, 60.7049141, -9.6093559, 36.3794479, -52.6359100, 70.3142700

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2283485, upper bound: 60.2354902
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2238390, upper bound: 60.2348302
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16.2470169, 48.3629837, -8.6991749, 27.7366791, -43.9836960, 57.0621567
1: -22.5515976, 50.0848351, -12.4092007, 28.8249302, -51.3765259, 62.4940300
2: -19.3376179, 55.7057762, -10.6860304, 32.1388817, -51.4764938, 66.3917999
3: -21.3147125, 71.4049072, -11.6377096, 41.3525887, -62.6672974, 83.0426178
4: -17.8041382, 66.4190826, -10.1306763, 38.2451363, -56.0492744, 76.5497589

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2327585, upper bound: 60.2314415
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2327585, upper bound: 60.2330276
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.7739029, 44.0974045, -9.4987583, 30.2317543, -45.0056534, 53.5961571
1: -20.3295860, 45.6811409, -13.4690104, 31.3657417, -51.6953278, 59.1501503
2: -17.5125771, 50.9196777, -11.6068840, 35.0222092, -52.5347862, 62.5265617
3: -19.3952198, 65.3102570, -12.6874399, 45.0393448, -64.4345627, 77.9976959
4: -16.2564678, 60.7049141, -10.9760017, 41.6223984, -57.8788643, 71.6809158

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117922, upper bound: 60.2273431
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183792, upper bound: 60.2298670
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16.2470169, 48.3629837, -9.9511909, 31.5481949, -47.7952118, 58.3141747
1: -22.5515976, 50.0848351, -14.1082268, 32.7253380, -55.2769356, 64.1930618
2: -19.3376179, 55.7057762, -12.1443062, 36.5262375, -55.8638458, 67.8500748
3: -21.3147125, 71.4049072, -13.2896729, 46.9455948, -68.2603073, 84.6945801
4: -17.8041382, 66.4190826, -11.4511833, 43.4067154, -61.2108536, 77.8702698

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2194964, upper bound: 60.2268795
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117922, upper bound: 60.2285198
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.9473248, 25.4955311, -16.5643425, 49.3281631, -57.2754822, 42.0598755
1: -11.3680954, 26.5178776, -23.0496655, 51.0872040, -62.4552841, 49.5675430
2: -9.7957926, 29.5887432, -19.7618008, 56.8108177, -66.6066132, 49.3505440
3: -10.6560183, 38.0537453, -21.7438297, 72.8408813, -83.4968719, 59.7975769
4: -9.3301964, 35.2053337, -18.1938057, 67.7537079, -77.0839081, 53.3991280

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308553, upper bound: 60.2297173
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2182676, upper bound: 60.2237412
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2182676, upper bound: 60.2297677
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.4299145, 29.5511837, -16.5531406, 49.3136826, -58.7435989, 46.1043205
1: -13.3265362, 30.6757679, -23.0194359, 51.0691910, -64.3957291, 53.6951981
2: -11.4935360, 34.2454948, -19.7388096, 56.7983475, -68.2918854, 53.9843025
3: -12.5065241, 44.0347290, -21.7314892, 72.8410797, -85.3476028, 65.7662201
4: -10.8536158, 40.8196640, -18.1895885, 67.7266464, -78.5802612, 59.0092506

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163301, upper bound: 60.2168447
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2163301, upper bound: 60.2192972
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.4071975, 23.9491329, -15.7811613, 47.7636032, -55.1707993, 39.7302933
1: -10.5605774, 24.9188671, -21.9785786, 49.3924294, -59.9530067, 46.8974419
2: -9.1286802, 27.8826504, -18.8535156, 54.9785881, -64.1072693, 46.7361488
3: -9.9030905, 35.9287338, -20.7966881, 70.6061401, -80.5092316, 56.7254105
4: -8.7557793, 33.2018623, -17.4714108, 65.4821930, -74.2379761, 50.6732635

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286164, upper bound: 60.2188078
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286164, upper bound: 60.2253749
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.4208174, 26.9780807, -16.3950844, 49.4747887, -57.8956070, 43.3731575
1: -12.0280809, 28.0432243, -22.8638802, 51.1643333, -63.1924057, 50.9071007
2: -10.3656874, 31.2567635, -19.5991325, 56.9245262, -67.2902145, 50.8558960
3: -11.2880955, 40.2329979, -21.5911999, 73.0606918, -84.3487854, 61.8241959
4: -9.8443117, 37.1850586, -18.1113377, 67.8160400, -77.6603470, 55.2963943

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2298859, upper bound: 60.2188078
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2298859, upper bound: 60.2279945
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.0264816, 41.5607224, -16.2499657, 48.3076897, -62.3341713, 57.8106880
1: -19.4275627, 43.0964813, -22.6016197, 50.0412750, -69.4688416, 65.6980972
2: -16.6255379, 48.0640945, -19.3688354, 55.6726570, -72.2981949, 67.4329300
3: -18.3753910, 61.1247749, -21.3230705, 71.3164368, -89.6918182, 82.4478378
4: -15.2870178, 57.2594147, -17.8133068, 66.4069061, -81.6939240, 75.0727234

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2141046, upper bound: 60.2141046
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2141046, upper bound: 60.2182676
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17.7583504, 52.7161827, -16.1907825, 48.3972702, -66.1556168, 68.9069672
1: -24.6639824, 54.5493889, -22.5029907, 50.1119728, -74.7759399, 77.0523758
2: -21.0661926, 60.6673622, -19.2800102, 55.7299461, -76.7961426, 79.9473572
3: -23.2926273, 77.4969101, -21.2789536, 71.4950943, -94.7877197, 98.7758560
4: -19.3064880, 72.2635345, -17.8081436, 66.4556961, -85.7621765, 90.0716705

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2315703, upper bound: 60.2274277
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2230519, upper bound: 60.2274260
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.0264816, 41.5607224, -15.7770824, 47.5518494, -61.5783310, 57.3378067
1: -19.4275627, 43.0964813, -21.9844246, 49.1937943, -68.6213531, 65.0809021
2: -16.6255379, 48.0640945, -18.8414764, 54.7584229, -71.3839569, 66.9055634
3: -18.3753910, 61.1247749, -20.7717686, 70.2238922, -88.5992661, 81.8965149
4: -15.2870178, 57.2594147, -17.4064407, 65.2618256, -80.5488434, 74.6658554

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2094261, upper bound: 60.2094261
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2094261, upper bound: 60.2094261
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.7583504, 52.7161827, -15.8083849, 47.9627571, -65.7210999, 68.5245667
1: -24.6639824, 54.5493889, -22.0459614, 49.5936890, -74.2576752, 76.5953522
2: -21.0661926, 60.6673622, -18.8815460, 55.1911240, -76.2573166, 79.5488968
3: -23.2926273, 77.4969101, -20.8602009, 70.8628006, -94.1554260, 98.3571091
4: -19.3064880, 72.2635345, -17.5104294, 65.7194672, -85.0259552, 89.7739563

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113196, upper bound: 60.2113196
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152466, upper bound: 60.2184828
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.3170586, 29.7991142, -8.4539728, 27.0218468, -36.3389053, 38.2530823
1: -13.2208233, 30.9148979, -12.0886927, 28.0910015, -41.3118172, 43.0035896
2: -11.3778601, 34.5305481, -10.4074154, 31.3298779, -42.7077370, 44.9379539
3: -12.4723749, 44.4027405, -11.3438549, 40.2586708, -52.7310448, 55.7465973
4: -10.7711830, 41.0302963, -9.8770962, 37.2661629, -48.0373459, 50.9073906

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1952408, upper bound: 60.1926744
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1851789, upper bound: 60.1851919
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.4296064, 30.1646137, -8.4131174, 26.9148827, -36.3444901, 38.5777321
1: -13.3704138, 31.2840176, -12.0319939, 27.9801464, -41.3505592, 43.3160095
2: -11.5101395, 34.9556236, -10.3597775, 31.2083130, -42.7184525, 45.3153992
3: -12.6192875, 44.9711990, -11.2915287, 40.1112823, -52.7305679, 56.2627220
4: -10.8931608, 41.5404205, -9.8351927, 37.1211967, -48.0143585, 51.3756142

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1955474, upper bound: 60.1929594
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1854854, upper bound: 60.1854771
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9161539, 28.5332737, -9.8079529, 30.6983566, -39.6145096, 38.3412247
1: -12.6159725, 29.5904236, -13.8782148, 31.8544083, -44.4703827, 43.4686356
2: -10.8581429, 33.0531349, -11.9479771, 35.5505447, -46.4086838, 45.0011139
3: -11.9586287, 42.4661980, -13.0573196, 45.6432037, -57.6018333, 55.5235062
4: -10.2857075, 39.2861977, -11.2456207, 42.3356514, -52.6213570, 50.5318184

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
time: 1.28 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.6825857, 33.8578300, -9.8029814, 30.7944717, -41.4770546, 43.6608124
1: -15.0734386, 35.0701523, -13.8772240, 31.9506149, -47.0240555, 48.9473724
2: -12.9553204, 39.1727257, -11.9406681, 35.6645279, -48.6198502, 51.1133957
3: -14.2432547, 50.3636627, -13.0723743, 45.8091698, -60.0524216, 63.4360199
4: -12.1948233, 46.5769691, -11.2485142, 42.4652138, -54.6600380, 57.8254852

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207325, upper bound: 60.2207325
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207325, upper bound: 60.2207325
time: 1.18 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.9466972, 37.3598442, -8.5719185, 27.3454933, -40.2921906, 45.9317589
1: -17.9134922, 38.9192886, -12.2496243, 28.4244652, -46.3379593, 51.1689072
2: -15.5488110, 43.2466469, -10.5459433, 31.6978016, -47.2466087, 53.7925911
3: -16.7241039, 55.1905479, -11.4941177, 40.7280464, -57.4521446, 66.6846466
4: -14.4517164, 51.5502625, -10.0002117, 37.7046280, -52.1563416, 61.5504761

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1957802, upper bound: 60.2000450
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1961200, upper bound: 60.2009641
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.6497698, 47.2753525, -8.5719185, 27.3454933, -42.9952621, 55.8472710
1: -21.8154945, 48.8985176, -12.2496243, 28.4244652, -50.2399597, 61.1481400
2: -18.7017670, 54.3979950, -10.5459433, 31.6978016, -50.3995667, 64.9439392
3: -20.6180286, 69.8063202, -11.4941177, 40.7280464, -61.3460655, 81.3004227
4: -17.3044968, 64.7805862, -10.0002117, 37.7046280, -55.0091248, 74.7807922

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2026672, upper bound: 60.2117574
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2030111, upper bound: 60.2135434
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.3411875, 43.4400063, -9.4934311, 29.8463478, -44.1875343, 52.9334373
1: -19.8349323, 44.9445953, -13.4412861, 30.9782848, -50.8132172, 58.3858795
2: -17.0622406, 50.1185608, -11.5817471, 34.5817604, -51.6439972, 61.7002983
3: -18.8881016, 64.3532867, -12.6410065, 44.4488983, -63.3369980, 76.9942932
4: -15.8966932, 59.6970749, -10.9299984, 41.1883087, -57.0849991, 70.6270752

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2027902, upper bound: 60.2194630
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2098436, upper bound: 60.2287065
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2098436, upper bound: 60.2287065
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.5943918, 47.2090340, -10.0309877, 31.3629513, -46.9573364, 57.2400208
1: -21.7177887, 48.8225288, -14.1997900, 32.5445366, -54.2623253, 63.0223198
2: -18.6191692, 54.3377037, -12.2250910, 36.3165894, -54.9357567, 66.5627975
3: -20.5576363, 69.7501373, -13.3469543, 46.6344604, -67.1920929, 83.0970840
4: -17.2536831, 64.7128906, -11.4994755, 43.2459297, -60.4996109, 76.2123642

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073641, upper bound: 60.2167034
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207499, upper bound: 60.2269631
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207499, upper bound: 60.2269631
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9365969, 28.5883827, -16.4597111, 48.9909210, -57.9275169, 45.0480957
1: -12.6416416, 29.6450901, -22.8955460, 50.7335510, -63.3751907, 52.5406342
2: -10.8776150, 33.1078873, -19.6281910, 56.4277649, -67.3053818, 52.7360764
3: -11.9829111, 42.5303879, -21.6016426, 72.3020477, -84.2849503, 64.1320343
4: -10.2974758, 39.3461494, -18.0408516, 67.2950287, -77.5925064, 57.3870010

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2236979, upper bound: 60.2152336
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2236979, upper bound: 60.2175400
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -10.7285433, 33.9901962, -16.4650936, 49.1909828, -59.9195251, 50.4552841
1: -15.1347942, 35.2033424, -22.8813095, 50.9291649, -66.0639496, 58.0846519
2: -13.0068378, 39.3147087, -19.5969582, 56.6425819, -69.6494141, 58.9116669
3: -14.2995472, 50.5393524, -21.6586361, 72.6446381, -86.9441833, 72.1979828
4: -12.2354679, 46.7396278, -18.0804539, 67.5422516, -79.7777176, 64.8200836

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2298670, upper bound: 60.2183792
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2256686, upper bound: 60.2224818
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9365969, 28.5883827, -16.1514091, 48.7116394, -57.6482353, 44.7397919
1: -12.6416416, 29.6450901, -22.5156193, 50.3776932, -63.0193329, 52.1607056
2: -10.8776150, 33.1078873, -19.3013000, 56.0547905, -66.9324036, 52.4091873
3: -11.9829111, 42.5303879, -21.2634144, 71.9146957, -83.8975983, 63.7938004
4: -10.2974758, 39.3461494, -17.8178215, 66.7943115, -77.0917892, 57.1639709

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2229441, upper bound: 60.2062832
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2255393, upper bound: 60.2173937
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.7285433, 33.9901962, -16.1220913, 48.8338089, -59.5623512, 50.1122856
1: -15.1347942, 35.2033424, -22.4719181, 50.4915962, -65.6263885, 57.6752586
2: -13.0068378, 39.3147087, -19.2416344, 56.1883812, -69.1952209, 58.5563431
3: -14.2995472, 50.5393524, -21.2831955, 72.1276474, -86.4271927, 71.8225403
4: -12.2354679, 46.7396278, -17.8234997, 66.9236832, -79.1591339, 64.5631256

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2282595, upper bound: 60.2098436
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2265190, upper bound: 60.2203681
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.6242218, 44.3043480, -16.0552692, 47.9523468, -62.5765686, 60.3596153
1: -20.2457829, 45.8350677, -22.2755680, 49.6494026, -69.8951874, 68.1106339
2: -17.4124851, 51.0891838, -19.1205654, 55.2374725, -72.6499557, 70.2097397
3: -19.2638702, 65.6291351, -21.0856686, 70.8764191, -90.1402893, 86.7148056
4: -16.2089767, 60.8570938, -17.6558819, 65.8586121, -82.0675888, 78.5129776

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2055348, upper bound: 60.2221444
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2150209, upper bound: 60.2270873
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8565083, 47.9964638, -16.7040405, 49.7543106, -65.6108093, 64.7004929
1: -22.0979061, 49.6354141, -23.2350788, 51.5204048, -73.6183090, 72.8704910
2: -18.9476433, 55.2243576, -19.9190445, 57.2917175, -76.2393570, 75.1434021
3: -20.9038620, 70.9059448, -21.9273682, 73.4507294, -94.3545837, 92.8332977
4: -17.5394630, 65.7744293, -18.3297539, 68.3172836, -85.8567429, 84.1041870

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101746, upper bound: 60.2193335
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2195623, upper bound: 60.2237943
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.6242218, 44.3043480, -15.7521372, 47.6777954, -62.3020134, 60.0564842
1: -20.2457829, 45.8350677, -21.9354630, 49.3028488, -69.5486298, 67.7705307
2: -17.4124851, 51.0891838, -18.8160934, 54.8793182, -72.2918015, 69.9052734
3: -19.2638702, 65.6291351, -20.7598515, 70.4745102, -89.7383804, 86.3889847
4: -16.2089767, 60.8570938, -17.4349174, 65.3647919, -81.5737686, 78.2920074

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2047355, upper bound: 60.2209833
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113197, upper bound: 60.2225906
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8565083, 47.9964638, -16.3657169, 49.3871498, -65.2436523, 64.3621826
1: -22.0979061, 49.6354141, -22.8206711, 51.0731087, -73.1710129, 72.4560852
2: -18.9476433, 55.2243576, -19.5613575, 56.8234863, -75.7711258, 74.7857132
3: -20.9038620, 70.9059448, -21.5540047, 72.9262695, -93.8301163, 92.4599380
4: -17.5394630, 65.7744293, -18.0741940, 67.6964035, -85.2358704, 83.8486176

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2160873, upper bound: 60.2160044
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2160873, upper bound: 60.2269635
time: 1.12 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.65 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2304300, upper bound: 60.2244532
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2304300, upper bound: 60.2250084
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2304300, upper bound: 60.2315973
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2320861, upper bound: 60.2320861
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1970448, upper bound: 60.1928184
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1866351, upper bound: 60.1853469
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1813694, upper bound: 60.1911341
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1813694, upper bound: 60.2280849
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2283485, upper bound: 60.2354902
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2238390, upper bound: 60.2348302
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2327585, upper bound: 60.2314415
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2327585, upper bound: 60.2330276
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2117922, upper bound: 60.2273431
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2183792, upper bound: 60.2298670
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2194964, upper bound: 60.2268795
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2117922, upper bound: 60.2285198
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2182676, upper bound: 60.2237412
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2182676, upper bound: 60.2297677
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2163301, upper bound: 60.2168447
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2163301, upper bound: 60.2192972
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2286164, upper bound: 60.2188078
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2286164, upper bound: 60.2253749
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2298859, upper bound: 60.2188078
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2298859, upper bound: 60.2279945
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2141046, upper bound: 60.2141046
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2141046, upper bound: 60.2182676
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2315703, upper bound: 60.2274277
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2230519, upper bound: 60.2274260
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2094261, upper bound: 60.2094261
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2094261, upper bound: 60.2094261
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2113196, upper bound: 60.2113196
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2152466, upper bound: 60.2184828
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1952408, upper bound: 60.1926744
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1851789, upper bound: 60.1851919
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1955474, upper bound: 60.1929594
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1854854, upper bound: 60.1854771
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2197240, upper bound: 60.2177615
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2207325, upper bound: 60.2207325
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2207325, upper bound: 60.2207325
IS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1957802, upper bound: 60.2000450
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.1961200, upper bound: 60.2009641
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2026672, upper bound: 60.2117574
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2030111, upper bound: 60.2135434
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2098436, upper bound: 60.2287065
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2098436, upper bound: 60.2287065
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2207499, upper bound: 60.2269631
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2207499, upper bound: 60.2269631
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2236979, upper bound: 60.2152336
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2236979, upper bound: 60.2175400
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2298670, upper bound: 60.2183792
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2256686, upper bound: 60.2224818
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2229441, upper bound: 60.2062832
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2255393, upper bound: 60.2173937
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2282595, upper bound: 60.2098436
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2265190, upper bound: 60.2203681
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2055348, upper bound: 60.2221444
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2150209, upper bound: 60.2270873
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2101746, upper bound: 60.2193335
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2195623, upper bound: 60.2237943
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2047355, upper bound: 60.2209833
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2113197, upper bound: 60.2225906
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2160873, upper bound: 60.2160044
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 4, lower bound: -60.2160873, upper bound: 60.2269635

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.7922506, 24.2612305, -7.5241141, 24.3725681, -32.1648178, 31.7853432
1: -11.1346073, 25.3483028, -10.8009777, 25.3525791, -36.4871864, 36.1492805
2: -9.6524334, 28.2541542, -9.3038254, 28.2927227, -37.9451561, 37.5579796
3: -10.3057480, 36.1886520, -10.1236763, 36.4297333, -46.7354813, 46.3123245
4: -9.1612072, 33.6231422, -8.8907633, 33.6745453, -42.8357544, 42.5139046

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2208079, upper bound: 60.2129682
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2208079, upper bound: 60.2242960
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.5413394, 23.6235046, -8.5015144, 27.2126102, -34.7539482, 32.1250191
1: -10.7885637, 24.6731892, -12.1287508, 28.2441063, -39.0326691, 36.8019409
2: -9.3455963, 27.5080891, -10.4369717, 31.4914322, -40.8370285, 37.9450493
3: -9.9993572, 35.2639084, -11.3794317, 40.6149902, -50.6143456, 46.6433334
4: -8.8929796, 32.7483253, -9.9257269, 37.5363007, -46.4292793, 42.6740532

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2208079, upper bound: 60.2133143
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2208079, upper bound: 60.2250084
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4703608, 27.1021118, -8.0124798, 25.8991966, -34.3695564, 35.1145859
1: -12.0919371, 28.1649151, -11.4759483, 26.9079971, -38.9999352, 39.6408615
2: -10.4100361, 31.4071712, -9.8741512, 30.0176525, -40.4276886, 41.2813225
3: -11.3458710, 40.4219971, -10.7776766, 38.6751404, -50.0210037, 51.1996727
4: -9.8835945, 37.3729286, -9.4093781, 35.7256699, -45.6092644, 46.7823067

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2314700, upper bound: 60.2314700
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2314700, upper bound: 60.2314700
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.3173208, 26.6370335, -9.0249634, 28.8327293, -37.1500397, 35.6619949
1: -11.8747272, 27.6768837, -12.8521852, 29.8926868, -41.7674141, 40.5290680
2: -10.2232609, 30.8666954, -11.0500860, 33.3156586, -43.5389175, 41.9167824
3: -11.1370926, 39.7374802, -12.0788307, 42.9909897, -54.1280823, 51.8163109
4: -9.7182798, 36.7462158, -10.4818535, 39.7062187, -49.4244995, 47.2280693

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2292021, upper bound: 60.2320039
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2315818, upper bound: 60.2315818
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.5392914, 27.2805252, -9.7370825, 30.8880539, -39.4273300, 37.0176048
1: -12.1856346, 28.3512688, -13.8120041, 32.0481262, -44.2337608, 42.1632729
2: -10.4923859, 31.6131878, -11.8939190, 35.7687950, -46.2611809, 43.5071030
3: -11.4304800, 40.6814766, -13.0009775, 45.9720306, -57.4025116, 53.6824532
4: -9.9563656, 37.6183701, -11.2234163, 42.5065231, -52.4628868, 48.8417854

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1805920, upper bound: 60.2070081
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1711379, upper bound: 60.1879822
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14.5813293, 43.5178719, -7.5224142, 24.3295155, -38.9108429, 51.0402794
1: -20.0615196, 45.0858574, -10.7710686, 25.3083858, -45.3699036, 55.8569260
2: -17.2847366, 50.2593575, -9.2912197, 28.2539444, -45.5386810, 59.5505753
3: -19.1445866, 64.4588470, -10.1076460, 36.3885193, -55.5331039, 74.5664902
4: -16.0476418, 59.9180527, -8.8885365, 33.6266060, -49.6742477, 68.8065872

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2262498, upper bound: 60.2301133
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2239927, upper bound: 60.2341952
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2245937, upper bound: 60.2337309
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.5358982, 43.4080086, -8.9262829, 28.1429977, -42.6788940, 52.3342819
1: -19.9942684, 44.9696045, -12.6151676, 29.2192383, -49.2135048, 57.5847702
2: -17.2268524, 50.1415863, -10.8896236, 32.6376724, -49.8645210, 61.0312119
3: -19.0894909, 64.3118973, -11.8489332, 41.9981728, -61.0876579, 76.1608276
4: -16.0095921, 59.7814598, -10.3216600, 38.9060783, -54.9156723, 70.1031189

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2219963, upper bound: 60.2339342
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2225954, upper bound: 60.2337309
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16.2470169, 48.3629837, -7.4733706, 24.1488380, -40.3958549, 55.8363495
1: -22.5515976, 50.0848351, -10.6540947, 25.1270370, -47.6786346, 60.7389297
2: -19.3376179, 55.7057762, -9.2093201, 28.1137600, -47.4513741, 64.9151001
3: -21.3147125, 71.4049072, -9.9912634, 36.2262306, -57.5409393, 81.3961716
4: -17.8041382, 66.4190826, -8.8301640, 33.4773788, -51.2815132, 75.2492447

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308498, upper bound: 60.2290040
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2231429, upper bound: 60.2178417
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2279581, upper bound: 60.2274203
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16.2470169, 48.3629837, -8.4589252, 27.0919533, -43.3389702, 56.8219070
1: -22.5515976, 50.0848351, -12.0820923, 28.1615906, -50.7131882, 62.1669273
2: -19.3376179, 55.7057762, -10.4120913, 31.3883038, -50.7259140, 66.1178665
3: -21.3147125, 71.4049072, -11.3392544, 40.4026985, -61.7174110, 82.7441635
4: -17.8041382, 66.4190826, -9.8867702, 37.3420067, -55.1461449, 76.3058548

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308498, upper bound: 60.2290040
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2231430, upper bound: 60.2178685
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2279581, upper bound: 60.2274203
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14.5447960, 43.3847046, -8.6088009, 27.6789246, -42.2237206, 51.9935074
1: -19.9988155, 44.9458427, -12.1736212, 28.7098713, -48.7086868, 57.1194649
2: -17.2338867, 50.1046867, -10.4875174, 32.0768738, -49.3107605, 60.5921936
3: -19.0886650, 64.2493362, -11.5543184, 41.2340469, -60.3227119, 75.8036575
4: -15.9816694, 59.7357140, -9.9588966, 38.1159782, -54.0976486, 69.6945953

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2094311, upper bound: 60.2247794
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2085978, upper bound: 60.2251111
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14.5935192, 43.7092819, -10.4146404, 33.0919647, -47.6854820, 54.1239204
1: -20.0780849, 45.2701187, -14.6930895, 34.2822037, -54.3602829, 59.9632072
2: -17.2730789, 50.4662895, -12.6378269, 38.2996521, -55.5727310, 63.1041031
3: -19.2076988, 64.7530518, -13.8819456, 49.2662697, -68.4739685, 78.6349869
4: -16.0759544, 60.1665382, -11.9162598, 45.5392723, -61.6152153, 72.0827942

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1985008, upper bound: 60.2097187
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164500, upper bound: 60.2292165
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16.0043888, 47.6061897, -9.0973234, 29.0640278, -45.0684128, 56.7035103
1: -22.2146263, 49.3045273, -12.8702602, 30.1397648, -52.3543930, 62.1747894
2: -19.0487728, 54.8491364, -11.0743856, 33.6567726, -52.7055359, 65.9235229
3: -20.9922714, 70.2689362, -12.1959352, 43.2404213, -64.2326889, 82.4648743
4: -17.5180683, 65.4074173, -10.4772635, 40.0022278, -57.5202942, 75.8846741

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2150070, upper bound: 60.2234854
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2171856, upper bound: 60.2249530
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16.0248108, 47.8609314, -10.8696327, 34.4083939, -50.4332047, 58.7305641
1: -22.2263279, 49.5563278, -15.3343544, 35.6384964, -57.8648224, 64.8906631
2: -19.0416851, 55.1238861, -13.1785946, 39.7972412, -58.8389282, 68.3024826
3: -21.0705929, 70.6932220, -14.4879837, 51.1623955, -72.2329865, 85.1812057
4: -17.5772438, 65.7236938, -12.3919563, 47.3149300, -64.8921661, 78.1156387

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2150070, upper bound: 60.2262062
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201478, upper bound: 60.2268068
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -7.3165860, 23.5979481, -14.1437168, 41.9415741, -49.2581596, 37.7416573
1: -10.4644079, 24.5583134, -19.6311188, 43.5074959, -53.9719048, 44.1894264
2: -9.0218010, 27.4330101, -16.8054848, 48.4976082, -57.5194054, 44.2384949
3: -9.8191910, 35.2550735, -18.5448036, 61.7214127, -71.5406036, 53.7998619
4: -8.6318407, 32.6342468, -15.4591503, 57.8076706, -66.4395065, 48.0933990

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2178431, upper bound: 60.2233191
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2178685, upper bound: 60.2228910
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -7.4766092, 24.2408237, -18.0358315, 53.5971260, -61.0737267, 42.2766533
1: -10.7127972, 25.2177620, -25.1292000, 55.4924355, -66.2052307, 50.3469620
2: -9.2253437, 28.1516094, -21.4710350, 61.6878624, -70.9132004, 49.6226425
3: -10.0732422, 36.2259865, -23.6876984, 78.8148499, -88.8880920, 59.9136848
4: -8.8297815, 33.4859924, -19.6624622, 73.4906006, -82.3203812, 53.1484451

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2259986, upper bound: 60.2293098
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2178685, upper bound: 60.2258504
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.8151550, 27.6531944, -14.0871859, 41.7401695, -50.5553207, 41.7403755
1: -12.4440136, 28.7242146, -19.5406761, 43.2914734, -55.7354889, 48.2648888
2: -10.7344007, 32.1002464, -16.7284279, 48.2883110, -59.0227127, 48.8286743
3: -11.6980724, 41.2370491, -18.4732113, 61.4797935, -73.1778488, 59.7102585
4: -10.1648264, 38.2646065, -15.3985710, 57.5645180, -67.7293472, 53.6631699

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=65.54161834716797
rel_dist={4: [-60.236541379106946, 60.23654137910691]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2273789, upper bound: 60.2254262
time: 0.90 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2245495, upper bound: 60.2245495
time: 1.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.10
Output dim: 4, lower bound: -60.2273789, upper bound: 60.2254262
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.10
Output dim: 4, lower bound: -60.2245495, upper bound: 60.2245495

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.4466705, 33.0043793, -11.1851501, 35.0803833, -45.5270424, 44.1895180
1: -14.8643341, 34.2352486, -15.8839388, 36.3763924, -51.2407265, 50.1191864
2: -12.7760410, 38.1106606, -13.6471653, 40.4780807, -53.2541199, 51.7578239
3: -13.9698524, 49.0998573, -14.9259052, 52.0954628, -66.0653000, 64.0257416
4: -12.0348778, 45.3377380, -12.8004007, 48.1516418, -60.1865158, 58.1381378

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226727, upper bound: 60.2253981
time: 0.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2273789, upper bound: 60.2254262
time: 0.94 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.5646820, 36.3494148, -10.8048944, 34.0338707, -45.5985527, 47.1543045
1: -16.3725929, 37.6687965, -15.3755426, 35.3007126, -51.6733055, 53.0443344
2: -14.0860319, 41.9790382, -13.2212753, 39.2853279, -53.3713493, 55.2003059
3: -15.4223356, 53.9925766, -14.4571342, 50.5656357, -65.9879684, 68.4497070
4: -13.2208614, 49.8709221, -12.4222422, 46.7201920, -59.9410515, 62.2931633

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2245022
time: 0.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2245495, upper bound: 60.2245495
time: 0.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.68 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 4, lower bound: -60.2226727, upper bound: 60.2253981
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 4, lower bound: -60.2273789, upper bound: 60.2254262
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2245022
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.68
Output dim: 4, lower bound: -60.2245495, upper bound: 60.2245495

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.4672174, 30.0572453, -9.3751278, 29.6644630, -39.1316795, 39.4323730
1: -13.4863625, 31.2039490, -13.3429766, 30.8149776, -44.3013382, 44.5469208
2: -11.6011353, 34.7654724, -11.4833860, 34.3370895, -45.9382172, 46.2488556
3: -12.6612263, 44.7709389, -12.5194311, 44.1346893, -56.7959137, 57.2903709
4: -10.9614296, 41.3684845, -10.8408489, 40.8578949, -51.8193169, 52.2093353

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2205423, upper bound: 60.2227529
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2189073, upper bound: 60.2232311
time: 0.82 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.1813774, 32.1828041, -17.5856438, 52.2330513, -62.4144287, 49.7684402
1: -14.4867420, 33.3902130, -24.4952412, 54.0863495, -68.5730896, 57.8854523
2: -12.4543438, 37.1749115, -20.9794998, 60.1464691, -72.6008148, 58.1544113
3: -13.6159983, 47.8808250, -23.0726814, 76.9944992, -90.6104965, 70.9534988
4: -11.7448587, 44.2198296, -19.2327614, 71.7109528, -83.4558105, 63.4525909

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2273789, upper bound: 60.2236761
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2273789, upper bound: 60.2254042
time: 1.10 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -10.6923285, 33.7469902, -8.9543991, 28.5164833, -39.2088089, 42.7013893
1: -15.1421385, 34.9863586, -12.7810354, 29.6316032, -44.7737389, 47.7673950
2: -13.0293388, 39.0228348, -11.0082102, 33.0257874, -46.0551262, 50.0310440
3: -14.2637110, 50.1733589, -11.9989080, 42.4614296, -56.7251396, 62.1722679
4: -12.2573977, 46.3668175, -10.4168720, 39.2818222, -51.5392189, 56.7836914

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2245022
time: 0.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -10.9670753, 34.5360794, -16.3314896, 48.9222183, -59.8892860, 50.8675690
1: -15.5506439, 35.7936859, -22.7674160, 50.6392555, -66.1898956, 58.5611000
2: -13.3780289, 39.8806458, -19.5142288, 56.3077431, -69.6857758, 59.3948746
3: -14.6416149, 51.3028374, -21.4846821, 72.2072601, -86.8488770, 72.7875214
4: -12.5790634, 47.3896980, -17.9640274, 67.0741119, -79.6531754, 65.3536987

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2245022, upper bound: 60.2203013
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2245022, upper bound: 60.2245495
time: 0.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 5.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 4, lower bound: -60.2205423, upper bound: 60.2227529
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 4, lower bound: -60.2189073, upper bound: 60.2232311
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 4, lower bound: -60.2273789, upper bound: 60.2236761
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 4, lower bound: -60.2273789, upper bound: 60.2254042
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2245022
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 4, lower bound: -60.2245022, upper bound: 60.2203013
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.88
Output dim: 4, lower bound: -60.2245022, upper bound: 60.2245495

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.8553267, 28.3344612, -9.2725420, 29.3746319, -38.2299576, 37.6070023
1: -12.6549244, 29.4409370, -13.2030754, 30.5181942, -43.1731186, 42.6440125
2: -10.8979311, 32.8131714, -11.3649273, 34.0088692, -44.9067993, 44.1780968
3: -11.8628435, 42.2517471, -12.3855963, 43.7116737, -55.5745125, 54.6373444
4: -10.3311682, 39.0331726, -10.7345371, 40.4652710, -50.7964363, 49.7677078

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2097989, upper bound: 60.2119509
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2199284
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2223757
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.3039818, 29.7487259, -9.1275072, 29.0132275, -38.3171997, 38.8762321
1: -13.3057241, 30.8814507, -13.0203342, 30.1409416, -43.4466667, 43.9017792
2: -11.4372253, 34.4042587, -11.2031784, 33.5888557, -45.0260811, 45.6074371
3: -12.4918308, 44.3404121, -12.2122974, 43.1949348, -55.6867638, 56.5527115
4: -10.8136549, 40.9327431, -10.5908775, 39.9700813, -50.7837372, 51.5236206

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070821, upper bound: 60.2127822
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2201840
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184835, upper bound: 60.2228032
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.4384260, 30.0077515, -17.0898552, 50.7494736, -60.1878967, 47.0976067
1: -13.4604979, 31.1456528, -23.8084278, 52.5574837, -66.0179825, 54.9540787
2: -11.5769281, 34.6995964, -20.3948097, 58.4777222, -70.0546494, 55.0944061
3: -12.6499252, 44.7010345, -22.4261436, 74.8108521, -87.4607773, 67.1271820
4: -10.9593449, 41.2792587, -18.6982594, 69.7159805, -80.6753235, 59.9775162

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2252786, upper bound: 60.2170656
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2269370, upper bound: 60.2234341
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.5441589, 32.9790421, -17.1302147, 50.9075623, -61.4517059, 50.1092567
1: -14.8934422, 34.1892319, -23.8353310, 52.7183533, -67.6117935, 58.0245628
2: -12.8240395, 38.1117668, -20.4207039, 58.6337509, -71.4577866, 58.5324707
3: -14.0226860, 49.0766220, -22.4871826, 75.0909882, -89.1136780, 71.5638046
4: -12.0665970, 45.4033699, -18.7584496, 69.9251938, -81.9917908, 64.1618195

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2121888, upper bound: 60.2167097
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2217161, upper bound: 60.2192867
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -8.9543991, 28.5164833, -38.4676743, 40.5025940
1: -14.1082268, 32.7253380, -12.7810354, 29.6316032, -43.7398262, 45.5063705
2: -12.1443062, 36.5262375, -11.0082102, 33.0257874, -45.1700897, 47.5344429
3: -13.2896729, 46.9455948, -11.9989080, 42.4614296, -55.7511024, 58.9445038
4: -11.4511833, 43.4067154, -10.4168720, 39.2818222, -50.7330055, 53.8235855

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170395, upper bound: 60.2192109
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2199904, upper bound: 60.2199904
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.9047461, 48.1323166, -8.9543991, 28.5164833, -44.4212189, 57.0867157
1: -22.1632099, 49.7601585, -12.7810354, 29.6316032, -51.7948074, 62.5411949
2: -19.0034084, 55.3839111, -11.0082102, 33.0257874, -52.0291977, 66.3921204
3: -20.9170437, 71.1037216, -11.9989080, 42.4614296, -63.3784714, 83.1026306
4: -17.5928783, 65.9601059, -10.4168720, 39.2818222, -56.8747025, 76.3769760

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170395, upper bound: 60.2212044
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170395, upper bound: 60.2240785
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.7923231, 31.0757713, -16.3314896, 48.9222183, -58.7145424, 47.4072609
1: -13.8838530, 32.2330780, -22.7674160, 50.6392555, -64.5231094, 55.0004959
2: -11.9508610, 35.9805870, -19.5142288, 56.3077431, -68.2585907, 55.4948158
3: -13.0776491, 46.2406044, -21.4846821, 72.2072601, -85.2849121, 67.7252731
4: -11.2742805, 42.7561111, -17.9640274, 67.0741119, -78.3483887, 60.7201271

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.3314896, 48.9222183, -65.0415115, 65.0706406
1: -22.4673023, 50.3988075, -22.7674160, 50.6392555, -73.1065598, 73.1662216
2: -19.2585125, 56.0775299, -19.5142288, 56.3077431, -75.5662537, 75.5917511
3: -21.2393341, 71.9838791, -21.4846821, 72.2072601, -93.4465942, 93.4685516
4: -17.8179779, 66.7789383, -17.9640274, 67.0741119, -84.8920898, 84.7429428

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
time: 0.91 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.32 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2199284
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2223757
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2201840
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2184835, upper bound: 60.2228032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2252786, upper bound: 60.2170656
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2269370, upper bound: 60.2234341
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2121888, upper bound: 60.2167097
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2217161, upper bound: 60.2192867
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2170395, upper bound: 60.2192109
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2199904, upper bound: 60.2199904
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2170395, upper bound: 60.2212044
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2170395, upper bound: 60.2240785
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 4, lower bound: -60.2203013, upper bound: 60.2203013

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.3403873, 26.7963810, -8.3552284, 26.6222553, -34.9626427, 35.1516075
1: -11.9133816, 27.8504829, -11.8828621, 27.6683140, -39.5816956, 39.7333450
2: -10.2646046, 31.0467148, -10.2341051, 30.8485241, -41.1131287, 41.2808189
3: -11.1893721, 39.9535065, -11.1927242, 39.6089287, -50.7982979, 51.1462288
4: -9.7481613, 36.9260712, -9.6982174, 36.7104759, -46.4586372, 46.6242905

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2196062
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2199284
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.6005344, 27.6624794, -10.6711636, 33.4253693, -42.0258942, 38.3336411
1: -12.2941151, 28.7413502, -15.0062704, 34.6942635, -46.9883804, 43.7476158
2: -10.5831261, 32.0448952, -12.8873243, 38.7181396, -49.3012657, 44.9322166
3: -11.5484400, 41.2719803, -14.2154579, 49.6341324, -61.1825676, 55.4874306
4: -10.0494480, 38.1128883, -12.1212740, 46.0326462, -56.0820923, 50.2341614

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2216146
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2223757
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.7402821, 28.0860443, -8.2104988, 26.2702427, -35.0105247, 36.2965431
1: -12.4956799, 29.1575184, -11.6995087, 27.3008995, -39.7965775, 40.8570251
2: -10.7445393, 32.4977570, -10.0724916, 30.4377346, -41.1822739, 42.5702477
3: -11.7515240, 41.8715401, -11.0186768, 39.1068039, -50.8583145, 52.8902168
4: -10.1804733, 38.6666451, -9.5553703, 36.2265053, -46.4069786, 48.2220154

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2185937
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2201840
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0397148, 29.0722790, -10.4873877, 32.9710312, -42.0107460, 39.5596657
1: -12.9303589, 30.1771049, -14.7738466, 34.2179680, -47.1483192, 44.9509468
2: -11.1082249, 33.6315842, -12.6821384, 38.1857643, -49.2939911, 46.3137207
3: -12.1710644, 43.3540306, -13.9929790, 48.9806442, -61.1517067, 57.3470039
4: -10.5214596, 40.0017281, -11.9377890, 45.4020348, -55.9234924, 51.9395180

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184835, upper bound: 60.2192418
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2228032
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.7060633, 27.9133015, -15.3177395, 45.6299667, -54.3360291, 43.2310410
1: -12.4392548, 28.9800453, -21.1273327, 47.2803078, -59.7195625, 50.1073761
2: -10.7107487, 32.3195801, -18.1896858, 52.6629181, -63.3736649, 50.5092659
3: -11.6988106, 41.6772842, -20.1215878, 67.5282669, -79.2270660, 61.7988739
4: -10.1970959, 38.4544525, -16.8215942, 62.8061714, -73.0032654, 55.2760429

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2231749, upper bound: 60.2151006
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132344, upper bound: 60.2072658
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226150, upper bound: 60.2149156
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.3303947, 29.7141743, -16.8560028, 50.0768661, -59.4072571, 46.5701752
1: -13.3141108, 30.8420811, -23.4750061, 51.8661003, -65.1802139, 54.3170853
2: -11.4529991, 34.3577080, -20.1144142, 57.7208557, -69.1738586, 54.4721184
3: -12.5143251, 44.2690964, -22.1278419, 73.8354492, -86.3497772, 66.3969269
4: -10.8484077, 40.8695259, -18.4461040, 68.8094482, -79.6578522, 59.3156281

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2247394, upper bound: 60.2211052
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2241677, upper bound: 60.2193950
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2249917, upper bound: 60.2218841
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.8126383, 27.7799606, -14.6861305, 43.5665016, -52.3791389, 42.4660912
1: -12.4205990, 28.8315525, -20.3941936, 45.1876144, -57.6082153, 49.2257423
2: -10.6920710, 32.2155266, -17.4489765, 50.3502579, -61.0423279, 49.6645050
3: -11.7356586, 41.4209099, -19.2648067, 64.0428085, -75.7784653, 60.6857147
4: -10.1354246, 38.3813400, -16.0234184, 59.9918594, -70.1272812, 54.4047585

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2063268, upper bound: 60.2162755
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2145106, upper bound: 60.2158786
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7721310, 30.8263531, -18.5309887, 55.0393753, -64.8115082, 49.3573418
1: -13.8299322, 31.9765606, -25.8151951, 56.9831009, -70.8130341, 57.7917557
2: -11.9066277, 35.6542740, -22.0498466, 63.3534813, -75.2601089, 57.7041168
3: -13.0353785, 45.9234085, -24.3465748, 80.8448792, -93.8802567, 70.2699661
4: -11.2620125, 42.4630890, -20.1659927, 75.4405975, -86.7025986, 62.6290817

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201164, upper bound: 60.2186468
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207866, upper bound: 60.2166818
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.4351130, 30.0855389, -7.9733305, 25.6580486, -35.0931549, 38.0588646
1: -13.3620987, 31.2028694, -11.3745708, 26.6658440, -40.0279427, 42.5774345
2: -11.4977827, 34.8392105, -9.7988224, 29.7364845, -41.2342644, 44.6380310
3: -12.6348324, 44.7649155, -10.7337732, 38.2041931, -50.8390274, 55.4986877
4: -10.8657188, 41.3981590, -9.3111467, 35.3659973, -46.2317123, 50.7093048

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140216, upper bound: 60.2163929
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140216, upper bound: 60.2172189
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6504440, 30.7189255, -10.1381731, 31.9982643, -41.6487045, 40.8570938
1: -13.6913834, 31.8643436, -14.3234596, 33.2083397, -46.8997231, 46.1878014
2: -11.7887878, 35.5764542, -12.3167572, 37.0581665, -48.8469543, 47.8932076
3: -12.9010925, 45.7366676, -13.5239420, 47.5643044, -60.4653969, 59.2606010
4: -11.1328373, 42.2776527, -11.6040926, 44.0581818, -55.1910172, 53.8817444

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140216, upper bound: 60.2170052
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2169629, upper bound: 60.2178566
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15.4538975, 46.7159843, -7.9733305, 25.6580486, -41.1119423, 54.6893158
1: -21.5198288, 48.2955971, -11.3745708, 26.6658440, -48.1856728, 59.6701660
2: -18.4586639, 53.7574120, -9.7988224, 29.7364845, -48.1951485, 63.5562286
3: -20.3074474, 68.9856110, -10.7337732, 38.2041931, -58.5116425, 79.7193832
4: -17.0559998, 64.0632553, -9.3111467, 35.3659973, -52.4219971, 73.3744049

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1802918, upper bound: 60.1844749
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2139506, upper bound: 60.2179500
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148909, upper bound: 60.2192316
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.5392857, 47.3291245, -10.1381731, 31.9982643, -47.5375443, 57.4672928
1: -21.6477318, 48.9168091, -14.3234596, 33.2083397, -54.8560638, 63.2402611
2: -18.5563126, 54.4601021, -12.3167572, 37.0581665, -55.6144791, 66.7768555
3: -20.5204449, 69.9517746, -13.5239420, 47.5643044, -68.0847473, 83.4757156
4: -17.2213554, 64.8388596, -11.6040926, 44.0581818, -61.2795296, 76.4429550

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2169122, upper bound: 60.2201610
time: 1.10 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2139506, upper bound: 60.2219264
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.7923231, 31.0757713, -16.5652657, 49.3458900, -59.1382141, 47.6410370
1: -13.8838530, 32.2330780, -23.0251637, 51.0942726, -64.9781265, 55.2582397
2: -11.9508610, 35.9805870, -19.7440033, 56.8110428, -68.7619019, 55.7245903
3: -13.0776491, 46.2406044, -21.7395668, 72.8528137, -85.9304581, 67.9801712
4: -11.2742805, 42.7561111, -18.1693535, 67.7313232, -79.0056000, 60.9254646

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212044, upper bound: 60.2170395
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212044, upper bound: 60.2199904
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.7923231, 31.0757713, -16.2631149, 49.0958710, -58.8881950, 47.3388863
1: -13.8838530, 32.2330780, -22.6706581, 50.7684135, -64.6522675, 54.9037361
2: -11.9508610, 35.9805870, -19.4368763, 56.4822769, -68.4331207, 55.4174652
3: -13.0776491, 46.2406044, -21.4176693, 72.4964828, -85.5741272, 67.6582718
4: -11.2742805, 42.7561111, -17.9588089, 67.2755051, -78.5497894, 60.7149124

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2192109, upper bound: 60.2170395
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240785, upper bound: 60.2199904
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.5652657, 49.3458900, -65.4651871, 65.3044128
1: -22.4673023, 50.3988075, -23.0251637, 51.0942726, -73.5615768, 73.4239731
2: -19.2585125, 56.0775299, -19.7440033, 56.8110428, -76.0695572, 75.8215332
3: -21.2393341, 71.9838791, -21.7395668, 72.8528137, -94.0921478, 93.7234421
4: -17.8179779, 66.7789383, -18.1693535, 67.7313232, -85.5493011, 84.9482880

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153448, upper bound: 60.2239346
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202912, upper bound: 60.2243702
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.2631149, 49.0958710, -65.2151566, 65.0022659
1: -22.4673023, 50.3988075, -22.6706581, 50.7684135, -73.2357101, 73.0694656
2: -19.2585125, 56.0775299, -19.4368763, 56.4822769, -75.7407837, 75.5144043
3: -21.2393341, 71.9838791, -21.4176693, 72.4964828, -93.7358170, 93.4015503
4: -17.8179779, 66.7789383, -17.9588089, 67.2755051, -85.0934830, 84.7377319

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153448, upper bound: 60.2239346
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202912, upper bound: 60.2243702
time: 1.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.58 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2196062
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2199284
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2216146
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2223757
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2185937
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2201840
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2184835, upper bound: 60.2192418
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2228032
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2132344, upper bound: 60.2072658
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2226150, upper bound: 60.2149156
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2241677, upper bound: 60.2193950
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2249917, upper bound: 60.2218841
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2063268, upper bound: 60.2162755
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2145106, upper bound: 60.2158786
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2201164, upper bound: 60.2186468
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2207866, upper bound: 60.2166818
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2140216, upper bound: 60.2163929
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2140216, upper bound: 60.2172189
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2140216, upper bound: 60.2170052
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2169629, upper bound: 60.2178566
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2139506, upper bound: 60.2179500
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2148909, upper bound: 60.2192316
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2169122, upper bound: 60.2201610
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2139506, upper bound: 60.2219264
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2212044, upper bound: 60.2170395
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2212044, upper bound: 60.2199904
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2192109, upper bound: 60.2170395
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2240785, upper bound: 60.2199904
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2153448, upper bound: 60.2239346
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2202912, upper bound: 60.2243702
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2153448, upper bound: 60.2239346
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.58
Output dim: 4, lower bound: -60.2202912, upper bound: 60.2243702

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -7.6530442, 24.6899452, -8.3552284, 26.6222553, -34.2752991, 33.0451660
1: -10.9422283, 25.6896152, -11.8828621, 27.6683140, -38.6105423, 37.5724640
2: -9.4419165, 28.6616936, -10.2341051, 30.8485241, -40.2904396, 38.8957939
3: -10.2711849, 36.8425674, -11.1927242, 39.6089287, -49.8801117, 48.0352859
4: -8.9991207, 34.0922890, -9.6982174, 36.7104759, -45.7095947, 43.7905045

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156533, upper bound: 60.2191938
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156533, upper bound: 60.2196062
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15.4591036, 45.9499245, -8.3552284, 26.6222553, -42.0813599, 54.3051529
1: -21.4735241, 47.5979691, -11.8828621, 27.6683140, -49.1418381, 59.4808311
2: -18.4118843, 52.9971733, -10.2341051, 30.8485241, -49.2604065, 63.2312775
3: -20.2607193, 67.7989731, -11.1927242, 39.6089287, -59.8696480, 78.9916992
4: -16.9254570, 63.1915512, -9.6982174, 36.7104759, -53.6359329, 72.8897705

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156533, upper bound: 60.2195140
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156533, upper bound: 60.2199284
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.8442678, 25.3760033, -10.6711636, 33.4253693, -41.2696342, 36.0471611
1: -11.2341404, 26.3970814, -15.0062704, 34.6942635, -45.9284058, 41.4033432
2: -9.6845427, 29.4569588, -12.8873243, 38.7181396, -48.4026833, 42.3442764
3: -10.5401440, 37.9058571, -14.2154579, 49.6341324, -60.1742706, 52.1213150
4: -9.2322922, 35.0379143, -12.1212740, 46.0326462, -55.2649384, 47.1591873

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2072493, upper bound: 60.2085118
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2195127, upper bound: 60.2216146
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2216146
time: 1.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15.5407362, 46.5346947, -10.6711636, 33.4253693, -48.9660950, 57.2058563
1: -21.5614262, 48.1837196, -15.0062704, 34.6942635, -56.2556915, 63.1899872
2: -18.4577885, 53.6195564, -12.8873243, 38.7181396, -57.1759262, 66.5068817
3: -20.4590416, 68.7463455, -14.2154579, 49.6341324, -70.0931702, 82.9617920
4: -17.0672722, 63.9294014, -12.1212740, 46.0326462, -63.0999184, 76.0506516

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2072493, upper bound: 60.2100325
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2223757
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161118, upper bound: 60.2223757
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.0023518, 25.8558311, -8.2104988, 26.2702427, -34.2725945, 34.0663223
1: -11.4526672, 26.8695335, -11.6995087, 27.3008995, -38.7535667, 38.5690422
2: -9.8580408, 29.9781017, -10.0724916, 30.4377346, -40.2957764, 40.0505905
3: -10.7690592, 38.5922012, -11.0186768, 39.1068039, -49.8758621, 49.6108780
4: -9.3811684, 35.6741600, -9.5553703, 36.2265053, -45.6076736, 45.2295303

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2150581, upper bound: 60.2181734
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2150581, upper bound: 60.2181734
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.6392212, 46.7010193, -8.2104988, 26.2702427, -41.9094620, 54.9115181
1: -21.7686615, 48.3614349, -11.6995087, 27.3008995, -49.0695610, 60.0609436
2: -18.6593056, 53.8688660, -10.0724916, 30.4377346, -49.0970383, 63.9413567
3: -20.5670757, 68.9144058, -11.0186768, 39.1068039, -59.6738815, 79.9330826
4: -17.1469116, 64.1903992, -9.5553703, 36.2265053, -53.3734131, 73.7457733

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2150581, upper bound: 60.2197637
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2150581, upper bound: 60.2201840
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.2413101, 26.6869240, -10.4873877, 32.9710312, -41.2123337, 37.1743126
1: -11.8082571, 27.7280483, -14.7738466, 34.2179680, -46.0262222, 42.5018883
2: -10.1533432, 30.9326096, -12.6821384, 38.1857643, -48.3391037, 43.6147461
3: -11.1096554, 39.8496780, -13.9929790, 48.9806442, -60.0903015, 53.8426590
4: -9.6594791, 36.7958527, -11.9377890, 45.4020348, -55.0615120, 48.7336426

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2192418
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155168, upper bound: 60.2192418
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.7288265, 47.2928848, -10.4873877, 32.9710312, -48.6998558, 57.7802734
1: -21.8663445, 48.9527435, -14.7738466, 34.2179680, -56.0843124, 63.7265892
2: -18.7086964, 54.4650650, -12.6821384, 38.1857643, -56.8944626, 67.1472015
3: -20.7734985, 69.8788071, -13.9929790, 48.9806442, -69.7541428, 83.8717651
4: -17.2897472, 64.9446793, -11.9377890, 45.4020348, -62.6917801, 76.8824692

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184835, upper bound: 60.2228032
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184835, upper bound: 60.2228032
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.3698692, 27.0078545, -15.1308794, 45.0865440, -53.4564056, 42.1387329
1: -11.9698353, 28.0420742, -20.8624992, 46.7187881, -58.6886215, 48.9045677
2: -10.3029938, 31.2929497, -17.9645348, 52.0492668, -62.3522491, 49.2574768
3: -11.2705765, 40.3711052, -19.8776302, 66.7379303, -78.0085068, 60.2487335
4: -9.8370991, 37.2332726, -16.6212769, 62.0784073, -71.9154892, 53.8545494

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132344, upper bound: 60.2072658
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132344, upper bound: 60.2072658
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.0915365, 26.1906891, -14.8877907, 44.5245209, -52.6160583, 41.0784721
1: -11.6215878, 27.2057152, -20.5433140, 46.1357269, -57.7573128, 47.7490273
2: -10.0078983, 30.3526764, -17.6857414, 51.3988380, -61.4067383, 48.0384140
3: -10.9162416, 39.1739960, -19.5880184, 65.9432220, -76.8594666, 58.7620125
4: -9.5652514, 36.1064339, -16.3837891, 61.2732697, -70.8385086, 52.4902229

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226150, upper bound: 60.2149156
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226150, upper bound: 60.2149156
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0210562, 28.8622589, -16.6667194, 49.5223160, -58.5433693, 45.5289764
1: -12.8821249, 29.9620323, -23.2084179, 51.2926865, -64.1748123, 53.1704483
2: -11.0815573, 33.3941879, -19.8873672, 57.0922432, -68.1737976, 53.2815475
3: -12.1167984, 43.0424728, -21.8786354, 73.0293503, -85.1461411, 64.9211121
4: -10.5209942, 39.7285576, -18.2426987, 68.0637283, -78.5847168, 57.9712563

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2041169, upper bound: 60.2041955
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2174041, upper bound: 60.2127428
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.6645908, 27.9282932, -16.3563499, 48.7765198, -57.4411087, 44.2846451
1: -12.4215164, 29.0002270, -22.7756920, 50.5212173, -62.9427338, 51.7759171
2: -10.6869392, 32.3213997, -19.5125732, 56.2288170, -66.9157486, 51.8339691
3: -11.6869316, 41.6874847, -21.5071564, 71.9853439, -83.6722717, 63.1946411
4: -10.1670370, 38.4365158, -17.9369774, 67.0331116, -77.2001343, 56.3734818

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2249917, upper bound: 60.2218841
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2249917, upper bound: 60.2218841
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -7.7120543, 24.5171127, -13.4091892, 39.7430420, -47.4550972, 37.9263000
1: -10.8052225, 25.4774952, -18.4996147, 41.2255211, -52.0307388, 43.9771118
2: -9.3254871, 28.5518818, -15.8556576, 46.0490837, -55.3745728, 44.4075394
3: -10.2515450, 36.7178612, -17.6032658, 58.6350670, -68.8866119, 54.3211250
4: -8.9437838, 34.0188751, -14.6510849, 54.9046707, -63.8484535, 48.6699562

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2063268, upper bound: 60.2162755
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2063268, upper bound: 60.2162755
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.5830555, 27.1699581, -14.5987120, 43.3285789, -51.9116325, 41.7686691
1: -12.1073017, 28.1990490, -20.2702656, 44.9443283, -57.0516281, 48.4693146
2: -10.4251814, 31.5071621, -17.3452148, 50.0781670, -60.5033493, 48.8523788
3: -11.4443026, 40.5224800, -19.1552544, 63.6989365, -75.1432419, 59.6777344
4: -9.8976698, 37.5275383, -15.9323158, 59.6667023, -69.5643692, 53.4598503

Time for backsubstitution: 2.23 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=65.54161834716797
rel_dist={4: [-60.234725823217936, 60.23472582321793]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1121.78 seconds
