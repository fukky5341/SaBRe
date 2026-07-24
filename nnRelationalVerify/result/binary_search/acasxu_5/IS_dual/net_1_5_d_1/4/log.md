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
execution time: IAR + LP analysis = 2.12 + 2.17 = 4.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -60.2373092, upper bound: 60.2373092


# Binary Search by BASE starts (time budget: 1195.72 seconds, max iter: 100)

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
Binary search time: 77.09 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1118.62 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2343734
time: 0.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 0.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.93 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.93
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2343734
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.93
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.4466705, 33.0043793, -12.1294031, 37.7388992, -48.1855583, 45.1337776
1: -14.8643341, 34.2352486, -17.1822987, 39.1265984, -53.9909325, 51.4175491
2: -12.7760410, 38.1106606, -14.7555904, 43.5125732, -56.2886124, 52.8662491
3: -13.9698524, 49.0998573, -16.1523533, 55.9294815, -69.8993073, 65.2521973
4: -12.0348778, 45.3377380, -13.7831745, 51.7584686, -63.7933464, 59.1209106

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 0.90 seconds

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

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

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
- Time for IS candidates: 3.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.61
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
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2325729
time: 1.02 seconds

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

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2343734
time: 0.93 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
time: 0.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.5646820, 36.3494148, -10.4466705, 33.0043793, -44.5690536, 46.7960739
1: -16.3725929, 37.6687965, -14.8643341, 34.2352486, -50.6078415, 52.5331268
2: -14.0860319, 41.9790382, -12.7760410, 38.1106606, -52.1966858, 54.7550774
3: -15.4223356, 53.9925766, -13.9698524, 49.0998573, -64.5221863, 67.9624176
4: -13.2208614, 49.8709221, -12.0348778, 45.3377380, -58.5585976, 61.9057999

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

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
time: 1.20 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.5646820, 36.3494148, -11.5646820, 36.3494148, -47.9140968, 47.9140968
1: -16.3725929, 37.6687965, -16.3725929, 37.6687965, -54.0413857, 54.0413895
2: -14.0860319, 41.9790382, -14.0860319, 41.9790382, -56.0650673, 56.0650635
3: -15.4223356, 53.9925766, -15.4223356, 53.9925766, -69.4149094, 69.4149094
4: -13.2208614, 49.8709221, -13.2208614, 49.8709221, -63.0917816, 63.0917816

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2213024
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 1.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.06 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 4, lower bound: -60.2337385, upper bound: 60.2325729
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2343734
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2213024
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2213024
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.06
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -10.4466705, 33.0043793, -41.7035484, 38.1833496
1: -12.4092007, 28.8249302, -14.8643341, 34.2352486, -46.6444473, 43.6892624
2: -10.6860304, 32.1388817, -12.7760410, 38.1106606, -48.7966919, 44.9149246
3: -11.6377096, 41.3525887, -13.9698524, 49.0998573, -60.7375641, 55.3224411
4: -10.1306763, 38.2451363, -12.0348778, 45.3377380, -55.4684143, 50.2800064

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2353505
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2353505
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -10.3593292, 32.7462234, -49.3114891, 59.7052193
1: -23.0251637, 51.0942726, -14.7422695, 33.9663849, -56.9915428, 65.8365402
2: -19.7440033, 56.8110428, -12.6706734, 37.8118439, -57.5558472, 69.4817200
3: -21.7395668, 72.8528137, -13.8555994, 48.7171173, -70.4566803, 86.7084122
4: -18.1693535, 67.7313232, -11.9381847, 44.9827614, -63.1521149, 79.6695099

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2371510
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2371510
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -10.4466705, 33.0043793, -9.9511909, 31.5481949, -41.9948578, 42.9555626
1: -14.8643341, 34.2352486, -14.1082268, 32.7253380, -47.5896721, 48.3434715
2: -12.7760410, 38.1106606, -12.1443062, 36.5262375, -49.3022766, 50.2549629
3: -13.9698524, 49.0998573, -13.2896729, 46.9455948, -60.9154434, 62.3895302
4: -12.0348778, 45.3377380, -11.4511833, 43.4067154, -55.4415894, 56.7889137

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2325729
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2343734
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -10.3593292, 32.7462234, -16.1192989, 48.7391510, -59.0984802, 48.8655205
1: -14.7422695, 33.9663849, -22.4673023, 50.3988075, -65.1410751, 56.4336853
2: -12.6706734, 37.8118439, -19.2585125, 56.0775299, -68.7481842, 57.0703545
3: -13.8555994, 48.7171173, -21.2393341, 71.9838791, -85.8394775, 69.9564438
4: -11.9381847, 44.9827614, -17.8179779, 66.7789383, -78.7171173, 62.8007393

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2325729
time: 0.90 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -10.4466705, 33.0043793, -42.9555626, 41.9948578
1: -14.1082268, 32.7253380, -14.8643341, 34.2352486, -48.3434715, 47.5896721
2: -12.1443062, 36.5262375, -12.7760410, 38.1106606, -50.2549667, 49.3022766
3: -13.2896729, 46.9455948, -13.9698524, 49.0998573, -62.3895302, 60.9154472
4: -11.4511833, 43.4067154, -12.0348778, 45.3377380, -56.7889137, 55.4415894

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

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
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -10.3593292, 32.7462234, -48.8655205, 59.0984802
1: -22.4673023, 50.3988075, -14.7422695, 33.9663849, -56.4336853, 65.1410751
2: -19.2585125, 56.0775299, -12.6706734, 37.8118439, -57.0703545, 68.7481842
3: -21.2393341, 71.9838791, -13.8555994, 48.7171173, -69.9564514, 85.8394775
4: -17.8179779, 66.7789383, -11.9381847, 44.9827614, -62.8007393, 78.7171173

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2335933
time: 0.93 seconds

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

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

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

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2308157
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2308157
time: 0.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.95 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2353505
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2353505
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2371510
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2353505, upper bound: 60.2371510
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2325729
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2343734
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2325729
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2335933, upper bound: 60.2343734
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2240800
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2240800
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2335933
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2325729, upper bound: 60.2335933
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2213024
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2308157
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -60.2213024, upper bound: 60.2308157

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -8.6991749, 27.7366791, -36.4358521, 36.4358521
1: -12.4092007, 28.8249302, -12.4092007, 28.8249302, -41.2341309, 41.2341309
2: -10.6860304, 32.1388817, -10.6860304, 32.1388817, -42.8249130, 42.8249130
3: -11.6377096, 41.3525887, -11.6377096, 41.3525887, -52.9902954, 52.9902954
4: -10.1306763, 38.2451363, -10.1306763, 38.2451363, -48.3758125, 48.3758125

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290779, upper bound: 60.2196514
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290779, upper bound: 60.2319546
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -16.5652657, 49.3458900, -58.0450668, 44.3019447
1: -12.4092007, 28.8249302, -23.0251637, 51.0942726, -63.5034714, 51.8500900
2: -10.6860304, 32.1388817, -19.7440033, 56.8110428, -67.4970703, 51.8828850
3: -11.6377096, 41.3525887, -21.7395668, 72.8528137, -84.4905243, 63.0921516
4: -10.1306763, 38.2451363, -18.1693535, 67.7313232, -77.8619995, 56.4144897

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290779, upper bound: 60.2196514
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319546, upper bound: 60.2319546
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -8.6602230, 27.6217041, -44.1869659, 58.0061111
1: -23.0251637, 51.0942726, -12.3543396, 28.7050247, -51.7301865, 63.4486122
2: -19.7440033, 56.8110428, -10.6386271, 32.0058823, -51.7498856, 67.4496689
3: -21.7395668, 72.8528137, -11.5859261, 41.1813660, -62.9209328, 84.4387360
4: -18.1693535, 67.7313232, -10.0872440, 38.0864449, -56.2557983, 77.8185654

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2312285
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319546, upper bound: 60.2355284
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -16.4778557, 49.0316277, -65.5968933, 65.8237457
1: -23.0251637, 51.0942726, -22.8805523, 50.7702751, -73.7954407, 73.9748230
2: -19.7440033, 56.8110428, -19.6148682, 56.4666824, -76.2106857, 76.4259109
3: -21.7395668, 72.8528137, -21.6091194, 72.3687057, -94.1082687, 94.4619217
4: -18.1693535, 67.7313232, -18.0539436, 67.3143234, -85.4836731, 85.7852631

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2289078, upper bound: 60.2360855
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2288698, upper bound: 60.2364990
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -9.9511909, 31.5481949, -40.2473679, 37.6878700
1: -12.4092007, 28.8249302, -14.1082268, 32.7253380, -45.1345367, 42.9331551
2: -10.6860304, 32.1388817, -12.1443062, 36.5262375, -47.2122688, 44.2831879
3: -11.6377096, 41.3525887, -13.2896729, 46.9455948, -58.5833054, 54.6422615
4: -10.1306763, 38.2451363, -11.4511833, 43.4067154, -53.5373917, 49.6963081

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2157223, upper bound: 60.2248978
time: 1.21 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2214995, upper bound: 60.2323247
time: 0.98 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2236058, upper bound: 60.2316234
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -9.9511909, 31.5481949, -48.1134567, 59.2970810
1: -23.0251637, 51.0942726, -14.1082268, 32.7253380, -55.7504959, 65.2024994
2: -19.7440033, 56.8110428, -12.1443062, 36.5262375, -56.2702408, 68.9553528
3: -21.7395668, 72.8528137, -13.2896729, 46.9455948, -68.6851654, 86.1424866
4: -18.1693535, 67.7313232, -11.4511833, 43.4067154, -61.5760689, 79.1825104

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2157223, upper bound: 60.2284716
time: 1.12 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2214995, upper bound: 60.2341252
time: 1.04 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2236058, upper bound: 60.2335626
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.6602230, 27.6217041, -16.1192989, 48.7391510, -57.3993759, 43.7410011
1: -12.3543396, 28.7050247, -22.4673023, 50.3988075, -62.7531471, 51.1723251
2: -10.6386271, 32.0058823, -19.2585125, 56.0775299, -66.7161560, 51.2643890
3: -11.5859261, 41.1813660, -21.2393341, 71.9838791, -83.5698090, 62.4207001
4: -10.0872440, 38.0864449, -17.8179779, 66.7789383, -76.8661804, 55.9044228

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2318665
time: 1.06 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240421, upper bound: 60.2261301
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -16.4778557, 49.0316277, -16.1192989, 48.7391510, -65.2170105, 65.1509247
1: -22.8805523, 50.7702751, -22.4673023, 50.3988075, -73.2793579, 73.2375793
2: -19.6148682, 56.4666824, -19.2585125, 56.0775299, -75.6923981, 75.7251968
3: -21.6091194, 72.3687057, -21.2393341, 71.9838791, -93.5929947, 93.6080399
4: -18.0539436, 67.3143234, -17.8179779, 66.7789383, -84.8328857, 85.1323013

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2157223, upper bound: 60.2248978
time: 1.00 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2333458
time: 0.83 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2240421, upper bound: 60.2261301
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -8.6991749, 27.7366791, -37.6878700, 40.2473679
1: -14.1082268, 32.7253380, -12.4092007, 28.8249302, -42.9331551, 45.1345367
2: -12.1443062, 36.5262375, -10.6860304, 32.1388817, -44.2831879, 47.2122688
3: -13.2896729, 46.9455948, -11.6377096, 41.3525887, -54.6422615, 58.5833054
4: -11.4511833, 43.4067154, -10.1306763, 38.2451363, -49.6963081, 53.5373917

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2248978, upper bound: 60.2157223
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2323247, upper bound: 60.2214995
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2316234, upper bound: 60.2236058
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -16.5652657, 49.3458900, -59.2970810, 48.1134567
1: -14.1082268, 32.7253380, -23.0251637, 51.0942726, -65.2024994, 55.7504959
2: -12.1443062, 36.5262375, -19.7440033, 56.8110428, -68.9553528, 56.2702408
3: -13.2896729, 46.9455948, -21.7395668, 72.8528137, -86.1424789, 68.6851654
4: -11.4511833, 43.4067154, -18.1693535, 67.7313232, -79.1825104, 61.5760689

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2248978, upper bound: 60.2157223
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2323247, upper bound: 60.2214995
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2316234, upper bound: 60.2236058
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -8.6602230, 27.6217041, -43.7409973, 57.3993759
1: -22.4673023, 50.3988075, -12.3543396, 28.7050247, -51.1723251, 62.7531471
2: -19.2585125, 56.0775299, -10.6386271, 32.0058823, -51.2643890, 66.7161560
3: -21.2393341, 71.9838791, -11.5859261, 41.1813660, -62.4207001, 83.5698090
4: -17.8179779, 66.7789383, -10.0872440, 38.0864449, -55.9044228, 76.8661804

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2109184, upper bound: 60.2268193
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2241957, upper bound: 60.2311192
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.4778557, 49.0316277, -65.1509247, 65.2170105
1: -22.4673023, 50.3988075, -22.8805523, 50.7702751, -73.2375793, 73.2793579
2: -19.2585125, 56.0775299, -19.6148682, 56.4666824, -75.7251968, 75.6923981
3: -21.2393341, 71.9838791, -21.6091194, 72.3687057, -93.6080399, 93.5929947
4: -17.8179779, 66.7789383, -18.0539436, 67.3143234, -85.1323013, 84.8328857

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2248978, upper bound: 60.2177228
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2241957, upper bound: 60.2311192
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -9.9511909, 31.5481949, -41.4993858, 41.4993858
1: -14.1082268, 32.7253380, -14.1082268, 32.7253380, -46.8335609, 46.8335571
2: -12.1443062, 36.5262375, -12.1443062, 36.5262375, -48.6705399, 48.6705399
3: -13.2896729, 46.9455948, -13.2896729, 46.9455948, -60.2352676, 60.2352676
4: -11.4511833, 43.4067154, -11.4511833, 43.4067154, -54.8578911, 54.8578911

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1992214, upper bound: 60.1836546
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -16.1192989, 48.7391510, -58.6903419, 47.6674919
1: -14.1082268, 32.7253380, -22.4673023, 50.3988075, -64.5070267, 55.1926422
2: -12.1443062, 36.5262375, -19.2585125, 56.0775299, -68.2218170, 55.7847481
3: -13.2896729, 46.9455948, -21.2393341, 71.9838791, -85.2735519, 68.1849289
4: -11.4511833, 43.4067154, -17.8179779, 66.7789383, -78.2301178, 61.2246933

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198886, upper bound: 60.2179560
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2209397, upper bound: 60.2209397
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -9.9511909, 31.5481949, -47.6674919, 58.6903419
1: -22.4673023, 50.3988075, -14.1082268, 32.7253380, -55.1926422, 64.5070190
2: -19.2585125, 56.0775299, -12.1443062, 36.5262375, -55.7847481, 68.2218170
3: -21.2393341, 71.9838791, -13.2896729, 46.9455948, -68.1849289, 85.2735443
4: -17.8179779, 66.7789383, -11.4511833, 43.4067154, -61.2246933, 78.2301178

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2079634, upper bound: 60.2240625
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2304047
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210082, upper bound: 60.2272077
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -16.1192989, 48.7391510, -64.8584518, 64.8584518
1: -22.4673023, 50.3988075, -22.4673023, 50.3988075, -72.8661118, 72.8661118
2: -19.2585125, 56.0775299, -19.2585125, 56.0775299, -75.3360443, 75.3360443
3: -21.2393341, 71.9838791, -21.2393341, 71.9838791, -93.2232132, 93.2232132
4: -17.8179779, 66.7789383, -17.8179779, 66.7789383, -84.5969162, 84.5969162

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2079634, upper bound: 60.2240625
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2304047
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210082, upper bound: 60.2272077
time: 1.07 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 8.45 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2290779, upper bound: 60.2196514
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2290779, upper bound: 60.2319546
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2290779, upper bound: 60.2196514
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2319546, upper bound: 60.2319546
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2186772, upper bound: 60.2312285
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2319546, upper bound: 60.2355284
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2289078, upper bound: 60.2360855
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2288698, upper bound: 60.2364990
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2214995, upper bound: 60.2323247
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2236058, upper bound: 60.2316234
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2214995, upper bound: 60.2341252
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2236058, upper bound: 60.2335626
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2318665
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2240421, upper bound: 60.2261301
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2240800, upper bound: 60.2333458
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2240421, upper bound: 60.2261301
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2323247, upper bound: 60.2214995
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2316234, upper bound: 60.2236058
IS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2323247, upper bound: 60.2214995
IS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2316234, upper bound: 60.2236058
IS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2109184, upper bound: 60.2268193
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2241957, upper bound: 60.2311192
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2248978, upper bound: 60.2177228
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2241957, upper bound: 60.2311192
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.1992214, upper bound: 60.1836546
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2198886, upper bound: 60.2179560
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2209397, upper bound: 60.2209397
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2304047
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2210082, upper bound: 60.2272077
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2304047
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 8.45
Output dim: 4, lower bound: -60.2210082, upper bound: 60.2272077

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.0000706, 19.5023861, -8.6991749, 27.7366791, -33.7367477, 28.2015610
1: -8.5339622, 20.3191204, -12.4092007, 28.8249302, -37.3588943, 32.7283211
2: -7.3457561, 22.7663746, -10.6860304, 32.1388817, -39.4846382, 33.4524002
3: -8.0549335, 29.1703911, -11.6377096, 41.3525887, -49.4075165, 40.8080902
4: -7.1232290, 27.0667210, -10.1306763, 38.2451363, -45.3683662, 37.1973953

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167747, upper bound: 60.2167747
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167747, upper bound: 60.2196514
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.7635622, 30.8680763, -8.6514282, 27.6096249, -37.3731880, 39.5195045
1: -13.8402271, 32.0488358, -12.3425217, 28.6933136, -42.5335388, 44.3913574
2: -11.9041538, 35.7016144, -10.6276340, 31.9932613, -43.8974152, 46.3292465
3: -13.0373688, 45.8267136, -11.5787268, 41.1669006, -54.2042694, 57.4054413
4: -11.2463779, 42.4568138, -10.0794258, 38.0708008, -49.3171768, 52.5362396

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196514, upper bound: 60.2290779
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196514, upper bound: 60.2319546
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.0000706, 19.5023861, -16.5652657, 49.3458900, -55.3459625, 36.0676498
1: -8.5339622, 20.3191204, -23.0251637, 51.0942726, -59.6282349, 43.3442841
2: -7.3457561, 22.7663746, -19.7440033, 56.8110428, -64.1567993, 42.5103722
3: -8.0549335, 29.1703911, -21.7395668, 72.8528137, -80.9077454, 50.9099503
4: -7.1232290, 27.0667210, -18.1693535, 67.7313232, -74.8545532, 45.2360764

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2322653, upper bound: 60.2184684
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2313951, upper bound: 60.2191280
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.7635622, 30.8680763, -16.5094051, 49.1953278, -58.9588776, 47.3774719
1: -13.8402271, 32.0488358, -22.9445534, 50.9378014, -64.7780304, 54.9933891
2: -11.9041538, 35.7016144, -19.6739578, 56.6379318, -68.5420837, 55.3755684
3: -13.0373688, 45.8267136, -21.6686821, 72.6334686, -85.6708374, 67.4953918
4: -11.2463779, 42.4568138, -18.1107655, 67.5218124, -78.7681885, 60.5675812

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2322653, upper bound: 60.2290363
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2313951, upper bound: 60.2296959
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -16.5652657, 49.3458900, -5.9589076, 19.3780518, -35.9433174, 55.3047943
1: -23.0251637, 51.0942726, -8.4746065, 20.1900005, -43.2151527, 59.5688782
2: -19.7440033, 56.8110428, -7.2950258, 22.6231365, -42.3671379, 64.1060715
3: -21.7395668, 72.8528137, -7.9993563, 28.9849606, -50.7245255, 80.8521652
4: -18.1693535, 67.7313232, -7.0772061, 26.8955078, -45.0648613, 74.8085327

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184684, upper bound: 60.2322653
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2191280, upper bound: 60.2313951
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -16.5094051, 49.1953278, -9.6396761, 30.5063362, -47.0157280, 58.8349991
1: -22.9445534, 50.9378014, -13.6652851, 31.6738052, -54.6183586, 64.6030884
2: -19.6739578, 56.6379318, -11.7548065, 35.2844162, -54.9583702, 68.3927383
3: -21.6686821, 72.6334686, -12.8723869, 45.2892723, -66.9579544, 85.5058594
4: -18.1107655, 67.5218124, -11.1100101, 41.9576759, -60.0684395, 78.6318207

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290363, upper bound: 60.2350051
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2191280, upper bound: 60.2341264
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15.8703461, 47.2382812, -16.4778557, 49.0316277, -64.9019699, 63.7161369
1: -22.0555267, 48.9259911, -22.8805523, 50.7702751, -72.8257980, 71.8065414
2: -18.9196854, 54.4121628, -19.6148682, 56.4666824, -75.3863678, 74.0270309
3: -20.8321724, 69.7385635, -21.6091194, 72.3687057, -93.2008820, 91.3476715
4: -17.4116936, 64.8971024, -18.0539436, 67.3143234, -84.7260132, 82.9510498

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2288698, upper bound: 60.2360855
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2294022, upper bound: 60.2360855
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6163216, 51.6089325, -16.4757500, 49.0254784, -66.6417923, 68.0846710
1: -24.3151302, 53.4625702, -22.8775120, 50.7639084, -75.0790405, 76.3400803
2: -20.8988380, 59.5328369, -19.6123123, 56.4597397, -77.3585663, 79.1451340
3: -23.0201664, 76.1713943, -21.6064072, 72.3598328, -95.3799973, 97.7777863
4: -19.1552315, 71.0358429, -18.0517578, 67.3060074, -86.4612350, 89.0876007

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2288698, upper bound: 60.2364990
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2294022, upper bound: 60.2364990
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.4943419, 24.1824589, -9.8982983, 31.3935871, -38.8879280, 34.0807571
1: -10.7083960, 25.1602631, -14.0326862, 32.5658302, -43.2742271, 39.1929474
2: -9.2479057, 28.1220131, -12.0804205, 36.3501205, -45.5980263, 40.2024269
3: -10.0447016, 36.2160797, -13.2189999, 46.7217255, -56.7664261, 49.4350777
4: -8.8644810, 33.4776611, -11.3946114, 43.1969490, -52.0614319, 44.8722725

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155814, upper bound: 60.2247725
time: 1.04 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2238378, upper bound: 60.2316183
time: 0.93 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2229413, upper bound: 60.2318683
time: 0.82 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2182749, upper bound: 60.2316261
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.6516638, 27.6356068, -9.9511909, 31.5481949, -40.1998596, 37.5867996
1: -12.3500996, 28.7194252, -14.1082268, 32.7253380, -45.0754318, 42.8276520
2: -10.6335354, 32.0210762, -12.1443062, 36.5262375, -47.1597748, 44.1653748
3: -11.5867176, 41.2065887, -13.2896729, 46.9455948, -58.5323105, 54.4962616
4: -10.0839005, 38.1016922, -11.4511833, 43.4067154, -53.4906120, 49.5528641

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2151990, upper bound: 60.2230373
time: 0.98 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2231284, upper bound: 60.2218771
time: 0.81 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2235376, upper bound: 60.2316234
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -15.0562019, 45.0254517, -9.8982983, 31.3935871, -46.4497910, 54.9237518
1: -20.8223305, 46.6315765, -14.0326862, 32.5658302, -53.3881607, 60.6642609
2: -17.9141350, 51.9317551, -12.0804205, 36.3501205, -54.2642517, 64.0121689
3: -19.7496395, 66.6583252, -13.2189999, 46.7217255, -66.4713516, 79.8773041
4: -16.6299400, 61.9274216, -11.3946114, 43.1969490, -59.8268890, 73.3220291

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2145393, upper bound: 60.2283463
time: 0.97 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2207340, upper bound: 60.2337803
time: 1.12 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152579, upper bound: 60.2326990
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2182416, upper bound: 60.2337105
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -16.5011749, 49.2117462, -9.9511909, 31.5481949, -48.0493698, 59.1629372
1: -22.9424286, 50.9540787, -14.1082268, 32.7253380, -55.6677589, 65.0622864
2: -19.6721401, 56.6566772, -12.1443062, 36.5262375, -56.1983681, 68.8009796
3: -21.6692696, 72.6607208, -13.2896729, 46.9455948, -68.6148682, 85.9503860
4: -18.1088314, 67.5434189, -11.4511833, 43.4067154, -61.5155487, 78.9945984

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2151990, upper bound: 60.2274677
time: 1.08 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152579, upper bound: 60.2321877
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2223901, upper bound: 60.2331479
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -7.9473248, 25.4955311, -16.1192989, 48.7391510, -56.6864738, 41.6148300
1: -11.3680954, 26.5178776, -22.4673023, 50.3988075, -61.7668953, 48.9851799
2: -9.7957926, 29.5887432, -19.2585125, 56.0775299, -65.8733215, 48.8472481
3: -10.6560183, 38.0537453, -21.2393341, 71.9838791, -82.6398773, 59.2930794
4: -9.3301964, 35.2053337, -17.8179779, 66.7789383, -76.1091309, 53.0233116

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2280986, upper bound: 60.2118925
time: 1.05 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2311192, upper bound: 60.2238627
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -9.4299145, 29.5511837, -16.1172333, 48.7331619, -58.1630783, 45.6684151
1: -13.3265362, 30.6757679, -22.4643669, 50.3926239, -63.7191620, 53.1401329
2: -11.4935360, 34.2454948, -19.2559967, 56.0707474, -67.5642853, 53.5014915
3: -12.5065241, 44.0347290, -21.2366638, 71.9751816, -84.4817047, 65.2713928
4: -10.8536158, 40.8196640, -17.8158073, 66.7708740, -77.6244888, 58.6354713

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2277361, upper bound: 60.2112594
time: 1.23 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2292175, upper bound: 60.2137625
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -15.7860851, 46.9332581, -16.1192989, 48.7391510, -64.5252380, 63.0525551
1: -21.9160442, 48.6120148, -22.4673023, 50.3988075, -72.3148499, 71.0793152
2: -18.7947407, 54.0785255, -19.2585125, 56.0775299, -74.8722687, 73.3370361
3: -20.7068844, 69.2675858, -21.2393341, 71.9838791, -92.6907578, 90.5069199
4: -17.2996521, 64.4921265, -17.8179779, 66.7789383, -84.0785751, 82.3101044

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2162736, upper bound: 60.2272801
time: 0.97 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164461, upper bound: 60.2265780
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -17.5195408, 51.2551117, -16.1172333, 48.7331619, -66.2527008, 67.3723450
1: -24.1536808, 53.0977554, -22.4643669, 50.3926239, -74.5463028, 75.5621185
2: -20.7541580, 59.1470680, -19.2559967, 56.0707474, -76.8248978, 78.4030609
3: -22.8760738, 75.6257706, -21.2366638, 71.9751816, -94.8512573, 96.8624268
4: -19.0250797, 70.5671539, -17.8158073, 66.7708740, -85.7959518, 88.3829651

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2158910, upper bound: 60.2280041
time: 1.02 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164461, upper bound: 60.2273020
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.8982983, 31.3935871, -7.4943419, 24.1824589, -34.0807571, 38.8879280
1: -14.0326862, 32.5658302, -10.7083960, 25.1602631, -39.1929474, 43.2742271
2: -12.0804205, 36.3501205, -9.2479057, 28.1220131, -40.2024269, 45.5980263
3: -13.2189999, 46.7217255, -10.0447016, 36.2160797, -49.4350777, 56.7664261
4: -11.3946114, 43.1969490, -8.8644810, 33.4776611, -44.8722687, 52.0614319

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2247725, upper bound: 60.2155814
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2316183, upper bound: 60.2238378
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2318683, upper bound: 60.2229413
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2316261, upper bound: 60.2182749
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -8.6516638, 27.6356068, -37.5867958, 40.1998596
1: -14.1082268, 32.7253380, -12.3500996, 28.7194252, -42.8276520, 45.0754318
2: -12.1443062, 36.5262375, -10.6335354, 32.0210762, -44.1653748, 47.1597748
3: -13.2896729, 46.9455948, -11.5867176, 41.2065887, -54.4962616, 58.5323105
4: -11.4511833, 43.4067154, -10.0839005, 38.1016922, -49.5528641, 53.4906120

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2230373, upper bound: 60.2151990
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2218771, upper bound: 60.2231284
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2316234, upper bound: 60.2235376
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.8982983, 31.3935871, -15.0562019, 45.0254517, -54.9237518, 46.4497910
1: -14.0326862, 32.5658302, -20.8223305, 46.6315765, -60.6642609, 53.3881607
2: -12.0804205, 36.3501205, -17.9141350, 51.9317551, -64.0121613, 54.2642517
3: -13.2189999, 46.7217255, -19.7496395, 66.6583252, -79.8773041, 66.4713440
4: -11.3946114, 43.1969490, -16.6299400, 61.9274216, -73.3220291, 59.8268890

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2283463, upper bound: 60.2145393
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2337803, upper bound: 60.2207340
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2326990, upper bound: 60.2152579
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2306659, upper bound: 60.2182416
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -16.5011749, 49.2117462, -59.1629333, 48.0493698
1: -14.1082268, 32.7253380, -22.9424286, 50.9540787, -65.0622940, 55.6677628
2: -12.1443062, 36.5262375, -19.6721401, 56.6566772, -68.8009796, 56.1983681
3: -13.2896729, 46.9455948, -21.6692696, 72.6607208, -85.9503860, 68.6148682
4: -11.4511833, 43.4067154, -18.1088314, 67.5434189, -78.9945984, 61.5155487

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2274677, upper bound: 60.2151990
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2321877, upper bound: 60.2194064
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2331479, upper bound: 60.2223901
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -16.1192989, 48.7391510, -5.9589076, 19.3780518, -35.4973526, 54.6980553
1: -22.4673023, 50.3988075, -8.4746065, 20.1900005, -42.6572990, 58.8734131
2: -19.2585125, 56.0775299, -7.2950258, 22.6231365, -41.8816376, 63.3725510
3: -21.2393341, 71.9838791, -7.9993563, 28.9849606, -50.2242966, 79.9832382
4: -17.8179779, 66.7789383, -7.0772061, 26.8955078, -44.7134857, 73.8561478

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2118925, upper bound: 60.2280986
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2112594, upper bound: 60.2277361
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -16.0664253, 48.6023941, -9.6396761, 30.5063362, -46.5727615, 58.2420654
1: -22.3939323, 50.2568207, -13.6652851, 31.6738052, -54.0677376, 63.9221039
2: -19.1939411, 55.9207115, -11.7548065, 35.2844162, -54.4783516, 67.6755219
3: -21.1733398, 71.7852325, -12.8723869, 45.2892723, -66.4626160, 84.6576157
4: -17.7641468, 66.5872040, -11.1100101, 41.9576759, -59.7218246, 77.6972122

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2238596, upper bound: 60.2177228
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2238596, upper bound: 60.2311192
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -13.3044767, 40.1446075, -16.4778557, 49.0316277, -62.3361053, 56.6224632
1: -18.4690800, 41.5440445, -22.8805523, 50.7702751, -69.2393341, 64.4245987
2: -15.8064594, 46.3984032, -19.6148682, 56.4666824, -72.2731247, 66.0132751
3: -17.5225220, 59.2194366, -21.6091194, 72.3687057, -89.8912201, 80.8285522
4: -14.6301203, 55.2837143, -18.0539436, 67.3143234, -81.9444427, 73.3376617

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2245649, upper bound: 60.2177228
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180704, upper bound: 60.2161611
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16.8450050, 51.1080399, -16.4223900, 48.8830070, -65.7280121, 67.5304260
1: -23.4218330, 52.8250275, -22.8005943, 50.6156540, -74.0374756, 75.6256180
2: -20.0210838, 58.7532997, -19.5453930, 56.2956467, -76.3167267, 78.2986832
3: -22.2593422, 75.3798065, -21.5388699, 72.1523514, -94.4116745, 96.9186783
4: -18.5738354, 69.9567947, -17.9959869, 67.1071548, -85.6809921, 87.9527740

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2255924, upper bound: 60.2311192
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2231074, upper bound: 60.2295575
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.7370825, 30.8880539, -9.9511909, 31.5481949, -41.2852783, 40.8392372
1: -13.8120041, 32.0481262, -14.1082268, 32.7253380, -46.5373421, 46.1563530
2: -11.8939190, 35.7687950, -12.1443062, 36.5262375, -48.4201546, 47.9130936
3: -13.0009775, 45.9720306, -13.2896729, 46.9455948, -59.9465714, 59.2617035
4: -11.2234163, 42.5065231, -11.4511833, 43.4067154, -54.6301308, 53.9576988

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2162801, upper bound: 60.2096310
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
time: 1.22 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.0973234, 29.0640278, -16.1192989, 48.7391510, -57.8364754, 45.1833191
1: -12.8702602, 30.1397648, -22.4673023, 50.3988075, -63.2690620, 52.6070671
2: -11.0743856, 33.6567726, -19.2585125, 56.0775299, -67.1519165, 52.9152832
3: -12.1959352, 43.2404213, -21.2393341, 71.9838791, -84.1798172, 64.4797516
4: -10.4772635, 40.0022278, -17.8179779, 66.7789383, -77.2561798, 57.8202057

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2206887, upper bound: 60.2042052
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290190, upper bound: 60.2071336
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2258220, upper bound: 60.2175960
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -10.8696327, 34.4083939, -16.0833206, 48.6556587, -59.5252914, 50.4917107
1: -15.3343544, 35.6384964, -22.4160194, 50.3113251, -65.6456757, 58.0545158
2: -13.1785946, 39.7972412, -19.2111607, 55.9817619, -69.1603470, 59.0084000
3: -14.4879837, 51.1623955, -21.1991253, 71.8633118, -86.3512878, 72.3615189
4: -12.3919563, 47.3149300, -17.7807560, 66.6622849, -79.0542297, 65.0956879

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2218072, upper bound: 60.2071701
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.2004224, upper bound: 60.1833558
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2301254, upper bound: 60.2178593
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.6242218, 44.3043480, -9.8296013, 31.1956444, -45.8198662, 54.1339417
1: -20.2457829, 45.8350677, -13.9361629, 32.3614120, -52.6071892, 59.7712326
2: -17.4124851, 51.0891838, -11.9998665, 36.1235733, -53.5360565, 63.0890427
3: -19.2638702, 65.6291351, -13.1277885, 46.4352493, -65.6991196, 78.7569122
4: -16.2089767, 60.8570938, -11.3237314, 42.9283142, -59.1372871, 72.1808167

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2030556, upper bound: 60.2236471
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2071336, upper bound: 60.2290190
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2299762
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.8565083, 47.9964638, -9.9511909, 31.5481949, -47.4047012, 57.9476509
1: -22.0979061, 49.6354141, -14.1082268, 32.7253380, -54.8232384, 63.7436371
2: -18.9476433, 55.2243576, -12.1443062, 36.5262375, -55.4738731, 67.3686523
3: -20.9038620, 70.9059448, -13.2896729, 46.9455948, -67.8494568, 84.1955948
4: -17.5394630, 65.7744293, -11.4511833, 43.4067154, -60.9461632, 77.2256165

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2074887, upper bound: 60.2168477
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B1_A2_A1
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
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.6242218, 44.3043480, -15.9651785, 48.3056831, -62.9299049, 60.2695160
1: -20.2457829, 45.8350677, -22.2454796, 49.9502563, -70.1960297, 68.0805511
2: -17.4124851, 51.0891838, -19.0712318, 55.5847397, -72.9972229, 70.1604156
3: -19.2638702, 65.6291351, -21.0392017, 71.3628387, -90.6267090, 86.6683350
4: -16.2089767, 60.8570938, -17.6571064, 66.1880646, -82.3970413, 78.5141983

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2049972, upper bound: 60.2236471
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2116103, upper bound: 60.2228429
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15.8565083, 47.9964638, -16.1192989, 48.7391510, -64.5956573, 64.1157608
1: -22.0979061, 49.6354141, -22.4673023, 50.3988075, -72.4967117, 72.1027145
2: -18.9476433, 55.2243576, -19.2585125, 56.0775299, -75.0251770, 74.4828720
3: -20.9038620, 70.9059448, -21.2393341, 71.9838791, -92.8877411, 92.1452637
4: -17.5394630, 65.7744293, -17.8179779, 66.7789383, -84.3183975, 83.5924072

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2094304, upper bound: 60.2168477
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125939, upper bound: 60.2160435
time: 1.21 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.99 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2167747, upper bound: 60.2167747
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2167747, upper bound: 60.2196514
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2196514, upper bound: 60.2290779
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2196514, upper bound: 60.2319546
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2322653, upper bound: 60.2184684
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2313951, upper bound: 60.2191280
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2322653, upper bound: 60.2290363
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2313951, upper bound: 60.2296959
IS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2184684, upper bound: 60.2322653
IS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2191280, upper bound: 60.2313951
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2290363, upper bound: 60.2350051
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2191280, upper bound: 60.2341264
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2288698, upper bound: 60.2360855
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2294022, upper bound: 60.2360855
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2288698, upper bound: 60.2364990
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2294022, upper bound: 60.2364990
IS_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2229413, upper bound: 60.2318683
IS_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2182749, upper bound: 60.2316261
IS_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2231284, upper bound: 60.2218771
IS_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2235376, upper bound: 60.2316234
IS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2152579, upper bound: 60.2326990
IS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2182416, upper bound: 60.2337105
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2152579, upper bound: 60.2321877
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2223901, upper bound: 60.2331479
IS_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2280986, upper bound: 60.2118925
IS_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2311192, upper bound: 60.2238627
IS_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2277361, upper bound: 60.2112594
IS_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2292175, upper bound: 60.2137625
IS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2162736, upper bound: 60.2272801
IS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2164461, upper bound: 60.2265780
IS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2158910, upper bound: 60.2280041
IS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2164461, upper bound: 60.2273020
IS_A2_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2318683, upper bound: 60.2229413
IS_A2_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2316261, upper bound: 60.2182749
IS_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2218771, upper bound: 60.2231284
IS_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2316234, upper bound: 60.2235376
IS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2326990, upper bound: 60.2152579
IS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2306659, upper bound: 60.2182416
IS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2321877, upper bound: 60.2194064
IS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2331479, upper bound: 60.2223901
IS_A2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2118925, upper bound: 60.2280986
IS_A2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2112594, upper bound: 60.2277361
IS_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2238596, upper bound: 60.2177228
IS_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2238596, upper bound: 60.2311192
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2245649, upper bound: 60.2177228
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2180704, upper bound: 60.2161611
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2255924, upper bound: 60.2311192
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2231074, upper bound: 60.2295575
IS_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2162801, upper bound: 60.2096310
IS_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2210875, upper bound: 60.2210875
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2290190, upper bound: 60.2071336
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2258220, upper bound: 60.2175960
IS_A2_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2004224, upper bound: 60.1833558
IS_A2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2301254, upper bound: 60.2178593
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2071336, upper bound: 60.2290190
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2101173, upper bound: 60.2299762
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2175960, upper bound: 60.2258220
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2205797, upper bound: 60.2267792
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2049972, upper bound: 60.2236471
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2116103, upper bound: 60.2228429
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2094304, upper bound: 60.2168477
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.99
Output dim: 4, lower bound: -60.2125939, upper bound: 60.2160435

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.0000706, 19.5023861, -6.0000706, 19.5023861, -25.5024529, 25.5024548
1: -8.5339622, 20.3191204, -8.5339622, 20.3191204, -28.8530807, 28.8530788
2: -7.3457561, 22.7663746, -7.3457561, 22.7663746, -30.1121311, 30.1121311
3: -8.0549335, 29.1703911, -8.0549335, 29.1703911, -37.2253151, 37.2253151
4: -7.1232290, 27.0667210, -7.1232290, 27.0667210, -34.1899490, 34.1899490

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161415, upper bound: 60.2162683
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156351, upper bound: 60.2156351
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.0000706, 19.5023861, -9.7635622, 30.8680763, -36.8681488, 29.2659492
1: -8.5339622, 20.3191204, -13.8402271, 32.0488358, -40.5827980, 34.1593475
2: -7.3457561, 22.7663746, -11.9041538, 35.7016144, -43.0473709, 34.6705246
3: -8.0549335, 29.1703911, -13.0373688, 45.8267136, -53.8816414, 42.2077522
4: -7.1232290, 27.0667210, -11.2463779, 42.4568138, -49.5800438, 38.3130989

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132134, upper bound: 60.2171169
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132134, upper bound: 60.2165033
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7635622, 30.8680763, -6.0000706, 19.5023861, -29.2659492, 36.8681488
1: -13.8402271, 32.0488358, -8.5339622, 20.3191204, -34.1593475, 40.5827980
2: -11.9041538, 35.7016144, -7.3457561, 22.7663746, -34.6705246, 43.0473709
3: -13.0373688, 45.8267136, -8.0549335, 29.1703911, -42.2077522, 53.8816376
4: -11.2463779, 42.4568138, -7.1232290, 27.0667210, -38.3130989, 49.5800438

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2171168, upper bound: 60.2152254
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2158181, upper bound: 60.2290673
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7635622, 30.8680763, -9.7635622, 30.8680763, -40.6316299, 40.6316299
1: -13.8402271, 32.0488358, -13.8402271, 32.0488358, -45.8890610, 45.8890610
2: -11.9041538, 35.7016144, -11.9041538, 35.7016144, -47.6057663, 47.6057663
3: -13.0373688, 45.8267136, -13.0373688, 45.8267136, -58.8640785, 58.8640785
4: -11.2463779, 42.4568138, -11.2463779, 42.4568138, -53.7031937, 53.7031937

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2160901, upper bound: 60.2296809
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193800, upper bound: 60.2319440
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -5.9377594, 19.3211899, -15.0562019, 45.0254517, -50.9632111, 34.3773880
1: -8.4458275, 20.1308193, -20.8223305, 46.6315765, -55.0773926, 40.9531479
2: -7.2717776, 22.5588818, -17.9141350, 51.9317551, -59.2035294, 40.4730148
3: -7.9715881, 28.9084816, -19.7496395, 66.6583252, -74.6299057, 48.6581192
4: -7.0578132, 26.8199654, -16.6299400, 61.9274216, -68.9852371, 43.4499054

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2317563, upper bound: 60.2081897
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2314668, upper bound: 60.2075565
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.0000706, 19.5023861, -16.5011749, 49.2117462, -55.2118149, 36.0035629
1: -8.5339622, 20.3191204, -22.9424286, 50.9540787, -59.4880409, 43.2615509
2: -7.3457561, 22.7663746, -19.6721401, 56.6566772, -64.0024338, 42.4385033
3: -8.0549335, 29.1703911, -21.6692696, 72.6607208, -80.7156525, 50.8396606
4: -7.1232290, 27.0667210, -18.1088314, 67.5434189, -74.6666412, 45.1755524

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308582, upper bound: 60.2176301
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2305639, upper bound: 60.2169970
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7049370, 30.7000332, -15.0087404, 44.9003677, -54.6053047, 45.7087708
1: -13.7583122, 31.8752861, -20.7554054, 46.5020752, -60.2603874, 52.6306839
2: -11.8341618, 35.5097733, -17.8549728, 51.7886772, -63.6228333, 53.3647461
3: -12.9595404, 45.5852470, -19.6904812, 66.4748993, -79.4344406, 65.2757187
4: -11.1842194, 42.2290115, -16.5804138, 61.7552338, -72.9394531, 58.8094254

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2345375, upper bound: 60.2184409
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2327810, upper bound: 60.2100597
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7635622, 30.8680763, -16.4454174, 49.0614471, -58.8250008, 47.3134880
1: -13.8402271, 32.0488358, -22.8620071, 50.7978668, -64.6380920, 54.9108429
2: -11.9041538, 35.7016144, -19.6022472, 56.4838753, -68.3880310, 55.3038635
3: -13.0373688, 45.8267136, -21.5985222, 72.4417267, -85.4790955, 67.4252319
4: -11.2463779, 42.4568138, -18.0503597, 67.3342361, -78.5806122, 60.5071716

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2181539, upper bound: 60.2257338
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2181539, upper bound: 60.2296960
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -15.0562019, 45.0254517, -5.8967075, 19.1971531, -34.2533531, 50.9221573
1: -20.8223305, 46.6315765, -8.3866167, 20.0020142, -40.8243446, 55.0181847
2: -17.9141350, 51.9317551, -7.2211628, 22.4159966, -40.3301277, 59.1529160
3: -19.7496395, 66.6583252, -7.9161596, 28.7234917, -48.4731293, 74.5744858
4: -16.6299400, 61.9274216, -7.0118980, 26.6491261, -43.2790680, 68.9393158

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2081897, upper bound: 60.2317563
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2075565, upper bound: 60.2314668
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -16.5011749, 49.2117462, -5.9589076, 19.3780518, -35.8792267, 55.1706505
1: -22.9424286, 50.9540787, -8.4746065, 20.1900005, -43.1324234, 59.4286842
2: -19.6721401, 56.6566772, -7.2950258, 22.6231365, -42.2952652, 63.9516983
3: -21.6692696, 72.6607208, -7.9993563, 28.9849606, -50.6542282, 80.6600723
4: -18.1088314, 67.5434189, -7.0772061, 26.8955078, -45.0043411, 74.6206284

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2176301, upper bound: 60.2308582
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2169970, upper bound: 60.2305639
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -15.0087404, 44.9003677, -9.5812569, 30.3389301, -45.3476715, 54.4816246
1: -20.7554054, 46.5020752, -13.5836620, 31.5009327, -52.2563400, 60.0857391
2: -17.8549728, 51.7886772, -11.6850586, 35.0933495, -52.9483223, 63.4737358
3: -19.6904812, 66.4748993, -12.7948961, 45.0487785, -64.7392578, 79.2697983
4: -16.5804138, 61.7552338, -11.0480547, 41.7307510, -58.3111649, 72.8032913

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2184409, upper bound: 60.2345375
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2100597, upper bound: 60.2327810
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -16.4454174, 49.0614471, -9.6396761, 30.5063362, -46.9517479, 58.7011223
1: -22.8620071, 50.7978668, -13.6652851, 31.6738052, -54.5358124, 64.4631424
2: -19.6022472, 56.4838753, -11.7548065, 35.2844162, -54.8866653, 68.2386780
3: -21.5985222, 72.4417267, -12.8723869, 45.2892723, -66.8877945, 85.3141174
4: -18.0503597, 67.3342361, -11.1100101, 41.9576759, -60.0080338, 78.4442291

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2151659, upper bound: 60.2181539
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2151659, upper bound: 60.2154419
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.8703461, 47.2382812, -15.7860851, 46.9332581, -62.8036041, 63.0243683
1: -22.0555267, 48.9259911, -21.9160442, 48.6120148, -70.6675339, 70.8420334
2: -18.9196854, 54.4121628, -18.7947407, 54.0785255, -72.9982147, 73.2069016
3: -20.8321724, 69.7385635, -20.7068844, 69.2675858, -90.0997620, 90.4454346
4: -17.4116936, 64.8971024, -17.2996521, 64.4921265, -81.9038086, 82.1967545

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2233447, upper bound: 60.2334805
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286137, upper bound: 60.2349132
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.8703461, 47.2382812, -17.5195408, 51.2551117, -67.1254578, 64.7578201
1: -22.0555267, 48.9259911, -24.1536808, 53.0977554, -75.1532822, 73.0796738
2: -18.9196854, 54.4121628, -20.7541580, 59.1470680, -78.0667572, 75.1663055
3: -20.8321724, 69.7385635, -22.8760738, 75.6257706, -96.4579468, 92.6146317
4: -17.4116936, 64.8971024, -19.0250797, 70.5671539, -87.9788437, 83.9221802

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2233447, upper bound: 60.2334805
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286137, upper bound: 60.2349132
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17.6163216, 51.6089325, -15.7860851, 46.9332581, -64.5495758, 67.3950195
1: -24.3151302, 53.4625702, -21.9160442, 48.6120148, -72.9271393, 75.3786163
2: -20.8988380, 59.5328369, -18.7947407, 54.0785255, -74.9773636, 78.3275757
3: -23.0201664, 76.1713943, -20.7068844, 69.2675858, -92.2877502, 96.8782806
4: -19.1552315, 71.0358429, -17.2996521, 64.4921265, -83.6473541, 88.3354950

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180704, upper bound: 60.2167755
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172214, upper bound: 60.2334992
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17.6163216, 51.6089325, -17.5195408, 51.2551117, -68.8714294, 69.1284637
1: -24.3151302, 53.4625702, -24.1536808, 53.0977554, -77.4128876, 77.6162415
2: -20.8988380, 59.5328369, -20.7541580, 59.1470680, -80.0459061, 80.2869873
3: -23.0201664, 76.1713943, -22.8760738, 75.6257706, -98.6459351, 99.0474701
4: -19.1552315, 71.0358429, -19.0250797, 70.5671539, -89.7223816, 90.0609207

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180704, upper bound: 60.2167755
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180704, upper bound: 60.2167755
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -6.6787472, 21.7114887, -9.8982983, 31.3935871, -38.0723343, 31.6097870
1: -9.5207081, 22.5992203, -14.0326862, 32.5658302, -42.0865402, 36.6319046
2: -8.2373705, 25.2666073, -12.0804205, 36.3501205, -44.5874901, 37.3470268
3: -8.9772997, 32.5146523, -13.2189999, 46.7217255, -55.6990242, 45.7336464
4: -7.9356890, 30.0801773, -11.3946114, 43.1969490, -51.1326370, 41.4747887

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140868, upper bound: 60.2223422
time: 0.99 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2229413, upper bound: 60.2311933
time: 0.99 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2227849, upper bound: 60.2226373
time: 1.01 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2227849, upper bound: 60.2318683
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -8.8190660, 27.9204865, -9.8712873, 31.3187065, -40.1377716, 37.7917747
1: -12.3835669, 29.0167999, -13.9951315, 32.4880905, -44.8716583, 43.0119324
2: -10.6684256, 32.4699593, -12.0484257, 36.2644234, -46.9328461, 44.5183868
3: -11.7180309, 41.6783829, -13.1840553, 46.6126480, -58.3306732, 54.8624382
4: -10.1649132, 38.6378288, -11.3658743, 43.0948868, -53.2598000, 50.0037041

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110780, upper bound: 60.2215278
time: 1.02 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2182749, upper bound: 60.2309975
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2178460, upper bound: 60.2312660
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152912, upper bound: 60.2306659
time: 1.21 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152912, upper bound: 60.2316261
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -7.8018370, 24.3208351, -9.8037682, 31.1057377, -38.9075737, 34.1245995
1: -11.1546240, 25.4104347, -13.9048090, 32.2734909, -43.4281120, 39.3152428
2: -9.6689148, 28.3222370, -11.9694061, 36.0275154, -45.6964226, 40.2916374
3: -10.3286180, 36.2782669, -13.0964680, 46.2995720, -56.6281891, 49.3747330
4: -9.1755247, 33.7004814, -11.2927246, 42.8150101, -51.9905319, 44.9932022

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2231284, upper bound: 60.2218771
time: 1.07 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=65.54161834716797
rel_dist={4: [-60.23730919021928, 60.23730919021929]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2365414, upper bound: 60.2349259
time: 0.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2365414, upper bound: 60.2365414
time: 1.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.24 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.24
Output dim: 4, lower bound: -60.2365414, upper bound: 60.2349259
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.24
Output dim: 4, lower bound: -60.2365414, upper bound: 60.2365414

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.2668142, 32.2093468, -11.7342844, 36.5605621, -46.8273773, 43.9436188
1: -14.5713062, 33.4432640, -16.6261826, 37.9152184, -52.4865227, 50.0694427
2: -12.5309057, 37.2433014, -14.2820234, 42.1752167, -54.7061234, 51.5253258
3: -13.6885414, 47.8052521, -15.6261969, 54.1983261, -67.8868713, 63.4314499
4: -11.7807655, 44.3089943, -13.3540154, 50.1710587, -61.9518242, 57.6630096

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2349259, upper bound: 60.2349259
time: 0.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2349259, upper bound: 60.2349259
time: 1.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -18.7248726, 55.4986000, -11.9270954, 37.1409912, -55.8658562, 67.4256973
1: -26.1227779, 57.4683647, -16.8980808, 38.5070686, -64.6298447, 74.3664398
2: -22.3536682, 63.8967094, -14.5107450, 42.8237915, -65.1774597, 78.4074478
3: -24.5606728, 81.6774292, -15.8871717, 55.0427513, -79.6034088, 97.5645981
4: -20.4283028, 76.1416473, -13.5612068, 50.9363976, -71.3647003, 89.7028503

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2319147, upper bound: 60.2306415
time: 0.92 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.47 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 4, lower bound: -60.2349259, upper bound: 60.2349259
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 4, lower bound: -60.2349259, upper bound: 60.2349259
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 4, lower bound: -60.2319147, upper bound: 60.2306415
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.47
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.2668142, 32.2093468, -10.2668142, 32.2093468, -42.4761581, 42.4761581
1: -14.5713062, 33.4432640, -14.5713062, 33.4432640, -48.0145721, 48.0145721
2: -12.5309057, 37.2433014, -12.5309057, 37.2433014, -49.7742081, 49.7742081
3: -13.6885414, 47.8052521, -13.6885414, 47.8052521, -61.4937935, 61.4937935
4: -11.7807655, 44.3089943, -11.7807655, 44.3089943, -56.0897598, 56.0897598

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237957, upper bound: 60.2281785
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.2668142, 32.2093468, -18.7248325, 55.4984283, -65.7652435, 50.9341660
1: -14.5713062, 33.4432640, -26.1227226, 57.4682007, -72.0395050, 59.5659866
2: -12.5309057, 37.2433014, -22.3536282, 63.8965302, -76.4274368, 59.5969315
3: -13.6885414, 47.8052521, -24.5606079, 81.6771774, -95.3657227, 72.3658600
4: -11.7807655, 44.3089943, -20.4282570, 76.1414261, -87.9221878, 64.7372513

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2281785, upper bound: 60.2237269
time: 0.94 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.99 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -16.8046818, 50.0466156, -11.5431805, 36.0584946, -52.8631706, 61.5897942
1: -23.3835297, 51.8265610, -16.3705826, 37.3864441, -60.7699699, 68.1971436
2: -20.0460148, 57.6273384, -14.0607281, 41.5876617, -61.6336746, 71.6880569
3: -22.0566101, 73.8997879, -15.3883858, 53.4828568, -75.5394669, 89.2881775
4: -18.4532261, 68.7191925, -13.1611385, 49.4679489, -67.9211731, 81.8803177

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464
time: 0.92 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2290464
time: 0.83 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -16.3950844, 49.4747887, -11.1404209, 34.9374084, -51.3324928, 60.6152077
1: -22.8638802, 51.1643333, -15.8220987, 36.2292862, -59.0931664, 66.9864273
2: -19.5991325, 56.9245262, -13.5956917, 40.3093758, -59.9085045, 70.5202179
3: -21.5911999, 73.0606918, -14.8771534, 51.8498383, -73.4410400, 87.9378433
4: -18.1113377, 67.8160400, -12.7482948, 47.9364548, -66.0477905, 80.5643158

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528
time: 1.05 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528
time: 1.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.00 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 4, lower bound: -60.2237957, upper bound: 60.2281785
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 4, lower bound: -60.2281785, upper bound: 60.2237269
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2290464
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.00
Output dim: 4, lower bound: -60.2290221, upper bound: 60.2210528

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -9.9006662, 31.1688557, -39.8680305, 37.6373444
1: -12.4092007, 28.8249302, -14.0700378, 32.3676949, -44.7768936, 42.8949661
2: -10.6860304, 32.1388817, -12.1031590, 36.0536575, -46.7396889, 44.2420387
3: -11.6377096, 41.3525887, -13.2101288, 46.3044205, -57.9421272, 54.5627174
4: -10.1306763, 38.2451363, -11.3959522, 42.8965683, -53.0272446, 49.6410904

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -9.4664173, 29.9630108, -39.9141998, 41.0146103
1: -14.1082268, 32.7253380, -13.4767380, 31.1234074, -45.2316322, 46.2020760
2: -12.1443062, 36.5262375, -11.5961876, 34.6756439, -46.8199501, 48.1224251
3: -13.2896729, 46.9455948, -12.6572742, 44.5472107, -57.8368835, 59.6028671
4: -11.4511833, 43.4067154, -10.9415665, 41.2441177, -52.6952934, 54.3482742

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.9006662, 31.1688557, -16.8046379, 50.0464516, -59.9471130, 47.9734955
1: -14.0700378, 32.3676949, -23.3834743, 51.8263855, -65.8964233, 55.7511673
2: -12.1031590, 36.0536575, -20.0459709, 57.6271553, -69.7303162, 56.0996284
3: -13.2101288, 46.3044205, -22.0565472, 73.8995285, -87.1096573, 68.3609619
4: -11.3959522, 42.8965683, -18.4531860, 68.7189713, -80.1149139, 61.3497543

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 1.06 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.4664173, 29.9630108, -16.3949947, 49.4744644, -58.9408760, 46.3580017
1: -13.4767380, 31.1234074, -22.8637619, 51.1639824, -64.6407166, 53.9871635
2: -11.5961876, 34.6756439, -19.5990372, 56.9241180, -68.5203094, 54.2746811
3: -12.6572742, 44.5472107, -21.5910721, 73.0601807, -85.7174530, 66.1382828
4: -10.9415665, 41.2441177, -18.1112518, 67.8156204, -78.7571793, 59.3553619

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -16.8046818, 50.0466156, -10.2615223, 32.4407578, -49.2454376, 60.3081360
1: -23.3835297, 51.8265610, -14.6010656, 33.6536789, -57.0372086, 66.4276047
2: -20.0460148, 57.6273384, -12.5514145, 37.4663925, -57.5124054, 70.1787567
3: -22.0566101, 73.8997879, -13.7237110, 48.2649994, -70.3216095, 87.6234970
4: -18.4532261, 68.7191925, -11.8317356, 44.5688171, -63.0220413, 80.5509186

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2293410, upper bound: 60.2223833
time: 0.87 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2314459, upper bound: 60.2304737
time: 0.97 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -16.8046818, 50.0466156, -11.0889120, 34.8989944, -51.7036743, 61.1355286
1: -23.3835297, 51.8265610, -15.7143784, 36.1689301, -59.5524597, 67.5409164
2: -20.0460148, 57.6273384, -13.5199480, 40.3033752, -60.3493881, 71.1472855
3: -22.0566101, 73.8997879, -14.7961807, 51.8413086, -73.8979187, 88.6959686
4: -18.4532261, 68.7191925, -12.7073727, 47.8885651, -66.3417892, 81.4265442

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2174202, upper bound: 60.2236349
time: 0.94 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2278671, upper bound: 60.2249570
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -16.3950844, 49.4747887, -10.2615223, 32.4407578, -48.8358421, 59.7363129
1: -22.8638802, 51.1643333, -14.6010656, 33.6536789, -56.5175552, 65.7653732
2: -19.5991325, 56.9245262, -12.5514145, 37.4663925, -57.0655251, 69.4759369
3: -21.5911999, 73.0606918, -13.7237110, 48.2649994, -69.8562012, 86.7844009
4: -18.1113377, 67.8160400, -11.8317356, 44.5688171, -62.6801529, 79.6477737

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2097269, upper bound: 60.2213913
time: 1.08 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2229920, upper bound: 60.2229920
time: 0.90 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -16.3950844, 49.4747887, -11.0889120, 34.8989944, -51.2940788, 60.5637016
1: -22.8638802, 51.1643333, -15.7143784, 36.1689301, -59.0328064, 66.8786926
2: -19.5991325, 56.9245262, -13.5199480, 40.3033752, -59.9025078, 70.4444733
3: -21.5911999, 73.0606918, -14.7961807, 51.8413086, -73.4325104, 87.8568726
4: -18.1113377, 67.8160400, -12.7073727, 47.8885651, -65.9999008, 80.5233994

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2097269, upper bound: 60.2213913
time: 0.99 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2229920, upper bound: 60.2229920
time: 1.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2293410, upper bound: 60.2223833
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2314459, upper bound: 60.2304737
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2174202, upper bound: 60.2236349
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2278671, upper bound: 60.2249570
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2097269, upper bound: 60.2213913
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2229920, upper bound: 60.2229920
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2097269, upper bound: 60.2213913
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 4, lower bound: -60.2229920, upper bound: 60.2229920

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -8.6991749, 27.7366791, -36.4358521, 36.4358521
1: -12.4092007, 28.8249302, -12.4092007, 28.8249302, -41.2341309, 41.2341309
2: -10.6860304, 32.1388817, -10.6860304, 32.1388817, -42.8249130, 42.8249130
3: -11.6377096, 41.3525887, -11.6377096, 41.3525887, -52.9902954, 52.9902954
4: -10.1306763, 38.2451363, -10.1306763, 38.2451363, -48.3758125, 48.3758125

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2180946
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2193231
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2255980
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -9.9511909, 31.5481949, -40.2473679, 37.6878700
1: -12.4092007, 28.8249302, -14.1082268, 32.7253380, -45.1345367, 42.9331551
2: -10.6860304, 32.1388817, -12.1443062, 36.5262375, -47.2122688, 44.2831879
3: -11.6377096, 41.3525887, -13.2896729, 46.9455948, -58.5833054, 54.6422615
4: -10.1306763, 38.2451363, -11.4511833, 43.4067154, -53.5373917, 49.6963081

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2180946
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237957, upper bound: 60.2280775
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2237089, upper bound: 60.2255980
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -8.6991749, 27.7366791, -37.6878700, 40.2473679
1: -14.1082268, 32.7253380, -12.4092007, 28.8249302, -42.9331551, 45.1345367
2: -12.1443062, 36.5262375, -10.6860304, 32.1388817, -44.2831879, 47.2122688
3: -13.2896729, 46.9455948, -11.6377096, 41.3525887, -54.6422615, 58.5833054
4: -11.4511833, 43.4067154, -10.1306763, 38.2451363, -49.6963081, 53.5373917

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2164250
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2210528
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -9.9511909, 31.5481949, -41.4993858, 41.4993858
1: -14.1082268, 32.7253380, -14.1082268, 32.7253380, -46.8335609, 46.8335571
2: -12.1443062, 36.5262375, -12.1443062, 36.5262375, -48.6705399, 48.6705399
3: -13.2896729, 46.9455948, -13.2896729, 46.9455948, -60.2352676, 60.2352676
4: -11.4511833, 43.4067154, -11.4511833, 43.4067154, -54.8578911, 54.8578911

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164250, upper bound: 60.2096684
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -16.8046379, 50.0464516, -58.7456284, 44.5413170
1: -12.4092007, 28.8249302, -23.3834743, 51.8263855, -64.2355804, 52.2084045
2: -10.6860304, 32.1388817, -20.0459709, 57.6271553, -68.3131790, 52.1848526
3: -11.6377096, 41.3525887, -22.0565472, 73.8995285, -85.5372391, 63.4091339
4: -10.1306763, 38.2451363, -18.4531860, 68.7189713, -78.8496399, 56.6983185

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2223030, upper bound: 60.2152022
time: 0.96 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2196388, upper bound: 60.2116649
time: 1.02 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2305228, upper bound: 60.2237089
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -16.8046379, 50.0464516, -59.9976387, 48.3528328
1: -14.1082268, 32.7253380, -23.3834743, 51.8263855, -65.9345932, 56.1088066
2: -12.1443062, 36.5262375, -20.0459709, 57.6271553, -69.7714462, 56.5722084
3: -13.2896729, 46.9455948, -22.0565472, 73.8995285, -87.1891861, 69.0021439
4: -11.4511833, 43.4067154, -18.4531860, 68.7189713, -80.1701508, 61.8599014

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2223030, upper bound: 60.2152022
time: 0.95 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2275683, upper bound: 60.2197016
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2275683, upper bound: 60.2226870
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.6991749, 27.7366791, -16.3949947, 49.4744644, -58.1736374, 44.1316757
1: -12.4092007, 28.8249302, -22.8637619, 51.1639824, -63.5731812, 51.6886864
2: -10.6860304, 32.1388817, -19.5990372, 56.9241180, -67.6101456, 51.7379189
3: -11.6377096, 41.3525887, -21.5910721, 73.0601807, -84.6978912, 62.9436607
4: -10.1306763, 38.2451363, -18.1112518, 67.8156204, -77.9462891, 56.3563843

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198763, upper bound: 60.2077902
time: 1.18 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2176066, upper bound: 60.2096684
time: 0.96 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290138, upper bound: 60.2210528
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.9511909, 31.5481949, -16.3949947, 49.4744644, -59.4256516, 47.9431915
1: -14.1082268, 32.7253380, -22.8637619, 51.1639824, -65.2721939, 55.5890923
2: -12.1443062, 36.5262375, -19.5990372, 56.9241180, -69.0684128, 56.1252747
3: -13.2896729, 46.9455948, -21.5910721, 73.0601807, -86.3498459, 68.5366669
4: -11.4511833, 43.4067154, -18.1112518, 67.8156204, -79.2667999, 61.5179672

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198763, upper bound: 60.2077902
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2176066, upper bound: 60.2096684
time: 0.99 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290138, upper bound: 60.2210528
time: 1.07 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.2892036, 45.7070007, -10.0411892, 31.8072395, -47.0964432, 55.7481842
1: -21.1691895, 47.3431854, -14.2933474, 33.0010223, -54.1702118, 61.6365318
2: -18.2080307, 52.7248726, -12.2910480, 36.7464447, -54.9544754, 65.0159073
3: -20.0574760, 67.6858368, -13.4358625, 47.3504677, -67.4079285, 81.1216965
4: -16.9065418, 62.9016724, -11.6041842, 43.7172699, -60.6238098, 74.5058594

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2169110, upper bound: 60.2349050
time: 0.93 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172152, upper bound: 60.2123798
time: 0.91 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16.7311668, 49.8861961, -10.2615223, 32.4407578, -49.1719208, 60.1477165
1: -23.2865944, 51.6587791, -14.6010656, 33.6536789, -56.9402733, 66.2598190
2: -19.9622326, 57.4426842, -12.5514145, 37.4663925, -57.4286118, 69.9940796
3: -21.9739342, 73.6682281, -13.7237110, 48.2649994, -70.2389374, 87.3919373
4: -18.3817101, 68.4947433, -11.8317356, 44.5688171, -62.9505272, 80.3264771

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2177280, upper bound: 60.2283593
time: 0.95 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2108107, upper bound: 60.2328996
time: 1.10 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110529, upper bound: 60.2110529
time: 1.08 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -16.2499657, 48.3076897, -7.7260199, 24.9885464, -41.2385101, 56.0336952
1: -22.6016197, 50.0412750, -10.9245014, 25.9543266, -48.5559425, 60.9657669
2: -19.3688354, 55.6726570, -9.3954067, 29.0555344, -48.4243698, 65.0680618
3: -21.3230705, 71.3164368, -10.3913212, 37.2941666, -58.6172371, 81.7077332
4: -17.8133068, 66.4069061, -8.9998016, 34.4906425, -52.3039474, 75.4067078

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1700867, upper bound: 60.2005176
time: 1.07 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2223030
time: 0.91 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2236349
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -16.1907825, 48.3972702, -11.4574566, 36.0637321, -52.2545166, 59.8547287
1: -22.5029907, 50.1119728, -16.2133541, 37.3637505, -59.8667336, 66.3253250
2: -19.2800102, 55.7299461, -13.9275732, 41.6031532, -60.8831635, 69.6575165
3: -21.2789536, 71.4950943, -15.2694445, 53.4530830, -74.7320404, 86.7645340
4: -17.8081436, 66.4556961, -13.0726910, 49.4179764, -67.2261200, 79.5283813

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2258380, upper bound: 60.2217572
time: 1.18 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2271129, upper bound: 60.2246950
time: 1.10 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -15.7770824, 47.5518494, -7.0539804, 22.8847084, -38.6617889, 54.6058311
1: -21.9844246, 49.1937943, -10.0291052, 23.7935123, -45.7779312, 59.2228966
2: -18.8414764, 54.7584229, -8.6115036, 26.6010494, -45.4425278, 63.3699265
3: -20.7717686, 70.2238922, -9.4927950, 34.1966286, -54.9683990, 79.7166748
4: -17.4064407, 65.2618256, -8.2850895, 31.6198654, -49.0263062, 73.5469131

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2105692, upper bound: 60.2226982
time: 0.95 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2105692, upper bound: 60.2226982
time: 0.94 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -15.8083849, 47.9627571, -11.2535267, 35.2861290, -51.0945129, 59.2162781
1: -22.0459614, 49.5936890, -15.9369659, 36.5871506, -58.6331100, 65.5306549
2: -18.8815460, 55.1911240, -13.6858082, 40.6975708, -59.5791168, 68.8769302
3: -20.8602009, 70.8628006, -15.0089293, 52.3045273, -73.1647263, 85.8717270
4: -17.5104294, 65.7194672, -12.8367405, 48.3964386, -65.9068604, 78.5562057

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2234409, upper bound: 60.2135330
time: 1.20 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2000478, upper bound: 60.2068350
time: 1.15 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -15.7770824, 47.5518494, -7.7260199, 24.9885464, -40.7656288, 55.2778625
1: -21.9844246, 49.1937943, -10.9245014, 25.9543266, -47.9387360, 60.1182938
2: -18.8414764, 54.7584229, -9.3954067, 29.0555344, -47.8970108, 64.1538315
3: -20.7717686, 70.2238922, -10.3913212, 37.2941666, -58.0659332, 80.6151733
4: -17.4064407, 65.2618256, -8.9998016, 34.4906425, -51.8970833, 74.2616119

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2077902, upper bound: 60.2198763
time: 0.97 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2077902, upper bound: 60.2213913
time: 1.32 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -15.8083849, 47.9627571, -11.4574566, 36.0637321, -51.8721161, 59.4202118
1: -22.0459614, 49.5936890, -16.2133541, 37.3637505, -59.4097137, 65.8070450
2: -18.8815460, 55.1911240, -13.9275732, 41.6031532, -60.4846992, 69.1186981
3: -20.8602009, 70.8628006, -15.2694445, 53.4530830, -74.3132858, 86.1322479
4: -17.5104294, 65.7194672, -13.0726910, 49.4179764, -66.9283981, 78.7921600

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213913, upper bound: 60.2097269
time: 0.97 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2213913, upper bound: 60.2229920
time: 1.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.36 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2193231
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2255980
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2237957, upper bound: 60.2280775
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2237089, upper bound: 60.2255980
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2164250
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2210528
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2164250, upper bound: 60.2096684
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2210528, upper bound: 60.2210528
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2196388, upper bound: 60.2116649
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2305228, upper bound: 60.2237089
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2275683, upper bound: 60.2197016
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2275683, upper bound: 60.2226870
IS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2176066, upper bound: 60.2096684
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2290138, upper bound: 60.2210528
IS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2176066, upper bound: 60.2096684
IS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2290138, upper bound: 60.2210528
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2169110, upper bound: 60.2349050
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2172152, upper bound: 60.2123798
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2108107, upper bound: 60.2328996
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2110529, upper bound: 60.2110529
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2223030
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2236349
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2258380, upper bound: 60.2217572
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2271129, upper bound: 60.2246950
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2105692, upper bound: 60.2226982
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2105692, upper bound: 60.2226982
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2234409, upper bound: 60.2135330
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2000478, upper bound: 60.2068350
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2077902, upper bound: 60.2198763
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2077902, upper bound: 60.2213913
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2213913, upper bound: 60.2097269
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.36
Output dim: 4, lower bound: -60.2213913, upper bound: 60.2229920

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.4992399, 27.1379700, -7.9862747, 25.6116962, -34.1109314, 35.1242447
1: -12.1326866, 28.2093182, -11.4230928, 26.6387024, -38.7713890, 39.6324120
2: -10.4494219, 31.4617252, -9.8432503, 29.7230167, -40.1724358, 41.3049774
3: -11.3772221, 40.4745598, -10.7079391, 38.2268753, -49.6040955, 51.1824951
4: -9.9179468, 37.4370956, -9.3737078, 35.3655739, -45.2835121, 46.8107948

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2171937, upper bound: 60.2179630
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2277106, upper bound: 60.2209870
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.4868965, 27.1095200, -9.4350300, 29.5662556, -38.0531502, 36.5445442
1: -12.1132841, 28.1768799, -13.3337088, 30.6914234, -42.8047066, 41.5105896
2: -10.4329882, 31.4311256, -11.4997158, 34.2629051, -44.6958923, 42.9308395
3: -11.3600817, 40.4455910, -12.5133295, 44.0571518, -55.4172325, 52.9589195
4: -9.9057350, 37.4076920, -10.8592606, 40.8404045, -50.7461395, 48.2669525

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166298, upper bound: 60.2177509
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2190320, upper bound: 60.2190320
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.9862747, 25.6116962, -9.7470121, 30.9525509, -38.9388275, 35.3587074
1: -11.4230928, 26.6387024, -13.8253012, 32.1093864, -43.5324783, 40.4640045
2: -9.8432503, 29.7230167, -11.9031410, 35.8449860, -45.6882362, 41.6261520
3: -10.7079391, 38.2268753, -13.0212603, 46.0756302, -56.7835655, 51.2481346
4: -9.3737078, 35.3655739, -11.2350693, 42.6007767, -51.9744797, 46.6006355

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2152022, upper bound: 60.2180588
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2186870
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2255980
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.4350300, 29.5662556, -9.7768440, 31.0364990, -40.4715271, 39.3430977
1: -13.3337088, 30.6914234, -13.8651476, 32.1935692, -45.5272789, 44.5565720
2: -11.4997158, 34.2629051, -11.9377117, 35.9426842, -47.4423981, 46.2006149
3: -12.5133295, 44.0571518, -13.0608997, 46.1977425, -58.7110710, 57.1180496
4: -10.8592606, 40.8404045, -11.2678738, 42.7165642, -53.5758247, 52.1082764

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2136025, upper bound: 60.2140661
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2193231
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2255980
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.7470121, 30.9525509, -7.9862747, 25.6116962, -35.3587074, 38.9388275
1: -13.8253012, 32.1093864, -11.4230928, 26.6387024, -40.4640045, 43.5324783
2: -11.9031410, 35.8449860, -9.8432503, 29.7230167, -41.6261520, 45.6882362
3: -13.0212603, 46.0756302, -10.7079391, 38.2268753, -51.2481346, 56.7835655
4: -11.2350693, 42.6007767, -9.3737078, 35.3655739, -46.6006393, 51.9744797

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180588, upper bound: 60.2152022
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2186870, upper bound: 60.2122686
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2186870, upper bound: 60.2237089
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.7768440, 31.0364990, -9.4350300, 29.5662556, -39.3431015, 40.4715271
1: -13.8651476, 32.1935692, -13.3337088, 30.6914234, -44.5565720, 45.5272789
2: -11.9377117, 35.9426842, -11.4997158, 34.2629051, -46.2006149, 47.4423981
3: -13.0608997, 46.1977425, -12.5133295, 44.0571518, -57.1180496, 58.7110710
4: -11.2678738, 42.7165642, -10.8592606, 40.8404045, -52.1082764, 53.5758247

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140661, upper bound: 60.2136025
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193231, upper bound: 60.2122686
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193231, upper bound: 60.2237089
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.1844368, 29.3177929, -9.7470121, 30.9525509, -40.1369820, 39.0648041
1: -13.0450573, 30.4253445, -13.8253012, 32.1093864, -45.1544418, 44.2506447
2: -11.2369299, 33.9735527, -11.9031410, 35.8449860, -47.0819168, 45.8766899
3: -12.2785158, 43.6895447, -13.0212603, 46.0756302, -58.3541451, 56.7108002
4: -10.6380224, 40.3852081, -11.2350693, 42.6007767, -53.2387962, 51.6202660

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2051175, upper bound: 60.2051175
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2051175, upper bound: 60.2096684
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -10.7788277, 33.7450409, -9.7768440, 31.0364990, -41.8153267, 43.5218849
1: -15.1692543, 34.9568481, -13.8651476, 32.1935692, -47.3628235, 48.8219948
2: -13.0496235, 39.0255585, -11.9377117, 35.9426842, -48.9923096, 50.9632721
3: -14.3286877, 50.1739922, -13.0608997, 46.1977425, -60.5264282, 63.2348900
4: -12.2625866, 46.4229164, -11.2678738, 42.7165642, -54.9791451, 57.6907883

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2164250
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2210528
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -7.9862747, 25.6116962, -16.5643120, 49.3280334, -57.3143082, 42.1760101
1: -11.4230928, 26.6387024, -23.0496235, 51.0870781, -62.5101585, 49.6883240
2: -9.8432503, 29.7230167, -19.7617702, 56.8106766, -66.6539307, 49.4847794
3: -10.7079391, 38.2268753, -21.7437840, 72.8406982, -83.5486374, 59.9706573
4: -9.3737078, 35.3655739, -18.1937695, 67.7535553, -77.1272659, 53.5593414

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2296196, upper bound: 60.2174505
time: 0.96 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2342505, upper bound: 60.2297677
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -9.4350300, 29.5662556, -16.5531044, 49.3135490, -58.7485733, 46.1193581
1: -13.3337088, 30.6914234, -23.0193863, 51.0690536, -64.4027557, 53.7108078
2: -11.4997158, 34.2629051, -19.7387733, 56.7981873, -68.2979050, 54.0016747
3: -12.5133295, 44.0571518, -21.7314377, 72.8408661, -85.3541870, 65.7885742
4: -10.8592606, 40.8404045, -18.1895523, 67.7264709, -78.5857315, 59.0299530

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2293752, upper bound: 60.2168994
time: 1.27 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2314943, upper bound: 60.2192972
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -9.0973234, 29.0640278, -16.5578880, 49.2752380, -58.3725624, 45.6219139
1: -12.8702602, 30.1397648, -23.0402069, 51.0306587, -63.9009056, 53.1799698
2: -11.0743856, 33.6567726, -19.7518291, 56.7543449, -67.8287277, 53.4085999
3: -12.1959352, 43.2404213, -21.7274742, 72.7368393, -84.9327774, 64.9678955
4: -10.4772635, 40.0022278, -18.1609097, 67.6860199, -78.1632690, 58.1631393

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2162045, upper bound: 60.2089952
time: 0.89 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2195181, upper bound: 60.2105720
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2243922, upper bound: 60.2197016
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -10.8696327, 34.4083939, -16.5639153, 49.4840279, -60.3536606, 50.9723091
1: -15.3343544, 35.6384964, -23.0278053, 51.2335739, -66.5679245, 58.6663017
2: -13.1785946, 39.7972412, -19.7227516, 56.9795685, -70.1581573, 59.5199928
3: -14.4879837, 51.1623955, -21.7862415, 73.0911560, -87.5791397, 72.9486389
4: -12.3919563, 47.3149300, -18.2028160, 67.9426193, -80.3345718, 65.5177460

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2201182, upper bound: 60.2127102
time: 1.22 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2257358, upper bound: 60.2226870
time: 1.14 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2275495, upper bound: 60.2226870
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -7.9862747, 25.6116962, -16.1934509, 48.8689537, -56.8552284, 41.8051453
1: -11.4230928, 26.6387024, -22.5819664, 50.5407982, -61.9638901, 49.2206688
2: -9.8432503, 29.7230167, -19.3590202, 56.2308998, -66.0741501, 49.0820351
3: -10.7079391, 38.2268753, -21.3255177, 72.1731033, -82.8810425, 59.5523911
4: -9.3737078, 35.3655739, -17.8931179, 67.0021591, -76.3758621, 53.2586861

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2224935, upper bound: 60.2114153
time: 1.18 seconds

## Relational analysis of IS_A1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2270183, upper bound: 60.2197249
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -9.4350300, 29.5662556, -16.1630669, 48.8086662, -58.2436905, 45.7293205
1: -13.3337088, 30.6914234, -22.5345764, 50.4753075, -63.8090172, 53.2259979
2: -11.4997158, 34.2629051, -19.3189430, 56.1682205, -67.6679382, 53.5818481
3: -12.5133295, 44.0571518, -21.2933598, 72.0947113, -84.6080322, 65.3504868
4: -10.8592606, 40.8404045, -17.8690834, 66.9156113, -77.7748718, 58.7094879

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226974, upper bound: 60.2108826
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2250038, upper bound: 60.2132856
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -9.1844368, 29.3177929, -16.1934509, 48.8689537, -58.0533829, 45.5112419
1: -13.0450573, 30.4253445, -22.5819664, 50.5407982, -63.5858536, 53.0073090
2: -11.2369299, 33.9735527, -19.3590202, 56.2308998, -67.4678268, 53.3325729
3: -12.2785158, 43.6895447, -21.3255177, 72.1731033, -84.4516068, 65.0150452
4: -10.6380224, 40.3852081, -17.8931179, 67.0021591, -77.6401825, 58.2783127

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2175072, upper bound: 60.2090269
time: 1.03 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131164, upper bound: 60.2029616
time: 1.25 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2146548, upper bound: 60.2075789
time: 0.85 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148975, upper bound: 60.2077670
time: 0.95 seconds

## Relational analysis of IS_A1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2165087, upper bound: 60.2083358
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -10.7788277, 33.7450409, -16.1630669, 48.8086662, -59.5874939, 49.9081039
1: -15.1692543, 34.9568481, -22.5345764, 50.4753075, -65.6445618, 57.4914246
2: -13.0496235, 39.0255585, -19.3189430, 56.1682205, -69.2178421, 58.3445015
3: -14.3286877, 50.1739922, -21.2933598, 72.0947113, -86.4234009, 71.4673309
4: -12.2625866, 46.4229164, -17.8690834, 66.9156113, -79.1781998, 64.2919998

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198734, upper bound: 60.2077902
time: 1.18 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2257717, upper bound: 60.2177615
time: 1.18 seconds

## Relational analysis of IS_A1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2286425, upper bound: 60.2207325
time: 1.01 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.2892036, 45.7070007, -9.4250584, 29.9569683, -45.2461700, 55.1320572
1: -21.1691895, 47.3431854, -13.4392796, 31.1022053, -52.2713928, 60.7824593
2: -18.2080307, 52.7248726, -11.5567131, 34.6557617, -52.8637886, 64.2815781
3: -20.0574760, 67.6858368, -12.6168146, 44.6343803, -64.6918488, 80.3026505
4: -16.9065418, 62.9016724, -10.9325199, 41.2190208, -58.1255646, 73.8341904

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2162716, upper bound: 60.2344827
time: 0.91 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166810, upper bound: 60.2336326
time: 0.95 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14.9437532, 44.7663155, -13.2609940, 41.1749611, -56.1187096, 58.0273094
1: -20.6960850, 46.3697548, -18.8325615, 42.6335030, -63.3295822, 65.2023163
2: -17.7965546, 51.6448059, -16.1723442, 47.4088593, -65.2054138, 67.8171539
3: -19.6097412, 66.3070831, -17.6479778, 60.8912392, -80.5009766, 83.9550552
4: -16.5461597, 61.6019287, -15.0162983, 56.3970909, -72.9432449, 76.6182251

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172153, upper bound: 60.2114666
time: 0.83 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172153, upper bound: 60.2123798
time: 1.09 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16.7311668, 49.8861961, -9.6451540, 30.5899220, -47.3210793, 59.5313492
1: -23.2865944, 51.6587791, -13.7480450, 31.7528801, -55.0394745, 65.4068222
2: -19.9622326, 57.4426842, -11.8179941, 35.3725891, -55.3348160, 69.2606659
3: -21.9739342, 73.6682281, -12.9046116, 45.5451889, -67.5191193, 86.5728378
4: -18.3817101, 68.4947433, -11.1606903, 42.0679893, -60.4496994, 79.6554260

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101112, upper bound: 60.2326470
time: 1.00 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2105183, upper bound: 60.2326470
time: 1.17 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16.3491402, 48.8458710, -13.4950256, 41.8496284, -58.1987686, 62.3408966
1: -22.7587852, 50.5755157, -19.1608982, 43.3308296, -66.0896149, 69.7364120
2: -19.5080147, 56.2410812, -16.4503098, 48.1778908, -67.6858978, 72.6913910
3: -21.4754448, 72.1546860, -17.9571533, 61.8611374, -83.3365784, 90.1118393
4: -17.9855785, 67.0533752, -15.2616196, 57.3078346, -75.2934113, 82.3149948

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110529, upper bound: 60.2107896
time: 1.03 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110529, upper bound: 60.2110529
time: 1.13 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -16.2499657, 48.3076897, -6.9233918, 22.3787384, -38.6287041, 55.2310715
1: -22.6016197, 50.0412750, -9.7522612, 23.2585583, -45.8601761, 59.7935295
2: -19.3688354, 55.6726570, -8.3892403, 26.0997849, -45.4686203, 64.0618973
3: -21.3230705, 71.3164368, -9.2856054, 33.4360733, -54.7591438, 80.6020432
4: -17.8133068, 66.4069061, -8.0783005, 31.0123272, -48.8256340, 74.4852066

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2121182, upper bound: 60.2219146
time: 0.99 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147328, upper bound: 60.2191903
time: 0.97 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -16.2499657, 48.3076897, -12.1081676, 36.9493027, -53.1992683, 60.4158554
1: -22.6016197, 50.0412750, -16.8258667, 38.2562523, -60.8578720, 66.8671417
2: -19.3688354, 55.6726570, -14.3537149, 42.7132225, -62.0820580, 70.0263748
3: -21.3230705, 71.3164368, -15.9871111, 54.6179428, -75.9410095, 87.3035431
4: -17.8133068, 66.4069061, -13.3867302, 50.8890648, -68.7023697, 79.7936401

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2112984, upper bound: 60.2113331
time: 1.04 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2112984, upper bound: 60.2113331
time: 0.99 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -14.7926102, 44.3845711, -11.2635708, 35.5039978, -50.2966003, 55.6481400
1: -20.4715405, 45.9750061, -15.9440355, 36.7868576, -57.2584000, 61.9190407
2: -17.5941029, 51.2100677, -13.6982613, 40.9666290, -58.5607300, 64.9083252
3: -19.4380913, 65.7397385, -15.0158882, 52.6477509, -72.0858307, 80.7556305
4: -16.3850689, 61.0765038, -12.8701077, 48.6614990, -65.0465698, 73.9466095

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155983, upper bound: 60.1977352
time: 1.08 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2068284, upper bound: 60.1988038
time: 0.98 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -16.1077728, 48.2076874, -11.4574566, 36.0637321, -52.1715012, 59.6651459
1: -22.3930321, 49.9144783, -16.2133541, 37.3637505, -59.7567825, 66.1278305
2: -19.1851158, 55.5127258, -13.9275732, 41.6031532, -60.7882690, 69.4403000
3: -21.1835976, 71.2213974, -15.2694445, 53.4530830, -74.6366806, 86.4908371
4: -17.7267532, 66.1934509, -13.0726910, 49.4179764, -67.1447296, 79.2661438

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2130309, upper bound: 60.2103573
time: 1.04 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2130309, upper bound: 60.2246950
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -15.7770824, 47.5518494, -5.9589076, 19.3780518, -35.1551361, 53.5107536
1: -21.9844246, 49.1937943, -8.4746065, 20.1900005, -42.1744118, 57.6683960
2: -18.8414764, 54.7584229, -7.2950258, 22.6231365, -41.4646149, 62.0534477
3: -20.7717686, 70.2238922, -7.9993563, 28.9849606, -49.7567291, 78.2232285
4: -17.4064407, 65.2618256, -7.0772061, 26.8955078, -44.3019485, 72.3390350

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2055348, upper bound: 60.2221444
time: 1.00 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2101746, upper bound: 60.2193335
time: 1.03 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -15.7770824, 47.5518494, -12.9136457, 38.5023308, -54.2794113, 60.4654961
1: -21.9844246, 49.1937943, -17.8915863, 39.9260864, -61.9105072, 67.0853806
2: -18.8414764, 54.7584229, -15.2458172, 44.5763283, -63.4178047, 70.0042343
3: -20.7717686, 70.2238922, -16.9759617, 56.7354393, -77.5071716, 87.1998444
4: -17.4064407, 65.2618256, -14.0684938, 53.1006775, -70.5071182, 79.3303070

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103105, upper bound: 60.2132153
time: 0.99 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103105, upper bound: 60.2226982
time: 1.04 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -15.3316879, 46.4681969, -11.2535267, 35.2861290, -50.6178169, 57.7217216
1: -21.3805065, 48.0620651, -15.9369659, 36.5871506, -57.9676590, 63.9990196
2: -18.3118153, 53.4990005, -13.6858082, 40.6975708, -59.0093842, 67.1848068
3: -20.2252483, 68.6554031, -15.0089293, 52.3045273, -72.5297775, 83.6643143
4: -16.9803734, 63.7066193, -12.8367405, 48.3964386, -65.3768158, 76.5433578

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2133592, upper bound: 60.2135051
time: 1.24 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2234409, upper bound: 60.2135330
time: 1.18 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -19.2769279, 58.0226402, -11.0744286, 34.7997246, -54.0766411, 69.0970688
1: -26.9978714, 59.9640846, -15.6995583, 36.0881653, -63.0860291, 75.6636429
2: -23.0902023, 66.6789703, -13.4848223, 40.1409950, -63.2311897, 80.1637878
3: -25.3416634, 85.4448318, -14.7826996, 51.5950851, -76.9367447, 100.2275162
4: -21.1887035, 79.3472519, -12.6566753, 47.7300720, -68.9187775, 92.0039291

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1989739, upper bound: 60.2068350
time: 1.04 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1989739, upper bound: 60.2068350
time: 1.05 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -15.7770824, 47.5518494, -6.9233918, 22.3787384, -38.1558189, 54.4752388
1: -21.9844246, 49.1937943, -9.7522612, 23.2585583, -45.2429771, 58.9460526
2: -18.8414764, 54.7584229, -8.3892403, 26.0997849, -44.9412613, 63.1476631
3: -20.7717686, 70.2238922, -9.2856054, 33.4360733, -54.2078400, 79.5094910
4: -17.4064407, 65.2618256, -8.0783005, 31.0123272, -48.4187698, 73.3401184

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2027902, upper bound: 60.2194630
time: 1.01 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2073641, upper bound: 60.2167034
time: 1.07 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -15.7770824, 47.5518494, -12.1081676, 36.9493027, -52.7263832, 59.6600189
1: -21.9844246, 49.1937943, -16.8258667, 38.2562523, -60.2406769, 66.0196609
2: -18.8414764, 54.7584229, -14.3537149, 42.7132225, -61.5546989, 69.1121368
3: -20.7717686, 70.2238922, -15.9871111, 54.6179428, -75.3896866, 86.2109985
4: -17.4064407, 65.2618256, -13.3867302, 50.8890648, -68.2955017, 78.6485596

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2075010, upper bound: 60.2104464
time: 0.93 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2075010, upper bound: 60.2104464
time: 1.11 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -13.5856018, 40.9604416, -11.4574566, 36.0637321, -49.6493340, 52.4179001
1: -18.8775501, 42.3891830, -16.2133541, 37.3637505, -56.2412949, 58.6025314
2: -16.1569481, 47.3286133, -13.9275732, 41.6031532, -57.7601013, 61.2561836
3: -17.8854637, 60.3969116, -15.2694445, 53.4530830, -71.3385468, 75.6663513
4: -14.9399548, 56.4018059, -13.0726910, 49.4179764, -64.3579254, 69.4744949

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1859071, upper bound: 60.2013552
time: 1.16 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1775166, upper bound: 60.1808941
time: 0.94 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -17.6552353, 53.2634773, -11.4574566, 36.0637321, -53.7189636, 64.7209320
1: -24.5836754, 55.0594139, -16.2133541, 37.3637505, -61.9474220, 71.2727661
2: -21.0160446, 61.2094574, -13.9275732, 41.6031532, -62.6191978, 75.1370316
3: -23.2941093, 78.4945526, -15.2694445, 53.4530830, -76.7471924, 93.7639999
4: -19.4026127, 72.9213943, -13.0726910, 49.4179764, -68.8205795, 85.9940872

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1859071, upper bound: 60.2102002
time: 1.14 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1775166, upper bound: 60.1889455
time: 1.04 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 12.75 seconds
IS_A1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2171937, upper bound: 60.2179630
IS_A1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2277106, upper bound: 60.2209870
IS_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2166298, upper bound: 60.2177509
IS_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2190320, upper bound: 60.2190320
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2186870
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2255980
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2193231
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2122686, upper bound: 60.2255980
IS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2186870, upper bound: 60.2122686
IS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2186870, upper bound: 60.2237089
IS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2193231, upper bound: 60.2122686
IS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2193231, upper bound: 60.2237089
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2051175, upper bound: 60.2051175
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2051175, upper bound: 60.2096684
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2164250
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2096684, upper bound: 60.2210528
IS_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2296196, upper bound: 60.2174505
IS_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2342505, upper bound: 60.2297677
IS_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2293752, upper bound: 60.2168994
IS_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2314943, upper bound: 60.2192972
IS_A1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2195181, upper bound: 60.2105720
IS_A1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2243922, upper bound: 60.2197016
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2257358, upper bound: 60.2226870
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2275495, upper bound: 60.2226870
IS_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2224935, upper bound: 60.2114153
IS_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2270183, upper bound: 60.2197249
IS_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2226974, upper bound: 60.2108826
IS_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2250038, upper bound: 60.2132856
IS_A1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2148975, upper bound: 60.2077670
IS_A1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2165087, upper bound: 60.2083358
IS_A1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2257717, upper bound: 60.2177615
IS_A1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2286425, upper bound: 60.2207325
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2162716, upper bound: 60.2344827
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2166810, upper bound: 60.2336326
IS_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2172153, upper bound: 60.2114666
IS_A2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2172153, upper bound: 60.2123798
IS_A2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2101112, upper bound: 60.2326470
IS_A2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2105183, upper bound: 60.2326470
IS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2110529, upper bound: 60.2107896
IS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2110529, upper bound: 60.2110529
IS_A2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2121182, upper bound: 60.2219146
IS_A2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2147328, upper bound: 60.2191903
IS_A2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2112984, upper bound: 60.2113331
IS_A2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2112984, upper bound: 60.2113331
IS_A2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2155983, upper bound: 60.1977352
IS_A2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2068284, upper bound: 60.1988038
IS_A2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2130309, upper bound: 60.2103573
IS_A2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2130309, upper bound: 60.2246950
IS_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2055348, upper bound: 60.2221444
IS_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2101746, upper bound: 60.2193335
IS_A2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2103105, upper bound: 60.2132153
IS_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2103105, upper bound: 60.2226982
IS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2133592, upper bound: 60.2135051
IS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2234409, upper bound: 60.2135330
IS_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.1989739, upper bound: 60.2068350
IS_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.1989739, upper bound: 60.2068350
IS_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2027902, upper bound: 60.2194630
IS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2073641, upper bound: 60.2167034
IS_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2075010, upper bound: 60.2104464
IS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.2075010, upper bound: 60.2104464
IS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.1859071, upper bound: 60.2013552
IS_A2_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.1775166, upper bound: 60.1808941
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.1859071, upper bound: 60.2102002
IS_A2_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 12.75
Output dim: 4, lower bound: -60.1775166, upper bound: 60.1889455

## BFS IS instance: IS_A1_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -7.8472180, 25.1719513, -5.3627934, 17.5434265, -25.3906441, 30.5347443
1: -11.2021017, 26.1843300, -7.6456523, 18.2880783, -29.4901810, 33.8299828
2: -9.6511488, 29.2291622, -6.5861640, 20.5284653, -30.1796074, 35.8153191
3: -10.5126266, 37.5677643, -7.2171130, 26.2792511, -36.7918739, 44.7848778
4: -9.1961346, 34.7747688, -6.4380808, 24.4010353, -33.5971642, 41.2128448

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2159310, upper bound: 60.2155500
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2159310, upper bound: 60.2179630
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -7.9972563, 25.7931309, -8.9393549, 28.3548870, -36.3521423, 34.7324829
1: -11.4319620, 26.8172150, -12.6786098, 29.4496670, -40.8816299, 39.4958191
2: -9.8379517, 29.9220982, -10.8863373, 32.8398857, -42.6778374, 40.8084297
3: -10.7550526, 38.5088615, -11.9512005, 42.0846939, -52.8397446, 50.4600601
4: -9.3799458, 35.5931625, -10.3152943, 39.0431786, -48.4231186, 45.9084549

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2274530, upper bound: 60.2202977
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2272365, upper bound: 60.2204154
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -7.8330355, 25.1413860, -6.7993159, 21.1920853, -29.0251198, 31.9407005
1: -11.1798468, 26.1492443, -9.4736481, 22.0555534, -33.2353973, 35.6228867
2: -9.6325722, 29.1960659, -8.1622896, 24.7513580, -34.3839302, 37.3583527
3: -10.4928989, 37.5369987, -8.9919262, 31.6601238, -42.1530190, 46.5289192
4: -9.1819553, 34.7410088, -7.8281941, 29.5022087, -38.6841583, 42.5692024

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153347, upper bound: 60.2153347
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153347, upper bound: 60.2177509
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -7.9763966, 25.7359867, -9.9835091, 31.2488327, -39.2252235, 35.7194939
1: -11.4007654, 26.7545128, -14.0577354, 32.4243736, -43.8251305, 40.8122482
2: -9.8126001, 29.8575878, -12.0938091, 36.1974792, -46.0100784, 41.9513931
3: -10.7254772, 38.4353828, -13.2860537, 46.4408951, -57.1663704, 51.7214355
4: -9.3587818, 35.5231628, -11.4175577, 43.0951729, -52.4539490, 46.9407196

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2177509, upper bound: 60.2166298
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2177509, upper bound: 60.2190320
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -7.9862747, 25.6116962, -9.1844368, 29.3177929, -37.3040657, 34.7961311
1: -11.4230928, 26.6387024, -13.0450573, 30.4253445, -41.8484383, 39.6837616
2: -9.8432503, 29.7230167, -11.2369299, 33.9735527, -43.8168030, 40.9599457
3: -10.7079391, 38.2268753, -12.2785158, 43.6895447, -54.3974762, 50.5053902
4: -9.3737078, 35.3655739, -10.6380224, 40.3852081, -49.7589073, 46.0035934

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2079882, upper bound: 60.2077968
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2119669, upper bound: 60.2183731
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -7.9862747, 25.6116962, -10.7788277, 33.7450409, -41.7313156, 36.3905258
1: -11.4230928, 26.6387024, -15.1692543, 34.9568481, -46.3799400, 41.8079567
2: -9.8432503, 29.7230167, -13.0496235, 39.0255585, -48.8688087, 42.7726326
3: -10.7079391, 38.2268753, -14.3286877, 50.1739922, -60.8819275, 52.5555649
4: -9.3737078, 35.3655739, -12.2625866, 46.4229164, -55.7966118, 47.6281548

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2079882, upper bound: 60.2103733
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2079882, upper bound: 60.2279233
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.4350300, 29.5662556, -9.1844368, 29.3177929, -38.7528152, 38.7506866
1: -13.3337088, 30.6914234, -13.0450573, 30.4253445, -43.7590523, 43.7364807
2: -11.4997158, 34.2629051, -11.2369299, 33.9735527, -45.4732666, 45.4998360
3: -12.5133295, 44.0571518, -12.2785158, 43.6895447, -56.2028694, 56.3356667
4: -10.8592606, 40.8404045, -10.6380224, 40.3852081, -51.2444687, 51.4784241

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1999430, upper bound: 60.2046593
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2054027, upper bound: 60.2149541
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2107276, upper bound: 60.2191972
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2004593, upper bound: 60.2065683
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2019068, upper bound: 60.2084208
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2105544, upper bound: 60.2181495
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.4350300, 29.5662556, -10.7788277, 33.7450409, -43.1800652, 40.3450851
1: -13.3337088, 30.6914234, -15.1692543, 34.9568481, -48.2905579, 45.8606796
2: -11.4997158, 34.2629051, -13.0496235, 39.0255585, -50.5252762, 47.3125267
3: -12.5133295, 44.0571518, -14.3286877, 50.1739922, -62.6873131, 58.3858414
4: -10.8592606, 40.8404045, -12.2625866, 46.4229164, -57.2821770, 53.1029892

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1999430, upper bound: 60.2098238
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2107276, upper bound: 60.2217755
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2054027, upper bound: 60.2152515
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2004593, upper bound: 60.2065683
time: 1.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2019068, upper bound: 60.2084208
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2105544, upper bound: 60.2209057
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -9.1844368, 29.3177929, -7.9862747, 25.6116962, -34.7961311, 37.3040657
1: -13.0450573, 30.4253445, -11.4230928, 26.6387024, -39.6837616, 41.8484383
2: -11.2369299, 33.9735527, -9.8432503, 29.7230167, -40.9599457, 43.8168030
3: -12.2785158, 43.6895447, -10.7079391, 38.2268753, -50.5053902, 54.3974800
4: -10.6380224, 40.3852081, -9.3737078, 35.3655739, -46.0035934, 49.7589035

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2077968, upper bound: 60.2079882
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183731, upper bound: 60.2119669
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -10.7788277, 33.7450409, -7.9862747, 25.6116962, -36.3905258, 41.7313156
1: -15.1692543, 34.9568481, -11.4230928, 26.6387024, -41.8079567, 46.3799400
2: -13.0496235, 39.0255585, -9.8432503, 29.7230167, -42.7726326, 48.8688087
3: -14.3286877, 50.1739922, -10.7079391, 38.2268753, -52.5555649, 60.8819237
4: -12.2625866, 46.4229164, -9.3737078, 35.3655739, -47.6281548, 55.7966118

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2077968, upper bound: 60.2113389
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2077968, upper bound: 60.2235483
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -9.1844368, 29.3177929, -9.4350300, 29.5662556, -38.7506866, 38.7528114
1: -13.0450573, 30.4253445, -13.3337088, 30.6914234, -43.7364807, 43.7590523
2: -11.2369299, 33.9735527, -11.4997158, 34.2629051, -45.4998360, 45.4732666
3: -12.2785158, 43.6895447, -12.5133295, 44.0571518, -56.3356628, 56.2028694
4: -10.6380224, 40.3852081, -10.8592606, 40.8404045, -51.4784241, 51.2444687

Time for backsubstitution: 2.03 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=65.54161834716797
rel_dist={4: [-60.236541379106946, 60.23654137910691]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2347258, upper bound: 60.2334202
time: 0.86 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2347258, upper bound: 60.2347258
time: 1.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.04
Output dim: 4, lower bound: -60.2347258, upper bound: 60.2334202
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.04
Output dim: 4, lower bound: -60.2347258, upper bound: 60.2347258

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -10.2668142, 32.2093468, -11.1193314, 34.7366142, -45.0034294, 43.3286781
1: -14.5713062, 33.4432640, -15.7631454, 36.0405502, -50.6118546, 49.2064095
2: -12.5309057, 37.2433014, -13.5458183, 40.1080513, -52.6389580, 50.7891121
3: -13.6885414, 47.8052521, -14.8123779, 51.5176849, -65.2062225, 62.6176300
4: -11.7807655, 44.3089943, -12.6898689, 47.7126083, -59.4933739, 56.9988632

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334202, upper bound: 60.2334202
time: 1.15 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2334202, upper bound: 60.2334202
time: 0.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -18.7248726, 55.4986000, -11.8347521, 36.8566246, -55.5814934, 67.3333511
1: -26.1227779, 57.4683647, -16.7644978, 38.2160301, -64.3388062, 74.2328491
2: -22.3536682, 63.8967094, -14.3975792, 42.5019989, -64.8556671, 78.2942886
3: -24.5606728, 81.6774292, -15.7618332, 54.6204414, -79.1811142, 97.4392624
4: -20.4283028, 76.1416473, -13.4606457, 50.5488243, -70.9771194, 89.6022949

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2212878
time: 0.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2327125
time: 0.95 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 4, lower bound: -60.2334202, upper bound: 60.2334202
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 4, lower bound: -60.2334202, upper bound: 60.2334202
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2212878
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2327125

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -10.2668142, 32.2093468, -10.2668142, 32.2093468, -42.4761581, 42.4761581
1: -14.5713062, 33.4432640, -14.5713062, 33.4432640, -48.0145721, 48.0145721
2: -12.5309057, 37.2433014, -12.5309057, 37.2433014, -49.7742081, 49.7742081
3: -13.6885414, 47.8052521, -13.6885414, 47.8052521, -61.4937935, 61.4937935
4: -11.7807655, 44.3089943, -11.7807655, 44.3089943, -56.0897598, 56.0897598

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2190912
time: 1.27 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308247, upper bound: 60.2307118
time: 0.99 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -10.2668142, 32.2093468, -18.7248325, 55.4984283, -65.7652435, 50.9341660
1: -14.5713062, 33.4432640, -26.1227226, 57.4682007, -72.0395050, 59.5659866
2: -12.5309057, 37.2433014, -22.3536282, 63.8965302, -76.4274368, 59.5969315
3: -13.6885414, 47.8052521, -24.5606079, 81.6771774, -95.3657227, 72.3658600
4: -11.7807655, 44.3089943, -20.4282570, 76.1414261, -87.9221878, 64.7372513

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2233647, upper bound: 60.2181292
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308248, upper bound: 60.2307118
time: 1.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -17.4155636, 51.6655502, -8.5632610, 27.1309204, -44.5464859, 60.2288132
1: -24.2821426, 53.5415878, -12.1202679, 28.1951675, -52.4773102, 65.6618576
2: -20.7646255, 59.5434418, -10.3797131, 31.4654942, -52.2301140, 69.9231567
3: -22.8374710, 75.8678131, -11.4786844, 40.2979088, -63.1353798, 87.3464890
4: -18.9766026, 70.9118576, -9.8491869, 37.4059792, -56.3825836, 80.7610474

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2151669, upper bound: 60.2167097
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
time: 1.09 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -17.6393681, 52.5239105, -12.9824114, 40.2203407, -57.8597069, 65.5063248
1: -24.5615463, 54.3826561, -18.3470001, 41.6900520, -66.2516022, 72.7296600
2: -20.9898109, 60.4339905, -15.7394218, 46.3124428, -67.3022385, 76.1734009
3: -23.1825333, 77.3704529, -17.2892818, 59.4044456, -82.5869751, 94.6597366
4: -19.2832451, 72.0513992, -14.6737556, 55.0842934, -74.3675385, 86.7251587

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212878, upper bound: 60.2166958
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212878, upper bound: 60.2327125
time: 1.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.65 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2190912
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 4, lower bound: -60.2308247, upper bound: 60.2307118
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 4, lower bound: -60.2233647, upper bound: 60.2181292
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 4, lower bound: -60.2308248, upper bound: 60.2307118
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 4, lower bound: -60.2151669, upper bound: 60.2167097
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 4, lower bound: -60.2212878, upper bound: 60.2166958
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.65
Output dim: 4, lower bound: -60.2212878, upper bound: 60.2327125

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.6757097, 27.4309902, -7.5174160, 23.8916359, -32.5673409, 34.9484062
1: -12.3062782, 28.5243645, -10.6438217, 24.8755379, -37.1818161, 39.1681862
2: -10.5774956, 31.8314247, -9.1326923, 27.8068905, -38.3843842, 40.9641113
3: -11.5997849, 40.7548523, -10.0773335, 35.5273323, -47.1271172, 50.8321838
4: -10.0222235, 37.8564339, -8.7215147, 33.0632362, -43.0854568, 46.5779495

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155360, upper bound: 60.2155360
time: 1.17 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2155360, upper bound: 60.2234027
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.5097847, 30.0959129, -11.4300442, 35.7608910, -45.2706757, 41.5259399
1: -13.5123158, 31.2646255, -16.1550903, 37.1057549, -50.6180725, 47.4197159
2: -11.6191063, 34.8305779, -13.8617659, 41.2794495, -52.8985481, 48.6923370
3: -12.7174416, 44.7178841, -15.2650948, 52.8817329, -65.5991669, 59.9829750
4: -10.9733906, 41.4255791, -13.0084991, 49.0694733, -60.0428619, 54.4340744

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2300869, upper bound: 60.2286400
time: 0.95 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2282501, upper bound: 60.2282501
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -7.5174160, 23.8916359, -17.4155636, 51.6655502, -59.1829681, 41.3071938
1: -10.6438217, 24.8755379, -24.2821426, 53.5415878, -64.1854095, 49.1576805
2: -9.1326923, 27.8068905, -20.7646255, 59.5434418, -68.6761322, 48.5715103
3: -10.0773335, 35.5273323, -22.8374710, 75.8678131, -85.9451447, 58.3648033
4: -8.7215147, 33.0632362, -18.9766026, 70.9118576, -79.6333618, 52.0398407

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2226957, upper bound: 60.2166601
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2223812, upper bound: 60.2161304
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.4300442, 35.7608910, -17.6393356, 52.5237694, -63.9538116, 53.4002266
1: -16.1550903, 37.1057549, -24.5615005, 54.3825073, -70.5375977, 61.6672554
2: -13.8617659, 41.2794495, -20.9897709, 60.4338264, -74.2955780, 62.2692184
3: -15.2650948, 52.8817329, -23.1824837, 77.3702469, -92.6353378, 76.0642166
4: -13.0084991, 49.0694733, -19.2832146, 72.0512314, -85.0597305, 68.3526917

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2291707, upper bound: 60.2261647
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2254053, upper bound: 60.2185553
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15.4560003, 45.8631821, -7.6549187, 24.5610104, -40.0170059, 53.5180969
1: -21.4846230, 47.5241013, -10.8635321, 25.5313263, -47.0159492, 58.3876343
2: -18.4006214, 52.9451103, -9.3170576, 28.5186024, -46.9192162, 62.2621613
3: -20.2751713, 67.6545715, -10.2797136, 36.6038475, -56.8790207, 77.9342804
4: -16.9117584, 63.1266060, -8.9067688, 33.9066200, -50.8183784, 72.0333710

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -14.8257456, 44.6510544, -7.1857576, 23.3562603, -38.1819954, 51.8368034
1: -20.6330700, 46.2077293, -10.2321997, 24.2863312, -44.9193993, 56.4399300
2: -17.6738434, 51.5202065, -8.7829666, 27.1435585, -44.8174019, 60.3031654
3: -19.5196896, 65.9270401, -9.7133331, 34.8634567, -54.3831406, 75.6403656
4: -16.3241749, 61.3931503, -8.4483280, 32.2411880, -48.5653610, 69.8414764

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -16.2717113, 48.4006004, -12.9824114, 40.2203407, -56.4920502, 61.3830109
1: -22.6913528, 50.1986313, -18.3470001, 41.6900520, -64.3814087, 68.5456314
2: -19.3859882, 55.8040314, -15.7394218, 46.3124428, -65.6984253, 71.5434341
3: -21.3407516, 70.9930115, -17.2892818, 59.4044456, -80.7451935, 88.2822952
4: -17.7321625, 66.4244690, -14.6737556, 55.0842934, -72.8164520, 81.0982132

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127224, upper bound: 60.2099880
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2089690, upper bound: 60.2091787
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.2108002, 60.1085243, -12.9824114, 40.2203407, -60.4311409, 73.0909348
1: -28.2083969, 62.2145920, -18.3470001, 41.6900520, -69.8984528, 80.5615921
2: -24.0790520, 69.0647659, -15.7394218, 46.3124428, -70.3914948, 84.8041840
3: -26.5538406, 88.0503006, -17.2892818, 59.4044456, -85.9582825, 105.3395844
4: -21.9430542, 82.2009811, -14.6737556, 55.0842934, -77.0273438, 96.8747330

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127224, upper bound: 60.2193438
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2089690, upper bound: 60.2181831
time: 1.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 5.99 seconds
IS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2155360, upper bound: 60.2155360
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2155360, upper bound: 60.2234027
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2300869, upper bound: 60.2286400
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2282501, upper bound: 60.2282501
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2226957, upper bound: 60.2166601
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2223812, upper bound: 60.2161304
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2291707, upper bound: 60.2261647
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2254053, upper bound: 60.2185553
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2091787, upper bound: 60.2154803
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2127224, upper bound: 60.2099880
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2089690, upper bound: 60.2091787
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2127224, upper bound: 60.2193438
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 5.99
Output dim: 4, lower bound: -60.2089690, upper bound: 60.2181831

## BFS IS instance: IS_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -7.5174160, 23.8916359, -7.5174160, 23.8916359, -31.4090500, 31.4090481
1: -10.6438217, 24.8755379, -10.6438217, 24.8755379, -35.5193596, 35.5193596
2: -9.1326923, 27.8068905, -9.1326923, 27.8068905, -36.9395752, 36.9395752
3: -10.0773335, 35.5273323, -10.0773335, 35.5273323, -45.6046677, 45.6046677
4: -8.7215147, 33.0632362, -8.7215147, 33.0632362, -41.7847519, 41.7847519

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148342, upper bound: 60.2153583
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147674, upper bound: 60.2147674
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -11.3529682, 35.4878502, -7.5174160, 23.8916359, -35.2445946, 43.0052643
1: -16.0524940, 36.8420067, -10.6438217, 24.8755379, -40.9280319, 47.4858246
2: -13.7718735, 40.9950600, -9.1326923, 27.8068905, -41.5787582, 50.1277542
3: -15.1633635, 52.4761314, -10.0773335, 35.5273323, -50.6906967, 62.5534668
4: -12.9175920, 48.7452736, -8.7215147, 33.0632362, -45.9808273, 57.4667892

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148342, upper bound: 60.2205742
time: 0.91 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147674, upper bound: 60.2172080
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -9.0871582, 28.8605576, -10.0303469, 31.6403484, -40.7275085, 38.8908958
1: -12.9181900, 29.9934216, -14.1784630, 32.8629951, -45.7811813, 44.1718826
2: -11.1144028, 33.4281578, -12.2023067, 36.6056366, -47.7200394, 45.6304626
3: -12.1575174, 42.9476128, -13.3751621, 47.0019455, -59.1594543, 56.3227768
4: -10.5257168, 39.7624016, -11.5354128, 43.5362358, -54.0619507, 51.2978134

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2266130, upper bound: 60.2246665
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2264602, upper bound: 60.2247941
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -9.5097847, 30.0959129, -11.3660059, 35.6128006, -45.1225853, 41.4619102
1: -13.5123158, 31.2646255, -16.0665855, 36.9516716, -50.4639893, 47.3312111
2: -11.6191063, 34.8305779, -13.7866611, 41.1097527, -52.7288589, 48.6172333
3: -12.7174416, 44.7178841, -15.1903563, 52.6686401, -65.3860703, 59.9082413
4: -10.9733906, 41.4255791, -12.9449902, 48.8660202, -59.8394089, 54.3705597

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2228096, upper bound: 60.2193377
time: 1.17 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180197, upper bound: 60.2180197
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -6.6845942, 21.3685608, -16.9088936, 50.1850510, -56.8696442, 38.2774506
1: -9.4924765, 22.2724018, -23.5766792, 52.0250244, -61.5175018, 45.8490677
2: -8.1440449, 24.9384079, -20.1663933, 57.8534927, -65.9975357, 45.1048012
3: -8.9932966, 31.8456116, -22.1773701, 73.6778564, -82.6711349, 54.0229797
4: -7.8372211, 29.6645374, -18.4417000, 68.8992615, -76.7364731, 48.1062355

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2195334, upper bound: 60.2147086
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2141019, upper bound: 60.2146268
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -7.9846687, 24.6592941, -16.9599133, 50.2500114, -58.2346802, 41.6192093
1: -11.1343088, 25.6553383, -23.6243591, 52.0743599, -63.2086678, 49.2796974
2: -9.5540485, 28.7395229, -20.2054214, 57.9575462, -67.5115967, 48.9449463
3: -10.5845137, 36.6657715, -22.2469006, 73.8490982, -84.4336090, 58.9126740
4: -9.0737028, 34.2209816, -18.4900475, 69.0422592, -78.1159592, 52.7110291

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148202, upper bound: 60.2121888
time: 1.11 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134760, upper bound: 60.2072158
time: 1.13 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -10.5829353, 33.1954765, -17.1321030, 50.9958344, -61.5787697, 50.3275719
1: -14.9808502, 34.4668999, -23.8587456, 52.8068810, -67.7877350, 58.3256416
2: -12.8593082, 38.3798714, -20.3928566, 58.7107620, -71.5700607, 58.7727280
3: -14.1552792, 49.1394577, -22.5209789, 75.1248322, -89.2801132, 71.6604233
4: -12.1076508, 45.6256638, -18.7354202, 70.0037231, -82.1113739, 64.3610764

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2190911
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2261647
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -11.4920979, 35.5845413, -17.1756935, 51.1622963, -62.6543922, 52.7602348
1: -16.1414547, 36.9067802, -23.8904209, 52.9810448, -69.1224976, 60.7971992
2: -13.8512897, 41.1478271, -20.4222679, 58.9046860, -72.7559662, 61.5700874
3: -15.2957268, 52.6893082, -22.5863800, 75.4269638, -90.7226868, 75.2756882
4: -12.9969368, 48.9727364, -18.8009186, 70.2296448, -83.2265778, 67.7736511

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156971, upper bound: 60.2163977
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2156971, upper bound: 60.2184175
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.4560003, 45.8631821, -6.9620099, 22.6051159, -38.0611153, 52.8251877
1: -21.4846230, 47.5241013, -9.8951836, 23.5049248, -44.9895439, 57.4192810
2: -18.4006214, 52.9451103, -8.4977646, 26.2823906, -44.6830025, 61.4428711
3: -20.2751713, 67.6545715, -9.3681488, 33.7857666, -54.0609360, 77.0227203
4: -16.9117584, 63.1266060, -8.1843386, 31.2388859, -48.1506424, 71.3109436

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2151669, upper bound: 60.2137186
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148282, upper bound: 60.2167097
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.4560003, 45.8631821, -7.4715652, 24.2221375, -39.6781311, 53.3347435
1: -21.4846230, 47.5241013, -10.5729570, 25.1648464, -46.6494637, 58.0970573
2: -18.4006214, 52.9451103, -9.0967140, 28.1707993, -46.5714149, 62.0418205
3: -20.2751713, 67.6545715, -10.0565710, 36.1666870, -56.4418564, 77.7111359
4: -16.9117584, 63.1266060, -8.7295208, 33.4408646, -50.3526192, 71.8561249

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2151669, upper bound: 60.2137186
time: 1.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148282, upper bound: 60.2167097
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.8257456, 44.6510544, -6.9620099, 22.6051159, -37.4308624, 51.6130562
1: -20.6330700, 46.2077293, -9.8951836, 23.5049248, -44.1379929, 56.1029091
2: -17.6738434, 51.5202065, -8.4977646, 26.2823906, -43.9562340, 60.0179710
3: -19.5196896, 65.9270401, -9.3681488, 33.7857666, -53.3054466, 75.2951889
4: -16.3241749, 61.3931503, -8.1843386, 31.2388859, -47.5630608, 69.5774918

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1977769, upper bound: 60.2083271
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2088190, upper bound: 60.2150075
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.8257456, 44.6510544, -7.4896488, 24.2770329, -39.1027756, 52.1407013
1: -20.6330700, 46.2077293, -10.5982838, 25.2219067, -45.8549767, 56.8060150
2: -17.6738434, 51.5202065, -9.1184864, 28.2342014, -45.9080429, 60.6386948
3: -19.5196896, 65.9270401, -10.0807705, 36.2469673, -55.7666512, 76.0078049
4: -16.3241749, 61.3931503, -8.7494125, 33.5158195, -49.8399963, 70.1425629

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.1977769, upper bound: 60.2105083
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2088190, upper bound: 60.2150075
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -14.3323498, 42.5070267, -11.9912415, 37.3878212, -51.7201614, 54.4982681
1: -19.8979225, 44.0862541, -16.9639797, 38.7631836, -58.6611061, 61.0502319
2: -17.0317307, 49.1434784, -14.5649471, 43.0888557, -60.1205864, 63.7084274
3: -18.7912884, 62.5671997, -15.9817076, 55.3257751, -74.1170578, 78.5489044
4: -15.6619244, 58.5765915, -13.6241512, 51.2421646, -66.9040909, 72.2007370

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154802, upper bound: 60.2091787
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154802, upper bound: 60.2091787
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -13.5843573, 40.9571800, -11.3791924, 35.7151070, -49.2994652, 52.3363686
1: -18.8757629, 42.3858299, -16.1628551, 37.0363083, -55.9120674, 58.5486832
2: -16.1553764, 47.3249359, -13.8870125, 41.1730003, -57.3283768, 61.2119446
3: -17.8838730, 60.3922539, -15.2091036, 52.8894081, -70.7732697, 75.6013565
4: -14.9385757, 56.3974495, -13.0088778, 48.9394493, -63.8780136, 69.4063263

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154802, upper bound: 60.2091787
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2154802, upper bound: 60.2091787
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -18.2456055, 54.2350540, -11.9912415, 37.3878212, -55.6334229, 66.2262955
1: -25.4212532, 56.1403465, -16.9639797, 38.7631836, -64.1844330, 73.1043243
2: -21.7202415, 62.4037132, -14.5649471, 43.0888557, -64.8090973, 76.9686584
3: -23.9617577, 79.7581482, -15.9817076, 55.3257751, -79.2875366, 95.7398529
4: -19.8924942, 74.3483582, -13.6241512, 51.2421646, -71.1346588, 87.9725037

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126778, upper bound: 60.2181830
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126778, upper bound: 60.2181831
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -17.6552353, 53.2634773, -11.3791924, 35.7151070, -53.3703346, 64.6426544
1: -24.5836754, 55.0594139, -16.1628551, 37.0363083, -61.6199837, 71.2222672
2: -21.0160446, 61.2094574, -13.8870125, 41.1730003, -62.1890450, 75.0964661
3: -23.2941093, 78.4945526, -15.2091036, 52.8894081, -76.1835175, 93.7036514
4: -19.4026127, 72.9213943, -13.0088778, 48.9394493, -68.3420639, 85.9302673

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126778, upper bound: 60.2181830
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126778, upper bound: 60.2181830
time: 1.04 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.04 seconds
IS_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2148342, upper bound: 60.2153583
IS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2147674, upper bound: 60.2147674
IS_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2148342, upper bound: 60.2205742
IS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2147674, upper bound: 60.2172080
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2266130, upper bound: 60.2246665
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2264602, upper bound: 60.2247941
IS_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2228096, upper bound: 60.2193377
IS_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2180197, upper bound: 60.2180197
IS_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2195334, upper bound: 60.2147086
IS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2141019, upper bound: 60.2146268
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2148202, upper bound: 60.2121888
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2134760, upper bound: 60.2072158
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2190911
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2166958, upper bound: 60.2261647
IS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2156971, upper bound: 60.2163977
IS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2156971, upper bound: 60.2184175
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2151669, upper bound: 60.2137186
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2148282, upper bound: 60.2167097
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2151669, upper bound: 60.2137186
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2148282, upper bound: 60.2167097
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.1977769, upper bound: 60.2083271
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2088190, upper bound: 60.2150075
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.1977769, upper bound: 60.2105083
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2088190, upper bound: 60.2150075
IS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2154802, upper bound: 60.2091787
IS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2154802, upper bound: 60.2091787
IS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2154802, upper bound: 60.2091787
IS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2154802, upper bound: 60.2091787
IS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2126778, upper bound: 60.2181830
IS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2126778, upper bound: 60.2181831
IS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2126778, upper bound: 60.2181830
IS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.04
Output dim: 4, lower bound: -60.2126778, upper bound: 60.2181830

## BFS IS instance: IS_A1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -6.6845942, 21.3685608, -7.0401478, 22.4450760, -29.1296692, 28.4087067
1: -9.4924765, 22.2724018, -9.9836550, 23.3832836, -32.8757591, 32.2560539
2: -8.1440449, 24.9384079, -8.5655918, 26.1655998, -34.3096428, 33.5039978
3: -8.9932966, 31.8456116, -9.4548874, 33.4191742, -42.4124718, 41.3004951
4: -7.8372211, 29.6645374, -8.2147732, 31.1199360, -38.9571571, 37.8792953

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147674, upper bound: 60.2147674
time: 0.96 seconds

## Relational analysis of IS_A1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147674, upper bound: 60.2147674
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -7.9846687, 24.6592941, -7.1300812, 22.7052116, -30.6898785, 31.7893753
1: -11.1343088, 25.6553383, -10.0938644, 23.6459122, -34.7802200, 35.7492027
2: -9.5540485, 28.7395229, -8.6649647, 26.4695072, -36.0235558, 37.4044876
3: -10.5845137, 36.6657715, -9.5626678, 33.8086853, -44.3931923, 46.2284355
4: -9.0737028, 34.2209816, -8.3075523, 31.4765968, -40.5502930, 42.5285339

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147674, upper bound: 60.2147674
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147674, upper bound: 60.2147674
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -10.5310612, 33.0076103, -7.0401478, 22.4450760, -32.9761276, 40.0477600
1: -14.9118528, 34.2865067, -9.9836550, 23.3832836, -38.2951355, 44.2701607
2: -12.7989817, 38.1851044, -8.5655918, 26.1655998, -38.9645729, 46.7506943
3: -14.0864353, 48.8605423, -9.4548874, 33.4191742, -47.5056000, 58.3154297
4: -12.0459251, 45.4043770, -8.2147732, 31.1199360, -43.1658630, 53.6191368

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2133055, upper bound: 60.2157432
time: 0.89 seconds

## Relational analysis of IS_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2149421, upper bound: 60.2172908
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -11.4446983, 35.4083786, -7.1300812, 22.7052116, -34.1499100, 42.5384598
1: -16.0799885, 36.7389221, -10.0938644, 23.6459122, -39.7258987, 46.8327866
2: -13.7976933, 40.9671669, -8.6649647, 26.4695072, -40.2672005, 49.6321335
3: -15.2359943, 52.4269714, -9.5626678, 33.8086853, -49.0446777, 61.9896355
4: -12.9400959, 48.7696495, -8.3075523, 31.4765968, -44.4166794, 57.0772018

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2121945, upper bound: 60.2123778
time: 1.28 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2121690, upper bound: 60.2135426
time: 1.19 seconds

## Relational analysis of IS_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134385, upper bound: 60.2141018
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.0392351, 25.8137569, -9.4600649, 29.9131966, -37.9524307, 35.2738228
1: -11.4161921, 26.8347321, -13.3560982, 31.0771103, -42.4932899, 40.1908302
2: -9.8261356, 29.9275799, -11.4930305, 34.6350746, -44.4612122, 41.4206085
3: -10.8070860, 38.4179344, -12.6308966, 44.4180450, -55.2251282, 51.0488319
4: -9.3526144, 35.5916328, -10.8857965, 41.1818161, -50.5344315, 46.4774284

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2262637, upper bound: 60.2246652
time: 1.09 seconds

## Relational analysis of IS_A1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2262637, upper bound: 60.2246665
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -10.3883686, 32.6530228, -9.8329306, 31.0955849, -41.4839554, 42.4859543
1: -14.6031942, 33.8972893, -13.9043045, 32.2988853, -46.9020805, 47.8015938
2: -12.5469265, 37.8225517, -11.9665079, 35.9830093, -48.5299377, 49.7890587
3: -13.8441725, 48.4858475, -13.1250286, 46.2041397, -60.0483131, 61.6108665
4: -11.8315516, 44.9658165, -11.3234024, 42.7938271, -54.6253777, 56.2892189

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2264602, upper bound: 60.2247923
time: 0.94 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2264602, upper bound: 60.2247941
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -8.9851866, 28.5412769, -10.5209007, 33.0500641, -42.0352478, 39.0621796
1: -12.7900400, 29.6649323, -14.8950577, 34.3161087, -47.1061478, 44.5599747
2: -11.0011997, 33.0687523, -12.7866030, 38.2139626, -49.2151604, 45.8553543
3: -12.0353107, 42.4519081, -14.0827742, 48.9307861, -60.9660950, 56.5346832
4: -10.4192820, 39.3315697, -12.0463495, 45.4264793, -55.8457565, 51.3779182

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180197, upper bound: 60.2180197
time: 0.97 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180197, upper bound: 60.2180197
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -9.1216927, 28.9248180, -11.4131126, 35.4017525, -44.5234451, 40.3379288
1: -12.9683704, 30.0562954, -16.0365601, 36.7170753, -49.6854477, 46.0928497
2: -11.1558418, 33.5142288, -13.7623730, 40.9396286, -52.0954704, 47.2766037
3: -12.2062540, 43.0263176, -15.2048712, 52.4269180, -64.6331558, 58.2311897
4: -10.5604506, 39.8678284, -12.9215202, 48.7243309, -59.2847824, 52.7893486

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180197, upper bound: 60.2180197
time: 1.26 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2180197, upper bound: 60.2180197
time: 1.65 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -6.0151253, 19.2322540, -16.4854603, 48.9686699, -54.9837914, 35.7177048
1: -8.4907103, 20.0557346, -22.9870205, 50.7871437, -59.2778511, 43.0427475
2: -7.2694263, 22.4775620, -19.6701736, 56.4504547, -63.7198792, 42.1477242
3: -8.0957794, 28.6040707, -21.6145878, 71.8842621, -79.9800415, 50.2186584
4: -7.0251560, 26.7257442, -17.9896793, 67.2153091, -74.2404633, 44.7154236

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2191931, upper bound: 60.2136014
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2191366, upper bound: 60.2141611
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -8.1216898, 25.6525383, -16.5079727, 49.1815872, -57.3032684, 42.1605072
1: -11.3275318, 26.6610680, -22.9815369, 50.9559479, -62.2834778, 49.6426048
2: -9.6611891, 29.8491669, -19.6181755, 56.7013054, -66.3624954, 49.4673424
3: -10.9383516, 38.0406990, -21.7250061, 72.2371368, -83.1754913, 59.7657013
4: -9.2247801, 35.4690819, -17.9648132, 67.5446548, -76.7694168, 53.4338951

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2190018, upper bound: 60.2132885
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2135431, upper bound: 60.2140263
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -7.1549425, 22.3000069, -14.9960184, 44.4852791, -51.6402206, 37.2960205
1: -9.9846935, 23.2027607, -20.8154697, 46.1058426, -56.0905380, 44.0182304
2: -8.5814981, 26.0230865, -17.8355770, 51.3795586, -59.9610481, 43.8586655
3: -9.4837322, 33.2494087, -19.6819248, 65.6835785, -75.1673050, 52.9313354
4: -8.2032366, 30.9986629, -16.4221840, 61.2829514, -69.4861908, 47.4208450

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134760, upper bound: 60.2072158
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134760, upper bound: 60.2072158
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -6.4774637, 20.6671791, -14.4124174, 43.4359589, -49.9134216, 35.0795937
1: -9.1463099, 21.5013542, -20.0441189, 44.9573975, -54.1037064, 41.5454674
2: -7.8650312, 24.1236458, -17.1761856, 50.1321907, -57.9972229, 41.2998276
3: -8.6785316, 30.8901653, -18.9855518, 64.1769714, -72.8554993, 49.8757095
4: -7.5690155, 28.7219067, -15.8928289, 59.7576675, -67.3266830, 44.6147308

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134760, upper bound: 60.2072158
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134760, upper bound: 60.2072158
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -10.5829353, 33.1954765, -15.7810001, 46.9630585, -57.5459938, 48.9764671
1: -14.9808502, 34.4668999, -22.0052528, 48.7282257, -63.7090721, 56.4721527
2: -12.8593082, 38.3798714, -18.8067207, 54.1643219, -67.0236282, 57.1865883
3: -14.1552792, 49.1394577, -20.7016296, 68.9027710, -83.0580521, 69.8410797
4: -12.1076508, 45.6256638, -17.2193184, 64.4698639, -76.5775146, 62.8449631

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2139404, upper bound: 60.2152770
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2144613, upper bound: 60.2162124
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -10.5829353, 33.1954765, -19.7030621, 58.6302338, -69.2131653, 52.8985252
1: -14.9808502, 34.4668999, -27.5065365, 60.6951981, -75.6760483, 61.9734344
2: -12.8593082, 38.3798714, -23.4807625, 67.3772736, -80.2365799, 61.8606339
3: -14.1552792, 49.1394577, -25.8919411, 85.8565445, -100.0118256, 75.0313950
4: -12.1076508, 45.6256638, -21.3997822, 80.1888580, -92.2965088, 67.0254288

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2139404, upper bound: 60.2256910
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2144613, upper bound: 60.2254070
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -11.4920979, 35.5845413, -15.8215408, 46.9833832, -58.4754753, 51.4060822
1: -16.1414547, 36.9067802, -22.0391636, 48.7293549, -64.8708115, 58.9459457
2: -13.8512897, 41.1478271, -18.8327408, 54.2230377, -68.0743103, 59.9805603
3: -15.2957268, 52.6893082, -20.7580585, 68.9564438, -84.2521667, 73.4473572
4: -12.9969368, 48.9727364, -17.2476120, 64.5638123, -77.5607452, 66.2203522

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148282, upper bound: 60.2125101
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2129360, upper bound: 60.2121281
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 23
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 21

Time for candidate selection: 15.34 seconds

### Candidate
type: A, layer: 3, pos: 38

## Relational analysis of IS_A1_B2_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2063863, upper bound: 60.2033531
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2118067, upper bound: 60.2130291
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -11.4920979, 35.5845413, -19.7084637, 58.5201721, -70.0122604, 55.2930069
1: -16.1414547, 36.9067802, -27.4857140, 60.5764618, -76.7179184, 64.3924942
2: -13.8512897, 41.1478271, -23.4667702, 67.2959366, -81.1472168, 64.6145935
3: -15.2957268, 52.6893082, -25.8886070, 85.8029175, -101.0986481, 78.5778961
4: -12.9969368, 48.9727364, -21.4132805, 80.1193008, -93.1162338, 70.3860168

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2148282, upper bound: 60.2126589
time: 1.22 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2129361, upper bound: 60.2124156
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=65.54161834716797
rel_dist={4: [-60.234725823217936, 60.23472582321793]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1119.44 seconds
