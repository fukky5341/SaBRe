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
execution time: IAR + LP analysis = 2.65 + 1.97 = 4.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -54.3001214, upper bound: 54.3001214


# Binary Search by BASE starts (time budget: 1195.38 seconds, max iter: 100)

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
Binary search time: 103.33 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1092.05 seconds

## Binary search (step 0) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2601609, upper bound: 54.2657107
time: 1.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2654878, upper bound: 54.2654879
time: 1.65 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.89
Output dim: 0, lower bound: -54.2601609, upper bound: 54.2657107
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.89
Output dim: 0, lower bound: -54.2654878, upper bound: 54.2654879

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.4868755, 34.6143112, -22.1557789, 40.7848358, -59.2717133, 56.7700882
1: -20.8144035, 32.0120316, -24.9570370, 38.1305771, -58.9449768, 56.9690628
2: -21.3472996, 31.4293518, -25.5280037, 37.2813034, -58.6285934, 56.9573555
3: -25.5772781, 36.8826523, -30.7027779, 44.2187729, -69.7960358, 67.5854263
4: -24.1322937, 34.9039764, -28.9112663, 41.7009773, -65.8332672, 63.8152390

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2558830, upper bound: 54.2558830
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2558830, upper bound: 54.2654879
time: 0.92 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -23.0712452, 42.9994736, -22.1557789, 40.7848358, -63.8560791, 65.1552505
1: -25.9661083, 39.6324463, -24.9570370, 38.1305771, -64.0966873, 64.5894852
2: -26.6078491, 38.8119164, -25.5280037, 37.2813034, -63.8891525, 64.3399200
3: -31.9601364, 45.9458580, -30.7027779, 44.2187729, -76.1789093, 76.6486359
4: -30.0656071, 43.3875542, -28.9112663, 41.7009773, -71.7665863, 72.2988205

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2654879, upper bound: 54.2558830
time: 0.74 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2654879, upper bound: 54.2654879
time: 0.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 0, lower bound: -54.2558830, upper bound: 54.2558830
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 0, lower bound: -54.2558830, upper bound: 54.2654879
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 0, lower bound: -54.2654879, upper bound: 54.2558830
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.26
Output dim: 0, lower bound: -54.2654879, upper bound: 54.2654879

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -18.4868755, 34.6143112, -18.4868755, 34.6143112, -53.1011887, 53.1011887
1: -20.8144035, 32.0120316, -20.8144035, 32.0120316, -52.8264313, 52.8264313
2: -21.3472996, 31.4293518, -21.3472996, 31.4293518, -52.7766495, 52.7766495
3: -25.5772781, 36.8826523, -25.5772781, 36.8826523, -62.4599152, 62.4599152
4: -24.1322937, 34.9039764, -24.1322937, 34.9039764, -59.0362549, 59.0362625

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2348666, upper bound: 53.7845762
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
time: 0.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -18.4868755, 34.6143112, -23.0712452, 42.9994736, -61.4863434, 57.6855545
1: -20.8144035, 32.0120316, -25.9661083, 39.6324463, -60.4468384, 57.9781265
2: -21.3472996, 31.4293518, -26.6078491, 38.8119164, -60.1592178, 58.0372009
3: -25.5772781, 36.8826523, -31.9601364, 45.9458580, -71.5231323, 68.8427887
4: -24.1322937, 34.9039764, -30.0656071, 43.3875542, -67.5198364, 64.9695816

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2348666, upper bound: 54.2655282
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7595180, upper bound: 54.2404974
time: 0.96 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -23.0712452, 42.9994736, -18.4868755, 34.6143112, -57.6855545, 61.4863472
1: -25.9661083, 39.6324463, -20.8144035, 32.0120316, -57.9781265, 60.4468384
2: -26.6078491, 38.8119164, -21.3472996, 31.4293518, -58.0372009, 60.1592178
3: -31.9601364, 45.9458580, -25.5772781, 36.8826523, -68.8427887, 71.5231323
4: -30.0656071, 43.3875542, -24.1322937, 34.9039764, -64.9695740, 67.5198364

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2394364, upper bound: 53.7843045
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2404974, upper bound: 53.7840963
time: 1.13 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -23.0712452, 42.9994736, -23.0712452, 42.9994736, -66.0707169, 66.0707169
1: -25.9661083, 39.6324463, -25.9661083, 39.6324463, -65.5985489, 65.5985489
2: -26.6078491, 38.8119164, -26.6078491, 38.8119164, -65.4197617, 65.4197617
3: -31.9601364, 45.9458580, -31.9601364, 45.9458580, -77.9059906, 77.9059906
4: -30.0656071, 43.3875542, -30.0656071, 43.3875542, -73.4531631, 73.4531631

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2394364, upper bound: 54.2647119
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2404974, upper bound: 54.2649645
time: 0.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -54.2348666, upper bound: 53.7845762
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -54.2348666, upper bound: 54.2655282
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -53.7595180, upper bound: 54.2404974
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -54.2394364, upper bound: 53.7843045
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -54.2404974, upper bound: 53.7840963
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -54.2394364, upper bound: 54.2647119
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.66
Output dim: 0, lower bound: -54.2404974, upper bound: 54.2649645

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -18.4868755, 34.6143112, -49.6788254, 47.3828583
1: -16.9958324, 26.4444580, -20.8144035, 32.0120316, -49.0078659, 47.2588577
2: -17.4429550, 26.0576000, -21.3472996, 31.4293518, -48.8723068, 47.4048996
3: -20.9023075, 30.3311520, -25.5772781, 36.8826523, -57.7849426, 55.9084320
4: -19.7728100, 28.7748013, -24.1322937, 34.9039764, -54.6767769, 52.9070892

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
time: 1.06 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -18.4033165, 34.4601097, -57.1487503, 59.7989349
1: -25.5468903, 38.5653343, -20.7217922, 31.8583221, -57.4052086, 59.2871246
2: -26.1332932, 37.7651024, -21.2513447, 31.2822342, -57.4155273, 59.0164490
3: -31.4312077, 44.6071281, -25.4646530, 36.6996689, -68.1308670, 70.0717621
4: -29.4743004, 42.2834320, -24.0197620, 34.7431984, -64.2174988, 66.3031921

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -23.0712452, 42.9994736, -58.0639877, 51.9672241
1: -16.9958324, 26.4444580, -25.9661083, 39.6324463, -56.6282806, 52.4105530
2: -17.4429550, 26.0576000, -26.6078491, 38.8119164, -56.2548676, 52.6654510
3: -20.9023075, 30.3311520, -31.9601364, 45.9458580, -66.8481598, 62.2912903
4: -19.7728100, 28.7748013, -30.0656071, 43.3875542, -63.1603584, 58.8404083

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7840963, upper bound: 54.2394364
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7840963, upper bound: 54.2404973
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -22.9954433, 42.8613434, -65.5499802, 64.3910675
1: -25.5468903, 38.5653343, -25.8818092, 39.5031013, -65.0499878, 64.4471436
2: -26.1332932, 37.7651024, -26.5212555, 38.6871414, -64.8204269, 64.2863388
3: -31.4312077, 44.6071281, -31.8573036, 45.7928696, -77.2240753, 76.4644089
4: -29.4743004, 42.2834320, -29.9669685, 43.2475853, -72.7218781, 72.2503815

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7840963, upper bound: 54.2394364
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7840963, upper bound: 54.2404973
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -18.4868755, 34.6143112, -54.5165787, 56.0654259
1: -22.4450703, 34.2541313, -20.8144035, 32.0120316, -54.4570999, 55.0685349
2: -22.9841385, 33.6525192, -21.3472996, 31.4293518, -54.4134827, 54.9998131
3: -27.6680088, 39.5840874, -25.5772781, 36.8826523, -64.5506592, 65.1613617
4: -25.9807587, 37.5789795, -24.1322937, 34.9039764, -60.8847351, 61.7112694

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2394364, upper bound: 53.7840963
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2394364, upper bound: 53.7840963
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -18.4033165, 34.4601097, -61.3874435, 67.9273605
1: -30.3515205, 45.6756859, -20.7217922, 31.8583221, -62.2098427, 66.3974533
2: -31.0041237, 44.6888275, -21.2513447, 31.2822342, -62.2863579, 65.9401627
3: -37.3951302, 52.9274445, -25.4646530, 36.6996689, -74.0947952, 78.3920975
4: -34.8959274, 50.2624893, -24.0197620, 34.7431984, -69.6391068, 74.2822495

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2404973, upper bound: 53.7840963
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2404973, upper bound: 53.7840963
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -23.0712452, 42.9994736, -62.9017410, 60.6497955
1: -22.4450703, 34.2541313, -25.9661083, 39.6324463, -62.0775070, 60.2202377
2: -22.9841385, 33.6525192, -26.6078491, 38.8119164, -61.7960396, 60.2603683
3: -27.6680088, 39.5840874, -31.9601364, 45.9458580, -73.6138687, 71.5442200
4: -25.9807587, 37.5789795, -30.0656071, 43.3875542, -69.3683167, 67.6445847

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2640147, upper bound: 54.2640147
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2640147, upper bound: 54.2647118
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -22.9954433, 42.8613434, -69.7886810, 72.5195007
1: -30.3515205, 45.6756859, -25.8818092, 39.5031013, -69.8546219, 71.5574951
2: -31.0041237, 44.6888275, -26.5212555, 38.6871414, -69.6912613, 71.2100601
3: -37.3951302, 52.9274445, -31.8573036, 45.7928696, -83.1880035, 84.7847443
4: -34.8959274, 50.2624893, -29.9669685, 43.2475853, -78.1434937, 80.2294464

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2650756, upper bound: 54.2640147
time: 1.13 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2650756, upper bound: 54.2649644
time: 0.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.66 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -53.7595180, upper bound: 53.7595180
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -53.7840963, upper bound: 54.2394364
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -53.7840963, upper bound: 54.2404973
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -53.7840963, upper bound: 54.2394364
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -53.7840963, upper bound: 54.2404973
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -54.2394364, upper bound: 53.7840963
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -54.2394364, upper bound: 53.7840963
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -54.2404973, upper bound: 53.7840963
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -54.2404973, upper bound: 53.7840963
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -54.2640147, upper bound: 54.2640147
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -54.2640147, upper bound: 54.2647118
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -54.2650756, upper bound: 54.2640147
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 0, lower bound: -54.2650756, upper bound: 54.2649644

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -15.0645208, 28.8959808, -43.9605026, 43.9605026
1: -16.9958324, 26.4444580, -16.9958324, 26.4444580, -43.4402847, 43.4402809
2: -17.4429550, 26.0576000, -17.4429550, 26.0576000, -43.5005569, 43.5005531
3: -20.9023075, 30.3311520, -20.9023075, 30.3311520, -51.2334595, 51.2334595
4: -19.7728100, 28.7748013, -19.7728100, 28.7748013, -48.5476074, 48.5476074

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2104694, upper bound: 53.6275351
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -22.6886387, 41.3956223, -56.4601440, 51.5846176
1: -16.9958324, 26.4444580, -25.5468903, 38.5653343, -55.5611649, 51.9913445
2: -17.4429550, 26.0576000, -26.1332932, 37.7651024, -55.2080574, 52.1908913
3: -20.9023075, 30.3311520, -31.4312077, 44.6071281, -65.5094147, 61.7623596
4: -19.7728100, 28.7748013, -29.4743004, 42.2834320, -62.0562286, 58.2490997

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2104694, upper bound: 53.6275351
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -15.0645208, 28.8959808, -51.5846138, 56.4601440
1: -25.5468903, 38.5653343, -16.9958324, 26.4444580, -51.9913445, 55.5611649
2: -26.1332932, 37.7651024, -17.4429550, 26.0576000, -52.1908913, 55.2080574
3: -31.4312077, 44.6071281, -20.9023075, 30.3311520, -61.7623596, 65.5094147
4: -29.4743004, 42.2834320, -19.7728100, 28.7748013, -58.2490959, 62.0562286

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7351208, upper bound: 53.6025004
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -22.6886387, 41.3956223, -64.0842590, 64.0842590
1: -25.5468903, 38.5653343, -25.5468903, 38.5653343, -64.1122284, 64.1122284
2: -26.1332932, 37.7651024, -26.1332932, 37.7651024, -63.8983955, 63.8983955
3: -31.4312077, 44.6071281, -31.4312077, 44.6071281, -76.0383148, 76.0383224
4: -29.4743004, 42.2834320, -29.4743004, 42.2834320, -71.7577209, 71.7577209

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7351208, upper bound: 53.6025004
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -19.9022694, 37.5785522, -52.6430702, 48.7982483
1: -16.9958324, 26.4444580, -22.4450703, 34.2541313, -51.2499619, 48.8895187
2: -17.4429550, 26.0576000, -22.9841385, 33.6525192, -51.0954666, 49.0417290
3: -20.9023075, 30.3311520, -27.6680088, 39.5840874, -60.4863892, 57.9991608
4: -19.7728100, 28.7748013, -25.9807587, 37.5789795, -57.3517838, 54.7555618

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2536947, upper bound: 54.1704611
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1286434, upper bound: 54.1677407
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -26.9273338, 49.5240555, -64.5885773, 55.8233147
1: -16.9958324, 26.4444580, -30.3515205, 45.6756859, -62.6715050, 56.7959785
2: -17.4429550, 26.0576000, -31.0041237, 44.6888275, -62.1317635, 57.0617218
3: -20.9023075, 30.3311520, -37.3951302, 52.9274445, -73.8297501, 67.7262802
4: -19.7728100, 28.7748013, -34.8959274, 50.2624893, -70.0352936, 63.6707191

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2536947, upper bound: 54.1704611
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1806131, upper bound: 54.1677407
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -19.9022694, 37.5785522, -60.2671814, 61.2978897
1: -25.5468903, 38.5653343, -22.4450703, 34.2541313, -59.8010216, 61.0104065
2: -26.1332932, 37.7651024, -22.9841385, 33.6525192, -59.7857971, 60.7492409
3: -31.4312077, 44.6071281, -27.6680088, 39.5840874, -71.0152893, 72.2751236
4: -29.4743004, 42.2834320, -25.9807587, 37.5789795, -67.0532837, 68.2641907

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7783461, upper bound: 54.1382376
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6231886, upper bound: 54.1210998
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -26.9273338, 49.5240555, -72.2126846, 68.3229523
1: -25.5468903, 38.5653343, -30.3515205, 45.6756859, -71.2225723, 68.9168549
2: -26.1332932, 37.7651024, -31.0041237, 44.6888275, -70.8221054, 68.7692261
3: -31.4312077, 44.6071281, -37.3951302, 52.9274445, -84.3586502, 82.0022507
4: -29.4743004, 42.2834320, -34.8959274, 50.2624893, -79.7367859, 77.1793365

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7783461, upper bound: 54.1382376
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6231886, upper bound: 54.1210998
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -15.0645208, 28.8959808, -48.7982483, 52.6430664
1: -22.4450703, 34.2541313, -16.9958324, 26.4444580, -48.8895149, 51.2499619
2: -22.9841385, 33.6525192, -17.4429550, 26.0576000, -49.0417290, 51.0954666
3: -27.6680088, 39.5840874, -20.9023075, 30.3311520, -57.9991608, 60.4863968
4: -25.9807587, 37.5789795, -19.7728100, 28.7748013, -54.7555618, 57.3517838

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2150392, upper bound: 53.6272611
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -22.6886387, 41.3956223, -61.2978897, 60.2671814
1: -22.4450703, 34.2541313, -25.5468903, 38.5653343, -61.0104065, 59.8010216
2: -22.9841385, 33.6525192, -26.1332932, 37.7651024, -60.7492409, 59.7858086
3: -27.6680088, 39.5840874, -31.4312077, 44.6071281, -72.2751236, 71.0152817
4: -25.9807587, 37.5789795, -29.4743004, 42.2834320, -68.2641907, 67.0532837

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2150392, upper bound: 53.6272611
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -15.0645208, 28.8959808, -55.8233147, 64.5885773
1: -30.3515205, 45.6756859, -16.9958324, 26.4444580, -56.7959785, 62.6715012
2: -31.0041237, 44.6888275, -17.4429550, 26.0576000, -57.0617218, 62.1317635
3: -37.3951302, 52.9274445, -20.9023075, 30.3311520, -67.7262802, 73.8297501
4: -34.8959274, 50.2624893, -19.7728100, 28.7748013, -63.6707153, 70.0352936

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2251106, upper bound: 53.7429172
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -22.6886387, 41.3956223, -68.3229523, 72.2126770
1: -30.3515205, 45.6756859, -25.5468903, 38.5653343, -68.9168549, 71.2225647
2: -31.0041237, 44.6888275, -26.1332932, 37.7651024, -68.7692261, 70.8221054
3: -37.3951302, 52.9274445, -31.4312077, 44.6071281, -82.0022507, 84.3586502
4: -34.8959274, 50.2624893, -29.4743004, 42.2834320, -77.1793365, 79.7367859

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2251106, upper bound: 53.7429159
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -19.9022694, 37.5785522, -57.4808197, 57.4808197
1: -22.4450703, 34.2541313, -22.4450703, 34.2541313, -56.6991959, 56.6992035
2: -22.9841385, 33.6525192, -22.9841385, 33.6525192, -56.6366425, 56.6366425
3: -27.6680088, 39.5840874, -27.6680088, 39.5840874, -67.2520981, 67.2520981
4: -25.9807587, 37.5789795, -25.9807587, 37.5789795, -63.5597382, 63.5597382

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2494872, upper bound: 54.1688396
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1680579, upper bound: 54.1661851
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -26.9273338, 49.5240555, -69.4263229, 64.5058899
1: -22.4450703, 34.2541313, -30.3515205, 45.6756859, -68.1207352, 64.6056519
2: -22.9841385, 33.6525192, -31.0041237, 44.6888275, -67.6729507, 64.6566391
3: -27.6680088, 39.5840874, -37.3951302, 52.9274445, -80.5954514, 76.9792175
4: -25.9807587, 37.5789795, -34.8959274, 50.2624893, -76.2432480, 72.4748917

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2494873, upper bound: 54.1688396
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1680578, upper bound: 54.1661851
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -19.9022694, 37.5785522, -64.5058899, 69.4263229
1: -30.3515205, 45.6756859, -22.4450703, 34.2541313, -64.6056519, 68.1207352
2: -31.0041237, 44.6888275, -22.9841385, 33.6525192, -64.6566391, 67.6729507
3: -37.3951302, 52.9274445, -27.6680088, 39.5840874, -76.9792099, 80.5954514
4: -34.8959274, 50.2624893, -25.9807587, 37.5789795, -72.4748917, 76.2432480

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2530221, upper bound: 54.2517572
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -26.9273338, 49.5240555, -76.4513855, 76.4513855
1: -30.3515205, 45.6756859, -30.3515205, 45.6756859, -76.0271912, 76.0271835
2: -31.0041237, 44.6888275, -31.0041237, 44.6888275, -75.6929398, 75.6929474
3: -37.3951302, 52.9274445, -37.3951302, 52.9274445, -90.3225708, 90.3225708
4: -34.8959274, 50.2624893, -34.8959274, 50.2624893, -85.1583939, 85.1583939

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2530221, upper bound: 54.2529484
time: 0.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.81 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2104694, upper bound: 53.6275351
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2104694, upper bound: 53.6275351
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -53.7351208, upper bound: 53.6025004
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -53.7351208, upper bound: 53.6025004
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2536947, upper bound: 54.1704611
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.1286434, upper bound: 54.1677407
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2536947, upper bound: 54.1704611
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.1806131, upper bound: 54.1677407
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -53.7783461, upper bound: 54.1382376
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -53.6231886, upper bound: 54.1210998
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -53.7783461, upper bound: 54.1382376
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -53.6231886, upper bound: 54.1210998
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2150392, upper bound: 53.6272611
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2150392, upper bound: 53.6272611
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2251106, upper bound: 53.7429172
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2251106, upper bound: 53.7429159
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2494872, upper bound: 54.1688396
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.1680579, upper bound: 54.1661851
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2494873, upper bound: 54.1688396
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.1680578, upper bound: 54.1661851
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2530221, upper bound: 54.2517572
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.81
Output dim: 0, lower bound: -54.2530221, upper bound: 54.2529484

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -15.0645208, 28.8959808, -40.3866348, 37.7912445
1: -12.9931536, 20.8490124, -16.9958324, 26.4444580, -39.4376106, 37.8448372
2: -13.3803749, 20.5927849, -17.4429550, 26.0576000, -39.4379730, 38.0357399
3: -16.0014496, 23.8214302, -20.9023075, 30.3311520, -46.3326035, 44.7237396
4: -15.2732220, 22.4852295, -19.7728100, 28.7748013, -44.0480194, 42.2580338

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -15.0645208, 28.8959808, -43.0083046, 42.3050041
1: -15.9184780, 24.8546124, -16.9958324, 26.4444580, -42.3629265, 41.8504333
2: -16.3577156, 24.5101395, -17.4429550, 26.0576000, -42.4153137, 41.9530945
3: -19.5635452, 28.4796600, -20.9023075, 30.3311520, -49.8946991, 49.3819580
4: -18.5248909, 27.0098228, -19.7728100, 28.7748013, -47.2996902, 46.7826233

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -22.6886387, 41.3956223, -52.8862762, 45.4153595
1: -12.9931536, 20.8490124, -25.5468903, 38.5653343, -51.5584869, 46.3958969
2: -13.3803749, 20.5927849, -26.1332932, 37.7651024, -51.1454773, 46.7260780
3: -16.0014496, 23.8214302, -31.4312077, 44.6071281, -60.6085663, 55.2526398
4: -15.2732220, 22.4852295, -29.4743004, 42.2834320, -57.5566444, 51.9595299

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -22.6886387, 41.3956223, -55.5079460, 49.9291153
1: -15.9184780, 24.8546124, -25.5468903, 38.5653343, -54.4838104, 50.4015007
2: -16.3577156, 24.5101395, -26.1332932, 37.7651024, -54.1228180, 50.6434326
3: -19.5635452, 28.4796600, -31.4312077, 44.6071281, -64.1706619, 59.9108543
4: -18.5248909, 27.0098228, -29.4743004, 42.2834320, -60.8083153, 56.4841232

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -15.0645208, 28.8959808, -47.7957802, 50.0370789
1: -21.2962494, 32.6550941, -16.9958324, 26.4444580, -47.7406998, 49.6509247
2: -21.7990875, 32.0393410, -17.4429550, 26.0576000, -47.8566895, 49.4822960
3: -26.2362270, 37.6525040, -20.9023075, 30.3311520, -56.5673790, 58.5548096
4: -24.6484203, 35.6282730, -19.7728100, 28.7748013, -53.4232140, 55.4010773

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6242801, upper bound: 54.1286434
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6242801, upper bound: 54.1286434
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -15.0645208, 28.8959808, -50.4148254, 54.4011459
1: -24.2212391, 36.5742035, -16.9958324, 26.4444580, -50.6656914, 53.5700378
2: -24.7897606, 35.8414726, -17.4429550, 26.0576000, -50.8473587, 53.2844276
3: -29.7945175, 42.2615585, -20.9023075, 30.3311520, -60.1256714, 63.1638641
4: -27.9134693, 40.0790443, -19.7728100, 28.7748013, -56.6882668, 59.8518486

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6242801, upper bound: 54.1286434
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6242801, upper bound: 54.1286434
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -22.6886387, 41.3956223, -60.2954216, 57.6611977
1: -21.2962494, 32.6550941, -25.5468903, 38.5653343, -59.8615837, 58.2019806
2: -21.7990875, 32.0393410, -26.1332932, 37.7651024, -59.5641899, 58.1726341
3: -26.2362270, 37.6525040, -31.4312077, 44.6071281, -70.8433456, 69.0837021
4: -24.6484203, 35.6282730, -29.4743004, 42.2834320, -66.9318542, 65.1025696

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -22.6886387, 41.3956223, -62.9144554, 62.0252609
1: -24.2212391, 36.5742035, -25.5468903, 38.5653343, -62.7865753, 62.1210938
2: -24.7897606, 35.8414726, -26.1332932, 37.7651024, -62.5548630, 61.9747620
3: -29.7945175, 42.2615585, -31.4312077, 44.6071281, -74.4016342, 73.6927567
4: -27.9134693, 40.0790443, -29.4743004, 42.2834320, -70.1968918, 69.5533447

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -19.9022694, 37.5785522, -49.0692024, 42.6289902
1: -12.9931536, 20.8490124, -22.4450703, 34.2541313, -47.2472839, 43.2940712
2: -13.3803749, 20.5927849, -22.9841385, 33.6525192, -47.0328865, 43.5769157
3: -16.0014496, 23.8214302, -27.6680088, 39.5840874, -55.5855331, 51.4894371
4: -15.2732220, 22.4852295, -25.9807587, 37.5789795, -52.8521957, 48.4659882

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2059022, upper bound: 54.1798389
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2059022, upper bound: 54.1798389
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -19.9022694, 37.5785522, -51.6908760, 47.1427536
1: -15.9184780, 24.8546124, -22.4450703, 34.2541313, -50.1726074, 47.2996712
2: -16.3577156, 24.5101395, -22.9841385, 33.6525192, -50.0102272, 47.4942741
3: -19.5635452, 28.4796600, -27.6680088, 39.5840874, -59.1476326, 56.1476555
4: -18.5248909, 27.0098228, -25.9807587, 37.5789795, -56.1038704, 52.9905815

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2059022, upper bound: 54.1798389
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2059022, upper bound: 54.1798389
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -26.9273338, 49.5240555, -61.0147018, 49.6540565
1: -12.9931536, 20.8490124, -30.3515205, 45.6756859, -58.6688347, 51.2005310
2: -13.3803749, 20.5927849, -31.0041237, 44.6888275, -58.0692024, 51.5969086
3: -16.0014496, 23.8214302, -37.3951302, 52.9274445, -68.9288940, 61.2165604
4: -15.2732220, 22.4852295, -34.8959274, 50.2624893, -65.5357056, 57.3811455

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3571967, upper bound: 52.7942391
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2393465, upper bound: 54.1674479
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -26.9273338, 49.5240555, -63.6363754, 54.1678162
1: -15.9184780, 24.8546124, -30.3515205, 45.6756859, -61.5941505, 55.2061272
2: -16.3577156, 24.5101395, -31.0041237, 44.6888275, -61.0465355, 55.5142632
3: -19.5635452, 28.4796600, -37.3951302, 52.9274445, -72.4909897, 65.8747787
4: -18.5248909, 27.0098228, -34.8959274, 50.2624893, -68.7873688, 61.9057350

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.2208645, upper bound: 52.7888780
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1790022, upper bound: 54.1648544
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -19.9022694, 37.5785522, -56.4783516, 54.8748283
1: -21.2962494, 32.6550941, -22.4450703, 34.2541313, -55.5503807, 55.1001625
2: -21.7990875, 32.0393410, -22.9841385, 33.6525192, -55.4516029, 55.0234756
3: -26.2362270, 37.6525040, -27.6680088, 39.5840874, -65.8203125, 65.3205109
4: -24.6484203, 35.6282730, -25.9807587, 37.5789795, -62.2274017, 61.6090317

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6249369, upper bound: 54.1315645
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6249369, upper bound: 54.1315645
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -19.9022694, 37.5785522, -59.0973816, 59.2388916
1: -24.2212391, 36.5742035, -22.4450703, 34.2541313, -58.4753723, 59.0192642
2: -24.7897606, 35.8414726, -22.9841385, 33.6525192, -58.4422684, 58.8256035
3: -29.7945175, 42.2615585, -27.6680088, 39.5840874, -69.3785934, 69.9295654
4: -27.9134693, 40.0790443, -25.9807587, 37.5789795, -65.4924469, 66.0597992

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6249369, upper bound: 54.1315645
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6249369, upper bound: 54.1315645
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -26.9273338, 49.5240555, -68.4238510, 61.8998947
1: -21.2962494, 32.6550941, -30.3515205, 45.6756859, -66.9719315, 63.0066147
2: -21.7990875, 32.0393410, -31.0041237, 44.6888275, -66.4879074, 63.0434532
3: -26.2362270, 37.6525040, -37.3951302, 52.9274445, -79.1636734, 75.0476379
4: -24.6484203, 35.6282730, -34.8959274, 50.2624893, -74.9109116, 70.5241776

Time for backsubstitution: 2.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.8751202, upper bound: 51.7251509
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7372411, upper bound: 54.1300891
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -26.9273338, 49.5240555, -71.0428848, 66.2639618
1: -24.2212391, 36.5742035, -30.3515205, 45.6756859, -69.8969193, 66.9257202
2: -24.7897606, 35.8414726, -31.0041237, 44.6888275, -69.4785919, 66.8455887
3: -29.7945175, 42.2615585, -37.3951302, 52.9274445, -82.7219620, 79.6566925
4: -27.9134693, 40.0790443, -34.8959274, 50.2624893, -78.1759567, 74.9749527

Time for backsubstitution: 2.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9622111, upper bound: 51.9622111
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5772014, upper bound: 54.1121082
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.5780373, 31.9165230, -15.0645208, 28.8959808, -45.4740181, 46.9810410
1: -18.7406483, 29.0654297, -16.9958324, 26.4444580, -45.1851006, 46.0612640
2: -19.1804962, 28.6155891, -17.4429550, 26.0576000, -45.2380981, 46.0585442
3: -23.1438141, 33.4699211, -20.9023075, 30.3311520, -53.4749680, 54.3722305
4: -21.7336960, 31.7478294, -19.7728100, 28.7748013, -50.5084953, 51.5206337

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1798389, upper bound: 54.2059022
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1798389, upper bound: 54.2059022
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18.8832836, 35.7310181, -15.0645208, 28.8959808, -47.7792664, 50.7955399
1: -21.2898827, 32.5063171, -16.9958324, 26.4444580, -47.7343369, 49.5021515
2: -21.8138657, 31.9566841, -17.4429550, 26.0576000, -47.8714676, 49.3996391
3: -26.2260628, 37.5371437, -20.9023075, 30.3311520, -56.5572128, 58.4394379
4: -24.6329060, 35.6249886, -19.7728100, 28.7748013, -53.4077072, 55.3977890

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1798389, upper bound: 54.2059022
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1798389, upper bound: 54.2059022
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.5780373, 31.9165230, -22.6886387, 41.3956223, -57.9736595, 54.6051445
1: -18.7406483, 29.0654297, -25.5468903, 38.5653343, -57.3059845, 54.6123199
2: -19.1804962, 28.6155891, -26.1332932, 37.7651024, -56.9455986, 54.7488823
3: -23.1438141, 33.4699211, -31.4312077, 44.6071281, -67.7509460, 64.9011307
4: -21.7336960, 31.7478294, -29.4743004, 42.2834320, -64.0171204, 61.2221260

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.8832836, 35.7310181, -22.6886387, 41.3956223, -60.2788925, 58.4196510
1: -21.2898827, 32.5063171, -25.5468903, 38.5653343, -59.8552170, 58.0532074
2: -21.8138657, 31.9566841, -26.1332932, 37.7651024, -59.5789680, 58.0899696
3: -26.2260628, 37.5371437, -31.4312077, 44.6071281, -70.8331909, 68.9683304
4: -24.6329060, 35.6249886, -29.4743004, 42.2834320, -66.9163361, 65.0992737

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6249369
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.4691925, 39.9133835, -14.9250278, 28.6497498, -50.1189423, 54.8384094
1: -24.0923729, 35.6383591, -16.8379745, 26.1963940, -50.2887650, 52.4763336
2: -24.6987610, 35.0490875, -17.2828350, 25.8190613, -50.5178223, 52.3319130
3: -29.4798279, 41.1165810, -20.7048187, 30.0409431, -59.5207596, 61.8213997
4: -27.4715805, 39.2330818, -19.5860615, 28.5057869, -55.9773674, 58.8191414

Time for backsubstitution: 2.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7942391, upper bound: 53.3571967
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7888780, upper bound: 53.2208645
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -25.9870872, 47.8569641, -15.0645208, 28.8959808, -54.8830681, 62.9214859
1: -29.2872486, 44.0551414, -16.9958324, 26.4444580, -55.7317009, 61.0509644
2: -29.9218769, 43.1240997, -17.4429550, 26.0576000, -55.9794769, 60.5670471
3: -36.0748901, 51.0303078, -20.9023075, 30.3311520, -66.4060440, 71.9325943
4: -33.6610641, 48.4680595, -19.7728100, 28.7748013, -62.4358673, 68.2408676

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1674479, upper bound: 54.2393465
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1648544, upper bound: 54.1790022
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.4691925, 39.9133835, -22.5430965, 41.1439095, -62.6131020, 62.4564819
1: -24.0923729, 35.6383591, -25.3821220, 38.3102837, -62.4026566, 61.0204659
2: -24.6987610, 35.0490875, -25.9665699, 37.5203247, -62.2190781, 61.0156479
3: -29.4798279, 41.1165810, -31.2259331, 44.3061523, -73.7859497, 72.3425140
4: -27.4715805, 39.2330818, -29.2808399, 42.0034790, -69.4750595, 68.5139084

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7409102, upper bound: 52.7473044
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -25.9870872, 47.8569641, -22.6886387, 41.3956223, -67.3827057, 70.5455933
1: -29.2872486, 44.0551414, -25.5468903, 38.5653343, -67.8525848, 69.6020279
2: -29.9218769, 43.1240997, -26.1332932, 37.7651024, -67.6869812, 69.2573853
3: -36.0748901, 51.0303078, -31.4312077, 44.6071281, -80.6820221, 82.4615097
4: -33.6610641, 48.4680595, -29.4743004, 42.2834320, -75.9444962, 77.9423599

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1300891, upper bound: 53.7372411
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1121082, upper bound: 53.5772014
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.5780373, 31.9165230, -19.9022694, 37.5785522, -54.1565857, 51.8187904
1: -18.7406483, 29.0654297, -22.4450703, 34.2541313, -52.9947815, 51.5104942
2: -19.1804962, 28.6155891, -22.9841385, 33.6525192, -52.8330078, 51.5997238
3: -23.1438141, 33.4699211, -27.6680088, 39.5840874, -62.7279015, 61.1379318
4: -21.7336960, 31.7478294, -25.9807587, 37.5789795, -59.3126717, 57.7285805

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1784579, upper bound: 54.1783918
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1784579, upper bound: 54.1783918
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18.8832836, 35.7310181, -19.9022694, 37.5785522, -56.4618263, 55.6332855
1: -21.2898827, 32.5063171, -22.4450703, 34.2541313, -55.5440140, 54.9513855
2: -21.8138657, 31.9566841, -22.9841385, 33.6525192, -55.4663811, 54.9408150
3: -26.2260628, 37.5371437, -27.6680088, 39.5840874, -65.8101501, 65.2051392
4: -24.6329060, 35.6249886, -25.9807587, 37.5789795, -62.2118835, 61.6057472

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1783607, upper bound: 54.1783918
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1783607, upper bound: 54.1783918
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.5780373, 31.9165230, -26.9273338, 49.5240555, -66.1020737, 58.8438530
1: -18.7406483, 29.0654297, -30.3515205, 45.6756859, -64.4163284, 59.4169502
2: -19.1804962, 28.6155891, -31.0041237, 44.6888275, -63.8693123, 59.6197052
3: -23.1438141, 33.4699211, -37.3951302, 52.9274445, -76.0712585, 70.8650513
4: -21.7336960, 31.7478294, -34.8959274, 50.2624893, -71.9961853, 66.6437302

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3574429, upper bound: 52.7937952
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2380054, upper bound: 54.1659877
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.8832836, 35.7310181, -26.9273338, 49.5240555, -68.4073257, 62.6583519
1: -21.2898827, 32.5063171, -30.3515205, 45.6756859, -66.9655609, 62.8578377
2: -21.8138657, 31.9566841, -31.0041237, 44.6888275, -66.5026855, 62.9607964
3: -26.2260628, 37.5371437, -37.3951302, 52.9274445, -79.1535034, 74.9322586
4: -24.6329060, 35.6249886, -34.8959274, 50.2624893, -74.8953934, 70.5208817

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.2982872, upper bound: 52.7917474
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1651883, upper bound: 54.1633062
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.4691925, 39.9133835, -19.7537060, 37.3146667, -58.7838593, 59.6670876
1: -24.0923729, 35.6383591, -22.2777328, 33.9865150, -58.0788841, 57.9160919
2: -24.6987610, 35.0490875, -22.8134842, 33.3965073, -58.0952682, 57.8625717
3: -29.4798279, 41.1165810, -27.4591808, 39.2693329, -68.7491608, 68.5757599
4: -27.4715805, 39.2330818, -25.7824459, 37.2875366, -64.7591171, 65.0155258

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7937877, upper bound: 53.3488282
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7888780, upper bound: 53.2208645
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -25.9870872, 47.8569641, -19.9022694, 37.5785522, -63.5656242, 67.7592316
1: -29.2872486, 44.0551414, -22.4450703, 34.2541313, -63.5413818, 66.5001984
2: -29.9218769, 43.1240997, -22.9841385, 33.6525192, -63.5743828, 66.1082382
3: -36.0748901, 51.0303078, -27.6680088, 39.5840874, -75.6589813, 78.6983109
4: -33.6610641, 48.4680595, -25.9807587, 37.5789795, -71.2400436, 74.4488220

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1671041, upper bound: 54.2471805
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1650984, upper bound: 54.1747011
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.4691925, 39.9133835, -26.7851086, 49.2700043, -70.7391891, 66.6984940
1: -24.0923729, 35.6383591, -30.1895866, 45.4200249, -69.5123825, 65.8279190
2: -24.6987610, 35.0490875, -30.8401871, 44.4438896, -69.1426544, 65.8892746
3: -29.4798279, 41.1165810, -37.1938553, 52.6247101, -82.1045380, 78.3104248
4: -27.4715805, 39.2330818, -34.7038536, 49.9825668, -77.4541473, 73.9369125

Time for backsubstitution: 2.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9622111, upper bound: 51.9622111
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.9622111, upper bound: 52.9093269
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -25.9870872, 47.8569641, -26.9273338, 49.5240555, -75.5111313, 74.7842941
1: -29.2872486, 44.0551414, -30.3515205, 45.6756859, -74.9629211, 74.4066467
2: -29.9218769, 43.1240997, -31.0041237, 44.6888275, -74.6106949, 74.1282196
3: -36.0748901, 51.0303078, -37.3951302, 52.9274445, -89.0023346, 88.4254379
4: -33.6610641, 48.4680595, -34.8959274, 50.2624893, -83.9235535, 83.3639755

Time for backsubstitution: 2.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.2032577, upper bound: 52.7784307
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.2032577, upper bound: 54.2529484
time: 1.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.20 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6242801
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.6242801, upper bound: 54.1286434
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.6242801, upper bound: 54.1286434
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.6242801, upper bound: 54.1286434
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.6242801, upper bound: 54.1286434
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2059022, upper bound: 54.1798389
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2059022, upper bound: 54.1798389
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2059022, upper bound: 54.1798389
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2059022, upper bound: 54.1798389
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.3571967, upper bound: 52.7942391
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2393465, upper bound: 54.1674479
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.2208645, upper bound: 52.7888780
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1790022, upper bound: 54.1648544
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.6249369, upper bound: 54.1315645
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.6249369, upper bound: 54.1315645
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.6249369, upper bound: 54.1315645
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.6249369, upper bound: 54.1315645
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -51.8751202, upper bound: 51.7251509
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.7372411, upper bound: 54.1300891
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -51.9622111, upper bound: 51.9622111
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.5772014, upper bound: 54.1121082
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1798389, upper bound: 54.2059022
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1798389, upper bound: 54.2059022
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1798389, upper bound: 54.2059022
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1798389, upper bound: 54.2059022
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1286434, upper bound: 53.6249369
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1315645, upper bound: 53.6249369
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -52.7942391, upper bound: 53.3571967
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -52.7888780, upper bound: 53.2208645
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1674479, upper bound: 54.2393465
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1648544, upper bound: 54.1790022
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -52.7678898, upper bound: 52.9093269
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -52.7409102, upper bound: 52.7473044
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1300891, upper bound: 53.7372411
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1121082, upper bound: 53.5772014
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1784579, upper bound: 54.1783918
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1784579, upper bound: 54.1783918
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1783607, upper bound: 54.1783918
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1783607, upper bound: 54.1783918
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.3574429, upper bound: 52.7937952
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.2380054, upper bound: 54.1659877
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.2982872, upper bound: 52.7917474
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1651883, upper bound: 54.1633062
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -52.7937877, upper bound: 53.3488282
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -52.7888780, upper bound: 53.2208645
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1671041, upper bound: 54.2471805
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -54.1650984, upper bound: 54.1747011
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -51.9622111, upper bound: 51.9622111
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -51.9622111, upper bound: 52.9093269
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.2032577, upper bound: 52.7784307
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.20
Output dim: 0, lower bound: -53.2032577, upper bound: 54.2529484

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -11.4906597, 22.7267227, -34.2173805, 34.2173805
1: -12.9931536, 20.8490124, -12.9931536, 20.8490124, -33.8421669, 33.8421669
2: -13.3803749, 20.5927849, -13.3803749, 20.5927849, -33.9731598, 33.9731598
3: -16.0014496, 23.8214302, -16.0014496, 23.8214302, -39.8228798, 39.8228798
4: -15.2732220, 22.4852295, -15.2732220, 22.4852295, -37.7584534, 37.7584534

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2493111, upper bound: 54.2015040
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2343833, upper bound: 54.2011938
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -14.1123228, 27.2404823, -38.7311363, 36.8390465
1: -12.9931536, 20.8490124, -15.9184780, 24.8546124, -37.8477592, 36.7674828
2: -13.3803749, 20.5927849, -16.3577156, 24.5101395, -37.8905144, 36.9504967
3: -16.0014496, 23.8214302, -19.5635452, 28.4796600, -44.4811020, 43.3849754
4: -15.2732220, 22.4852295, -18.5248909, 27.0098228, -42.2830429, 41.0101204

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2493111, upper bound: 54.2015040
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2343833, upper bound: 54.2011938
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -11.4906597, 22.7267227, -36.8390465, 38.7311401
1: -15.9184780, 24.8546124, -12.9931536, 20.8490124, -36.7674828, 37.8477592
2: -16.3577156, 24.5101395, -13.3803749, 20.5927849, -36.9505005, 37.8905144
3: -19.5635452, 28.4796600, -16.0014496, 23.8214302, -43.3849754, 44.4811020
4: -18.5248909, 27.0098228, -15.2732220, 22.4852295, -41.0101204, 42.2830429

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1182978, upper bound: 54.1971245
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2004790, upper bound: 54.2004791
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -14.1123228, 27.2404823, -41.3528061, 41.3528061
1: -15.9184780, 24.8546124, -15.9184780, 24.8546124, -40.7730789, 40.7730827
2: -16.3577156, 24.5101395, -16.3577156, 24.5101395, -40.8678551, 40.8678551
3: -19.5635452, 28.4796600, -19.5635452, 28.4796600, -48.0432053, 48.0432053
4: -18.5248909, 27.0098228, -18.5248909, 27.0098228, -45.5347099, 45.5347099

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1182978, upper bound: 54.1971245
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2004790, upper bound: 54.2004791
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -18.8997993, 34.9725609, -46.4632111, 41.6265221
1: -12.9931536, 20.8490124, -21.2962494, 32.6550941, -45.6482468, 42.1452560
2: -13.3803749, 20.5927849, -21.7990875, 32.0393410, -45.4197159, 42.3918686
3: -16.0014496, 23.8214302, -26.2362270, 37.6525040, -53.6539497, 50.0576515
4: -15.2732220, 22.4852295, -24.6484203, 35.6282730, -50.9014931, 47.1336441

Time for backsubstitution: 2.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1302639, upper bound: 53.5978705
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2051629, upper bound: 53.6225901
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -21.5188446, 39.3366241, -50.8272820, 44.2455673
1: -12.9931536, 20.8490124, -24.2212391, 36.5742035, -49.5673561, 45.0702515
2: -13.3803749, 20.5927849, -24.7897606, 35.8414726, -49.2218475, 45.3825455
3: -16.0014496, 23.8214302, -29.7945175, 42.2615585, -58.2630005, 53.6159477
4: -15.2732220, 22.4852295, -27.9134693, 40.0790443, -55.3522644, 50.3986969

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1302639, upper bound: 53.5978705
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2051629, upper bound: 53.6225901
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -18.8997993, 34.9725609, -49.0848846, 46.1402817
1: -15.9184780, 24.8546124, -21.2962494, 32.6550941, -48.5735703, 46.1508522
2: -16.3577156, 24.5101395, -21.7990875, 32.0393410, -48.3970566, 46.3092270
3: -19.5635452, 28.4796600, -26.2362270, 37.6525040, -57.2160492, 54.7158737
4: -18.5248909, 27.0098228, -24.6484203, 35.6282730, -54.1531639, 51.6582336

Time for backsubstitution: 2.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9850772, upper bound: 53.2090255
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1208288, upper bound: 53.5782654
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -21.5188446, 39.3366241, -53.4489479, 48.7593269
1: -15.9184780, 24.8546124, -24.2212391, 36.5742035, -52.4926758, 49.0758400
2: -16.3577156, 24.5101395, -24.7897606, 35.8414726, -52.1991882, 49.2999001
3: -19.5635452, 28.4796600, -29.7945175, 42.2615585, -61.8250999, 58.2741776
4: -18.5248909, 27.0098228, -27.9134693, 40.0790443, -58.6039162, 54.9232864

Time for backsubstitution: 2.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9850772, upper bound: 53.2090255
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1208288, upper bound: 53.5782654
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -11.4906597, 22.7267227, -41.6265221, 46.4632149
1: -21.2962494, 32.6550941, -12.9931536, 20.8490124, -42.1452560, 45.6482468
2: -21.7990875, 32.0393410, -13.3803749, 20.5927849, -42.3918724, 45.4197159
3: -26.2362270, 37.6525040, -16.0014496, 23.8214302, -50.0576515, 53.6539497
4: -24.6484203, 35.6282730, -15.2732220, 22.4852295, -47.1336441, 50.9014931

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7756884, upper bound: 54.1509259
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7048310, upper bound: 54.0188575
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -14.1123228, 27.2404823, -46.1402817, 49.0848846
1: -21.2962494, 32.6550941, -15.9184780, 24.8546124, -46.1508484, 48.5735703
2: -21.7990875, 32.0393410, -16.3577156, 24.5101395, -46.3092270, 48.3970566
3: -26.2362270, 37.6525040, -19.5635452, 28.4796600, -54.7158661, 57.2160492
4: -24.6484203, 35.6282730, -18.5248909, 27.0098228, -51.6582336, 54.1531639

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7756884, upper bound: 54.1509259
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7048310, upper bound: 54.0188575
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -11.4906597, 22.7267227, -44.2455673, 50.8272820
1: -24.2212391, 36.5742035, -12.9931536, 20.8490124, -45.0702477, 49.5673561
2: -24.7897606, 35.8414726, -13.3803749, 20.5927849, -45.3825455, 49.2218475
3: -29.7945175, 42.2615585, -16.0014496, 23.8214302, -53.6159477, 58.2630043
4: -27.9134693, 40.0790443, -15.2732220, 22.4852295, -50.3986969, 55.3522568

Time for backsubstitution: 2.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9774855, upper bound: 53.6550735
time: 1.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5782652, upper bound: 54.1208292
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -14.1123228, 27.2404823, -48.7593269, 53.4489479
1: -24.2212391, 36.5742035, -15.9184780, 24.8546124, -49.0758400, 52.4926834
2: -24.7897606, 35.8414726, -16.3577156, 24.5101395, -49.2999001, 52.1991882
3: -29.7945175, 42.2615585, -19.5635452, 28.4796600, -58.2741699, 61.8251038
4: -27.9134693, 40.0790443, -18.5248909, 27.0098228, -54.9232903, 58.6039200

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9774855, upper bound: 53.6550735
time: 1.45 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5782652, upper bound: 54.1208292
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -18.8997993, 34.9725609, -53.8723602, 53.8723602
1: -21.2962494, 32.6550941, -21.2962494, 32.6550941, -53.9513435, 53.9513435
2: -21.7990875, 32.0393410, -21.7990875, 32.0393410, -53.8384285, 53.8384285
3: -26.2362270, 37.6525040, -26.2362270, 37.6525040, -63.8887253, 63.8887215
4: -24.6484203, 35.6282730, -24.6484203, 35.6282730, -60.2766953, 60.2766953

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1857154, upper bound: 52.9751680
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.0843565, upper bound: 52.9472694
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -21.5188446, 39.3366241, -58.2364235, 56.4914017
1: -21.2962494, 32.6550941, -24.2212391, 36.5742035, -57.8704529, 56.8763351
2: -21.7990875, 32.0393410, -24.7897606, 35.8414726, -57.6405602, 56.8291016
3: -26.2362270, 37.6525040, -29.7945175, 42.2615585, -68.4977875, 67.4470062
4: -24.6484203, 35.6282730, -27.9134693, 40.0790443, -64.7274628, 63.5417404

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1857154, upper bound: 52.9751680
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.0843565, upper bound: 52.9472694
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -18.8997993, 34.9725609, -56.4914017, 58.2364235
1: -24.2212391, 36.5742035, -21.2962494, 32.6550941, -56.8763351, 57.8704529
2: -24.7897606, 35.8414726, -21.7990875, 32.0393410, -56.8291016, 57.6405602
3: -29.7945175, 42.2615585, -26.2362270, 37.6525040, -67.4470062, 68.4977875
4: -27.9134693, 40.0790443, -24.6484203, 35.6282730, -63.5417404, 64.7274628

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9264563, upper bound: 53.1040943
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5260032, upper bound: 53.5260034
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -21.5188446, 39.3366241, -60.8554649, 60.8554688
1: -24.2212391, 36.5742035, -24.2212391, 36.5742035, -60.7954407, 60.7954407
2: -24.7897606, 35.8414726, -24.7897606, 35.8414726, -60.6312332, 60.6312332
3: -29.7945175, 42.2615585, -29.7945175, 42.2615585, -72.0560684, 72.0560684
4: -27.9134693, 40.0790443, -27.9134693, 40.0790443, -67.9925156, 67.9925079

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9264563, upper bound: 53.1040943
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5260032, upper bound: 53.5260034
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -16.5780373, 31.9165230, -43.4071655, 39.3047600
1: -12.9931536, 20.8490124, -18.7406483, 29.0654297, -42.0585823, 39.5896568
2: -13.3803749, 20.5927849, -19.1804962, 28.6155891, -41.9959641, 39.7732773
3: -16.0014496, 23.8214302, -23.1438141, 33.4699211, -49.4713707, 46.9652443
4: -15.2732220, 22.4852295, -21.7336960, 31.7478294, -47.0210457, 44.2189255

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2474690, upper bound: 54.1731727
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2330018, upper bound: 54.1728625
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -18.8832836, 35.7310181, -47.2216721, 41.6100082
1: -12.9931536, 20.8490124, -21.2898827, 32.5063171, -45.4994698, 42.1388931
2: -13.3803749, 20.5927849, -21.8138657, 31.9566841, -45.3370590, 42.4066505
3: -16.0014496, 23.8214302, -26.2260628, 37.5371437, -53.5385933, 50.0474930
4: -15.2732220, 22.4852295, -24.6329060, 35.6249886, -50.8982086, 47.1181335

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2474690, upper bound: 54.1731727
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2330018, upper bound: 54.1728625
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -16.5780373, 31.9165230, -46.0288467, 43.8185196
1: -15.9184780, 24.8546124, -18.7406483, 29.0654297, -44.9839058, 43.5952568
2: -16.3577156, 24.5101395, -19.1804962, 28.6155891, -44.9733047, 43.6906357
3: -19.5635452, 28.4796600, -23.1438141, 33.4699211, -53.0334663, 51.6234703
4: -18.5248909, 27.0098228, -21.7336960, 31.7478294, -50.2727165, 48.7435150

Time for backsubstitution: 2.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1169164, upper bound: 54.1687933
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1990976, upper bound: 54.1721478
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -18.8832836, 35.7310181, -49.8433418, 46.1237640
1: -15.9184780, 24.8546124, -21.2898827, 32.5063171, -48.4247971, 46.1444893
2: -16.3577156, 24.5101395, -21.8138657, 31.9566841, -48.3143997, 46.3240051
3: -19.5635452, 28.4796600, -26.2260628, 37.5371437, -57.1006889, 54.7057114
4: -18.5248909, 27.0098228, -24.6329060, 35.6249886, -54.1498795, 51.6427307

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1169164, upper bound: 54.1687933
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1990976, upper bound: 54.1721478
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.3470850, 22.4635010, -21.4691925, 39.9133835, -51.2604637, 43.9326935
1: -12.8309460, 20.5887737, -24.0923729, 35.6383591, -48.4693069, 44.6811447
2: -13.2166290, 20.3387623, -24.6987610, 35.0490875, -48.2657166, 45.0375214
3: -15.7994022, 23.5187149, -29.4798279, 41.1165810, -56.9159851, 52.9985352
4: -15.0839539, 22.2000656, -27.4715805, 39.2330818, -54.3170357, 49.6716461

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8525378, upper bound: 52.7416897
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3401635, upper bound: 52.7879960
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -25.9870872, 47.8569641, -59.3476219, 48.7138100
1: -12.9931536, 20.8490124, -29.2872486, 44.0551414, -57.0482903, 50.1362572
2: -13.3803749, 20.5927849, -29.9218769, 43.1240997, -56.5044708, 50.5146637
3: -16.0014496, 23.8214302, -36.0748901, 51.0303078, -67.0317535, 59.8963203
4: -15.2732220, 22.4852295, -33.6610641, 48.4680595, -63.7412796, 56.1462822

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2305024, upper bound: 54.1577830
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2218287, upper bound: 54.1575045
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -13.9738646, 26.9974422, -21.4691925, 39.9133835, -53.8872375, 48.4666328
1: -15.7618523, 24.6102581, -24.0923729, 35.6383591, -51.4002113, 48.7026291
2: -16.2004223, 24.2743378, -24.6987610, 35.0490875, -51.2495041, 48.9730988
3: -19.3674469, 28.1945114, -29.4798279, 41.1165810, -60.4840164, 57.6743317
4: -18.3413429, 26.7436256, -27.4715805, 39.2330818, -57.5744209, 54.2152023

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=62.94061279296875
rel_dist={0: [-54.30012139088531, 54.300121390885295]}

## Binary search (step 1) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2601010, upper bound: 54.2644138
time: 0.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2653816, upper bound: 54.2653817
time: 1.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.03
Output dim: 0, lower bound: -54.2601010, upper bound: 54.2644138
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.03
Output dim: 0, lower bound: -54.2653816, upper bound: 54.2653817

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.4868755, 34.6143112, -21.5081902, 39.6732483, -58.1601257, 56.1225014
1: -20.8144035, 32.0120316, -24.2265549, 37.0184097, -57.8328133, 56.2385826
2: -21.3472996, 31.4293518, -24.7906876, 36.2220840, -57.5693779, 56.2200317
3: -25.5772781, 36.8826523, -29.8022003, 42.9040718, -68.4813538, 66.6848450
4: -24.1322937, 34.9039764, -28.0556107, 40.4867325, -64.6190186, 62.9595795

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2558172, upper bound: 54.2558172
time: 1.20 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2558172, upper bound: 54.2644138
time: 1.09 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -23.0712452, 42.9994736, -21.3992023, 39.5110817, -62.5823288, 64.3986664
1: -25.9661083, 39.6324463, -24.0871258, 36.8941040, -62.8602142, 63.7195549
2: -26.6078491, 38.8119164, -24.6748428, 36.1065140, -62.7143555, 63.4867592
3: -31.9601364, 45.9458580, -29.6082268, 42.7722549, -74.7323914, 75.5540695
4: -30.0656071, 43.3875542, -27.9484043, 40.2693329, -70.3349304, 71.3359528

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2644138, upper bound: 54.2558172
time: 1.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2644138, upper bound: 54.2653817
time: 0.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.50 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.50
Output dim: 0, lower bound: -54.2558172, upper bound: 54.2558172
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.50
Output dim: 0, lower bound: -54.2558172, upper bound: 54.2644138
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.50
Output dim: 0, lower bound: -54.2644138, upper bound: 54.2558172
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.50
Output dim: 0, lower bound: -54.2644138, upper bound: 54.2653817

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -18.4868755, 34.6143112, -18.4868755, 34.6143112, -53.1011887, 53.1011887
1: -20.8144035, 32.0120316, -20.8144035, 32.0120316, -52.8264313, 52.8264313
2: -21.3472996, 31.4293518, -21.3472996, 31.4293518, -52.7766495, 52.7766495
3: -25.5772781, 36.8826523, -25.5772781, 36.8826523, -62.4599152, 62.4599152
4: -24.1322937, 34.9039764, -24.1322937, 34.9039764, -59.0362549, 59.0362625

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2317916, upper bound: 53.7836615
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7594598, upper bound: 53.7594598
time: 0.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -18.4868755, 34.6143112, -23.0712452, 42.9994736, -61.4863434, 57.6855545
1: -20.8144035, 32.0120316, -25.9661083, 39.6324463, -60.4468384, 57.9781265
2: -21.3472996, 31.4293518, -26.6078491, 38.8119164, -60.1592178, 58.0372009
3: -25.5772781, 36.8826523, -31.9601364, 45.9458580, -71.5231323, 68.8427887
4: -24.1322937, 34.9039764, -30.0656071, 43.3875542, -67.5198364, 64.9695816

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2317916, upper bound: 54.2643124
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7594598, upper bound: 54.2392734
time: 0.73 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -23.0712452, 42.9994736, -18.4868755, 34.6143112, -57.6855545, 61.4863472
1: -25.9661083, 39.6324463, -20.8144035, 32.0120316, -57.9781265, 60.4468384
2: -26.6078491, 38.8119164, -21.3472996, 31.4293518, -58.0372009, 60.1592178
3: -31.9601364, 45.9458580, -25.5772781, 36.8826523, -68.8427887, 71.5231323
4: -30.0656071, 43.3875542, -24.1322937, 34.9039764, -64.9695740, 67.5198364

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2293454, upper bound: 53.7836113
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2392732, upper bound: 53.7833965
time: 0.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -23.0712452, 42.9994736, -23.0712452, 42.9994736, -66.0707169, 66.0707169
1: -25.9661083, 39.6324463, -25.9661083, 39.6324463, -65.5985489, 65.5985489
2: -26.6078491, 38.8119164, -26.6078491, 38.8119164, -65.4197617, 65.4197617
3: -31.9601364, 45.9458580, -31.9601364, 45.9458580, -77.9059906, 77.9059906
4: -30.0656071, 43.3875542, -30.0656071, 43.3875542, -73.4531631, 73.4531631

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2382124, upper bound: 54.2647096
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2392733, upper bound: 54.2649439
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.35 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -54.2317916, upper bound: 53.7836615
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -53.7594598, upper bound: 53.7594598
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -54.2317916, upper bound: 54.2643124
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -53.7594598, upper bound: 54.2392734
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -54.2293454, upper bound: 53.7836113
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -54.2392732, upper bound: 53.7833965
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -54.2382124, upper bound: 54.2647096
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 0, lower bound: -54.2392733, upper bound: 54.2649439

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -17.7262897, 33.3611679, -48.4256859, 46.6222687
1: -16.9958324, 26.4444580, -19.9674664, 30.7858047, -47.7816391, 46.4119072
2: -17.4429550, 26.0576000, -20.4798756, 30.2479630, -47.6909103, 46.5374680
3: -20.9023075, 30.3311520, -24.5445423, 35.4331741, -56.3354797, 54.8756828
4: -19.7728100, 28.7748013, -23.1667061, 33.5508347, -53.3236351, 51.9415016

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7594598, upper bound: 53.7594598
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7594598, upper bound: 53.7594598
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -17.9830551, 33.6858978, -56.3745270, 59.3786774
1: -25.5468903, 38.5653343, -20.2545319, 31.0862694, -56.6331596, 58.8198662
2: -26.1332932, 37.7651024, -20.7698936, 30.5436668, -56.6769524, 58.5349960
3: -31.4312077, 44.6071281, -24.8967247, 35.7990417, -67.2302475, 69.5038452
4: -29.4743004, 42.2834320, -23.4569626, 33.9364319, -63.4107323, 65.7403870

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6025004, upper bound: 53.7349605
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -22.3039627, 41.6913033, -56.7558250, 51.1999397
1: -16.9958324, 26.4444580, -25.1158524, 38.3436852, -55.3395157, 51.5603065
2: -17.4429550, 26.0576000, -25.7319889, 37.5726166, -55.0155716, 51.7895851
3: -20.9023075, 30.3311520, -30.9268894, 44.4202881, -65.3225784, 61.2580414
4: -19.7728100, 28.7748013, -29.0826683, 41.9940147, -61.7668190, 57.8574677

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7995693, upper bound: 52.7497363
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2409538, upper bound: 54.2520856
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -22.5696735, 42.0886650, -64.7772980, 63.9652939
1: -25.5468903, 38.5653343, -25.4086132, 38.7819443, -64.3288345, 63.9739342
2: -26.1332932, 37.7651024, -26.0352268, 37.9914856, -64.1247711, 63.8003311
3: -31.4312077, 44.6071281, -31.2806778, 44.9415627, -76.3727570, 75.8877869
4: -29.4743004, 42.2834320, -29.4159775, 42.4662018, -71.9405060, 71.6994019

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7949795, upper bound: 52.7480904
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7422034, upper bound: 54.2239323
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -17.7262897, 33.3611679, -53.2634354, 55.3048401
1: -22.4450703, 34.2541313, -19.9674664, 30.7858047, -53.2308731, 54.2215919
2: -22.9841385, 33.6525192, -20.4798756, 30.2479630, -53.2320862, 54.1323814
3: -27.6680088, 39.5840874, -24.5445423, 35.4331741, -63.1011810, 64.1286316
4: -25.9807587, 37.5789795, -23.1667061, 33.5508347, -59.5315933, 60.7456741

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2382123, upper bound: 53.7833965
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2382123, upper bound: 53.7833965
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -17.9830551, 33.6858978, -60.6132317, 67.5071106
1: -30.3515205, 45.6756859, -20.2545319, 31.0862694, -61.4377899, 65.9302063
2: -31.0041237, 44.6888275, -20.7698936, 30.5436668, -61.5477905, 65.4587021
3: -37.3951302, 52.9274445, -24.8967247, 35.7990417, -73.1941681, 77.8241730
4: -34.8959274, 50.2624893, -23.4569626, 33.9364319, -68.8323364, 73.7194443

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9644818, upper bound: 53.7625512
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0085708, upper bound: 53.6147131
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -22.3039627, 41.6913033, -61.5935707, 59.8825035
1: -22.4450703, 34.2541313, -25.1158524, 38.3436852, -60.7887573, 59.3699837
2: -22.9841385, 33.6525192, -25.7319889, 37.5726166, -60.5567398, 59.3845024
3: -27.6680088, 39.5840874, -30.9268894, 44.4202881, -72.0882950, 70.5109787
4: -25.9807587, 37.5789795, -29.0826683, 41.9940147, -67.9747772, 66.6616440

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7882941, upper bound: 52.7470828
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2517502, upper bound: 54.2527370
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -22.5696735, 42.0886650, -69.0159988, 72.0937195
1: -30.3515205, 45.6756859, -25.4086132, 38.7819443, -69.1334686, 71.0842743
2: -31.0041237, 44.6888275, -26.0352268, 37.9914856, -68.9956055, 70.7240524
3: -37.3951302, 52.9274445, -31.2806778, 44.9415627, -82.3366928, 84.2081146
4: -34.8959274, 50.2624893, -29.4159775, 42.4662018, -77.3621216, 79.6784668

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7939578, upper bound: 52.7474536
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2530181, upper bound: 54.2529382
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.33 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -53.7594598, upper bound: 53.7594598
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -53.7594598, upper bound: 53.7594598
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -53.6025004, upper bound: 53.7349605
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -52.7995693, upper bound: 52.7497363
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -54.2409538, upper bound: 54.2520856
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -52.7949795, upper bound: 52.7480904
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -53.7422034, upper bound: 54.2239323
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -54.2382123, upper bound: 53.7833965
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -54.2382123, upper bound: 53.7833965
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -53.9644818, upper bound: 53.7625512
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -54.0085708, upper bound: 53.6147131
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -52.7882941, upper bound: 52.7470828
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -54.2517502, upper bound: 54.2527370
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -52.7939578, upper bound: 52.7474536
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.33
Output dim: 0, lower bound: -54.2530181, upper bound: 54.2529382

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -15.0645208, 28.8959808, -43.9605026, 43.9605026
1: -16.9958324, 26.4444580, -16.9958324, 26.4444580, -43.4402847, 43.4402809
2: -17.4429550, 26.0576000, -17.4429550, 26.0576000, -43.5005569, 43.5005531
3: -20.9023075, 30.3311520, -20.9023075, 30.3311520, -51.2334595, 51.2334595
4: -19.7728100, 28.7748013, -19.7728100, 28.7748013, -48.5476074, 48.5476074

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1001539, upper bound: 53.6229637
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9879569, upper bound: 53.6135342
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -22.6886387, 41.3956223, -56.4601440, 51.5846176
1: -16.9958324, 26.4444580, -25.5468903, 38.5653343, -55.5611649, 51.9913445
2: -17.4429550, 26.0576000, -26.1332932, 37.7651024, -55.2080574, 52.1908913
3: -20.9023075, 30.3311520, -31.4312077, 44.6071281, -65.5094147, 61.7623596
4: -19.7728100, 28.7748013, -29.4743004, 42.2834320, -62.0562286, 58.2490997

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1001539, upper bound: 53.6229637
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9879569, upper bound: 53.6135342
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.1735744, 40.4536667, -14.1795368, 27.2102909, -49.3838654, 54.6332016
1: -24.9678097, 37.7618866, -15.9920120, 25.1330872, -50.1008949, 53.7538948
2: -25.5432529, 36.9840622, -16.4357147, 24.7750473, -50.3182983, 53.4197731
3: -30.7247219, 43.6692009, -19.6724014, 28.8499527, -59.5746765, 63.3416023
4: -28.8320293, 41.3713188, -18.6232147, 27.2906284, -56.1226540, 59.9945259

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9378173, upper bound: 53.1694494
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.6154995, 41.2647095, -16.9253654, 31.8555069, -54.4710045, 58.1900711
1: -25.4640388, 38.4398422, -19.0558167, 29.3090401, -54.7730713, 57.4956589
2: -26.0492439, 37.6439667, -19.5580559, 28.8210125, -54.8702545, 57.2020226
3: -31.3288937, 44.4591980, -23.4104767, 33.7298431, -65.0587387, 67.8696594
4: -29.3765526, 42.1444778, -22.0626602, 31.9714603, -61.3480148, 64.2071381

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.1573372, 25.5415192, -16.7984924, 31.9844475, -45.1417847, 42.3400078
1: -14.8389606, 23.0743694, -18.8279457, 28.1917534, -43.0307159, 41.9023132
2: -15.2664061, 22.8050270, -19.3691444, 27.7948418, -43.0612488, 42.1741714
3: -18.2035694, 26.3889160, -22.9624577, 32.5395660, -50.7431335, 49.3513718
4: -17.2332630, 25.1046200, -21.5983162, 30.9085541, -48.1418152, 46.7029266

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7995693, upper bound: 52.7497363
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7836722, upper bound: 52.7471660
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15.0645208, 28.8959808, -21.3621407, 40.0312691, -55.0957909, 50.2581215
1: -16.9958324, 26.4444580, -24.0533676, 36.7257309, -53.7215652, 50.4978180
2: -17.4429550, 26.0576000, -24.6515255, 36.0096474, -53.4526024, 50.7091255
3: -20.9023075, 30.3311520, -29.6106834, 42.5240974, -63.4264030, 59.9418335
4: -19.7728100, 28.7748013, -27.8500595, 40.2059212, -59.9787292, 56.6248550

Time for backsubstitution: 2.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2205829, upper bound: 54.1561691
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0474053, upper bound: 54.1537721
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -20.7667236, 38.0892105, -16.9123707, 32.1303177, -52.8970337, 55.0015755
1: -23.3658791, 35.2087669, -18.9600563, 28.3687134, -51.7345886, 54.1688232
2: -23.9299965, 34.5416870, -19.5022812, 27.9627380, -51.8927116, 54.0439644
3: -28.7107162, 40.6532211, -23.1261082, 32.7467270, -61.4574432, 63.7793236
4: -26.9199791, 38.5900764, -21.7377396, 31.1066132, -58.0265923, 60.3278084

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7949795, upper bound: 52.7480904
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7037247, upper bound: 52.7312393
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.6886387, 41.3956223, -21.6140747, 40.4032669, -63.0918846, 63.0096970
1: -25.5468903, 38.5653343, -24.3307667, 37.1392136, -62.6861038, 62.8961029
2: -26.1332932, 37.7651024, -24.9391232, 36.4053955, -62.5386696, 62.7042236
3: -31.4312077, 44.6071281, -29.9461803, 43.0156059, -74.4468002, 74.5532837
4: -29.4743004, 42.2834320, -28.1648712, 40.6520615, -70.1263580, 70.4482880

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7194768, upper bound: 53.9448966
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5681997, upper bound: 53.9929289
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -15.0645208, 28.8959808, -48.7982483, 52.6430664
1: -22.4450703, 34.2541313, -16.9958324, 26.4444580, -48.8895149, 51.2499619
2: -22.9841385, 33.6525192, -17.4429550, 26.0576000, -49.0417290, 51.0954666
3: -27.6680088, 39.5840874, -20.9023075, 30.3311520, -57.9991608, 60.4863968
4: -25.9807587, 37.5789795, -19.7728100, 28.7748013, -54.7555618, 57.3517838

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0972317, upper bound: 53.6227026
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0110913, upper bound: 53.6160538
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -22.6886387, 41.3956223, -61.2978897, 60.2671814
1: -22.4450703, 34.2541313, -25.5468903, 38.5653343, -61.0104065, 59.8010216
2: -22.9841385, 33.6525192, -26.1332932, 37.7651024, -60.7492409, 59.7858086
3: -27.6680088, 39.5840874, -31.4312077, 44.6071281, -72.2751236, 71.0152817
4: -25.9807587, 37.5789795, -29.4743004, 42.2834320, -68.2641907, 67.0532837

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0972317, upper bound: 53.6227026
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0110913, upper bound: 53.6160538
time: 1.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -26.4240799, 48.5659599, -14.1795368, 27.2102909, -53.6343689, 62.7454987
1: -29.7855968, 44.8685226, -15.9920120, 25.1330872, -54.9186783, 60.8605232
2: -30.4237957, 43.9020309, -16.4357147, 24.7750473, -55.1988335, 60.3377457
3: -36.7051964, 51.9935112, -19.6724014, 28.8499527, -65.5551376, 71.6658936
4: -34.2674866, 49.3483467, -18.6232147, 27.2906284, -61.5581131, 67.9715576

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7480904, upper bound: 52.7949795
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9448966, upper bound: 53.7194768
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -26.8571014, 49.3908997, -16.9253654, 31.8555069, -58.7126007, 66.3162613
1: -30.2719021, 45.5512772, -19.0558167, 29.3090401, -59.5809402, 64.6070862
2: -30.9230328, 44.5686722, -19.5580559, 28.8210125, -59.7440453, 64.1267242
3: -37.2971649, 52.7829399, -23.4104767, 33.7298431, -71.0269928, 76.1934128
4: -34.8030319, 50.1253777, -22.0626602, 31.9714603, -66.7744827, 72.1880341

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7312393, upper bound: 52.7037247
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9929289, upper bound: 53.5681997
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17.9686852, 34.1765518, -16.7984924, 31.9844475, -49.9531326, 50.9750443
1: -20.2515030, 30.7857094, -18.8279457, 28.1917534, -48.4432564, 49.6136513
2: -20.7512856, 30.3308468, -19.3691444, 27.7948418, -48.5461273, 49.6999893
3: -24.9203167, 35.4998474, -22.9624577, 32.5395660, -57.4598846, 58.4623032
4: -23.3792572, 33.7958412, -21.5983162, 30.9085541, -54.2878113, 55.3941460

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7882941, upper bound: 52.7470828
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7876468, upper bound: 52.7469846
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -19.9022694, 37.5785522, -21.3621407, 40.0312691, -59.9335365, 58.9406929
1: -22.4450703, 34.2541313, -24.0533676, 36.7257309, -59.1707954, 58.3074913
2: -22.9841385, 33.6525192, -24.6515255, 36.0096474, -58.9937630, 58.3040352
3: -27.6680088, 39.5840874, -29.6106834, 42.5240974, -70.1921005, 69.1947708
4: -25.9807587, 37.5789795, -27.8500595, 40.2059212, -66.1866760, 65.4290314

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2380054, upper bound: 54.1659877
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1651883, upper bound: 54.1633062
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -25.0109787, 46.1318817, -16.9123707, 32.1303177, -57.1412964, 63.0442505
1: -28.1714668, 42.2596893, -18.9600563, 28.3687134, -56.5401726, 61.2197418
2: -28.8003483, 41.4147339, -19.5022812, 27.9627380, -56.7630730, 60.9170151
3: -34.6842232, 48.8824120, -23.1261082, 32.7467270, -67.4309540, 72.0085144
4: -32.3183784, 46.5211334, -21.7377396, 31.1066132, -63.4249878, 68.2588501

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7882941, upper bound: 52.7474536
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7900742, upper bound: 52.7464638
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -26.9273338, 49.5240555, -21.6140747, 40.4032669, -67.3305969, 71.1381302
1: -30.3515205, 45.6756859, -24.3307667, 37.1392136, -67.4907303, 70.0064468
2: -31.0041237, 44.6888275, -24.9391232, 36.4053955, -67.4095001, 69.6279373
3: -37.3951302, 52.9274445, -29.9461803, 43.0156059, -80.4107361, 82.8736191
4: -34.8959274, 50.2624893, -28.1648712, 40.6520615, -75.5479813, 78.4273605

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7474536, upper bound: 52.7914898
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7474536, upper bound: 54.2529384
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.14 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.1001539, upper bound: 53.6229637
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.9879569, upper bound: 53.6135342
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.1001539, upper bound: 53.6229637
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.9879569, upper bound: 53.6135342
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.9378173, upper bound: 53.1694494
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7995693, upper bound: 52.7497363
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7836722, upper bound: 52.7471660
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.2205829, upper bound: 54.1561691
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.0474053, upper bound: 54.1537721
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7949795, upper bound: 52.7480904
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7037247, upper bound: 52.7312393
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.7194768, upper bound: 53.9448966
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.5681997, upper bound: 53.9929289
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.0972317, upper bound: 53.6227026
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.0110913, upper bound: 53.6160538
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.0972317, upper bound: 53.6227026
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.0110913, upper bound: 53.6160538
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7480904, upper bound: 52.7949795
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.9448966, upper bound: 53.7194768
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7312393, upper bound: 52.7037247
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -53.9929289, upper bound: 53.5681997
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7882941, upper bound: 52.7470828
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7876468, upper bound: 52.7469846
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.2380054, upper bound: 54.1659877
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -54.1651883, upper bound: 54.1633062
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7882941, upper bound: 52.7474536
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7900742, upper bound: 52.7464638
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7474536, upper bound: 52.7914898
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.14
Output dim: 0, lower bound: -52.7474536, upper bound: 54.2529384

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -14.5599203, 27.9834175, -39.4740677, 37.2866440
1: -12.9931536, 20.8490124, -16.4297619, 25.6742134, -38.6673660, 37.2787743
2: -13.3803749, 20.5927849, -16.8679676, 25.3052940, -38.6856537, 37.4607468
3: -16.0014496, 23.8214302, -20.2104073, 29.4367676, -45.4382095, 44.0318375
4: -15.2732220, 22.4852295, -19.1528549, 27.8993111, -43.1725235, 41.6380806

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2306948, upper bound: 54.1190121
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2313976, upper bound: 54.2010919
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -14.9943523, 28.7740726, -42.8863945, 42.2348328
1: -15.9184780, 24.8546124, -16.9170227, 26.3280640, -42.2465363, 41.7716255
2: -16.3577156, 24.5101395, -17.3627224, 25.9443417, -42.3020554, 41.8728638
3: -19.5635452, 28.4796600, -20.8051300, 30.1953812, -49.7589264, 49.2847786
4: -18.5248909, 27.0098228, -19.6811256, 28.6462555, -47.1711464, 46.6909485

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -22.1735744, 40.4536667, -51.9443245, 44.9002991
1: -12.9931536, 20.8490124, -24.9678097, 37.7618866, -50.7550392, 45.8168221
2: -13.3803749, 20.5927849, -25.5432529, 36.9840622, -50.3644333, 46.1360397
3: -16.0014496, 23.8214302, -30.7247219, 43.6692009, -59.6706505, 54.5461502
4: -15.2732220, 22.4852295, -28.8320293, 41.3713188, -56.6445389, 51.3172493

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1874475, upper bound: 52.9432926
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0735857, upper bound: 53.5763900
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -22.6154995, 41.2647095, -55.3770332, 49.8559799
1: -15.9184780, 24.8546124, -25.4640388, 38.4398422, -54.3583183, 50.3186417
2: -16.3577156, 24.5101395, -26.0492439, 37.6439667, -54.0016785, 50.5593834
3: -19.5635452, 28.4796600, -31.3288937, 44.4591980, -64.0227356, 59.8085480
4: -18.5248909, 27.0098228, -29.3765526, 42.1444778, -60.6693611, 56.3863754

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9759394, upper bound: 53.6135342
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9759394, upper bound: 53.6135342
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -16.9955292, 31.4307480, -12.1702232, 23.6434784, -40.6390076, 43.6009674
1: -19.0474510, 28.3562393, -13.7246428, 21.5781441, -40.6255951, 42.0808830
2: -19.5802612, 27.9446602, -14.1476650, 21.3224716, -40.9027290, 42.0923195
3: -23.2360268, 32.6596794, -16.8367805, 24.7001057, -47.9361343, 49.4964600
4: -21.8171463, 31.0525780, -15.9731979, 23.3990479, -45.2161865, 47.0257683

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9378173, upper bound: 53.1694494
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9378173, upper bound: 53.1694494
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.2581940, 38.8678398, -14.1795368, 27.2102909, -48.4684792, 53.0473785
1: -23.9263000, 36.1767044, -15.9920120, 25.1330872, -49.0593834, 52.1687164
2: -24.4926319, 35.4565048, -16.4357147, 24.7750473, -49.2676773, 51.8922195
3: -29.4232903, 41.8080063, -19.6724014, 28.8499527, -58.2732430, 61.4804077
4: -27.6137218, 39.6126175, -18.6232147, 27.2906284, -54.9043503, 58.2358208

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -16.9253654, 31.8555069, -50.7553024, 51.8979225
1: -21.2962494, 32.6550941, -19.0558167, 29.3090401, -50.6052780, 51.7109108
2: -21.7990875, 32.0393410, -19.5580559, 28.8210125, -50.6201019, 51.5973969
3: -26.2362270, 37.6525040, -23.4104767, 33.7298431, -59.9660683, 61.0629654
4: -24.6484203, 35.6282730, -22.0626602, 31.9714603, -56.6198730, 57.6909294

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -16.9253654, 31.8555069, -53.3743515, 56.2619858
1: -24.2212391, 36.5742035, -19.0558167, 29.3090401, -53.5302696, 55.6300201
2: -24.7897606, 35.8414726, -19.5580559, 28.8210125, -53.6107712, 55.3995285
3: -29.7945175, 42.2615585, -23.4104767, 33.7298431, -63.5243530, 65.6720200
4: -27.9134693, 40.0790443, -22.0626602, 31.9714603, -59.8849297, 62.1416931

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.5783987, 19.1222038, -16.3331852, 31.1085052, -40.6869049, 35.4553909
1: -10.8226967, 17.2935429, -18.3060932, 27.4781418, -38.3008385, 35.5996361
2: -11.1932211, 17.1231270, -18.8379040, 27.0985107, -38.2917328, 35.9610252
3: -13.2758369, 19.6837635, -22.3266125, 31.7077999, -44.9836349, 42.0103645
4: -12.7250004, 18.6032982, -21.0232677, 30.0897007, -42.8147011, 39.6265640

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.0217462, upper bound: 51.9668710
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7658579, upper bound: 52.7448737
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0554867, upper bound: 52.4391226
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7811695, upper bound: 52.7424015
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.2104034, 23.8924599, -16.7483349, 31.8932800, -44.1036835, 40.6407928
1: -13.7698116, 21.5079422, -18.7718849, 28.1076603, -41.8774643, 40.2798271
2: -14.1916609, 21.2636547, -19.3111820, 27.7130299, -41.9046898, 40.5748367
3: -16.8770905, 24.5696297, -22.8938198, 32.4401932, -49.3172836, 47.4634476
4: -16.0110855, 23.3445320, -21.5312023, 30.8160248, -46.8271103, 44.8757210

Time for backsubstitution: 2.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.0425079, upper bound: 51.7652923
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.6938536, upper bound: 50.7967206
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.1454683, upper bound: 50.5910466
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -20.8611736, 39.0893936, -50.5800476, 43.5878983
1: -12.9931536, 20.8490124, -23.4883480, 35.9333153, -48.9264679, 44.3373489
2: -13.3803749, 20.5927849, -24.0778732, 35.2381516, -48.6185265, 44.6706543
3: -16.0014496, 23.8214302, -28.9221554, 41.6024208, -57.6038704, 52.7435837
4: -15.2732220, 22.4852295, -27.2251892, 39.3023758, -54.5755959, 49.7104187

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2205829, upper bound: 54.1561691
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2205829, upper bound: 54.1561689
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -21.2916794, 39.9001999, -54.0125237, 48.5321579
1: -15.9184780, 24.8546124, -23.9737625, 36.6023293, -52.5208054, 48.8283577
2: -16.3577156, 24.5101395, -24.5705719, 35.8904724, -52.2481880, 49.0807114
3: -19.5635452, 28.4796600, -29.5124683, 42.3796844, -61.9432144, 57.9921150
4: -18.5248909, 27.0098228, -27.7570133, 40.0695419, -58.5944252, 54.7668228

Time for backsubstitution: 2.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0474053, upper bound: 54.1537720
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0474053, upper bound: 54.1537720
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -16.8578339, 31.3709641, -16.4512005, 31.2597485, -48.1175766, 47.8221664
1: -18.9779472, 29.0129738, -18.4427872, 27.6620998, -46.6400452, 47.4557610
2: -19.4568291, 28.5502281, -18.9762115, 27.2734871, -46.7303162, 47.5264359
3: -23.3346539, 33.3619919, -22.4961109, 31.9234085, -55.2580643, 55.8581009
4: -21.9063911, 31.6608067, -21.1692657, 30.2947350, -52.2011185, 52.8300705

Time for backsubstitution: 2.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -50.6025806, upper bound: 51.4797376
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7594125, upper bound: 52.7424360
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0529014, upper bound: 52.4381774
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7769839, upper bound: 52.7408317
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -19.5651665, 35.9848518, -16.8616734, 32.0378685, -51.6030350, 52.8465195
1: -22.0042305, 33.1786346, -18.9033546, 28.2836037, -50.2878342, 52.0819893
2: -22.5499840, 32.5748367, -19.4437141, 27.8799572, -50.4299393, 52.0185509
3: -27.0253048, 38.2681503, -23.0565834, 32.6461716, -59.6714783, 61.3247299
4: -25.3242493, 36.3369751, -21.6699314, 31.0128555, -56.3370895, 58.0069046

Time for backsubstitution: 2.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -50.7305051, upper bound: 51.4043880
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.7315104, upper bound: 50.8188574
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.1566162, upper bound: 50.6109180
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -21.1190224, 39.4698715, -58.3696709, 56.0915756
1: -21.2962494, 32.6550941, -23.7724209, 36.3579178, -57.6541672, 56.4275055
2: -21.7990875, 32.0393410, -24.3721924, 35.6446457, -57.4437294, 56.4115334
3: -26.2362270, 37.6525040, -29.2663250, 42.1073799, -68.3436050, 66.9188232
4: -24.6484203, 35.6282730, -27.5483570, 39.7604980, -64.4089127, 63.1766281

Time for backsubstitution: 2.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7194768, upper bound: 53.9448968
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6878929, upper bound: 53.9448966
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -21.5426750, 40.2702560, -61.7890778, 60.8792992
1: -24.2212391, 36.5742035, -24.2501221, 37.0141602, -61.2353973, 60.8243256
2: -24.7897606, 35.8414726, -24.8571301, 36.2846451, -61.0744057, 60.6985893
3: -29.7945175, 42.2615585, -29.8467636, 42.8692055, -72.6637268, 72.1083145
4: -27.9134693, 40.0790443, -28.0706463, 40.5139732, -68.4274368, 68.1496887

Time for backsubstitution: 2.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5681997, upper bound: 53.9929289
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5681997, upper bound: 53.9929289
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16.5780373, 31.9165230, -14.5599203, 27.9834175, -44.5614548, 46.4764404
1: -18.7406483, 29.0654297, -16.4297619, 25.6742134, -44.4148636, 45.4951935
2: -19.1804962, 28.6155891, -16.8679676, 25.3052940, -44.4857788, 45.4835587
3: -23.1438141, 33.4699211, -20.2104073, 29.4367676, -52.5805779, 53.6803284
4: -21.7336960, 31.7478294, -19.1528549, 27.8993111, -49.6329994, 50.9006767

Time for backsubstitution: 2.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2245388, upper bound: 54.1154324
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2337400, upper bound: 54.1967491
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18.8832836, 35.7310181, -14.9943523, 28.7740726, -47.6573563, 50.7253723
1: -21.2898827, 32.5063171, -16.9170227, 26.3280640, -47.6179428, 49.4233398
2: -21.8138657, 31.9566841, -17.3627224, 25.9443417, -47.7582092, 49.3194046
3: -26.2260628, 37.5371437, -20.8051300, 30.1953812, -56.4214439, 58.3422585
4: -24.6329060, 35.6249886, -19.6811256, 28.6462555, -53.2791595, 55.3061142

Time for backsubstitution: 2.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1794983, upper bound: 54.2013493
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1794983, upper bound: 54.2013493
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.5780373, 31.9165230, -22.1735744, 40.4536667, -57.0317039, 54.0900879
1: -18.7406483, 29.0654297, -24.9678097, 37.7618866, -56.5025291, 54.0332413
2: -19.1804962, 28.6155891, -25.5432529, 36.9840622, -56.1645508, 54.1588440
3: -23.1438141, 33.4699211, -30.7247219, 43.6692009, -66.8130188, 64.1946411
4: -21.7336960, 31.7478294, -28.8320293, 41.3713188, -63.1050148, 60.5798531

Time for backsubstitution: 2.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1628304, upper bound: 52.9370227
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0705814, upper bound: 53.5761006
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.8832836, 35.7310181, -22.6154995, 41.2647095, -60.1479874, 58.3465195
1: -21.2898827, 32.5063171, -25.4640388, 38.4398422, -59.7297249, 57.9703560
2: -21.8138657, 31.9566841, -26.0492439, 37.6439667, -59.4578323, 58.0059242
3: -26.2260628, 37.5371437, -31.3288937, 44.4591980, -70.6852570, 68.8660202
4: -24.6329060, 35.6249886, -29.3765526, 42.1444778, -66.7773819, 65.0015411

Time for backsubstitution: 2.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9736499, upper bound: 53.6160538
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9736499, upper bound: 53.6160538
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -20.9906769, 38.9875908, -12.1702232, 23.6434784, -44.6341515, 51.1578140
1: -23.5563202, 34.8806992, -13.7246428, 21.5781441, -45.1344643, 48.6053429
2: -24.1495247, 34.3087502, -14.1476650, 21.3224716, -45.4719925, 48.4564133
3: -28.8277397, 40.2391281, -16.8367805, 24.7001057, -53.5278397, 57.0759087
4: -26.8794594, 38.3673820, -15.9731979, 23.3990479, -50.2784996, 54.3405762

Time for backsubstitution: 2.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7480904, upper bound: 52.7949795
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7480904, upper bound: 52.7949795
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -25.4969425, 46.9239922, -14.1795368, 27.2102909, -52.7072334, 61.1035233
1: -28.7361279, 43.2721252, -15.9920120, 25.1330872, -53.8692131, 59.2641373
2: -29.3582783, 42.3605347, -16.4357147, 24.7750473, -54.1333199, 58.7962494
3: -35.4030685, 50.1249962, -19.6724014, 28.8499527, -64.2530060, 69.7973938
4: -33.0499153, 47.5793724, -18.6232147, 27.2906284, -60.3405457, 66.2025909

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9448966, upper bound: 53.7194768
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9448966, upper bound: 53.7194768
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.4144440, 39.8041153, -15.0410728, 28.5634613, -49.9779053, 54.8451881
1: -24.0307846, 35.5404816, -16.9191952, 25.9849243, -50.0157051, 52.4596748
2: -24.6355057, 34.9542770, -17.4041252, 25.6217384, -50.2572403, 52.3583946
3: -29.4043121, 41.0026169, -20.7295551, 29.8395424, -59.2438545, 61.7321625
4: -27.3987236, 39.1260605, -19.5538940, 28.3533859, -55.7521095, 58.6799469

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7312393, upper bound: 52.7037247
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7312393, upper bound: 52.7037247
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -25.9151077, 47.7208557, -16.9253654, 31.8555069, -57.7706070, 64.6462250
1: -29.2056484, 43.9278870, -19.0558167, 29.3090401, -58.5146790, 62.9837036
2: -29.8389282, 43.0013657, -19.5580559, 28.8210125, -58.6599426, 62.5594101
3: -35.9744797, 50.8823891, -23.4104767, 33.7298431, -69.7043228, 74.2928467
4: -33.5658913, 48.3280373, -22.0626602, 31.9714603, -65.5373535, 70.3907013

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9929289, upper bound: 53.5681997
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9929289, upper bound: 53.5681997
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14.6527462, 28.4891777, -16.3331852, 31.1085052, -45.7612495, 44.8223648
1: -16.5521927, 25.5925064, -18.3060932, 27.4781418, -44.0303345, 43.8985977
2: -16.9544601, 25.2674217, -18.8379040, 27.0985107, -44.0529671, 44.1053162
3: -20.3944798, 29.4170628, -22.3266125, 31.7077999, -52.1022797, 51.7436752
4: -19.1556625, 27.9590588, -21.0232677, 30.0897007, -49.2453575, 48.9823265

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7526769, upper bound: 52.7418444
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0398884, upper bound: 52.4357760
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7709742, upper bound: 52.7399196
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16.9437923, 32.3259010, -16.7483349, 31.8932800, -48.8370743, 49.0742340
1: -19.0906372, 29.0419483, -18.7718849, 28.1076603, -47.1982956, 47.8138351
2: -19.5795135, 28.6301384, -19.3111820, 27.7130299, -47.2925415, 47.9413223
3: -23.4713097, 33.4658127, -22.8938198, 32.4401932, -55.9115028, 56.3596344
4: -22.0297527, 31.8547611, -21.5312023, 30.8160248, -52.8457794, 53.3859520

Time for backsubstitution: 2.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7505017, upper bound: 52.7418977
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0015649, upper bound: 52.4279096
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7707306, upper bound: 52.7398814
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -16.5780373, 31.9165230, -20.8611736, 39.0893936, -55.6674309, 52.7776871
1: -18.7406483, 29.0654297, -23.4883480, 35.9333153, -54.6739655, 52.5537796
2: -19.1804962, 28.6155891, -24.0778732, 35.2381516, -54.4186478, 52.6934624
3: -23.1438141, 33.4699211, -28.9221554, 41.6024208, -64.7462311, 62.3920746
4: -21.7336960, 31.7478294, -27.2251892, 39.3023758, -61.0360718, 58.9730110

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2380054, upper bound: 54.1659877
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2380054, upper bound: 54.1659877
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.8832836, 35.7310181, -21.2916794, 39.9001999, -58.7834663, 57.0226974
1: -21.2898827, 32.5063171, -23.9737625, 36.6023293, -57.8922119, 56.4800758
2: -21.8138657, 31.9566841, -24.5705719, 35.8904724, -57.7043381, 56.5272560
3: -26.2260628, 37.5371437, -29.5124683, 42.3796844, -68.6057434, 67.0495911
4: -24.6329060, 35.6249886, -27.7570133, 40.0695419, -64.7024460, 63.3819771

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1651883, upper bound: 54.1633062
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1651883, upper bound: 54.1633062
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.5215797, 40.0351677, -16.4512005, 31.2597485, -52.7813263, 56.4863625
1: -24.2690544, 36.7058296, -18.4427872, 27.6620998, -51.9311523, 55.1486168
2: -24.7971802, 36.0255013, -18.9762115, 27.2734871, -52.0706673, 55.0017128
3: -29.9228897, 42.3549805, -22.4961109, 31.9234085, -61.8462982, 64.8510818
4: -27.8625126, 40.2867165, -21.1692657, 30.2947350, -58.1572342, 61.4559822

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7527262, upper bound: 52.7413726
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0424892, upper bound: 52.4358202
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7765513, upper bound: 52.7402502
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -23.9168224, 44.0330849, -16.8616734, 32.0378685, -55.9546890, 60.8947601
1: -26.9250317, 40.3184280, -18.9033546, 28.2836037, -55.2086334, 59.2217827
2: -27.5417709, 39.5367241, -19.4437141, 27.8799572, -55.4217300, 58.9804382
3: -33.1392670, 46.6385422, -23.0565834, 32.6461716, -65.7854309, 69.6951294
4: -30.8757000, 44.3760605, -21.6699314, 31.0128555, -61.8885422, 66.0459900

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.6592335, upper bound: 50.8286423
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.1615771, upper bound: 50.6146162
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.4691925, 39.9133835, -21.6140747, 40.4032669, -61.8724480, 61.5274582
1: -24.0923729, 35.6383591, -24.3307667, 37.1392136, -61.2315712, 59.9691124
2: -24.6987610, 35.0490875, -24.9391232, 36.4053955, -61.1041451, 59.9882011
3: -29.4798279, 41.1165810, -29.9461803, 43.0156059, -72.4954224, 71.0627441
4: -27.4715805, 39.2330818, -28.1648712, 40.6520615, -68.1236420, 67.3979416

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.6575237, upper bound: 50.8354814
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -49.8091960, upper bound: 50.1566159
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -25.9870872, 47.8569641, -21.6140747, 40.4032669, -66.3903351, 69.4710312
1: -29.2872486, 44.0551414, -24.3307667, 37.1392136, -66.4264526, 68.3858948
2: -29.9218769, 43.1240997, -24.9391232, 36.4053955, -66.3272552, 68.0632248
3: -36.0748901, 51.0303078, -29.9461803, 43.0156059, -79.0904999, 80.9764709
4: -33.6610641, 48.4680595, -28.1648712, 40.6520615, -74.3131256, 76.6329346

Time for backsubstitution: 2.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -50.6575237, upper bound: 51.4148433
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -49.8091960, upper bound: 50.9669972
time: 1.17 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.11 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2306948, upper bound: 54.1190121
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2313976, upper bound: 54.2010919
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2072832, upper bound: 54.2072832
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.1874475, upper bound: 52.9432926
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.0735857, upper bound: 53.5763900
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.9759394, upper bound: 53.6135342
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.9759394, upper bound: 53.6135342
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.9378173, upper bound: 53.1694494
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.9378173, upper bound: 53.1694494
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.5781031, upper bound: 53.5781031
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.0554867, upper bound: 52.4391226
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7811695, upper bound: 52.7424015
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -50.6938536, upper bound: 50.7967206
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -50.1454683, upper bound: 50.5910466
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2205829, upper bound: 54.1561691
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2205829, upper bound: 54.1561689
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.0474053, upper bound: 54.1537720
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.0474053, upper bound: 54.1537720
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.0529014, upper bound: 52.4381774
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7769839, upper bound: 52.7408317
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -50.7315104, upper bound: 50.8188574
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -50.1566162, upper bound: 50.6109180
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.7194768, upper bound: 53.9448968
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.6878929, upper bound: 53.9448966
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.5681997, upper bound: 53.9929289
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.5681997, upper bound: 53.9929289
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2245388, upper bound: 54.1154324
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2337400, upper bound: 54.1967491
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.1794983, upper bound: 54.2013493
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.1794983, upper bound: 54.2013493
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.1628304, upper bound: 52.9370227
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.0705814, upper bound: 53.5761006
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.9736499, upper bound: 53.6160538
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.9736499, upper bound: 53.6160538
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7480904, upper bound: 52.7949795
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7480904, upper bound: 52.7949795
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.9448966, upper bound: 53.7194768
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.9448966, upper bound: 53.7194768
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7312393, upper bound: 52.7037247
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7312393, upper bound: 52.7037247
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.9929289, upper bound: 53.5681997
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -53.9929289, upper bound: 53.5681997
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.0398884, upper bound: 52.4357760
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7709742, upper bound: 52.7399196
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.0015649, upper bound: 52.4279096
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7707306, upper bound: 52.7398814
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2380054, upper bound: 54.1659877
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.2380054, upper bound: 54.1659877
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.1651883, upper bound: 54.1633062
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -54.1651883, upper bound: 54.1633062
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.0424892, upper bound: 52.4358202
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -52.7765513, upper bound: 52.7402502
IS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -50.6592335, upper bound: 50.8286423
IS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -50.1615771, upper bound: 50.6146162
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -50.6575237, upper bound: 50.8354814
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -49.8091960, upper bound: 50.1566159
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.11
Output dim: 0, lower bound: -50.6575237, upper bound: 51.4148433
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.11
Output dim: 0, lower bound: -49.8091960, upper bound: 50.9669972

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -13.0984602, 25.5609436, -37.0516052, 35.8251839
1: -12.9931536, 20.8490124, -14.8080769, 23.3034058, -36.2965584, 35.6570816
2: -13.3803749, 20.5927849, -15.2141285, 23.0057163, -36.3860855, 35.8069000
3: -16.0014496, 23.8214302, -18.2130032, 26.6391182, -42.6405640, 42.0344315
4: -15.2732220, 22.4852295, -17.2754612, 25.2983437, -40.5715637, 39.7606888

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2306948, upper bound: 54.1190121
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2306948, upper bound: 54.1190121
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.9082861, 21.7448769, -14.0445080, 27.1689224, -38.0772095, 35.7893829
1: -12.3404970, 19.9252815, -15.8506737, 24.9322376, -37.2727356, 35.7759552
2: -12.7244921, 19.6817036, -16.2994747, 24.5649776, -37.2894630, 35.9811745
3: -15.1908503, 22.7401123, -19.4416981, 28.5428333, -43.7336807, 42.1818085
4: -14.5545177, 21.4448280, -18.5609226, 27.0019245, -41.5564423, 40.0057487

Time for backsubstitution: 2.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2313976, upper bound: 54.2004323
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2313976, upper bound: 54.2010919
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -11.4906597, 22.7267227, -36.8390465, 38.7311401
1: -15.9184780, 24.8546124, -12.9931536, 20.8490124, -36.7674828, 37.8477592
2: -16.3577156, 24.5101395, -13.3803749, 20.5927849, -36.9505005, 37.8905144
3: -19.5635452, 28.4796600, -16.0014496, 23.8214302, -43.3849754, 44.4811020
4: -18.5248909, 27.0098228, -15.2732220, 22.4852295, -41.0101204, 42.2830429

Time for backsubstitution: 2.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1182978, upper bound: 54.1971245
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2004790, upper bound: 54.2004791
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -14.1123228, 27.2404823, -41.3528061, 41.3528061
1: -15.9184780, 24.8546124, -15.9184780, 24.8546124, -40.7730789, 40.7730827
2: -16.3577156, 24.5101395, -16.3577156, 24.5101395, -40.8678551, 40.8678551
3: -19.5635452, 28.4796600, -19.5635452, 28.4796600, -48.0432053, 48.0432053
4: -18.5248909, 27.0098228, -18.5248909, 27.0098228, -45.5347099, 45.5347099

Time for backsubstitution: 2.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1182978, upper bound: 54.1971245
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2004790, upper bound: 54.2004791
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.5783987, 19.1222038, -16.9955292, 31.4307480, -41.0091476, 36.1177330
1: -10.8226967, 17.2935429, -19.0474510, 28.3562393, -39.1789284, 36.3409920
2: -11.1932211, 17.1231270, -19.5802612, 27.9446602, -39.1378822, 36.7033844
3: -13.2758369, 19.6837635, -23.2360268, 32.6596794, -45.9355164, 42.9197884
4: -12.7250004, 18.6032982, -21.8171463, 31.0525780, -43.7775764, 40.4204407

Time for backsubstitution: 2.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8846486, upper bound: 52.8272782
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1727198, upper bound: 52.9362268
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -21.2581940, 38.8678398, -50.3584976, 43.9849167
1: -12.9931536, 20.8490124, -23.9263000, 36.1767044, -49.1698570, 44.7753067
2: -13.3803749, 20.5927849, -24.4926319, 35.4565048, -48.8368797, 45.0854187
3: -16.0014496, 23.8214302, -29.4232903, 41.8080063, -57.8094559, 53.2447166
4: -15.2732220, 22.4852295, -27.6137218, 39.6126175, -54.8858337, 50.0989532

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9092900, upper bound: 53.4247972
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0674461, upper bound: 53.5714630
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -18.8997993, 34.9725609, -49.0848846, 46.1402817
1: -15.9184780, 24.8546124, -21.2962494, 32.6550941, -48.5735703, 46.1508522
2: -16.3577156, 24.5101395, -21.7990875, 32.0393410, -48.3970566, 46.3092270
3: -19.5635452, 28.4796600, -26.2362270, 37.6525040, -57.2160492, 54.7158737
4: -18.5248909, 27.0098228, -24.6484203, 35.6282730, -54.1531639, 51.6582336

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9568738, upper bound: 53.1594448
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9565185, upper bound: 53.5668429
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -21.5188446, 39.3366241, -53.4489479, 48.7593269
1: -15.9184780, 24.8546124, -24.2212391, 36.5742035, -52.4926758, 49.0758400
2: -16.3577156, 24.5101395, -24.7897606, 35.8414726, -52.1991882, 49.2999001
3: -19.5635452, 28.4796600, -29.7945175, 42.2615585, -61.8250999, 58.2741776
4: -18.5248909, 27.0098228, -27.9134693, 40.0790443, -58.6039162, 54.9232864

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9568738, upper bound: 53.1594448
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9565185, upper bound: 53.5668429
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16.9955292, 31.4307480, -9.5783987, 19.1222038, -36.1177330, 41.0091476
1: -19.0474510, 28.3562393, -10.8226967, 17.2935429, -36.3409882, 39.1789322
2: -19.5802612, 27.9446602, -11.1932211, 17.1231270, -36.7033882, 39.1378822
3: -23.2360268, 32.6596794, -13.2758369, 19.6837635, -42.9197884, 45.9355164
4: -21.8171463, 31.0525780, -12.7250004, 18.6032982, -40.4204407, 43.7775764

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7095153, upper bound: 52.5354702
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6599294, upper bound: 52.4077220
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -16.9955292, 31.4307480, -16.0136604, 29.9076042, -46.9031334, 47.4444046
1: -19.0474510, 28.3562393, -18.0346069, 27.5633450, -46.6107941, 46.3908463
2: -19.5802612, 27.9446602, -18.4906292, 27.1347771, -46.7150345, 46.4352837
3: -23.2360268, 32.6596794, -22.1305408, 31.6597500, -54.8957748, 54.7902184
4: -21.8171463, 31.0525780, -20.8091221, 30.0383682, -51.8555107, 51.8617020

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7095153, upper bound: 52.5354702
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6599294, upper bound: 52.4077220
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -21.2581940, 38.8678398, -11.4906597, 22.7267227, -43.9849167, 50.3584976
1: -23.9263000, 36.1767044, -12.9931536, 20.8490124, -44.7753105, 49.1698570
2: -24.4926319, 35.4565048, -13.3803749, 20.5927849, -45.0854187, 48.8368797
3: -29.4232903, 41.8080063, -16.0014496, 23.8214302, -53.2447166, 57.8094559
4: -27.6137218, 39.6126175, -15.2732220, 22.4852295, -50.0989532, 54.8858376

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
time: 1.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.2581940, 38.8678398, -18.3416824, 33.8961067, -55.1543007, 57.2095222
1: -23.9263000, 36.1767044, -20.6681404, 31.5833549, -55.5096550, 56.8448448
2: -24.4926319, 35.4565048, -21.1485443, 31.0053368, -55.4979706, 56.6050491
3: -29.4232903, 41.8080063, -25.4536095, 36.4101868, -65.8334808, 67.2616119
4: -27.6137218, 39.6126175, -23.8948879, 34.4529533, -62.0666733, 63.5074997

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5529828, upper bound: 53.6878929
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -14.1123228, 27.2404823, -46.1402817, 49.0848846
1: -21.2962494, 32.6550941, -15.9184780, 24.8546124, -46.1508484, 48.5735703
2: -21.7990875, 32.0393410, -16.3577156, 24.5101395, -46.3092270, 48.3970566
3: -26.2362270, 37.6525040, -19.5635452, 28.4796600, -54.7158661, 57.2160492
4: -24.6484203, 35.6282730, -18.5248909, 27.0098228, -51.6582336, 54.1531639

Time for backsubstitution: 2.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1288451, upper bound: 52.9621333
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9332570, upper bound: 52.9227103
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -18.8997993, 34.9725609, -20.3894386, 37.3347435, -56.2345428, 55.3619995
1: -21.2962494, 32.6550941, -22.9784584, 34.5795403, -55.8757896, 55.6335526
2: -21.7990875, 32.0393410, -23.4825687, 33.9048004, -55.7038841, 55.5219040
3: -26.2362270, 37.6525040, -28.2879829, 39.8983345, -66.1345596, 65.9404755
4: -24.6484203, 35.6282730, -26.3920212, 37.9236145, -62.5720367, 62.0202904

Time for backsubstitution: 2.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.1288451, upper bound: 52.9621333
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9332570, upper bound: 52.9227103
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -14.1123228, 27.2404823, -48.7593269, 53.4489479
1: -24.2212391, 36.5742035, -15.9184780, 24.8546124, -49.0758400, 52.4926834
2: -24.7897606, 35.8414726, -16.3577156, 24.5101395, -49.2999001, 52.1991882
3: -29.7945175, 42.2615585, -19.5635452, 28.4796600, -58.2741699, 61.8251038
4: -27.9134693, 40.0790443, -18.5248909, 27.0098228, -54.9232903, 58.6039200

Time for backsubstitution: 2.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9262232, upper bound: 53.1040943
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5260032, upper bound: 53.5260034
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5188446, 39.3366241, -20.3894386, 37.3347435, -58.8535881, 59.7260628
1: -24.2212391, 36.5742035, -22.9784584, 34.5795403, -58.8007812, 59.5526505
2: -24.7897606, 35.8414726, -23.4825687, 33.9048004, -58.6945610, 59.3240280
3: -29.7945175, 42.2615585, -28.2879829, 39.8983345, -69.6928406, 70.5495377
4: -27.9134693, 40.0790443, -26.3920212, 37.9236145, -65.8370819, 66.4710617

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.9262232, upper bound: 53.1040943
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5260032, upper bound: 53.5260034
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.4012470, 18.7822590, -15.4377556, 29.5055103, -38.9067574, 34.2200127
1: -10.6178713, 16.9743195, -17.2882118, 25.9514885, -36.5693588, 34.2625313
2: -10.9888000, 16.8088474, -17.8075523, 25.6108322, -36.5996170, 34.6164017
3: -13.0179892, 19.3143959, -21.0557785, 29.9196358, -42.9376259, 40.3701744
4: -12.4910479, 18.2470760, -19.8410912, 28.4075356, -40.8985825, 38.0881653

Time for backsubstitution: 2.69 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=62.94061279296875
rel_dist={0: [-54.300068832219395, 54.30006883221938]}

## Binary search (step 2) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2599357, upper bound: 54.2610626
time: 0.73 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2650522, upper bound: 54.2650523
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -54.2599357, upper bound: 54.2610626
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -54.2650522, upper bound: 54.2650523

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -18.4868755, 34.6143112, -20.2760925, 37.5954819, -56.0823593, 54.8903961
1: -20.8144035, 32.0120316, -22.8364048, 34.9631996, -55.7775993, 54.8484344
2: -21.3472996, 31.4293518, -23.3855381, 34.2576447, -55.6049423, 54.8148880
3: -25.5772781, 36.8826523, -28.0821171, 40.4225235, -65.9998016, 64.9647675
4: -24.1322937, 34.9039764, -26.4360714, 38.2008171, -62.3330956, 61.3400497

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7750925, upper bound: 54.0522116
time: 0.65 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7749727, upper bound: 54.0534231
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -23.0712452, 42.9994736, -20.7409878, 38.3652725, -61.4365158, 63.7404594
1: -25.9661083, 39.6324463, -23.3004074, 35.7874031, -61.7535095, 62.9328499
2: -26.6078491, 38.8119164, -23.9231968, 35.0485077, -61.6563568, 62.7351151
3: -31.9601364, 45.9458580, -28.6117115, 41.4680023, -73.4281387, 74.5575638
4: -30.0656071, 43.3875542, -27.0799427, 38.9733391, -69.0389404, 70.4674988

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2650097, upper bound: 54.2638507
time: 0.98 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2649560, upper bound: 54.2649561
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -53.7750925, upper bound: 54.0522116
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -53.7749727, upper bound: 54.0534231
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -54.2650097, upper bound: 54.2638507
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.48
Output dim: 0, lower bound: -54.2649560, upper bound: 54.2649561

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -16.5842857, 31.4502945, -16.7889042, 31.8010082, -48.3852921, 48.2391968
1: -18.6925430, 28.9172668, -18.9533329, 29.3083477, -48.0008812, 47.8705978
2: -19.1761055, 28.4486637, -19.4042816, 28.8136215, -47.9897270, 47.8529434
3: -22.9838333, 33.2298317, -23.3441238, 33.6866188, -56.6704521, 56.5739555
4: -21.7086563, 31.4988327, -21.9937401, 31.9684086, -53.6770630, 53.4925690

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6059500, upper bound: 53.8065456
time: 0.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -17.6497459, 33.0559959, -24.3801613, 44.1683769, -61.8181000, 57.4361572
1: -19.8834400, 30.4581985, -27.4453545, 41.2796936, -61.1631203, 57.9035530
2: -20.3877487, 29.9437065, -28.0530720, 40.3757324, -60.7634811, 57.9967728
3: -24.4449310, 35.0852280, -33.7670860, 47.8000298, -72.2449646, 68.8523102
4: -23.0095768, 33.2862091, -31.6138706, 45.3063583, -68.3159332, 64.9000778

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7359781, upper bound: 53.7800164
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6041082, upper bound: 53.8065456
time: 0.69 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -21.1899261, 39.7947083, -17.1834087, 32.4464417, -53.6363640, 56.9781189
1: -23.8794594, 36.4696922, -19.3675156, 29.8953705, -53.7748299, 55.8372078
2: -24.4598618, 35.7714462, -19.8700581, 29.3695202, -53.8293686, 55.6415024
3: -29.4223919, 42.2065544, -23.8265648, 34.4926453, -63.9150391, 66.0331192
4: -27.6570454, 39.9657364, -22.5389099, 32.5816154, -60.2386627, 62.5046463

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2428487, upper bound: 54.1750584
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1668161, upper bound: 54.1725736
time: 0.79 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.1894913, 41.4026070, -24.3663597, 44.2001076, -66.3895874, 65.7689667
1: -24.9855957, 38.1396561, -27.4224072, 41.3056030, -66.2911911, 65.5620651
2: -25.6013317, 37.3728027, -28.0530319, 40.3851929, -65.9865189, 65.4258118
3: -30.7658539, 44.1848907, -33.7387505, 48.0140457, -78.7798996, 77.9236450
4: -28.9257908, 41.7698135, -31.6571312, 45.3417130, -74.2675018, 73.4269257

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6445045, upper bound: 52.4035925
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7331372, upper bound: 54.2529247
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -53.6059500, upper bound: 53.8065456
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -53.7359781, upper bound: 53.7800164
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -53.6041082, upper bound: 53.8065456
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -54.2428487, upper bound: 54.1750584
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -54.1668161, upper bound: 54.1725736
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -52.6445045, upper bound: 52.4035925
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.19
Output dim: 0, lower bound: -53.7331372, upper bound: 54.2529247

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.8830805, 25.1311016, -15.6281281, 29.7159939, -42.5990753, 40.7592201
1: -14.5471048, 23.1460762, -17.6479874, 27.5266590, -42.0737648, 40.7940636
2: -14.9644260, 22.8335800, -18.0800991, 27.0783176, -42.0427361, 40.9136696
3: -17.9040298, 26.4914932, -21.7524853, 31.6161289, -49.5201569, 48.2439728
4: -17.0220585, 25.0357056, -20.5604954, 29.9421577, -46.9642181, 45.5961990

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15.5915403, 29.7336845, -16.3179016, 30.9953995, -46.5869408, 46.0515785
1: -17.5682640, 27.2609444, -18.4215488, 28.5239468, -46.0922089, 45.6824875
2: -18.0409908, 26.8382149, -18.8645992, 28.0524025, -46.0933914, 45.7028122
3: -21.5867920, 31.2940540, -22.6875229, 32.7665329, -54.3533249, 53.9815750
4: -20.4039211, 29.6645699, -21.3740082, 31.0984821, -51.5024033, 51.0385704

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6026783, upper bound: 53.7916097
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6026783, upper bound: 53.8065456
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -13.8668394, 26.6131897, -23.2001400, 42.0108566, -55.8776932, 49.8133316
1: -15.6459961, 24.5399742, -26.1167202, 39.4397278, -55.0857201, 50.6566925
2: -16.0804520, 24.2066879, -26.7000084, 38.5881920, -54.6686325, 50.9066963
3: -19.2482281, 28.1847649, -32.1492996, 45.6531219, -64.9013519, 60.3340607
4: -18.2019634, 26.6813850, -30.1370602, 43.2221718, -61.4241333, 56.8184433

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.4119408, upper bound: 52.6493826
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6892079, upper bound: 53.7467420
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -16.6036377, 31.2419891, -23.8871288, 43.2838402, -59.8874779, 55.1291199
1: -18.6975479, 28.6983337, -26.8848915, 40.4290390, -59.1265869, 55.5832214
2: -19.1899624, 28.2366352, -27.4851742, 39.5558548, -58.7458115, 55.7218094
3: -22.9721146, 33.0409813, -33.0749664, 46.7967453, -69.7688599, 66.1159439
4: -21.6303387, 31.3418770, -30.9499588, 44.3672371, -65.9975586, 62.2918320

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.4091893, upper bound: 52.6479207
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5558019, upper bound: 53.7753337
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17.7970028, 34.0092278, -16.0263767, 30.3466263, -48.1436195, 50.0356064
1: -20.1008263, 31.1648102, -18.0576401, 28.1160412, -48.2168655, 49.2224503
2: -20.5783043, 30.6260185, -18.5494595, 27.6379852, -48.2162895, 49.1754761
3: -24.8209305, 35.9611664, -22.2268600, 32.4197006, -57.2406311, 58.1880150
4: -23.3319550, 34.0164986, -21.1049671, 30.5431194, -53.8750610, 55.1214561

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0500829, upper bound: 53.8489395
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0500833, upper bound: 54.1750584
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.1371193, 37.8709068, -16.7227459, 31.6618881, -51.7990074, 54.5936508
1: -22.6855583, 34.6526031, -18.8487225, 29.1346474, -51.8202057, 53.5013237
2: -23.2515335, 34.0151291, -19.3432999, 28.6301460, -51.8816719, 53.3584290
3: -27.9415264, 40.0812378, -23.1875706, 33.5978622, -61.5393906, 63.2688065
4: -26.2660828, 37.9435158, -21.9337749, 31.7354851, -58.0015640, 59.8772888

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1536180, upper bound: 54.0743498
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1570996, upper bound: 54.1636546
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -16.5578117, 31.5026054, -20.3924217, 37.2590561, -53.8168678, 51.8950195
1: -18.5654697, 27.7749023, -22.9147282, 34.3064613, -52.8719254, 50.6896286
2: -19.0977459, 27.3892498, -23.4953632, 33.6584625, -52.7562065, 50.8846130
3: -22.6423988, 32.0498199, -28.1189976, 39.7268333, -62.3692322, 60.1688080
4: -21.2815571, 30.4649220, -26.3377419, 37.6417007, -58.9232559, 56.8026657

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6445045, upper bound: 52.4035925
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6445045, upper bound: 52.4034635
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.2291279, 39.7105598, -24.1825237, 43.8826599, -65.1117859, 63.8930702
1: -23.9023304, 36.4898453, -27.2133179, 40.9897156, -64.8920441, 63.7031631
2: -24.4997444, 35.7794952, -27.8422203, 40.0806351, -64.5803833, 63.6217155
3: -29.4249477, 42.2512283, -33.4778099, 47.6425972, -77.0675354, 75.7290344
4: -27.6693363, 39.9478989, -31.4130478, 44.9906158, -72.6599426, 71.3609467

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1660470, upper bound: 54.2386367
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1633060, upper bound: 54.1633066
time: 0.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.29 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -53.6026783, upper bound: 53.7916097
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -53.6026783, upper bound: 53.8065456
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -52.4119408, upper bound: 52.6493826
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -53.6892079, upper bound: 53.7467420
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -52.4091893, upper bound: 52.6479207
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -53.5558019, upper bound: 53.7753337
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -54.0500829, upper bound: 53.8489395
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -54.0500833, upper bound: 54.1750584
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -54.1536180, upper bound: 54.0743498
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -54.1570996, upper bound: 54.1636546
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -52.6445045, upper bound: 52.4035925
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -52.6445045, upper bound: 52.4034635
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -54.1660470, upper bound: 54.2386367
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.29
Output dim: 0, lower bound: -54.1633060, upper bound: 54.1633066

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -12.8830805, 25.1311016, -13.9387941, 26.8548717, -39.7379456, 39.0698929
1: -14.5471048, 23.1460762, -15.7320099, 24.7193108, -39.2664146, 38.8780861
2: -14.9644260, 22.8335800, -16.1609840, 24.3732967, -39.3377228, 38.9945526
3: -17.9040298, 26.4914932, -19.3580685, 28.3290482, -46.2330704, 45.8495598
4: -17.0220585, 25.0357056, -18.3864365, 26.8138123, -43.8358688, 43.4221420

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3662201, upper bound: 53.5734849
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -12.8830805, 25.1311016, -18.6726837, 35.2943687, -48.1774445, 43.8037796
1: -14.5471048, 23.1460762, -21.0612679, 32.2930946, -46.8401985, 44.2073441
2: -14.9644260, 22.8335800, -21.5753098, 31.7414761, -46.7058983, 44.4088860
3: -17.9040298, 26.4914932, -25.9731541, 37.2977371, -55.2017632, 52.4646454
4: -17.0220585, 25.0357056, -24.4370499, 35.3352585, -52.3573151, 49.4727554

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3662201, upper bound: 53.5734849
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15.5915403, 29.7336845, -14.5992975, 28.0919647, -43.6835022, 44.3329773
1: -17.5682640, 27.2609444, -16.4727116, 25.6685715, -43.2368279, 43.7336502
2: -18.0409908, 26.8382149, -16.9113159, 25.3030701, -43.3440475, 43.7495308
3: -21.5867920, 31.2940540, -20.2564812, 29.4238663, -51.0106583, 51.5505371
4: -20.4039211, 29.6645699, -19.1599579, 27.9194336, -48.3233566, 48.8245277

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3608130, upper bound: 53.5876783
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6026783, upper bound: 53.7916097
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6026783, upper bound: 53.7916097
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15.5915403, 29.7336845, -19.3164806, 36.4998169, -52.0913544, 49.0501633
1: -17.5682640, 27.2609444, -21.7840309, 33.2254448, -50.7936974, 49.0449715
2: -18.0409908, 26.8382149, -22.3099022, 32.6569252, -50.6979065, 49.1481094
3: -21.5867920, 31.2940540, -26.8443775, 38.3786621, -59.9654541, 58.1384315
4: -20.4039211, 29.6645699, -25.2041817, 36.4266319, -56.8305511, 54.8687515

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3608130, upper bound: 53.5876783
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6026783, upper bound: 53.8065456
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6026783, upper bound: 53.7916097
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.0228472, 19.6864433, -17.9653034, 32.8679619, -42.8908081, 37.6517372
1: -11.2963448, 17.6795597, -20.1392212, 29.9588585, -41.2552032, 37.8187790
2: -11.6970196, 17.5050163, -20.6668015, 29.4787312, -41.1757431, 38.1718178
3: -13.8138695, 20.1995163, -24.6009579, 34.5374565, -48.3513260, 44.8004761
4: -13.1423883, 19.1495247, -23.0650177, 32.7986565, -45.9410439, 42.2145424

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.6068604, upper bound: 51.9792625
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3937419, upper bound: 52.6406840
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -13.6923637, 26.3083477, -22.3074989, 40.4664345, -54.1587944, 48.6158447
1: -15.4473963, 24.2376900, -25.1016693, 37.8981018, -53.3454971, 49.3393555
2: -15.8820705, 23.9136314, -25.6753674, 37.1046219, -52.9866905, 49.5889969
3: -18.9981346, 27.8347549, -30.8821373, 43.8419685, -62.8401031, 58.7168922
4: -17.9702759, 26.3485947, -28.9475784, 41.5141144, -59.4843788, 55.2961693

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3213627, upper bound: 53.5259697
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6892079, upper bound: 53.7467420
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6892079, upper bound: 53.7467420
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.7915850, 24.5824432, -18.6768570, 34.2317429, -47.0233269, 43.2592964
1: -14.3846521, 21.9987583, -20.9346581, 30.9911194, -45.3757706, 42.9334183
2: -14.8440495, 21.7650127, -21.4799671, 30.4932327, -45.3372726, 43.2449799
3: -17.5673180, 25.2215958, -25.5615311, 35.7306786, -53.2979965, 50.7831268
4: -16.5836239, 24.0389004, -23.9052448, 33.9909401, -50.5745621, 47.9441452

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0513253, upper bound: 52.1554859
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0513253, upper bound: 52.6479207
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16.4197865, 30.9190216, -23.0206013, 41.7768555, -58.1966400, 53.9396210
1: -18.4870586, 28.3752098, -25.8990707, 38.9290886, -57.4161453, 54.2742805
2: -18.9791622, 27.9252281, -26.4897804, 38.1118050, -57.0909576, 54.4150085
3: -22.7068520, 32.6649590, -31.8432159, 45.0360565, -67.7428894, 64.5081787
4: -21.3850136, 30.9870262, -29.7940826, 42.7039604, -64.0889587, 60.7811089

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5558019, upper bound: 53.7753337
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5558019, upper bound: 53.7753337
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17.7970028, 34.0092278, -13.9387941, 26.8548717, -44.6518707, 47.9480209
1: -20.1008263, 31.1648102, -15.7320099, 24.7193108, -44.8201332, 46.8968163
2: -20.5783043, 30.6260185, -16.1609840, 24.3732967, -44.9515991, 46.7869987
3: -24.8209305, 35.9611664, -19.3580685, 28.3290482, -53.1499748, 55.3192291
4: -23.3319550, 34.0164986, -18.3864365, 26.8138123, -50.1457558, 52.4029274

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6201633, upper bound: 53.6324539
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0500829, upper bound: 53.8489395
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0500829, upper bound: 53.8489395
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17.7970028, 34.0092278, -18.6726837, 35.2943687, -53.0913658, 52.6819115
1: -20.1008263, 31.1648102, -21.0612679, 32.2930946, -52.3939209, 52.2260780
2: -20.5783043, 30.6260185, -21.5753098, 31.7414761, -52.3197784, 52.2013283
3: -24.8209305, 35.9611664, -25.9731541, 37.2977371, -62.1186676, 61.9343147
4: -23.3319550, 34.0164986, -24.4370499, 35.3352585, -58.6672134, 58.4535370

Time for backsubstitution: 2.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6201634, upper bound: 54.1612974
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8806116, upper bound: 54.1661840
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -19.6469269, 37.0629959, -15.2520590, 29.2517014, -48.8986282, 52.3150520
1: -22.1422119, 33.8611984, -17.2155190, 26.7630329, -48.9052429, 51.0767174
2: -22.6950874, 33.2494736, -17.6743927, 26.3343163, -49.0293999, 50.9238663
3: -27.2719059, 39.1463280, -21.1770649, 30.7941628, -58.0660706, 60.3233757
4: -25.6410198, 37.0699425, -20.0565929, 29.1337471, -54.7747574, 57.1265335

Time for backsubstitution: 2.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1536180, upper bound: 54.0743498
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1536180, upper bound: 54.0743498
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18.8823357, 35.7719078, -16.0535736, 30.5426064, -49.4249382, 51.8254814
1: -21.2831268, 32.6370010, -18.0910416, 28.0503349, -49.3334579, 50.7280312
2: -21.8165550, 32.0599442, -18.5909901, 27.5660667, -49.3826180, 50.6509323
3: -26.2054634, 37.6893272, -22.1961174, 32.3047371, -58.5102005, 59.8854446
4: -24.6579800, 35.6917992, -21.1116486, 30.5018139, -55.1597939, 56.8034477

Time for backsubstitution: 2.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1570996, upper bound: 54.1636546
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1570996, upper bound: 54.1636546
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15.5266380, 29.5548973, -16.6624508, 30.8254204, -46.3520584, 46.2173462
1: -17.4081535, 26.1904716, -18.7298260, 28.4043903, -45.8125305, 44.9202957
2: -17.9216900, 25.8432751, -19.2264118, 27.9494171, -45.8711014, 45.0696869
3: -21.2329102, 30.2046375, -22.9954224, 32.7837677, -54.0166779, 53.2000580
4: -20.0093384, 28.6440964, -21.5640640, 31.0460663, -51.0554047, 50.2081566

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.1302884, upper bound: 51.6360376
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.6360411, upper bound: 52.3858484
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -16.2263813, 30.8975258, -19.2566433, 35.2722054, -51.4985886, 50.1541672
1: -18.1938400, 27.2115326, -21.6257763, 32.3818321, -50.5756721, 48.8373032
2: -18.7148647, 26.8423061, -22.1909618, 31.7921677, -50.5070343, 49.0332680
3: -22.1862221, 31.3839397, -26.5225220, 37.4830704, -59.6692924, 57.9064598
4: -20.8358936, 29.8450928, -24.8298836, 35.5088692, -56.3447647, 54.6749763

Time for backsubstitution: 2.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of IS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.4952368, upper bound: 50.2401070
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -50.4861670, upper bound: 50.1173280
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -20.1259995, 37.6480942, -20.6826916, 37.9379921, -58.0639877, 58.3307724
1: -22.6574402, 34.7559853, -23.2955303, 35.5764275, -58.2338676, 58.0515137
2: -23.2365303, 34.0913048, -23.8402290, 34.8274651, -58.0639877, 57.9315224
3: -27.9086418, 40.2340508, -28.7056255, 41.2626343, -69.1712646, 68.9396744
4: -26.2938576, 37.9664268, -26.9639702, 38.8965111, -65.1903610, 64.9303894

Time for backsubstitution: 2.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0634741, upper bound: 54.2282336
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1564705, upper bound: 54.2303090
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20.7388840, 38.8040123, -23.0527840, 41.8725548, -62.6114388, 61.8567924
1: -23.3474731, 35.6344490, -25.9292793, 39.0605240, -62.4079857, 61.5637245
2: -23.9366035, 34.9540634, -26.5414886, 38.2072258, -62.1438179, 61.4955292
3: -28.7397575, 41.2482834, -31.8876419, 45.3663139, -74.1060715, 73.1359177
4: -27.0194721, 39.0012131, -29.8953762, 42.8449440, -69.8644180, 68.8965759

Time for backsubstitution: 2.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0604024, upper bound: 54.1498429
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1536148, upper bound: 54.1536154
time: 0.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.52 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.6026783, upper bound: 53.7916097
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.6026783, upper bound: 53.7916097
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.6026783, upper bound: 53.8065456
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.6026783, upper bound: 53.7916097
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -51.6068604, upper bound: 51.9792625
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -52.3937419, upper bound: 52.6406840
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.6892079, upper bound: 53.7467420
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.6892079, upper bound: 53.7467420
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -52.0513253, upper bound: 52.1554859
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -52.0513253, upper bound: 52.6479207
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.5558019, upper bound: 53.7753337
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.5558019, upper bound: 53.7753337
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.0500829, upper bound: 53.8489395
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.0500829, upper bound: 53.8489395
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.6201634, upper bound: 54.1612974
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -53.8806116, upper bound: 54.1661840
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.1536180, upper bound: 54.0743498
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.1536180, upper bound: 54.0743498
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.1570996, upper bound: 54.1636546
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.1570996, upper bound: 54.1636546
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -52.1302884, upper bound: 51.6360376
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -52.6360411, upper bound: 52.3858484
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.52
Output dim: 0, lower bound: -50.4952368, upper bound: 50.2401070
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.52
Output dim: 0, lower bound: -50.4861670, upper bound: 50.1173280
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.0634741, upper bound: 54.2282336
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.1564705, upper bound: 54.2303090
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.0604024, upper bound: 54.1498429
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 0, lower bound: -54.1536148, upper bound: 54.1536154

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -13.9387941, 26.8548717, -38.3455276, 36.6655159
1: -12.9931536, 20.8490124, -15.7320099, 24.7193108, -37.7124634, 36.5810127
2: -13.3803749, 20.5927849, -16.1609840, 24.3732967, -37.7536697, 36.7537651
3: -16.0014496, 23.8214302, -19.3580685, 28.3290482, -44.3304901, 43.1794968
4: -15.2732220, 22.4852295, -18.3864365, 26.8138123, -42.0870323, 40.8716660

Time for backsubstitution: 2.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7950611, upper bound: 52.8474325
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6937795, upper bound: 53.7526565
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18.2756729, 33.8391953, -13.9387941, 26.8548717, -45.1305466, 47.7779884
1: -20.5992928, 31.5115490, -15.7320099, 24.7193108, -45.3186035, 47.2435570
2: -21.0710773, 30.9448738, -16.1609840, 24.3732967, -45.4443741, 47.1058540
3: -25.3643303, 36.2957077, -19.3580685, 28.3290482, -53.6933784, 55.6537781
4: -23.7730541, 34.3740425, -18.3864365, 26.8138123, -50.5868607, 52.7604752

Time for backsubstitution: 2.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.7950611, upper bound: 52.8474325
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6937795, upper bound: 53.7526565
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.4906597, 22.7267227, -18.6726837, 35.2943687, -46.7850227, 41.3994064
1: -12.9931536, 20.8490124, -21.0612679, 32.2930946, -45.2862473, 41.9102669
2: -13.3803749, 20.5927849, -21.5753098, 31.7414761, -45.1218452, 42.1680946
3: -16.0014496, 23.8214302, -25.9731541, 37.2977371, -53.2991867, 49.7945862
4: -15.2732220, 22.4852295, -24.4370499, 35.3352585, -50.6084747, 46.9222794

Time for backsubstitution: 2.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.2756729, 33.8391953, -18.6726837, 35.2943687, -53.5700417, 52.5118752
1: -20.5992928, 31.5115490, -21.0612679, 32.2930946, -52.8923874, 52.5728111
2: -21.0710773, 30.9448738, -21.5753098, 31.7414761, -52.8125496, 52.5201836
3: -25.3643303, 36.2957077, -25.9731541, 37.2977371, -62.6620674, 62.2688560
4: -23.7730541, 34.3740425, -24.4370499, 35.3352585, -59.1083145, 58.8110809

Time for backsubstitution: 2.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7362107, upper bound: 53.7845109
time: 2.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -14.5992975, 28.0919647, -42.2042885, 41.8397789
1: -15.9184780, 24.8546124, -16.4727116, 25.6685715, -41.5870438, 41.3273163
2: -16.3577156, 24.5101395, -16.9113159, 25.3030701, -41.6607780, 41.4214554
3: -19.5635452, 28.4796600, -20.2564812, 29.4238663, -48.9874115, 48.7361336
4: -18.5248909, 27.0098228, -19.1599579, 27.9194336, -46.4443245, 46.1697807

Time for backsubstitution: 2.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8827413, upper bound: 52.8874955
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5539214, upper bound: 53.7593542
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20.3892612, 37.3345947, -14.5992975, 28.0919647, -48.4812241, 51.9338913
1: -22.9782505, 34.5793762, -16.4727116, 25.6685715, -48.6468124, 51.0520821
2: -23.4823799, 33.9046478, -16.9113159, 25.3030701, -48.7854424, 50.8159637
3: -28.2877445, 39.8981094, -20.2564812, 29.4238663, -57.7116089, 60.1545906
4: -26.3917770, 37.9234200, -19.1599579, 27.9194336, -54.3112106, 57.0833778

Time for backsubstitution: 2.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.8827413, upper bound: 52.8874955
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.5539214, upper bound: 53.7593542
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14.1123228, 27.2404823, -19.3164806, 36.4998169, -50.6121407, 46.5569611
1: -15.9184780, 24.8546124, -21.7840309, 33.2254448, -49.1439171, 46.6386337
2: -16.3577156, 24.5101395, -22.3099022, 32.6569252, -49.0146332, 46.8200417
3: -19.5635452, 28.4796600, -26.8443775, 38.3786621, -57.9422073, 55.3240166
4: -18.5248909, 27.0098228, -25.2041817, 36.4266319, -54.9515228, 52.2140045

Time for backsubstitution: 2.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6059500, upper bound: 53.8065454
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6059500, upper bound: 53.8065454
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20.3892612, 37.3345947, -19.3164806, 36.4998169, -56.8890724, 56.6510773
1: -22.9782505, 34.5793762, -21.7840309, 33.2254448, -56.2036934, 56.3634033
2: -23.4823799, 33.9046478, -22.3099022, 32.6569252, -56.1393013, 56.2145386
3: -28.2877445, 39.8981094, -26.8443775, 38.3786621, -66.6664047, 66.7424850
4: -26.3917770, 37.9234200, -25.2041817, 36.4266319, -62.8183975, 63.1276016

Time for backsubstitution: 2.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6059500, upper bound: 53.8065454
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6059500, upper bound: 53.8065454
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.3433323, 18.3868923, -17.5917149, 32.2131805, -41.5565071, 35.9786072
1: -10.5062618, 16.4586811, -19.7143059, 29.3373222, -39.8435822, 36.1729851
2: -10.9107857, 16.3086014, -20.2418938, 28.8809471, -39.7917328, 36.5504951
3: -12.8258018, 18.7873745, -24.0727844, 33.8106918, -46.6364937, 42.8601608
4: -12.2389488, 17.8110218, -22.5847740, 32.1109276, -44.3498764, 40.3957977

Time for backsubstitution: 2.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.6068604, upper bound: 51.9792625
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -51.6068604, upper bound: 51.9792625
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.6648560, 18.9851589, -17.7883377, 32.5500717, -42.2149277, 36.7734985
1: -10.8772316, 17.0418739, -19.9351616, 29.6563053, -40.5335312, 36.9770355
2: -11.2884941, 16.8642693, -20.4653492, 29.1875572, -40.4760513, 37.3296204
3: -13.2864447, 19.4724598, -24.3458271, 34.1856079, -47.4720535, 43.8182869
4: -12.6693878, 18.4325409, -22.8328876, 32.4631844, -45.1325645, 41.2654266

Time for backsubstitution: 2.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3937419, upper bound: 52.6406840
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.3937419, upper bound: 52.6406840
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.3281441, 22.4347954, -22.3074989, 40.4664345, -51.7945786, 44.7422943
1: -12.8078461, 20.5624294, -25.1016693, 37.8981018, -50.7059479, 45.6640930
2: -13.1951189, 20.3123493, -25.6753674, 37.1046219, -50.2997398, 45.9877129
3: -15.7693405, 23.4899292, -30.8821373, 43.8419685, -59.6113091, 54.3720589
4: -15.0586796, 22.1680984, -28.9475784, 41.5141144, -56.5727882, 51.1156731

Time for backsubstitution: 2.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6871493, upper bound: 53.5525501
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6871493, upper bound: 53.7467420
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18.1670151, 33.5953369, -22.3074989, 40.4664345, -58.6334496, 55.9028358
1: -20.4690361, 31.2833176, -25.1016693, 37.8981018, -58.3671379, 56.3849869
2: -20.9490776, 30.7164116, -25.6753674, 37.1046219, -58.0536842, 56.3917770
3: -25.2049751, 36.0595169, -30.8821373, 43.8419685, -69.0469284, 66.9416428
4: -23.6631165, 34.1221199, -28.9475784, 41.5141144, -65.1772232, 63.0696983

Time for backsubstitution: 2.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6871493, upper bound: 53.5525501
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6871493, upper bound: 53.7467420
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.3663168, 21.9989719, -18.6768570, 34.2317429, -45.5980606, 40.6758270
1: -12.7415857, 19.1561356, -20.9346581, 30.9911194, -43.7327042, 40.0907898
2: -13.1712570, 18.9806023, -21.4799671, 30.4932327, -43.6644821, 40.4605675
3: -15.4639874, 21.9047527, -25.5615311, 35.7306786, -51.1946640, 47.4662819
4: -14.5332193, 20.9419708, -23.9052448, 33.9909401, -48.5241585, 44.8472137

Time for backsubstitution: 2.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0513253, upper bound: 52.1554859
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -52.0513253, upper bound: 52.1554859
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -15.7345610, 29.7194901, -18.6768570, 34.2317429, -49.9663010, 48.3963470
1: -17.7042294, 27.1811428, -20.9346581, 30.9911194, -48.6953430, 48.1157875
2: -18.1944504, 26.7729149, -21.4799671, 30.4932327, -48.6876755, 48.2528801
3: -21.7213821, 31.2786942, -25.5615311, 35.7306786, -57.4520607, 56.8402252
4: -20.4752922, 29.6744862, -23.9052448, 33.9909401, -54.4662323, 53.5797310

Time for backsubstitution: 2.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=62.94061279296875
rel_dist={0: [-54.29988917735716, 54.29988917735716]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1092.11 seconds
