## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 743.673742927666


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.5269012, 721.5264282, -138.5269012, 721.5264282, -860.0532837, 860.0533447)
1: (-226.4326935, 857.1390381, -226.4326935, 857.1390381, -1083.5717773, 1083.5716553)
2: (-160.1910706, 887.5496826, -160.1910706, 887.5496826, -1047.7407227, 1047.7406006)
3: (-390.1859741, 752.6910400, -390.1859741, 752.6910400, -1142.8769531, 1142.8769531)
4: (-263.8327942, 761.5472412, -263.8327942, 761.5472412, -1025.3800049, 1025.3800049)

## BASE Result
execution time: IAR + LP analysis = 2.03 + 1.94 = 3.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -743.6896151, upper bound: 743.6896151


# Binary Search by BASE starts (time budget: 1196.03 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=860.0533447265625
rel_dist={0: [-743.6895975863799, 743.6895975863799]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=860.0533447265625
rel_dist={0: [-743.6893068054288, 743.6893068054287]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=860.0533447265625
rel_dist={0: [-743.6887393664056, 743.6887393664056]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=860.0533447265625
rel_dist={0: [-743.6883248579472, 743.6883248579472]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=860.0533447265625
rel_dist={0: [-743.6880449320028, 743.6880449320029]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=860.0533447265625
rel_dist={0: [-743.6877737489342, 743.6877737489342]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=860.0533447265625
rel_dist={0: [-743.6876008715927, 743.6876008715928]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=860.0533447265625
rel_dist={0: [-743.6875013877294, 743.6875013877293]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=860.0533447265625
rel_dist={0: [-743.6874472022885, 743.6874472022885]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=860.0533447265625
rel_dist={0: [-743.6874172086395, 743.6874172086395]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=860.0533447265625
rel_dist={0: [-743.6874005417767, 743.6874005417767]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=860.0533447265625
rel_dist={0: [-743.6873920794325, 743.6873920794324]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=860.0533447265625
rel_dist={0: [-743.6873878482721, 743.687387848272]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=860.0533447265625
rel_dist={0: [-743.6873857327159, 743.6873857327159]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=860.0533447265625
rel_dist={0: [-743.6873846764196, 743.6873846749841]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=860.0533447265625
rel_dist={0: [-743.6873841475402, 743.6873841492904]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=860.0533447265625
rel_dist={0: [-743.6873838833546, 743.6873838817382]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=860.0533447265625
rel_dist={0: [-743.6873837518972, 743.6873837516682]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=860.0533447265625
rel_dist={0: [-743.6873836840066, 743.6873836867503]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=860.0533447265625
rel_dist={0: [-743.6873836604632, 743.6873836596296]}

## Binary Search Result
Binary search time: 77.20 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1118.83 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6883770, upper bound: 743.6867587
time: 0.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6881450, upper bound: 743.6881450
time: 0.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.50 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 0, lower bound: -743.6883770, upper bound: 743.6867587
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 0, lower bound: -743.6881450, upper bound: 743.6881450

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -133.1433258, 692.8184204, -138.5269012, 721.5264282, -854.6695557, 831.3453369
1: -217.3697815, 823.0866089, -226.4326935, 857.1390381, -1074.5087891, 1049.5192871
2: -153.8224335, 852.0822754, -160.1910706, 887.5496826, -1041.3720703, 1012.2733154
3: -374.4883728, 722.8275757, -390.1859741, 752.6910400, -1127.1794434, 1113.0135498
4: -253.3396606, 731.3729858, -263.8327942, 761.5472412, -1014.8869019, 995.2058105

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6795367, upper bound: 743.6774439
time: 0.75 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
time: 0.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -136.5654449, 712.0169678, -138.5269012, 721.5264282, -858.0917969, 850.5438843
1: -223.1357117, 845.3634644, -226.4326935, 857.1390381, -1080.2745361, 1071.7961426
2: -157.8641815, 876.0595093, -160.1910706, 887.5496826, -1045.4136963, 1036.2506104
3: -384.4993286, 741.7769165, -390.1859741, 752.6910400, -1137.1903076, 1131.9628906
4: -259.9345093, 750.8963623, -263.8327942, 761.5472412, -1021.4817505, 1014.7291260

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793079, upper bound: 743.6784223
time: 0.79 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6702943, upper bound: 743.6702943
time: 0.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.34 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -743.6795367, upper bound: 743.6774439
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -743.6793079, upper bound: 743.6784223
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 3.34
Output dim: 0, lower bound: -743.6702943, upper bound: 743.6702943

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -138.5269012, 721.5264282, -844.9579468, 781.3366699
1: -201.7657318, 763.8026123, -226.4326935, 857.1390381, -1058.9047852, 990.2352905
2: -142.5318909, 790.5067139, -160.1910706, 887.5496826, -1030.0815430, 950.6977539
3: -347.4673157, 670.4584961, -390.1859741, 752.6910400, -1100.1580811, 1060.6445312
4: -234.7127075, 678.4046021, -263.8327942, 761.5472412, -996.2599487, 942.2373657

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
time: 0.71 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -136.9550476, 714.2769775, -957.8859863, 1439.4020996
1: -401.4721985, 1546.7423096, -223.8997040, 848.4136353, -1249.8857422, 1770.6419678
2: -282.7309570, 1598.5117188, -158.4233246, 878.5940552, -1161.3249512, 1756.9349365
3: -693.1887207, 1357.5162354, -385.8987732, 744.7566528, -1437.9453125, 1743.4150391
4: -466.2039795, 1371.2233887, -260.9156494, 753.7153931, -1219.9194336, 1632.1390381

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
time: 0.67 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -127.0210266, 662.5469360, -138.5269012, 721.5264282, -848.5472412, 801.0738525
1: -207.7908630, 786.8043823, -226.4326935, 857.1390381, -1064.9296875, 1013.2370605
2: -146.7888184, 815.2058716, -160.1910706, 887.5496826, -1034.3385010, 975.3969116
3: -357.9414368, 690.1265259, -390.1859741, 752.6910400, -1110.6323242, 1080.3125000
4: -241.6536560, 698.6527100, -263.8327942, 761.5472412, -1003.2008667, 962.4854736

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6702943, upper bound: 743.6702943
time: 0.72 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6702943, upper bound: 743.6702943
time: 0.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.29 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -743.6770094, upper bound: 743.6714788
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.29
Output dim: 0, lower bound: -743.6702943, upper bound: 743.6702943
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.29
Output dim: 0, lower bound: -743.6702943, upper bound: 743.6702943

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -128.5790863, 670.3025513, -793.7341309, 771.3887939
1: -201.7657318, 763.8026123, -210.4309387, 796.4016113, -998.1673584, 974.2334595
2: -142.5318909, 790.5067139, -148.6245575, 824.4069214, -966.9387817, 939.1312256
3: -347.4673157, 670.4584961, -362.4870300, 699.0101929, -1046.4772949, 1032.9454346
4: -234.7127075, 678.4046021, -244.7495728, 707.2666626, -941.9793701, 923.1541748

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6733881, upper bound: 743.6730087
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6792358, upper bound: 743.6762802
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -250.1162415, 1337.2148438, -1460.6462402, 892.9260254
1: -201.7657318, 763.8026123, -412.2479858, 1588.0107422, -1789.7763672, 1176.0505371
2: -142.5318909, 790.5067139, -290.3746338, 1641.2213135, -1783.7531738, 1080.8812256
3: -347.4673157, 670.4584961, -711.9147949, 1393.7313232, -1741.1982422, 1382.3732910
4: -234.7127075, 678.4046021, -478.8499146, 1407.6657715, -1642.3784180, 1157.2545166

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6790638, upper bound: 743.6754037
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6792358, upper bound: 743.6762802
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -128.5790863, 670.3025513, -913.9114990, 1431.0261230
1: -401.4721985, 1546.7423096, -210.4309387, 796.4016113, -1197.8737793, 1757.1730957
2: -282.7309570, 1598.5117188, -148.6245575, 824.4069214, -1107.1375732, 1747.1362305
3: -693.1887207, 1357.5162354, -362.4870300, 699.0101929, -1392.1989746, 1720.0031738
4: -466.2039795, 1371.2233887, -244.7495728, 707.2666626, -1173.4707031, 1615.9727783

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6413376, upper bound: 743.6432633
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6758975, upper bound: 743.6702654
time: 0.59 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6759132, upper bound: 743.6704876
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -250.1162415, 1337.2148438, -1580.8238525, 1552.5627441
1: -401.4721985, 1546.7423096, -412.2479858, 1588.0107422, -1989.4829102, 1958.9902344
2: -282.7309570, 1598.5117188, -290.3746338, 1641.2213135, -1923.9520264, 1888.8861084
3: -693.1887207, 1357.5162354, -711.9147949, 1393.7313232, -2086.9194336, 2069.4311523
4: -466.2039795, 1371.2233887, -478.8499146, 1407.6657715, -1873.8697510, 1850.0732422

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743709, upper bound: 743.6686943
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6657768, upper bound: 743.6606473
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.28 seconds
IS_A1_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -743.6733881, upper bound: 743.6730087
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 0, lower bound: -743.6792358, upper bound: 743.6762802
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 0, lower bound: -743.6790638, upper bound: 743.6754037
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 0, lower bound: -743.6792358, upper bound: 743.6762802
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 0, lower bound: -743.6758975, upper bound: 743.6702654
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 0, lower bound: -743.6759132, upper bound: 743.6704876
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 0, lower bound: -743.6743709, upper bound: 743.6686943
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 3.28
Output dim: 0, lower bound: -743.6657768, upper bound: 743.6606473

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -128.5790863, 670.3025513, -787.6891479, 740.6250610
1: -191.7628937, 727.0043945, -210.4309387, 796.4016113, -988.1644897, 937.4351807
2: -135.5527344, 752.8033447, -148.6245575, 824.4069214, -959.9596558, 901.4277954
3: -330.2159424, 637.6373901, -362.4870300, 699.0101929, -1029.2260742, 1000.1243896
4: -223.1825562, 645.3632812, -244.7495728, 707.2666626, -930.4492188, 890.1128540

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6846886, upper bound: 743.6833995
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787504, upper bound: 743.6760579
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -244.4674072, 1308.1789551, -1431.6103516, 887.2771606
1: -201.7657318, 763.8026123, -402.8388672, 1553.4019775, -1755.1674805, 1166.6414795
2: -142.5318909, 790.5067139, -283.7665100, 1605.3963623, -1747.9282227, 1074.2730713
3: -347.4673157, 670.4584961, -695.8218994, 1362.9735107, -1710.4404297, 1366.2800293
4: -234.7127075, 678.4046021, -467.9937439, 1376.6451416, -1611.3577881, 1146.3983154

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6788747, upper bound: 743.6745373
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786221, upper bound: 743.6744032
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -245.5195770, 1313.1480713, -1436.5794678, 888.3293457
1: -201.7657318, 763.8026123, -404.5788879, 1559.3142090, -1761.0797119, 1168.3814697
2: -142.5318909, 790.5067139, -285.0334473, 1611.6280518, -1754.1599121, 1075.5400391
3: -347.4673157, 670.4584961, -698.6735840, 1368.2851562, -1715.7520752, 1369.1319580
4: -234.7127075, 678.4046021, -470.0249939, 1382.0180664, -1616.7307129, 1148.4295654

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6790709, upper bound: 743.6753584
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787600, upper bound: 743.6752243
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -118.2663727, 615.8584595, -859.4674072, 1420.7132568
1: -401.4721985, 1546.7423096, -193.5168457, 731.5498657, -1133.0219727, 1740.2591553
2: -282.7309570, 1598.5117188, -136.6161346, 757.9348755, -1040.6657715, 1735.1278076
3: -693.1887207, 1357.5162354, -333.4353638, 641.7996826, -1334.9882812, 1690.9515381
4: -466.2039795, 1371.2233887, -224.9028168, 649.7612915, -1115.9653320, 1596.1259766

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6826575, upper bound: 743.6793744
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6828121, upper bound: 743.6792200
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -123.4280624, 643.4694824, -887.0784912, 1425.8748779
1: -401.4721985, 1546.7423096, -202.0025482, 764.3729248, -1165.8450928, 1748.7448730
2: -282.7309570, 1598.5117188, -142.6591034, 791.5834351, -1074.3144531, 1741.1705322
3: -693.1887207, 1357.5162354, -347.8949585, 670.5841675, -1363.7729492, 1705.4111328
4: -466.2039795, 1371.2233887, -234.8611450, 678.7764282, -1144.9802246, 1606.0843506

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6824422, upper bound: 743.6789868
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6828008, upper bound: 743.6789287
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -250.7165222, 1339.7071533, -1583.3161621, 1553.1633301
1: -401.4721985, 1546.7423096, -412.9922180, 1591.0345459, -1992.5065918, 1959.7344971
2: -282.7309570, 1598.5117188, -290.9667664, 1644.2225342, -1926.9532471, 1889.4783936
3: -693.1887207, 1357.5162354, -713.2125244, 1396.5231934, -2089.7116699, 2070.7287598
4: -466.2039795, 1371.2233887, -479.7973022, 1410.4566650, -1876.6606445, 1851.0203857

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6734836, upper bound: 743.6677492
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5979616, upper bound: 743.6170922
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743709, upper bound: 743.6686943
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6678482, upper bound: 743.6605315
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6624555, upper bound: 743.6556719
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6734774, upper bound: 743.6678642
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743709, upper bound: 743.6686943
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.90 seconds
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6846886, upper bound: 743.6833995
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6787504, upper bound: 743.6760579
IS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6788747, upper bound: 743.6745373
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6786221, upper bound: 743.6744032
IS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6790709, upper bound: 743.6753584
IS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6787600, upper bound: 743.6752243
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6826575, upper bound: 743.6793744
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6828121, upper bound: 743.6792200
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6824422, upper bound: 743.6789868
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6828008, upper bound: 743.6789287
IS_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6734774, upper bound: 743.6678642
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.90
Output dim: 0, lower bound: -743.6743709, upper bound: 743.6686943

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -127.5799332, 665.7973022, -783.1838989, 739.6259766
1: -191.7628937, 727.0043945, -208.5430145, 790.9661865, -982.7290649, 935.5474243
2: -135.5527344, 752.8033447, -147.3827362, 818.6778564, -954.2305908, 900.1860352
3: -330.2159424, 637.6373901, -359.1884766, 693.9826660, -1024.1986084, 996.8257446
4: -223.1825562, 645.3632812, -242.6873474, 702.1574097, -925.3399658, 888.0505981

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6840095, upper bound: 743.6825748
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6831559, upper bound: 743.6821540
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -126.6055450, 659.9930420, -777.3796387, 738.6513672
1: -191.7628937, 727.0043945, -207.1665497, 784.1243286, -975.8872070, 934.1708984
2: -135.5527344, 752.8033447, -146.3279114, 811.7086792, -947.2614136, 899.1311646
3: -330.2159424, 637.6373901, -356.8464661, 688.1320190, -1018.3479004, 994.4836426
4: -223.1825562, 645.3632812, -240.9527283, 696.3119507, -919.4945068, 886.3160400

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775320, upper bound: 743.6741164
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774690, upper bound: 743.6748705
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -102.8738937, 539.4071045, -243.8543396, 1305.0135498, -1407.8873291, 783.2613525
1: -167.7988281, 640.0291138, -401.8410339, 1549.6354980, -1717.4343262, 1041.8701172
2: -118.7764435, 663.3266602, -283.0545349, 1601.5018311, -1720.2783203, 946.3812256
3: -289.1987610, 560.0819092, -694.0940552, 1359.6447754, -1648.8435059, 1254.1757812
4: -195.4968872, 567.6482544, -466.8258667, 1373.2978516, -1568.7946777, 1034.4741211

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786438, upper bound: 743.6740757
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6781129, upper bound: 743.6738882
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -120.7842331, 628.9497681, -244.4674072, 1308.1789551, -1428.9631348, 873.4171143
1: -197.4676971, 747.3856201, -402.8388672, 1553.4019775, -1750.8696289, 1150.2244873
2: -139.5008698, 773.4520874, -283.7665100, 1605.3963623, -1744.8972168, 1057.2186279
3: -340.1154480, 655.9765015, -695.8218994, 1362.9735107, -1703.0888672, 1351.7983398
4: -229.7405701, 663.6655884, -467.9937439, 1376.6451416, -1606.3857422, 1131.6593018

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6784983, upper bound: 743.6739416
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780316, upper bound: 743.6737279
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -102.8738937, 539.4071045, -244.9624481, 1310.2556152, -1413.1295166, 784.3695679
1: -167.7988281, 640.0291138, -403.6732788, 1555.8756104, -1723.6744385, 1043.7022705
2: -118.7764435, 663.3266602, -284.3862610, 1608.0706787, -1726.8470459, 947.7128906
3: -289.1987610, 560.0819092, -697.1050415, 1365.2519531, -1654.4506836, 1257.1867676
4: -195.4968872, 567.6482544, -468.9622192, 1378.9670410, -1574.4638672, 1036.6104736

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6757328, upper bound: 743.6725674
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6570283, upper bound: 743.6569464
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -120.7842331, 628.9497681, -245.5195770, 1313.1480713, -1433.9322510, 874.4692993
1: -197.4676971, 747.3856201, -404.5788879, 1559.3142090, -1756.7818604, 1151.9643555
2: -139.5008698, 773.4520874, -285.0334473, 1611.6280518, -1751.1289062, 1058.4854736
3: -340.1154480, 655.9765015, -698.6735840, 1368.2851562, -1708.4006348, 1354.6501465
4: -229.7405701, 663.6655884, -470.0249939, 1382.0180664, -1611.7586670, 1133.6905518

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6758896, upper bound: 743.6726445
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6571472, upper bound: 743.6569405
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -243.0082397, 1299.3513184, -98.4456558, 516.2005615, -759.2088013, 1397.7968750
1: -400.4924927, 1543.0550537, -160.7731476, 612.1020508, -1012.5945435, 1703.8281250
2: -282.0349731, 1594.7023926, -113.7264023, 635.3582153, -917.3931885, 1708.4288330
3: -691.4927368, 1354.2535400, -277.3016357, 535.2220459, -1226.7148438, 1631.5551758
4: -465.0600891, 1367.9425049, -187.1013641, 542.9273071, -1007.9874268, 1555.0435791

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751556, upper bound: 743.6780058
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816774, upper bound: 743.6790519
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -115.4791565, 601.3374634, -844.9464111, 1417.9259033
1: -401.4721985, 1546.7423096, -188.9927368, 714.3245239, -1115.7966309, 1735.7351074
2: -282.7309570, 1598.5117188, -133.4235687, 740.0919189, -1022.8226929, 1731.9350586
3: -693.1887207, 1357.5162354, -325.6819458, 626.5873413, -1319.7760010, 1683.1982422
4: -466.2039795, 1371.2233887, -219.6637726, 634.2804565, -1100.4841309, 1590.8870850

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752584, upper bound: 743.6778170
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817970, upper bound: 743.6788674
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -243.0082397, 1299.3513184, -103.4892807, 543.0014648, -786.0097046, 1402.8405762
1: -400.4924927, 1543.0550537, -169.0302582, 644.1295166, -1044.6220703, 1712.0852051
2: -282.0349731, 1594.7023926, -119.6135712, 668.0424805, -950.0774536, 1714.3159180
3: -691.4927368, 1354.2535400, -291.3856812, 563.4126587, -1254.9052734, 1645.6391602
4: -465.0600891, 1367.9425049, -196.8375854, 571.2093506, -1036.2694092, 1564.7797852

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751489, upper bound: 743.6776286
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816626, upper bound: 743.6786751
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -120.7921219, 629.6492310, -873.2581787, 1423.2391357
1: -401.4721985, 1546.7423096, -197.7229156, 748.0092773, -1149.4814453, 1744.4652100
2: -282.7309570, 1598.5117188, -139.6376801, 774.6166992, -1057.3475342, 1738.1494141
3: -693.1887207, 1357.5162354, -340.5773010, 656.1560669, -1349.3447266, 1698.0935059
4: -466.2039795, 1371.2233887, -229.9062195, 664.0927124, -1130.2966309, 1601.1295166

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776234, upper bound: 743.6756801
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6758102, upper bound: 743.6723175
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6730838, upper bound: 743.6693861
time: 0.68 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -260.1164246, 1395.6341553, -250.7165222, 1339.7071533, -1599.8234863, 1645.2591553
1: -429.1390686, 1657.0156250, -412.9922180, 1591.0345459, -2020.1733398, 2069.2023926
2: -302.0234375, 1712.9410400, -290.9667664, 1644.2225342, -1946.2459717, 2003.3355713
3: -740.7056274, 1452.8055420, -713.2125244, 1396.5231934, -2137.2282715, 2164.9663086
4: -497.8664246, 1467.6594238, -479.7973022, 1410.4566650, -1908.3231201, 1946.4467773

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6734836, upper bound: 743.6677492
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5979616, upper bound: 743.6170922
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6678482, upper bound: 743.6605315
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743709, upper bound: 743.6686943
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6624555, upper bound: 743.6556719
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741212, upper bound: 743.6682126
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6729941, upper bound: 743.6671710
time: 0.89 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 16.01 seconds
IS_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6840095, upper bound: 743.6825748
IS_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6831559, upper bound: 743.6821540
IS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6775320, upper bound: 743.6741164
IS_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6774690, upper bound: 743.6748705
IS_A1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6786438, upper bound: 743.6740757
IS_A1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6781129, upper bound: 743.6738882
IS_A1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6784983, upper bound: 743.6739416
IS_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6780316, upper bound: 743.6737279
IS_A1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6757328, upper bound: 743.6725674
IS_A1_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6570283, upper bound: 743.6569464
IS_A1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6758896, upper bound: 743.6726445
IS_A1_A1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6571472, upper bound: 743.6569405
IS_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6751556, upper bound: 743.6780058
IS_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6816774, upper bound: 743.6790519
IS_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6752584, upper bound: 743.6778170
IS_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6817970, upper bound: 743.6788674
IS_A1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6751489, upper bound: 743.6776286
IS_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6816626, upper bound: 743.6786751
IS_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6758102, upper bound: 743.6723175
IS_A1_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6730838, upper bound: 743.6693861
IS_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6741212, upper bound: 743.6682126
IS_A1_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 16.01
Output dim: 0, lower bound: -743.6729941, upper bound: 743.6671710

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -106.4925842, 554.9370728, -127.5799332, 665.7973022, -772.2899170, 682.5170288
1: -173.8893280, 658.7961426, -208.5430145, 790.9661865, -964.8554077, 867.3391724
2: -122.8767929, 682.9953003, -147.3827362, 818.6778564, -941.5546265, 830.3779907
3: -299.5489502, 577.4619141, -359.1884766, 693.9826660, -993.5315552, 936.6503296
4: -202.2262421, 584.8556519, -242.6873474, 702.1574097, -904.3836670, 827.5429688

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6822168, upper bound: 743.6782312
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6789249, upper bound: 743.6779863
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -112.4629211, 586.3134155, -127.5799332, 665.7973022, -778.2602539, 713.8933716
1: -183.7023315, 696.2958374, -208.5430145, 790.9661865, -974.6683960, 904.8388672
2: -129.8522797, 721.3862915, -147.3827362, 818.6778564, -948.5301514, 868.7689819
3: -316.2727051, 610.4154663, -359.1884766, 693.9826660, -1010.2552490, 969.6038818
4: -213.7347260, 618.0704956, -242.6873474, 702.1574097, -915.8921509, 860.7578125

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6813440, upper bound: 743.6780475
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780521, upper bound: 743.6778027
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -116.4187546, 606.2535400, -723.6401367, 728.4647217
1: -191.7628937, 727.0043945, -190.4518890, 720.1048584, -911.8677368, 917.4562988
2: -135.5527344, 752.8033447, -134.4618073, 746.0765991, -881.6293335, 887.2651367
3: -330.2159424, 637.6373901, -328.1390381, 631.6458130, -961.8615723, 965.7764282
4: -223.1825562, 645.3632812, -221.3377533, 639.5360107, -862.7185669, 866.7010498

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6717994, upper bound: 743.6653566
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770832, upper bound: 743.6727895
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6762557, upper bound: 743.6715134
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6764244, upper bound: 743.6739730
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6768174, upper bound: 743.6739670
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -121.4223862, 633.0242920, -750.4108887, 733.4683228
1: -191.7628937, 727.0043945, -198.6864166, 751.9244385, -943.6873169, 925.6906738
2: -135.5527344, 752.8033447, -140.3257751, 778.7070923, -914.2598267, 893.1291504
3: -330.2159424, 637.6373901, -342.1655579, 659.5394897, -989.7553101, 979.8029785
4: -223.1825562, 645.3632812, -230.9998169, 667.6664429, -890.8489990, 876.3630371

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6729171, upper bound: 743.6697486
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6656705, upper bound: 743.6654968
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -93.6532288, 490.5315857, -243.8543396, 1305.0135498, -1398.6667480, 734.3859253
1: -152.6970367, 581.8051147, -401.8410339, 1549.6354980, -1702.3325195, 983.6460571
2: -108.0520782, 603.7379150, -283.0545349, 1601.5018311, -1709.5539551, 886.7924194
3: -263.2998657, 508.7830505, -694.0940552, 1359.6447754, -1622.9445801, 1202.8769531
4: -177.7571411, 516.1376343, -466.8258667, 1373.2978516, -1551.0549316, 982.9634399

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6517666, upper bound: 743.6484284
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6774881, upper bound: 743.6707065
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782459, upper bound: 743.6725553
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -98.5702744, 516.7167358, -243.8543396, 1305.0135498, -1403.5838623, 760.5710449
1: -160.7362976, 612.9924316, -401.8410339, 1549.6354980, -1710.3718262, 1014.8333740
2: -113.7882843, 635.5537109, -283.0545349, 1601.5018311, -1715.2901611, 918.6082764
3: -277.0000610, 536.1976929, -694.0940552, 1359.6447754, -1636.6447754, 1230.2916260
4: -187.2500458, 543.6464844, -466.8258667, 1373.2978516, -1560.5478516, 1010.4722290

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6632750, upper bound: 743.6588418
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755637, upper bound: 743.6710501
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742213, upper bound: 743.6702995
time: 1.04 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6687347, upper bound: 743.6636936
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -110.5168381, 574.7239380, -244.4674072, 1308.1789551, -1418.6958008, 819.1913452
1: -180.6490021, 682.8037720, -402.8388672, 1553.4019775, -1734.0507812, 1085.6425781
2: -127.5510025, 707.2360229, -283.7665100, 1605.3963623, -1732.9472656, 991.0025635
3: -311.2001038, 599.0428467, -695.8218994, 1362.9735107, -1674.1734619, 1294.8643799
4: -209.9757385, 606.4170532, -467.9937439, 1376.6451416, -1586.6208496, 1074.4107666

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751089, upper bound: 743.6712289
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695402, upper bound: 743.6645565
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -115.6310959, 602.0667725, -244.4674072, 1308.1789551, -1423.8098145, 846.5341797
1: -189.0406952, 715.3134155, -402.8388672, 1553.4019775, -1742.4426270, 1118.1522217
2: -133.5306702, 740.5586548, -283.7665100, 1605.3963623, -1738.9270020, 1024.3251953
3: -325.5199890, 627.5319824, -695.8218994, 1362.9735107, -1688.4934082, 1323.3538818
4: -219.8414459, 635.1401978, -467.9937439, 1376.6451416, -1596.4864502, 1103.1339111

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746098, upper bound: 743.6707875
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6690708, upper bound: 743.6641328
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -102.8738937, 539.4071045, -246.9420166, 1320.2366943, -1423.1104736, 786.3489990
1: -167.7988281, 640.0291138, -406.7655945, 1567.7904053, -1735.5892334, 1046.7945557
2: -118.7764435, 663.3266602, -286.6111145, 1620.2916260, -1739.0681152, 949.9377441
3: -289.1987610, 560.0819092, -702.4229736, 1375.8458252, -1665.0445557, 1262.5048828
4: -195.4968872, 567.6482544, -472.6008606, 1389.5914307, -1585.0882568, 1040.2491455

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751842, upper bound: 743.6720660
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745056, upper bound: 743.6712014
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -120.7842331, 628.9497681, -247.4863892, 1323.0605469, -1443.8447266, 876.4360962
1: -197.4676971, 747.3856201, -407.6497803, 1571.1477051, -1768.6153564, 1155.0354004
2: -139.5008698, 773.4520874, -287.2434387, 1623.7653809, -1763.2662354, 1060.6953125
3: -340.1154480, 655.9765015, -703.9545898, 1378.8055420, -1718.9210205, 1359.9311523
4: -229.7405701, 663.6655884, -473.6381226, 1392.5699463, -1622.3105469, 1137.3037109

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754131, upper bound: 743.6721455
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6748926, upper bound: 743.6717010
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -238.0637207, 1273.8032227, -98.4456558, 516.2005615, -754.2642212, 1372.2487793
1: -392.2139587, 1512.6197510, -160.7731476, 612.1020508, -1004.3159790, 1673.3929443
2: -276.2398987, 1563.1684570, -113.7264023, 635.3582153, -911.5981445, 1676.8948975
3: -677.3527222, 1327.2672119, -277.3016357, 535.2220459, -1212.5743408, 1604.5688477
4: -455.5540466, 1340.6821289, -187.1013641, 542.9273071, -998.4812622, 1527.7832031

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745648, upper bound: 743.6759601
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6696877, upper bound: 743.6757821
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745106, upper bound: 743.6776479
time: 0.70 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -238.3988037, 1275.1735840, -98.4456558, 516.2005615, -754.5993652, 1373.6191406
1: -392.8224792, 1514.2291260, -160.7731476, 612.1020508, -1004.9244995, 1675.0023193
2: -276.6891479, 1564.9869385, -113.7264023, 635.3582153, -912.0473633, 1678.7133789
3: -678.2596436, 1328.7338867, -277.3016357, 535.2220459, -1213.4816895, 1606.0355225
4: -456.2293396, 1342.2238770, -187.1013641, 542.9273071, -999.1566162, 1529.3250732

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785140, upper bound: 743.6760174
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6816686, upper bound: 743.6790519
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797858, upper bound: 743.6771367
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6798208, upper bound: 743.6757705
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -238.6806641, 1276.9854736, -115.4791565, 601.3374634, -840.0181274, 1392.4643555
1: -393.2183838, 1516.4067383, -188.9927368, 714.3245239, -1107.5428467, 1705.3994141
2: -276.9560852, 1567.0826416, -133.4235687, 740.0919189, -1017.0479736, 1700.5061035
3: -679.0913086, 1330.6146240, -325.6819458, 626.5873413, -1305.6785889, 1656.2966309
4: -456.7281494, 1344.0480957, -219.6637726, 634.2804565, -1091.0083008, 1563.7117920

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746678, upper bound: 743.6757350
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6697638, upper bound: 743.6755666
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746153, upper bound: 743.6774364
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -238.9429169, 1278.0006104, -115.4791565, 601.3374634, -840.2803955, 1393.4796143
1: -393.7060242, 1517.5902100, -188.9927368, 714.3245239, -1108.0303955, 1706.5830078
2: -277.3214722, 1568.4636230, -133.4235687, 740.0919189, -1017.4133301, 1701.8870850
3: -679.7905884, 1331.6966553, -325.6819458, 626.5873413, -1306.3778076, 1657.3786621
4: -457.2668762, 1345.2036133, -219.6637726, 634.2804565, -1091.5471191, 1564.8673096

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786187, upper bound: 743.6757931
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6767766, upper bound: 743.6710717
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6747335, upper bound: 743.6730968
time: 1.06 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817968, upper bound: 743.6788674
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6802259, upper bound: 743.6767846
time: 0.70 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -238.0637207, 1273.8032227, -103.4892807, 543.0014648, -781.0650635, 1377.2924805
1: -392.2139587, 1512.6197510, -169.0302582, 644.1295166, -1036.3435059, 1681.6500244
2: -276.2398987, 1563.1684570, -119.6135712, 668.0424805, -944.2823486, 1682.7819824
3: -677.3527222, 1327.2672119, -291.3856812, 563.4126587, -1240.7648926, 1618.6528320
4: -455.5540466, 1340.6821289, -196.8375854, 571.2093506, -1026.7633057, 1537.5194092

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745577, upper bound: 743.6755429
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751433, upper bound: 743.6776286
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6668940, upper bound: 743.6603202
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6678169, upper bound: 743.6621678
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -238.3988037, 1275.1735840, -103.4892807, 543.0014648, -781.4002686, 1378.6628418
1: -392.8224792, 1514.2291260, -169.0302582, 644.1295166, -1036.9520264, 1683.2593994
2: -276.6891479, 1564.9869385, -119.6135712, 668.0424805, -944.7316284, 1684.6003418
3: -678.2596436, 1328.7338867, -291.3856812, 563.4126587, -1241.6722412, 1620.1196289
4: -456.2293396, 1342.2238770, -196.8375854, 571.2093506, -1027.4384766, 1539.0612793

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785024, upper bound: 743.6756003
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6760959, upper bound: 743.6700991
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797780, upper bound: 743.6754327
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -244.4261932, 1305.8933105, -120.7921219, 629.6492310, -874.0753174, 1426.6854248
1: -402.5630798, 1550.9205322, -197.7229156, 748.0092773, -1150.5723877, 1748.6434326
2: -283.5724792, 1602.6917725, -139.6376801, 774.6166992, -1058.1892090, 1742.3294678
3: -695.0786743, 1361.3898926, -340.5773010, 656.1560669, -1351.2346191, 1701.9670410
4: -467.5620422, 1375.0917969, -229.9062195, 664.0927124, -1131.6545410, 1604.9979248

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6694495, upper bound: 743.6699633
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744252, upper bound: 743.6710059
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -250.0434265, 1342.4066162, -250.7165222, 1339.7071533, -1589.7506104, 1591.9254150
1: -412.7971191, 1593.6484375, -412.9922180, 1591.0345459, -2003.8314209, 2005.6892090
2: -290.3471680, 1647.6481934, -290.9667664, 1644.2225342, -1934.5693359, 1937.9049072
3: -712.5440674, 1396.9146729, -713.2125244, 1396.5231934, -2108.9924316, 2109.0065918
4: -478.5559082, 1411.5390625, -479.7973022, 1410.4566650, -1889.0124512, 1890.2609863

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6731372, upper bound: 743.6673004
time: 0.98 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.5979303, upper bound: 743.6166089
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741212, upper bound: 743.6682126
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6615410, upper bound: 743.6573821
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6547465, upper bound: 743.6508273
time: 0.70 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741212, upper bound: 743.6682126
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741212, upper bound: 743.6682126
time: 0.90 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 16.17 seconds
IS_A1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6822168, upper bound: 743.6782312
IS_A1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6789249, upper bound: 743.6779863
IS_A1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6813440, upper bound: 743.6780475
IS_A1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6780521, upper bound: 743.6778027
IS_A1_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6764244, upper bound: 743.6739730
IS_A1_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6768174, upper bound: 743.6739670
IS_A1_A1_B1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6729171, upper bound: 743.6697486
IS_A1_A1_B1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6656705, upper bound: 743.6654968
IS_A1_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6774881, upper bound: 743.6707065
IS_A1_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6782459, upper bound: 743.6725553
IS_A1_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6742213, upper bound: 743.6702995
IS_A1_A1_B2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6687347, upper bound: 743.6636936
IS_A1_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6751089, upper bound: 743.6712289
IS_A1_A1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6695402, upper bound: 743.6645565
IS_A1_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6746098, upper bound: 743.6707875
IS_A1_A1_B2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6690708, upper bound: 743.6641328
IS_A1_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6751842, upper bound: 743.6720660
IS_A1_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6745056, upper bound: 743.6712014
IS_A1_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6754131, upper bound: 743.6721455
IS_A1_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6748926, upper bound: 743.6717010
IS_A1_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6696877, upper bound: 743.6757821
IS_A1_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6745106, upper bound: 743.6776479
IS_A1_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6797858, upper bound: 743.6771367
IS_A1_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6798208, upper bound: 743.6757705
IS_A1_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6697638, upper bound: 743.6755666
IS_A1_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6746153, upper bound: 743.6774364
IS_A1_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6817968, upper bound: 743.6788674
IS_A1_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6802259, upper bound: 743.6767846
IS_A1_A2_B1_B2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6668940, upper bound: 743.6603202
IS_A1_A2_B1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6678169, upper bound: 743.6621678
IS_A1_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6760959, upper bound: 743.6700991
IS_A1_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6797780, upper bound: 743.6754327
IS_A1_A2_B1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6694495, upper bound: 743.6699633
IS_A1_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6744252, upper bound: 743.6710059
IS_A1_A2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6741212, upper bound: 743.6682126
IS_A1_A2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 16.17
Output dim: 0, lower bound: -743.6741212, upper bound: 743.6682126

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -106.4925842, 554.9370728, -122.1900177, 636.8364868, -743.3291016, 677.1270142
1: -173.8893280, 658.7961426, -199.7045746, 756.4603882, -930.3496704, 858.5006714
2: -122.8767929, 682.9953003, -141.1188812, 783.4572144, -906.3339844, 824.1141968
3: -299.5489502, 577.4619141, -343.9291992, 663.4416504, -962.9906006, 921.3909912
4: -202.2262421, 584.8556519, -232.3058929, 671.5488892, -873.7751465, 817.1614990

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6797987, upper bound: 743.6745443
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6817771, upper bound: 743.6778921
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6781389, upper bound: 743.6729392
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6821086, upper bound: 743.6781027
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -106.2503891, 553.6389771, -127.7602386, 666.8062134, -773.0565796, 681.3992310
1: -173.4863739, 657.2608032, -208.7126160, 791.9608765, -965.4472046, 865.9733887
2: -122.5931015, 681.4016113, -147.5619354, 820.0437622, -942.6368408, 828.9635620
3: -298.8548584, 576.1057129, -359.7370300, 694.5027466, -993.3576050, 935.8427124
4: -201.7567749, 583.4891968, -242.9948883, 702.9088745, -904.6655884, 826.4840698

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754011, upper bound: 743.6739917
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6784853, upper bound: 743.6774288
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6789249, upper bound: 743.6779863
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6789249, upper bound: 743.6779863
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -112.4629211, 586.3134155, -122.1900177, 636.8364868, -749.2994385, 708.5033569
1: -183.7023315, 696.2958374, -199.7045746, 756.4603882, -940.1627197, 896.0003662
2: -129.8522797, 721.3862915, -141.1188812, 783.4572144, -913.3095093, 862.5051880
3: -316.2727051, 610.4154663, -343.9291992, 663.4416504, -979.7142334, 954.3446045
4: -213.7347260, 618.0704956, -232.3058929, 671.5488892, -885.2836304, 850.3762817

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6799378, upper bound: 743.6744934
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6806799, upper bound: 743.6771470
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6805584, upper bound: 743.6775175
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6803317, upper bound: 743.6773488
time: 1.21 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -112.2148361, 584.9869385, -127.7602386, 666.8062134, -779.0210571, 712.7471924
1: -183.2905273, 694.7182617, -208.7126160, 791.9608765, -975.2513428, 903.4307861
2: -129.5617218, 719.7493286, -147.5619354, 820.0437622, -949.6054688, 867.3112183
3: -315.5617371, 609.0219727, -359.7370300, 694.5027466, -1010.0643311, 968.7589111
4: -213.2532349, 616.6671753, -242.9948883, 702.9088745, -916.1621094, 859.6620483

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755412, upper bound: 743.6739669
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6773880, upper bound: 743.6766837
time: 0.97 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6777822, upper bound: 743.6778027
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6777822, upper bound: 743.6778027
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -97.7733688, 513.7284546, -115.8136063, 603.0981445, -700.8713989, 629.5419922
1: -159.3895721, 609.2998047, -189.4631500, 716.3464355, -875.7360229, 798.7628174
2: -112.8911057, 631.9371338, -133.7614136, 742.1966553, -855.0877686, 765.6985474
3: -274.7174072, 532.7036133, -326.4377136, 628.3359985, -903.0534058, 859.1412354
4: -185.8033142, 540.0395508, -220.1850586, 636.2050781, -822.0083618, 760.2246094

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6710490, upper bound: 743.6652178
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755924, upper bound: 743.6715134
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763797, upper bound: 743.6738791
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6763797, upper bound: 743.6739730
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -114.6738663, 597.7830200, -116.4187546, 606.2535400, -720.9274292, 714.2017212
1: -187.3528900, 710.1228638, -190.4518890, 720.1048584, -907.4577637, 900.5747681
2: -132.4465637, 735.3177490, -134.4618073, 746.0765991, -878.5231323, 869.7795410
3: -322.6878357, 622.8237305, -328.1390381, 631.6458130, -954.3334961, 950.9627686
4: -218.1035614, 630.2503052, -221.3377533, 639.5360107, -857.6395874, 851.5880737

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6756006, upper bound: 743.6715134
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6767699, upper bound: 743.6738731
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6767699, upper bound: 743.6739670
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -92.3664474, 483.9183655, -229.4125824, 1229.5303955, -1321.8968506, 713.3309326
1: -150.5876617, 573.8440552, -378.1433105, 1459.1627197, -1609.7503662, 951.9873657
2: -106.5717621, 595.6366577, -266.4448547, 1509.2376709, -1615.8094482, 862.0813599
3: -259.6806946, 501.6614685, -653.5508423, 1278.8580322, -1538.5386963, 1155.2120361
4: -175.3153076, 509.0587158, -439.3456726, 1293.0384521, -1468.3537598, 948.4042969

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6734632, upper bound: 743.6677109
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6686121, upper bound: 743.6627861
time: 0.94 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -93.6532288, 490.5315857, -243.2976227, 1302.1055908, -1395.7587891, 733.8292236
1: -152.6970367, 581.8051147, -400.9390259, 1546.1535645, -1698.8505859, 982.7440796
2: -108.0520782, 603.7379150, -282.4139709, 1597.9449463, -1705.9969482, 886.1516724
3: -263.2998657, 508.7830505, -692.5370483, 1356.5466309, -1619.8464355, 1201.3200684
4: -177.7571411, 516.1376343, -465.7606506, 1370.2196045, -1547.9768066, 981.8981934

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6515464, upper bound: 743.6465317
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743968, upper bound: 743.6694961
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6621699, upper bound: 743.6590699
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -98.5702744, 516.7167358, -244.5006104, 1307.6458740, -1406.2161865, 761.2173462
1: -160.7362976, 612.9924316, -402.6793518, 1552.8074951, -1713.5437012, 1015.6717529
2: -113.7882843, 635.5537109, -283.7077942, 1604.7091064, -1718.4974365, 919.2614136
3: -277.0000610, 536.1976929, -695.5402222, 1362.5753174, -1639.5753174, 1231.7379150
4: -187.2500458, 543.6464844, -467.8513489, 1376.2795410, -1563.5295410, 1011.4978027

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6605032, upper bound: 743.6545838
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6718347, upper bound: 743.6668560
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6596031, upper bound: 743.6573865
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6729125, upper bound: 743.6667419
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6738079, upper bound: 743.6685144
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -110.5168381, 574.7239380, -245.1050720, 1310.7550049, -1421.2718506, 819.8289795
1: -180.6490021, 682.8037720, -403.6629944, 1556.5074463, -1737.1563721, 1086.4665527
2: -127.5510025, 707.2360229, -284.4091492, 1608.5339355, -1736.0848389, 991.6450806
3: -311.2001038, 599.0428467, -697.2430420, 1365.8476562, -1677.0477295, 1296.2855225
4: -209.9757385, 606.4170532, -469.0012512, 1379.5718994, -1589.5474854, 1075.4183350

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6672159, upper bound: 743.6638229
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6705624, upper bound: 743.6643190
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6737704, upper bound: 743.6677059
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746573, upper bound: 743.6694969
time: 0.63 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -115.6310959, 602.0667725, -245.1050720, 1310.7550049, -1426.3858643, 847.1718750
1: -189.0406952, 715.3134155, -403.6629944, 1556.5074463, -1745.5480957, 1118.9761963
2: -133.5306702, 740.5586548, -284.4091492, 1608.5339355, -1742.0645752, 1024.9676514
3: -325.5199890, 627.5319824, -697.2430420, 1365.8476562, -1691.3676758, 1324.7750244
4: -219.8414459, 635.1401978, -469.0012512, 1379.5718994, -1599.4130859, 1104.1413574

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6718386, upper bound: 743.6664010
time: 0.99 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6719276, upper bound: 743.6643168
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6730454, upper bound: 743.6670104
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6740779, upper bound: 743.6687941
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -93.6532288, 490.5315857, -246.9420166, 1320.2366943, -1413.8898926, 737.4736328
1: -152.6970367, 581.8051147, -406.7655945, 1567.7904053, -1720.4874268, 988.5706177
2: -108.0520782, 603.7379150, -286.6111145, 1620.2916260, -1728.3436279, 890.3489990
3: -263.2998657, 508.7830505, -702.4229736, 1375.8458252, -1639.1456299, 1211.2060547
4: -177.7571411, 516.1376343, -472.6008606, 1389.5914307, -1567.3485107, 988.7384644

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6605625, upper bound: 743.6571017
time: 1.06 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6725991, upper bound: 743.6679759
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6730290, upper bound: 743.6685020
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751842, upper bound: 743.6719939
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708823, upper bound: 743.6704888
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6708823, upper bound: 743.6720660
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -98.5702744, 516.7167358, -246.9420166, 1320.2366943, -1418.8070068, 763.6587524
1: -160.7362976, 612.9924316, -406.7655945, 1567.7904053, -1728.5267334, 1019.7579346
2: -113.7882843, 635.5537109, -286.6111145, 1620.2916260, -1734.0799561, 922.1647949
3: -277.0000610, 536.1976929, -702.4229736, 1375.8458252, -1652.8457031, 1238.6206055
4: -187.2500458, 543.6464844, -472.6008606, 1389.5914307, -1576.8414307, 1016.2473145

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6676969, upper bound: 743.6617158
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6724287, upper bound: 743.6679053
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6672803, upper bound: 743.6648351
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745056, upper bound: 743.6712014
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6682709, upper bound: 743.6662584
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6682709, upper bound: 743.6712014
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -110.5168381, 574.7239380, -247.4863892, 1323.0605469, -1433.5773926, 822.2102661
1: -180.6490021, 682.8037720, -407.6497803, 1571.1477051, -1751.7965088, 1090.4534912
2: -127.5510025, 707.2360229, -287.2434387, 1623.7653809, -1751.3161621, 994.4793701
3: -311.2001038, 599.0428467, -703.9545898, 1378.8055420, -1690.0056152, 1302.9971924
4: -209.9757385, 606.4170532, -473.6381226, 1392.5699463, -1602.5456543, 1080.0550537

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6717144, upper bound: 743.6675179
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6722778, upper bound: 743.6661371
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746122, upper bound: 743.6696640
time: 1.14 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754131, upper bound: 743.6721455
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6726367, upper bound: 743.6704179
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -115.6310959, 602.0667725, -247.4863892, 1323.0605469, -1438.6914062, 849.5531616
1: -189.0406952, 715.3134155, -407.6497803, 1571.1477051, -1760.1883545, 1122.9630127
2: -133.5306702, 740.5586548, -287.2434387, 1623.7653809, -1757.2960205, 1027.8018799
3: -325.5199890, 627.5319824, -703.9545898, 1378.8055420, -1704.3255615, 1331.4865723
4: -219.8414459, 635.1401978, -473.6381226, 1392.5699463, -1612.4112549, 1108.7781982

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6717144, upper bound: 743.6676840
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6732504, upper bound: 743.6661371
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6748926, upper bound: 743.6717010
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6725783, upper bound: 743.6701795
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -224.0162964, 1200.3963623, -97.1601028, 509.5699768, -733.5863037, 1297.5563965
1: -369.1511841, 1424.5958252, -158.6625977, 604.1230469, -973.2742310, 1583.2581787
2: -260.0831909, 1473.4626465, -112.2457581, 627.2377319, -887.3209229, 1585.7083740
3: -637.9225464, 1248.6026611, -273.6815491, 528.0825195, -1166.0048828, 1522.2841797
4: -428.8231506, 1262.5526123, -184.6594086, 535.8335571, -964.6566772, 1447.2117920

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6674387, upper bound: 743.6741596
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6696415, upper bound: 743.6757821
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6680663, upper bound: 743.6736557
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6696877, upper bound: 743.6757523
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6695334, upper bound: 743.6757821
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -237.5209808, 1270.9678955, -98.4456558, 516.2005615, -753.7214966, 1369.4134521
1: -391.3341064, 1509.2241211, -160.7731476, 612.1020508, -1003.4361572, 1669.9973145
2: -275.6153259, 1559.7001953, -113.7264023, 635.3582153, -910.9735107, 1673.4266357
3: -675.8341064, 1324.2457275, -277.3016357, 535.2220459, -1211.0560303, 1601.5473633
4: -454.5150757, 1337.6806641, -187.1013641, 542.9273071, -997.4423828, 1524.7818604

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6740082, upper bound: 743.6755479
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744677, upper bound: 743.6776479
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6726653, upper bound: 743.6754540
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6677521, upper bound: 743.6624217
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -231.4593048, 1237.7504883, -98.4456558, 516.2005615, -747.6598511, 1336.1960449
1: -381.3911743, 1469.6024170, -160.7731476, 612.1020508, -993.4932251, 1630.3756104
2: -268.5526123, 1519.3814697, -113.7264023, 635.3582153, -903.9108276, 1633.1079102
3: -658.4533081, 1289.0308838, -277.3016357, 535.2220459, -1193.6750488, 1566.3325195
4: -442.6774292, 1302.6938477, -187.1013641, 542.9273071, -985.6047363, 1489.7950439

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6767983, upper bound: 743.6736779
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=860.0533447265625
rel_dist={0: [-743.6895975863799, 743.6895975863799]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6879802, upper bound: 743.6867407
time: 0.63 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6880197, upper bound: 743.6880197
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.44 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -743.6879802, upper bound: 743.6867407
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -743.6880197, upper bound: 743.6880197

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -133.1433258, 692.8184204, -138.5269012, 721.5264282, -854.6695557, 831.3453369
1: -217.3697815, 823.0866089, -226.4326935, 857.1390381, -1074.5087891, 1049.5192871
2: -153.8224335, 852.0822754, -160.1910706, 887.5496826, -1041.3720703, 1012.2733154
3: -374.4883728, 722.8275757, -390.1859741, 752.6910400, -1127.1794434, 1113.0135498
4: -253.3396606, 731.3729858, -263.8327942, 761.5472412, -1014.8869019, 995.2058105

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6788359, upper bound: 743.6760473
time: 0.96 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -136.5654449, 712.0169678, -137.6375275, 716.8999023, -853.4653320, 849.6543579
1: -223.1357117, 845.3634644, -224.9833374, 851.6223145, -1074.7578125, 1070.3468018
2: -157.8641815, 876.0595093, -159.1597748, 881.9213867, -1039.7855225, 1035.2192383
3: -384.4993286, 741.7769165, -387.6859436, 747.8033447, -1132.3026123, 1129.4627686
4: -259.9345093, 750.8963623, -262.1313477, 756.6465454, -1016.5810547, 1013.0277100

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780608, upper bound: 743.6760777
time: 0.73 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701847, upper bound: 743.6701847
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.35 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -743.6788359, upper bound: 743.6760473
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 0, lower bound: -743.6780608, upper bound: 743.6760777
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 3.35
Output dim: 0, lower bound: -743.6701847, upper bound: 743.6701847

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -138.5269012, 721.5264282, -844.9579468, 781.3366699
1: -201.7657318, 763.8026123, -226.4326935, 857.1390381, -1058.9047852, 990.2352905
2: -142.5318909, 790.5067139, -160.1910706, 887.5496826, -1030.0815430, 950.6977539
3: -347.4673157, 670.4584961, -390.1859741, 752.6910400, -1100.1580811, 1060.6445312
4: -234.7127075, 678.4046021, -263.8327942, 761.5472412, -996.2599487, 942.2373657

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
time: 0.60 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -134.5297546, 702.8942261, -946.5031738, 1436.9765625
1: -401.4721985, 1546.7423096, -219.9175720, 834.7116089, -1236.1838379, 1766.6597900
2: -282.7309570, 1598.5117188, -155.6542053, 864.5371704, -1147.2680664, 1754.1657715
3: -693.1887207, 1357.5162354, -379.1771545, 732.3038330, -1425.4925537, 1736.6932373
4: -466.2039795, 1371.2233887, -256.3384705, 741.4115601, -1207.6153564, 1627.5617676

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
time: 0.65 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -127.0210266, 662.5469360, -137.6375275, 716.8999023, -843.9207764, 800.1843262
1: -207.7908630, 786.8043823, -224.9833374, 851.6223145, -1059.4129639, 1011.7877197
2: -146.7888184, 815.2058716, -159.1597748, 881.9213867, -1028.7102051, 974.3656616
3: -357.9414368, 690.1265259, -387.6859436, 747.8033447, -1105.7447510, 1077.8125000
4: -241.6536560, 698.6527100, -262.1313477, 756.6465454, -998.3001709, 960.7840576

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701847, upper bound: 743.6701847
time: 0.73 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701847, upper bound: 743.6701847
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.37 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.37
Output dim: 0, lower bound: -743.6739167, upper bound: 743.6710344
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.37
Output dim: 0, lower bound: -743.6701847, upper bound: 743.6701847
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.37
Output dim: 0, lower bound: -743.6701847, upper bound: 743.6701847

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -128.5790863, 670.3025513, -793.7341309, 771.3887939
1: -201.7657318, 763.8026123, -210.4309387, 796.4016113, -998.1673584, 974.2334595
2: -142.5318909, 790.5067139, -148.6245575, 824.4069214, -966.9387817, 939.1312256
3: -347.4673157, 670.4584961, -362.4870300, 699.0101929, -1046.4772949, 1032.9454346
4: -234.7127075, 678.4046021, -244.7495728, 707.2666626, -941.9793701, 923.1541748

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6733383, upper bound: 743.6726552
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6783823, upper bound: 743.6748741
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -250.0693207, 1336.9821777, -1460.4135742, 892.8789673
1: -201.7657318, 763.8026123, -412.1712952, 1587.7349854, -1789.5004883, 1175.9738770
2: -142.5318909, 790.5067139, -290.3203430, 1640.9337158, -1783.4655762, 1080.8270264
3: -347.4673157, 670.4584961, -711.7825317, 1393.4858398, -1740.9527588, 1382.2409668
4: -234.7127075, 678.4046021, -478.7600708, 1407.4163818, -1642.1290283, 1157.1646729

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6778494, upper bound: 743.6740020
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6783823, upper bound: 743.6748741
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -243.5512085, 1302.1607666, -128.5790863, 670.3025513, -913.8537598, 1430.7398682
1: -401.3778076, 1546.4030762, -210.4309387, 796.4016113, -1197.7792969, 1756.8339844
2: -282.6641235, 1598.1584473, -148.6245575, 824.4069214, -1107.0709229, 1746.7829590
3: -693.0258179, 1357.2143555, -362.4870300, 699.0101929, -1392.0360107, 1719.7014160
4: -466.0933838, 1370.9166260, -244.7495728, 707.2666626, -1173.3599854, 1615.6661377

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6643218, upper bound: 743.6619959
time: 0.71 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6620465, upper bound: 743.6593254
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -250.1162415, 1337.2148438, -1580.8238525, 1552.5627441
1: -401.4721985, 1546.7423096, -412.2479858, 1588.0107422, -1989.4829102, 1958.9902344
2: -282.7309570, 1598.5117188, -290.3746338, 1641.2213135, -1923.9520264, 1888.8861084
3: -693.1887207, 1357.5162354, -711.9147949, 1393.7313232, -2086.9194336, 2069.4311523
4: -466.2039795, 1371.2233887, -478.8499146, 1407.6657715, -1873.8697510, 1850.0732422

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6710944, upper bound: 743.6683545
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6620465, upper bound: 743.6593254
time: 1.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.61 seconds
IS_A1_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -743.6733383, upper bound: 743.6726552
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -743.6783823, upper bound: 743.6748741
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -743.6778494, upper bound: 743.6740020
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.61
Output dim: 0, lower bound: -743.6783823, upper bound: 743.6748741
IS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -743.6643218, upper bound: 743.6619959
IS_A1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -743.6620465, upper bound: 743.6593254
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -743.6710944, upper bound: 743.6683545
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 3.61
Output dim: 0, lower bound: -743.6620465, upper bound: 743.6593254

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -127.7160873, 665.8647461, -783.2513428, 739.7620850
1: -191.7628937, 727.0043945, -209.0059662, 791.1096191, -982.8724976, 936.0103760
2: -135.5527344, 752.8033447, -147.6264648, 818.9638062, -954.5165405, 900.4298096
3: -330.2159424, 637.6373901, -360.0421448, 694.3253174, -1024.5412598, 997.6794434
4: -223.1825562, 645.3632812, -243.1080017, 702.5476685, -925.7302246, 888.4712524

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6854488, upper bound: 743.6830724
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6831143, upper bound: 743.6825405
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -244.4051208, 1307.8698730, -1431.3012695, 887.2148438
1: -201.7657318, 763.8026123, -402.7369080, 1553.0355225, -1754.8011475, 1166.5395508
2: -142.5318909, 790.5067139, -283.6946106, 1605.0146484, -1747.5465088, 1074.2011719
3: -347.4673157, 670.4584961, -695.6462402, 1362.6473389, -1710.1143799, 1366.1046143
4: -234.7127075, 678.4046021, -467.8744202, 1376.3135986, -1611.0262451, 1146.2790527

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6732144, upper bound: 743.6717801
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6732144, upper bound: 743.6740020
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -122.5160065, 638.0969849, -245.5036774, 1313.0690918, -1435.5850830, 883.6006470
1: -200.2581024, 758.1837769, -404.5528259, 1559.2205811, -1759.4786377, 1162.7364502
2: -141.4712524, 784.7168579, -285.0150757, 1611.5306396, -1753.0018311, 1069.7315674
3: -344.8741455, 665.4780273, -698.6285400, 1368.2016602, -1713.0758057, 1364.1065674
4: -232.9658356, 673.3876343, -469.9944458, 1381.9331055, -1614.8986816, 1143.3818359

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6750319, upper bound: 743.6716083
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6540071, upper bound: 743.6502442
time: 0.57 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.47 seconds
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -743.6854488, upper bound: 743.6830724
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -743.6831143, upper bound: 743.6825405
IS_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 0, lower bound: -743.6732144, upper bound: 743.6717801
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -743.6732144, upper bound: 743.6740020
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 0, lower bound: -743.6750319, upper bound: 743.6716083
IS_A1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 0, lower bound: -743.6540071, upper bound: 743.6502442

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -116.9194565, 609.5358276, -122.2141418, 636.3038330, -753.2232666, 731.7498779
1: -190.9940338, 724.0135498, -199.9622803, 755.8794556, -946.8733521, 923.9758301
2: -135.0098419, 749.7550049, -141.2275543, 783.0245972, -918.0343628, 890.9825439
3: -328.8941956, 634.9904785, -344.4512024, 663.1257324, -992.0198975, 979.4416504
4: -222.2835541, 642.7123413, -232.5072021, 671.2997437, -893.5833130, 875.2195435

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830736, upper bound: 743.6824591
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830736, upper bound: 743.6824591
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.8782196, 603.9889526, -127.3483429, 664.1273804, -780.0055542, 731.3372192
1: -189.2604828, 717.4243774, -208.2507477, 788.8132324, -978.0736694, 925.6750488
2: -133.7859802, 742.8610840, -147.1669312, 816.9288940, -950.7148438, 890.0280151
3: -325.8961182, 629.1713257, -359.0077820, 691.8966675, -1017.7927856, 988.1790771
4: -220.2556000, 636.8395996, -242.3616180, 700.3497925, -920.6054077, 879.2011108

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794901, upper bound: 743.6787760
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6644553, upper bound: 743.6631381
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -244.3662567, 1307.6770020, -1425.0634766, 856.4122314
1: -191.7628937, 727.0043945, -402.6732483, 1552.8071289, -1744.5698242, 1129.6776123
2: -135.5527344, 752.8033447, -283.6496887, 1604.7764893, -1740.3291016, 1036.4530029
3: -330.2159424, 637.6373901, -695.5365601, 1362.4437256, -1692.6596680, 1333.1739502
4: -223.1825562, 645.3632812, -467.7999268, 1376.1068115, -1599.2891846, 1113.1632080

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6486419, upper bound: 743.6486261
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6446610, upper bound: 743.6441921
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6727328, upper bound: 743.6707127
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6665129, upper bound: 743.6635336
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -122.1696243, 636.3114624, -247.4592590, 1322.9256592, -1445.0953369, 883.7707520
1: -199.6856537, 756.0559692, -407.6053467, 1570.9877930, -1770.6733398, 1163.6612549
2: -141.0696259, 782.5231323, -287.2120056, 1623.5986328, -1764.6680908, 1069.7347412
3: -343.8877258, 663.5952759, -703.8779907, 1378.6633301, -1722.5510254, 1367.4732666
4: -232.3043365, 671.4921265, -473.5860901, 1392.4251709, -1624.7294922, 1145.0782471

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6549355, upper bound: 743.6505271
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744593, upper bound: 743.6708241
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6733669, upper bound: 743.6698571
time: 0.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.19 seconds
IS_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 0, lower bound: -743.6830736, upper bound: 743.6824591
IS_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 0, lower bound: -743.6830736, upper bound: 743.6824591
IS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 0, lower bound: -743.6794901, upper bound: 743.6787760
IS_A1_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -743.6644553, upper bound: 743.6631381
IS_A1_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -743.6727328, upper bound: 743.6707127
IS_A1_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -743.6665129, upper bound: 743.6635336
IS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.19
Output dim: 0, lower bound: -743.6744593, upper bound: 743.6708241
IS_A1_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.19
Output dim: 0, lower bound: -743.6733669, upper bound: 743.6698571

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -111.9797897, 583.0902100, -122.2141418, 636.3038330, -748.2836304, 705.3042603
1: -182.8664093, 692.4880371, -199.9622803, 755.8794556, -938.7457275, 892.4503174
2: -129.2702026, 717.6174927, -141.2275543, 783.0245972, -912.2947998, 858.8450317
3: -314.9225159, 607.0635376, -344.4512024, 663.1257324, -978.0482178, 951.5146484
4: -212.7743530, 614.7598267, -232.5072021, 671.2997437, -884.0740356, 847.2670288

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6832437, upper bound: 743.6800052
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6833794, upper bound: 743.6809098
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -116.7978821, 609.1127319, -122.2141418, 636.3038330, -753.1016846, 731.3267822
1: -190.6253662, 723.2798462, -199.9622803, 755.8794556, -946.5048218, 923.2421265
2: -134.8507233, 749.3662109, -141.2275543, 783.0245972, -917.8753052, 890.5937500
3: -328.5602722, 633.9815674, -344.4512024, 663.1257324, -991.6860352, 978.4327393
4: -222.0314026, 641.9342041, -232.5072021, 671.2997437, -893.3311768, 874.4414062

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6809456, upper bound: 743.6788956
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6832437, upper bound: 743.6800052
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6833794, upper bound: 743.6809098
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -115.5890045, 602.4942017, -126.8733063, 662.2459717, -777.8349609, 729.3674316
1: -188.7858887, 715.6423950, -207.2536774, 786.5241089, -975.3099976, 922.8960571
2: -133.4513092, 741.0276489, -146.5355530, 814.4353638, -947.8866577, 887.5631714
3: -325.0758972, 627.5936890, -357.2309875, 689.6895752, -1014.7655029, 984.8247070
4: -219.7032928, 635.2545166, -241.3061981, 698.0534668, -917.7567749, 876.5606689

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6760702, upper bound: 743.6736597
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793169, upper bound: 743.6787760
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6793169, upper bound: 743.6787759
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -111.9378586, 582.3029785, -246.6463470, 1318.6754150, -1430.6132812, 828.9493408
1: -182.9193115, 691.7217407, -406.2972412, 1565.9196777, -1748.8389893, 1098.0190430
2: -129.1608582, 716.5346069, -286.2727356, 1618.3917236, -1747.5526123, 1002.8071899
3: -315.0741272, 606.8560181, -701.6083374, 1374.1655273, -1689.2396240, 1308.4643555
4: -212.6065521, 614.4605103, -472.0285950, 1387.9188232, -1600.5253906, 1086.4890137

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6719332, upper bound: 743.6706436
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6719332, upper bound: 743.6708241
time: 0.69 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.57 seconds
IS_A1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.57
Output dim: 0, lower bound: -743.6832437, upper bound: 743.6800052
IS_A1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.57
Output dim: 0, lower bound: -743.6833794, upper bound: 743.6809098
IS_A1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.57
Output dim: 0, lower bound: -743.6832437, upper bound: 743.6800052
IS_A1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.57
Output dim: 0, lower bound: -743.6833794, upper bound: 743.6809098
IS_A1_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.57
Output dim: 0, lower bound: -743.6793169, upper bound: 743.6787760
IS_A1_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.57
Output dim: 0, lower bound: -743.6793169, upper bound: 743.6787759
IS_A1_A1_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.57
Output dim: 0, lower bound: -743.6719332, upper bound: 743.6706436
IS_A1_A1_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.57
Output dim: 0, lower bound: -743.6719332, upper bound: 743.6708241

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -111.7422180, 581.8432007, -114.7383881, 597.0875854, -708.8298340, 696.5816040
1: -182.4721832, 690.9969482, -187.5865936, 708.8566284, -891.3287354, 878.5835571
2: -128.9919281, 716.0993042, -132.4647369, 735.1064453, -864.0982666, 848.5640259
3: -314.2457886, 605.7302246, -323.1352844, 621.0758057, -935.3215942, 928.8654785
4: -212.3128357, 613.4350586, -217.9689178, 629.4948120, -841.8076172, 831.4039917

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6758543, upper bound: 743.6747563
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6758543, upper bound: 743.6823588
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -110.8494797, 577.1076050, -119.6630402, 624.1439209, -734.9932861, 696.7706299
1: -181.0065002, 685.3728638, -195.6524811, 741.3687134, -922.3752441, 881.0253296
2: -127.9477539, 710.2473145, -138.2488556, 767.8076172, -895.7553711, 848.4961548
3: -311.6886292, 600.7761841, -336.9811707, 649.9666748, -961.6551514, 937.7573242
4: -210.5751190, 608.4346313, -227.5605774, 657.8191528, -868.3942871, 835.9950562

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6604017, upper bound: 743.6577854
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6604017, upper bound: 743.6823833
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -116.5648956, 607.8717651, -114.7383881, 597.0875854, -713.6524658, 722.6101685
1: -190.2372589, 721.7974854, -187.5865936, 708.8566284, -899.0938721, 909.3840942
2: -134.5773926, 747.8570557, -132.4647369, 735.1064453, -869.6838379, 880.3217773
3: -327.8936768, 632.6607666, -323.1352844, 621.0758057, -948.9694824, 955.7960205
4: -221.5783539, 640.6193848, -217.9689178, 629.4948120, -851.0731812, 858.5883179

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6747807, upper bound: 743.6706225
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6545169, upper bound: 743.6510336
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6545169, upper bound: 743.6800052
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.5949173, 602.8661499, -119.6630402, 624.1439209, -739.7387695, 722.5291748
1: -188.6611786, 715.8370972, -195.6524811, 741.3687134, -930.0299072, 911.4895630
2: -133.4519806, 741.6561279, -138.2488556, 767.8076172, -901.2595825, 879.9049683
3: -325.1480103, 627.3775635, -336.9811707, 649.9666748, -975.1146851, 964.3587646
4: -219.7033386, 635.3098145, -227.5605774, 657.8191528, -877.5224609, 862.8703003

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782154, upper bound: 743.6759762
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6533333, upper bound: 743.6502358
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6533333, upper bound: 743.6502358
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -111.6929245, 581.6089478, -126.8733063, 662.2459717, -773.9388428, 708.4822388
1: -182.3955383, 690.7224121, -207.2536774, 786.5241089, -968.9196167, 897.9760132
2: -128.9382629, 715.8007202, -146.5355530, 814.4353638, -943.3736572, 862.3362427
3: -314.1089783, 605.5001831, -357.2309875, 689.6895752, -1003.7985229, 962.7312012
4: -212.2267761, 613.1884766, -241.3061981, 698.0534668, -910.2802734, 854.4946289

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6732172, upper bound: 743.6727635
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786323, upper bound: 743.6774635
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775461, upper bound: 743.6774711
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -116.4913559, 607.5222778, -126.8733063, 662.2459717, -778.7372437, 734.3955078
1: -190.1228333, 721.3847046, -207.2536774, 786.5241089, -976.6469116, 928.6383057
2: -134.4960480, 747.4137573, -146.5355530, 814.4353638, -948.9313965, 893.9492798
3: -327.6919250, 632.3068848, -357.2309875, 689.6895752, -1017.3814697, 989.5378418
4: -221.4464569, 640.2501221, -241.3061981, 698.0534668, -919.4999390, 881.5563354

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6732172, upper bound: 743.6736597
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6786323, upper bound: 743.6774635
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775461, upper bound: 743.6774510
time: 0.66 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.76 seconds
IS_A1_A1_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6758543, upper bound: 743.6747563
IS_A1_A1_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6758543, upper bound: 743.6823588
IS_A1_A1_B1_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6604017, upper bound: 743.6577854
IS_A1_A1_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6604017, upper bound: 743.6823833
IS_A1_A1_B1_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6545169, upper bound: 743.6510336
IS_A1_A1_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6545169, upper bound: 743.6800052
IS_A1_A1_B1_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6533333, upper bound: 743.6502358
IS_A1_A1_B1_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6533333, upper bound: 743.6502358
IS_A1_A1_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6786323, upper bound: 743.6774635
IS_A1_A1_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6775461, upper bound: 743.6774711
IS_A1_A1_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6786323, upper bound: 743.6774635
IS_A1_A1_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 6.76
Output dim: 0, lower bound: -743.6775461, upper bound: 743.6774510

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -111.7422180, 581.8432007, -111.1765747, 579.0368042, -690.7789307, 693.0197754
1: -182.4721832, 690.9969482, -181.6289368, 687.3578491, -869.8299561, 872.6258545
2: -128.9919281, 716.0993042, -128.2874298, 712.7778931, -841.7697754, 844.3867188
3: -314.2457886, 605.7302246, -312.9619446, 602.0298462, -916.2756348, 918.6921387
4: -212.3128357, 613.4350586, -211.1488190, 610.1497192, -822.4625244, 824.5838623

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6716738, upper bound: 743.6709667
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6696963, upper bound: 743.6687445
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -111.7422180, 581.8432007, -109.5474243, 570.9955444, -682.7377930, 691.3906250
1: -182.4721832, 690.9969482, -178.9551697, 677.4798584, -859.9520264, 869.9520264
2: -128.9919281, 716.0993042, -126.4774704, 703.0489502, -832.0407715, 842.5767822
3: -314.2457886, 605.7302246, -308.3374634, 592.9629517, -907.2086792, 914.0676880
4: -212.3128357, 613.4350586, -208.0803070, 601.2585449, -813.5713501, 821.5152588

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6729161, upper bound: 743.6725824
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6696963, upper bound: 743.6689128
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -110.8494797, 577.1076050, -112.8554459, 589.6697388, -700.5191650, 689.9630127
1: -181.0065002, 685.3728638, -184.3795624, 700.0742188, -881.0806885, 869.7524414
2: -127.9477539, 710.2473145, -130.3926392, 725.5932617, -853.5410156, 840.6399536
3: -311.6886292, 600.7761841, -317.5811768, 613.1365356, -924.8250122, 918.3573608
4: -210.5751190, 608.4346313, -214.5669556, 620.8038940, -831.3790283, 823.0015259

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6599898, upper bound: 743.6711917
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6582859, upper bound: 743.6676861
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -116.5648956, 607.8717651, -109.5474243, 570.9955444, -687.5604248, 717.4191895
1: -190.2372589, 721.7974854, -178.9551697, 677.4798584, -867.7171021, 900.7525635
2: -134.5773926, 747.8570557, -126.4774704, 703.0489502, -837.6263428, 874.3345337
3: -327.8936768, 632.6607666, -308.3374634, 592.9629517, -920.8566284, 940.9982300
4: -221.5783539, 640.6193848, -208.0803070, 601.2585449, -822.8369141, 848.6995850

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6440005, upper bound: 743.6706225
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6537389, upper bound: 743.6631679
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6534103, upper bound: 743.6609967
time: 0.94 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -100.7541275, 524.5238037, -126.0135040, 657.7810669, -758.5352173, 650.5372925
1: -164.4388123, 622.4212036, -205.8652191, 781.1970215, -945.6358643, 828.2863770
2: -116.2068253, 645.7384033, -145.5436859, 808.9602051, -925.1670532, 791.2819824
3: -283.3091736, 545.1054077, -354.8246765, 684.9608154, -968.2700195, 899.9300537
4: -191.1775208, 552.4777222, -239.6618805, 693.3170776, -884.4946289, 792.1395264

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746212, upper bound: 743.6779830
time: 1.05 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6757796, upper bound: 743.6771902
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -106.7628860, 555.9508667, -126.5048828, 660.3396606, -767.1024780, 682.4556274
1: -174.3182526, 660.0167236, -206.6531372, 784.2482910, -958.5665283, 866.6698608
2: -123.2321930, 684.3818359, -146.1094971, 812.0964966, -935.3286133, 830.4913330
3: -300.1461182, 578.2810669, -356.1883850, 687.6671143, -987.8132324, 934.4694214
4: -202.7708740, 585.8930664, -240.6005859, 696.0236206, -898.7944336, 826.4936523

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6725123, upper bound: 743.6725350
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771449, upper bound: 743.6796784
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6771449, upper bound: 743.6796784
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -105.5689926, 550.2665405, -126.0135040, 657.7810669, -763.3500366, 676.2800293
1: -172.1500397, 652.9240112, -205.8652191, 781.1970215, -953.3470459, 858.7891846
2: -121.7772675, 677.3750610, -145.5436859, 808.9602051, -930.7374268, 822.9187622
3: -296.8818359, 571.9130249, -354.8246765, 684.9608154, -981.8425903, 926.7376709
4: -200.4350433, 579.5048828, -239.6618805, 693.3170776, -893.7521362, 819.1667480

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746964, upper bound: 743.6733186
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782845, upper bound: 743.6769078
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774495
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774495
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -111.7286911, 582.6046753, -126.5048828, 660.3396606, -772.0681763, 709.1095581
1: -182.3279266, 691.6400757, -206.6531372, 784.2482910, -966.5762329, 898.2931519
2: -128.9882355, 716.9859619, -146.1094971, 812.0964966, -941.0847168, 863.0954590
3: -314.2033386, 605.9669189, -356.1883850, 687.6671143, -1001.8704224, 962.1552734
4: -212.3221893, 613.8260498, -240.6005859, 696.0236206, -908.3457642, 854.4266357

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751735, upper bound: 743.6733114
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
time: 0.72 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 4.78 seconds
IS_A1_A1_B1_A2_B1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6716738, upper bound: 743.6709667
IS_A1_A1_B1_A2_B1_A1_B1_B1_B2, status: Status.VERIFIED, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6696963, upper bound: 743.6687445
IS_A1_A1_B1_A2_B1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6729161, upper bound: 743.6725824
IS_A1_A1_B1_A2_B1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6696963, upper bound: 743.6689128
IS_A1_A1_B1_A2_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6599898, upper bound: 743.6711917
IS_A1_A1_B1_A2_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6582859, upper bound: 743.6676861
IS_A1_A1_B1_A2_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6537389, upper bound: 743.6631679
IS_A1_A1_B1_A2_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6534103, upper bound: 743.6609967
IS_A1_A1_B1_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6746212, upper bound: 743.6779830
IS_A1_A1_B1_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6757796, upper bound: 743.6771902
IS_A1_A1_B1_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6771449, upper bound: 743.6796784
IS_A1_A1_B1_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6771449, upper bound: 743.6796784
IS_A1_A1_B1_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774495
IS_A1_A1_B1_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774495
IS_A1_A1_B1_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
IS_A1_A1_B1_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.78
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -93.8905640, 488.2969360, -125.7824249, 656.5498047, -750.4403687, 614.0793457
1: -153.1530609, 579.1849365, -205.4827118, 779.7271118, -932.8800659, 784.6676025
2: -108.1851501, 601.4987793, -145.2726288, 807.4539185, -915.6390381, 746.7713623
3: -263.7633057, 506.3635254, -354.1629639, 683.6505737, -947.4138794, 860.5264893
4: -177.8286438, 514.1507568, -239.2126923, 692.0113525, -869.8399658, 753.3634033

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744076, upper bound: 743.6779131
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744076, upper bound: 743.6779830
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -96.0861206, 501.3072205, -124.7688751, 651.2944336, -747.3804932, 626.0759888
1: -156.7012939, 594.6295166, -203.8444214, 773.4731445, -930.1744385, 798.4739380
2: -110.7827835, 617.0957642, -144.0966797, 800.9873657, -911.7701416, 761.1924438
3: -269.8397217, 520.2304688, -351.2932129, 678.1173096, -947.9570312, 871.5236206
4: -182.1300659, 527.3391113, -237.2519684, 686.4431152, -868.5731812, 764.5910645

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6752406, upper bound: 743.6742701
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755390, upper bound: 743.6771356
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755390, upper bound: 743.6771902
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -106.7628860, 555.9508667, -116.7602844, 608.9218140, -715.6846313, 672.7111206
1: -174.3182526, 660.0167236, -190.7099609, 722.9573364, -897.2755737, 850.7266235
2: -123.2321930, 684.3818359, -134.7745056, 749.3273926, -872.5595093, 819.1563721
3: -300.1461182, 578.2810669, -328.8068848, 633.6053467, -933.7514648, 907.0878906
4: -202.7708740, 585.8930664, -221.8480835, 641.7368774, -844.5076904, 807.7411499

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6731056, upper bound: 743.6769912
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742419, upper bound: 743.6769537
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -106.7628860, 555.9508667, -121.6166229, 635.0413208, -741.8041382, 677.5674438
1: -174.3182526, 660.0167236, -198.6941071, 754.0400391, -928.3582764, 858.7108154
2: -123.2321930, 684.3818359, -140.4603729, 781.1187744, -904.3508911, 824.8422241
3: -300.1461182, 578.2810669, -342.3758545, 660.8359985, -960.9821167, 920.6567993
4: -202.7708740, 585.8930664, -231.2395172, 669.1063232, -871.8770752, 817.1325073

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6731056, upper bound: 743.6769912
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742419, upper bound: 743.6769537
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -105.5689926, 550.2665405, -116.7602844, 608.9218140, -714.4907837, 667.0268555
1: -172.1500397, 652.9240112, -190.7099609, 722.9573364, -895.1073608, 843.6339111
2: -121.7772675, 677.3750610, -134.7745056, 749.3273926, -871.1046143, 812.1495361
3: -296.8818359, 571.9130249, -328.8068848, 633.6053467, -930.4871216, 900.7199097
4: -200.4350433, 579.5048828, -221.8480835, 641.7368774, -842.1719360, 801.3529663

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780357, upper bound: 743.6767052
time: 0.82 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6774635
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6774635
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -105.5689926, 550.2665405, -121.6166229, 635.0413208, -740.6102905, 671.8831787
1: -172.1500397, 652.9240112, -198.6941071, 754.0400391, -926.1900635, 851.6181030
2: -121.7772675, 677.3750610, -140.4603729, 781.1187744, -902.8959961, 817.8354492
3: -296.8818359, 571.9130249, -342.3758545, 660.8359985, -957.7177124, 914.2888794
4: -200.4350433, 579.5048828, -231.2395172, 669.1063232, -869.5413818, 810.7443237

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780357, upper bound: 743.6769078
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6774635
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6774635
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -111.7286911, 582.6046753, -116.7602844, 608.9218140, -720.6503906, 699.3649902
1: -182.3279266, 691.6400757, -190.7099609, 722.9573364, -905.2852783, 882.3499146
2: -128.9882355, 716.9859619, -134.7745056, 749.3273926, -878.3156128, 851.7604980
3: -314.2033386, 605.9669189, -328.8068848, 633.6053467, -947.8087158, 934.7738037
4: -212.3221893, 613.8260498, -221.8480835, 641.7368774, -854.0590210, 835.6741333

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742905, upper bound: 743.6718656
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6768284, upper bound: 743.6759190
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -111.7286911, 582.6046753, -121.6166229, 635.0413208, -746.7698975, 704.2213135
1: -182.3279266, 691.6400757, -198.6941071, 754.0400391, -936.3679810, 890.3341675
2: -128.9882355, 716.9859619, -140.4603729, 781.1187744, -910.1069946, 857.4463501
3: -314.2033386, 605.9669189, -342.3758545, 660.8359985, -975.0393066, 948.3427124
4: -212.3221893, 613.8260498, -231.2395172, 669.1063232, -881.4284058, 845.0655518

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742905, upper bound: 743.6729129
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6768284, upper bound: 743.6759839
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
time: 0.68 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 10.30 seconds
IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6744076, upper bound: 743.6779131
IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6744076, upper bound: 743.6779830
IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6755390, upper bound: 743.6771356
IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6755390, upper bound: 743.6771902
IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6731056, upper bound: 743.6769912
IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6742419, upper bound: 743.6769537
IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6731056, upper bound: 743.6769912
IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6742419, upper bound: 743.6769537
IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6774635
IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6774635
IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6774635
IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6774635
IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510
IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 10.30
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6774510

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -93.8905640, 488.2969360, -116.5339355, 607.7105103, -701.6010742, 604.8308105
1: -153.1530609, 579.1849365, -190.3352051, 721.5106201, -874.6636353, 769.5201416
2: -108.1851501, 601.4987793, -134.5086365, 747.8549805, -856.0401611, 736.0074463
3: -263.7633057, 506.3635254, -328.1582336, 632.3173218, -896.0806274, 834.5217285
4: -177.8286438, 514.1507568, -221.4071655, 640.4547119, -818.2832642, 735.5579224

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744076, upper bound: 743.6779131
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744076, upper bound: 743.6779131
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -93.8905640, 488.2969360, -121.3820343, 633.7922363, -727.6828003, 609.6789551
1: -153.1530609, 579.1849365, -198.3058777, 752.5486450, -905.7015991, 777.4908447
2: -108.1851501, 601.4987793, -140.1851501, 779.5996704, -887.7847900, 741.6838379
3: -263.7633057, 506.3635254, -341.7041626, 659.5065308, -923.2698364, 848.0676880
4: -177.8286438, 514.1507568, -230.7833405, 667.7815552, -845.6101685, 744.9340820

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744076, upper bound: 743.6779830
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744076, upper bound: 743.6779830
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -96.0861206, 501.3072205, -115.5088425, 602.4033813, -698.4895020, 616.8160400
1: -156.7012939, 594.6295166, -188.6776123, 715.1955566, -871.8968506, 783.3071289
2: -110.7827835, 617.0957642, -133.3171082, 741.2879639, -852.0707397, 750.4128418
3: -269.8397217, 520.2304688, -325.2614746, 626.7319946, -896.5717163, 845.4918823
4: -182.1300659, 527.3391113, -219.4214020, 634.8335571, -816.9636230, 746.7604980

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6749375, upper bound: 743.6739343
time: 1.08 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755390, upper bound: 743.6771356
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755390, upper bound: 743.6771356
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -96.0861206, 501.3072205, -120.3905945, 628.6263428, -724.7124023, 621.6976929
1: -156.7012939, 594.6295166, -196.7025604, 746.4047241, -903.1060181, 791.3320923
2: -110.7827835, 617.0957642, -139.0357971, 773.2148438, -883.9976196, 756.1315918
3: -269.8397217, 520.2304688, -338.8952637, 654.0805664, -923.9202881, 859.1256714
4: -182.1300659, 527.3391113, -228.8686371, 662.3169556, -844.4470215, 756.2077637

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6749375, upper bound: 743.6742701
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755390, upper bound: 743.6771902
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6755390, upper bound: 743.6771902
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -99.7698517, 519.1476440, -116.5339355, 607.7105103, -707.4803467, 635.6815796
1: -162.7973938, 615.8474121, -190.3352051, 721.5106201, -884.3079834, 806.1826172
2: -115.0395203, 639.4166870, -134.5086365, 747.8549805, -862.8945312, 773.9252930
3: -280.2346802, 538.8372192, -328.1582336, 632.3173218, -912.5520020, 866.9954224
4: -189.1817474, 546.7881470, -221.4071655, 640.4547119, -829.6364136, 768.1953125

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6731056, upper bound: 743.6770715
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6731056, upper bound: 743.6770715
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -102.2184448, 533.5438232, -115.5088425, 602.4033813, -704.6218262, 649.0526733
1: -166.7402649, 633.2833862, -188.6776123, 715.1955566, -881.9357910, 821.9609985
2: -117.9496689, 656.6825562, -133.3171082, 741.2879639, -859.2376099, 789.9996338
3: -287.0317383, 554.3771362, -325.2614746, 626.7319946, -913.7637329, 879.6384888
4: -194.0307770, 561.6346436, -219.4214020, 634.8335571, -828.8643188, 781.0560303

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742419, upper bound: 743.6770370
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742419, upper bound: 743.6770370
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -99.7698517, 519.1476440, -121.3820343, 633.7922363, -733.5620728, 640.5296631
1: -162.7973938, 615.8474121, -198.3058777, 752.5486450, -915.3460693, 814.1533203
2: -115.0395203, 639.4166870, -140.1851501, 779.5996704, -894.6391602, 779.6017456
3: -280.2346802, 538.8372192, -341.7041626, 659.5065308, -939.7412109, 880.5413208
4: -189.1817474, 546.7881470, -230.7833405, 667.7815552, -856.9632568, 777.5713501

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6466315, upper bound: 743.6501079
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6733797, upper bound: 743.6769814
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6733797, upper bound: 743.6769912
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -102.2184448, 533.5438232, -120.3905945, 628.6263428, -730.8447266, 653.9342651
1: -166.7402649, 633.2833862, -196.7025604, 746.4047241, -913.1448975, 829.9859619
2: -117.9496689, 656.6825562, -139.0357971, 773.2148438, -891.1644897, 795.7183228
3: -287.0317383, 554.3771362, -338.8952637, 654.0805664, -941.1123047, 893.2722778
4: -194.0307770, 561.6346436, -228.8686371, 662.3169556, -856.3477173, 790.5032959

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6669032, upper bound: 743.6651149
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744614, upper bound: 743.6769537
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6744614, upper bound: 743.6769537
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -105.5689926, 550.2665405, -111.9499435, 583.2946167, -688.8635864, 662.2164307
1: -172.1500397, 652.9240112, -182.7245789, 692.5597534, -864.7097778, 835.6485596
2: -121.7772675, 677.3750610, -129.1518707, 717.6832275, -839.4604492, 806.5268555
3: -296.8818359, 571.9130249, -314.7863159, 607.0264282, -903.9082642, 886.6993408
4: -200.4350433, 579.5048828, -212.4824982, 614.8881836, -815.3232422, 791.9873047

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780357, upper bound: 743.6767052
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6781040, upper bound: 743.6775319
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6776096
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6776096
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -105.5689926, 550.2665405, -113.8316727, 594.3364868, -699.9054565, 664.0982056
1: -172.1500397, 652.9240112, -185.7338104, 705.1296997, -877.2797241, 838.6578369
2: -121.7772675, 677.3750610, -131.3334198, 731.4205322, -853.1978149, 808.7084961
3: -296.8818359, 571.9130249, -320.3638916, 617.1741943, -914.0559692, 892.2769165
4: -200.4350433, 579.5048828, -216.1201172, 625.4914551, -825.9265137, 795.6248779

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780357, upper bound: 743.6767052
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6781040, upper bound: 743.6775319
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6776096
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785700, upper bound: 743.6776096
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -105.5689926, 550.2665405, -116.6364746, 608.4291992, -713.9981689, 666.9030151
1: -172.1500397, 652.9240112, -190.4151611, 722.4934692, -894.6434937, 843.3391724
2: -121.7772675, 677.3750610, -134.6294098, 748.3213501, -870.0985718, 812.0044556
3: -296.8818359, 571.9130249, -327.8545532, 633.2802734, -930.1619873, 899.7675781
4: -200.4350433, 579.5048828, -221.5791473, 641.2385254, -841.6735840, 801.0840454

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746964, upper bound: 743.6733186
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782845, upper bound: 743.6769078
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6777684, upper bound: 743.6770209
time: 1.06 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6756402, upper bound: 743.6757499
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -105.5689926, 550.2665405, -119.8103256, 626.1521606, -731.7211304, 670.0768433
1: -172.1500397, 652.9240112, -195.5274048, 743.1024780, -915.2525024, 848.4514160
2: -121.7772675, 677.3750610, -138.3423920, 770.5074463, -892.2846680, 815.7174072
3: -296.8818359, 571.9130249, -337.0954285, 650.7700806, -947.6518555, 909.0084229
4: -200.4350433, 579.5048828, -227.7197723, 659.2333984, -859.6684570, 807.2246094

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746964, upper bound: 743.6733186
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782845, upper bound: 743.6769078
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6777684, upper bound: 743.6770209
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6756402, upper bound: 743.6757499
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -111.7286911, 582.6046753, -111.9499435, 583.2946167, -695.0231934, 694.5546265
1: -182.3279266, 691.6400757, -182.7245789, 692.5597534, -874.8876953, 874.3646240
2: -128.9882355, 716.9859619, -129.1518707, 717.6832275, -846.6714478, 846.1377563
3: -314.2033386, 605.9669189, -314.7863159, 607.0264282, -921.2297363, 920.7531128
4: -212.3221893, 613.8260498, -212.4824982, 614.8881836, -827.2102661, 826.3085327

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742905, upper bound: 743.6718656
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6768284, upper bound: 743.6759190
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6768416, upper bound: 743.6767234
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6772513, upper bound: 743.6776397
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6776397
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6776397
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -111.7286911, 582.6046753, -113.8316727, 594.3364868, -706.0651245, 696.4363403
1: -182.3279266, 691.6400757, -185.7338104, 705.1296997, -887.4576416, 877.3737793
2: -128.9882355, 716.9859619, -131.3334198, 731.4205322, -860.4087524, 848.3193970
3: -314.2033386, 605.9669189, -320.3638916, 617.1741943, -931.3775635, 926.3308105
4: -212.3221893, 613.8260498, -216.1201172, 625.4914551, -837.8135376, 829.9461060

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742905, upper bound: 743.6718656
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6768284, upper bound: 743.6759190
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6768416, upper bound: 743.6767234
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6772513, upper bound: 743.6776397
time: 0.93 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6776397
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776873, upper bound: 743.6776397
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -111.7286911, 582.6046753, -116.6364746, 608.4291992, -720.1578369, 699.2411499
1: -182.3279266, 691.6400757, -190.4151611, 722.4934692, -904.8214111, 882.0552368
2: -128.9882355, 716.9859619, -134.6294098, 748.3213501, -877.3095703, 851.6153564
3: -314.2033386, 605.9669189, -327.8545532, 633.2802734, -947.4836426, 933.8214722
4: -212.3221893, 613.8260498, -221.5791473, 641.2385254, -853.5606079, 835.4052124

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751735, upper bound: 743.6729129
time: 1.05 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770673, upper bound: 743.6759839
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=860.0533447265625
rel_dist={0: [-743.6893068054288, 743.6893068054287]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6865737, upper bound: 743.6872439
time: 0.69 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6876739, upper bound: 743.6876739
time: 0.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.78 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -743.6865737, upper bound: 743.6872439
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.78
Output dim: 0, lower bound: -743.6876739, upper bound: 743.6876739

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -138.5269012, 721.5264282, -133.1433258, 692.8184204, -831.3453369, 854.6695557
1: -226.4326935, 857.1390381, -217.3697815, 823.0866089, -1049.5192871, 1074.5087891
2: -160.1910706, 887.5496826, -153.8224335, 852.0822754, -1012.2733154, 1041.3720703
3: -390.1859741, 752.6910400, -374.4883728, 722.8275757, -1113.0135498, 1127.1794434
4: -263.8327942, 761.5472412, -253.3396606, 731.3729858, -995.2058105, 1014.8869019

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6737998, upper bound: 743.6755819
time: 0.66 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6703606, upper bound: 743.6716841
time: 0.80 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -135.8346252, 707.5307617, -136.5654449, 712.0169678, -847.8515625, 844.0961914
1: -222.0466461, 840.4483032, -223.1357117, 845.3634644, -1067.4100342, 1063.5837402
2: -157.0680695, 870.5262451, -157.8641815, 876.0595093, -1033.1275635, 1028.3903809
3: -382.6246033, 737.8988037, -384.4993286, 741.7769165, -1124.4014893, 1122.3978271
4: -258.6805115, 746.7246094, -259.9345093, 750.8963623, -1009.5769043, 1006.6591187

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6736744, upper bound: 743.6747338
time: 0.86 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6698726, upper bound: 743.6698726
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.42 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -743.6737998, upper bound: 743.6755819
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 3.42
Output dim: 0, lower bound: -743.6703606, upper bound: 743.6716841
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 3.42
Output dim: 0, lower bound: -743.6736744, upper bound: 743.6747338
IS_B2_B2, status: Status.VERIFIED, split count: 2, time: 3.42
Output dim: 0, lower bound: -743.6698726, upper bound: 743.6698726

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -135.3650513, 705.3255615, -123.4315948, 642.8097534, -778.1747437, 828.7570801
1: -221.3665009, 837.9066772, -201.7657318, 763.8026123, -985.1691284, 1039.6723633
2: -156.5181122, 867.5812988, -142.5318909, 790.5067139, -947.0247803, 1010.1131592
3: -381.4162598, 735.6503296, -347.4673157, 670.4584961, -1051.8746338, 1083.1176758
4: -257.7686768, 744.3495483, -234.7127075, 678.4046021, -936.1731567, 979.0622559

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6703606, upper bound: 743.6716841
time: 0.62 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6703606, upper bound: 743.6716841
time: 0.89 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -132.7151947, 691.5526733, -127.0210266, 662.5469360, -795.2621460, 818.5734863
1: -217.0476379, 821.4733276, -207.7908630, 786.8043823, -1003.8520508, 1029.2641602
2: -153.4456329, 850.8350220, -146.7888184, 815.2058716, -968.6514893, 997.6237793
3: -373.9700317, 721.0775757, -357.9414368, 690.1265259, -1064.0965576, 1079.0189209
4: -252.6955872, 729.7541504, -241.6536560, 698.6527100, -951.3482666, 971.4078369

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6698726, upper bound: 743.6698726
time: 0.91 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6698726, upper bound: 743.6698726
time: 0.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.41 seconds
IS_B1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.41
Output dim: 0, lower bound: -743.6703606, upper bound: 743.6716841
IS_B1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.41
Output dim: 0, lower bound: -743.6703606, upper bound: 743.6716841
IS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.41
Output dim: 0, lower bound: -743.6698726, upper bound: 743.6698726
IS_B2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.41
Output dim: 0, lower bound: -743.6698726, upper bound: 743.6698726
Binary search (step 2): status=Status.VERIFIED, low=0.1250000, high=0.2500000, mid=0.1250000, abs_max=860.0533447265625
rel_dist={0: [-743.6887393664056, 743.6887393664056]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6876787, upper bound: 743.6866751
time: 0.67 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6878903, upper bound: 743.6878903
time: 0.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -743.6876787, upper bound: 743.6866751
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -743.6878903, upper bound: 743.6878903

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -133.1433258, 692.8184204, -138.5269012, 721.5264282, -854.6695557, 831.3453369
1: -217.3697815, 823.0866089, -226.4326935, 857.1390381, -1074.5087891, 1049.5192871
2: -153.8224335, 852.0822754, -160.1910706, 887.5496826, -1041.3720703, 1012.2733154
3: -374.4883728, 722.8275757, -390.1859741, 752.6910400, -1127.1794434, 1113.0135498
4: -253.3396606, 731.3729858, -263.8327942, 761.5472412, -1014.8869019, 995.2058105

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6777097, upper bound: 743.6751243
time: 0.62 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6728595, upper bound: 743.6707471
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -136.5654449, 712.0169678, -136.9424896, 713.2861328, -849.8515625, 848.9593506
1: -223.1357117, 845.3634644, -223.8509979, 847.3130493, -1070.4484863, 1069.2143555
2: -157.8641815, 876.0595093, -158.3536224, 877.5255127, -1035.3896484, 1034.4130859
3: -384.4993286, 741.7769165, -385.7337036, 743.9847412, -1128.4837646, 1127.5104980
4: -259.9345093, 750.8963623, -260.8017273, 752.8196411, -1012.7541504, 1011.6979980

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6769371, upper bound: 743.6750273
time: 0.79 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6700627, upper bound: 743.6700627
time: 0.65 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.34 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -743.6777097, upper bound: 743.6751243
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 3.34
Output dim: 0, lower bound: -743.6728595, upper bound: 743.6707471
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -743.6769371, upper bound: 743.6750273
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 3.34
Output dim: 0, lower bound: -743.6700627, upper bound: 743.6700627

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -138.5269012, 721.5264282, -844.9579468, 781.3366699
1: -201.7657318, 763.8026123, -226.4326935, 857.1390381, -1058.9047852, 990.2352905
2: -142.5318909, 790.5067139, -160.1910706, 887.5496826, -1030.0815430, 950.6977539
3: -347.4673157, 670.4584961, -390.1859741, 752.6910400, -1100.1580811, 1060.6445312
4: -234.7127075, 678.4046021, -263.8327942, 761.5472412, -996.2599487, 942.2373657

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6728595, upper bound: 743.6707471
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6728595, upper bound: 743.6707471
time: 0.68 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -127.0210266, 662.5469360, -136.9424896, 713.2861328, -840.3070068, 799.4893188
1: -207.7908630, 786.8043823, -223.8509979, 847.3130493, -1055.1036377, 1010.6553955
2: -146.7888184, 815.2058716, -158.3536224, 877.5255127, -1024.3143311, 973.5595093
3: -357.9414368, 690.1265259, -385.7337036, 743.9847412, -1101.9259033, 1075.8602295
4: -241.6536560, 698.6527100, -260.8017273, 752.8196411, -994.4732056, 959.4544067

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6700627, upper bound: 743.6700627
time: 0.62 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6700627, upper bound: 743.6700627
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.26 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 0, lower bound: -743.6728595, upper bound: 743.6707471
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 0, lower bound: -743.6728595, upper bound: 743.6707471
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 0, lower bound: -743.6700627, upper bound: 743.6700627
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.26
Output dim: 0, lower bound: -743.6700627, upper bound: 743.6700627
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2500000, mid=0.1875000, abs_max=860.0533447265625
rel_dist={0: [-743.6890518099495, 743.6890518099494]}

## Binary search (step 4) starts
Candidate diff: 0.2187500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6878395, upper bound: 743.6867118
time: 1.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6879592, upper bound: 743.6879592
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.95 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.95
Output dim: 0, lower bound: -743.6878395, upper bound: 743.6867118
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.95
Output dim: 0, lower bound: -743.6879592, upper bound: 743.6879592

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -133.1433258, 692.8184204, -138.5269012, 721.5264282, -854.6695557, 831.3453369
1: -217.3697815, 823.0866089, -226.4326935, 857.1390381, -1074.5087891, 1049.5192871
2: -153.8224335, 852.0822754, -160.1910706, 887.5496826, -1041.3720703, 1012.2733154
3: -374.4883728, 722.8275757, -390.1859741, 752.6910400, -1127.1794434, 1113.0135498
4: -253.3396606, 731.3729858, -263.8327942, 761.5472412, -1014.8869019, 995.2058105

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6783528, upper bound: 743.6756378
time: 0.64 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6734067, upper bound: 743.6708963
time: 0.79 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -136.5654449, 712.0169678, -137.3253937, 715.2758179, -851.8411865, 849.3422241
1: -223.1357117, 845.3634644, -224.4746246, 849.6856079, -1072.8212891, 1069.8380127
2: -157.8641815, 876.0595093, -158.7977142, 879.9458008, -1037.8099365, 1034.8571777
3: -384.4993286, 741.7769165, -386.8089600, 746.0875854, -1130.5867920, 1128.5858154
4: -259.9345093, 750.8963623, -261.5342407, 754.9269409, -1014.8614502, 1012.4305420

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775587, upper bound: 743.6755860
time: 0.69 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701419, upper bound: 743.6701419
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.17 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 0, lower bound: -743.6783528, upper bound: 743.6756378
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 3.17
Output dim: 0, lower bound: -743.6734067, upper bound: 743.6708963
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 0, lower bound: -743.6775587, upper bound: 743.6755860
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 3.17
Output dim: 0, lower bound: -743.6701419, upper bound: 743.6701419

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -138.5269012, 721.5264282, -844.9579468, 781.3366699
1: -201.7657318, 763.8026123, -226.4326935, 857.1390381, -1058.9047852, 990.2352905
2: -142.5318909, 790.5067139, -160.1910706, 887.5496826, -1030.0815430, 950.6977539
3: -347.4673157, 670.4584961, -390.1859741, 752.6910400, -1100.1580811, 1060.6445312
4: -234.7127075, 678.4046021, -263.8327942, 761.5472412, -996.2599487, 942.2373657

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6734067, upper bound: 743.6708963
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6734067, upper bound: 743.6708963
time: 0.77 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -127.0210266, 662.5469360, -137.3253937, 715.2758179, -842.2966309, 799.8721924
1: -207.7908630, 786.8043823, -224.4746246, 849.6856079, -1057.4764404, 1011.2788696
2: -146.7888184, 815.2058716, -158.7977142, 879.9458008, -1026.7346191, 974.0036011
3: -357.9414368, 690.1265259, -386.8089600, 746.0875854, -1104.0288086, 1076.9355469
4: -241.6536560, 698.6527100, -261.5342407, 754.9269409, -996.5805664, 960.1869507

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701419, upper bound: 743.6701419
time: 0.80 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701419, upper bound: 743.6701419
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.46 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 0, lower bound: -743.6734067, upper bound: 743.6708963
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 0, lower bound: -743.6734067, upper bound: 743.6708963
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 0, lower bound: -743.6701419, upper bound: 743.6701419
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.46
Output dim: 0, lower bound: -743.6701419, upper bound: 743.6701419
Binary search (step 4): status=Status.VERIFIED, low=0.2187500, high=0.2500000, mid=0.2187500, abs_max=860.0533447265625
rel_dist={0: [-743.6891902221568, 743.6891902221569]}

## Binary search (step 5) starts
Candidate diff: 0.2343750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6867263, upper bound: 743.6879146
time: 0.59 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6879895, upper bound: 743.6879895
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.50 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 0, lower bound: -743.6867263, upper bound: 743.6879146
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 0, lower bound: -743.6879895, upper bound: 743.6879895

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -138.5269012, 721.5264282, -133.1433258, 692.8184204, -831.3453369, 854.6695557
1: -226.4326935, 857.1390381, -217.3697815, 823.0866089, -1049.5192871, 1074.5087891
2: -160.1910706, 887.5496826, -153.8224335, 852.0822754, -1012.2733154, 1041.3720703
3: -390.1859741, 752.6910400, -374.4883728, 722.8275757, -1113.0135498, 1127.1794434
4: -263.8327942, 761.5472412, -253.3396606, 731.3729858, -995.2058105, 1014.8869019

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6758581, upper bound: 743.6786157
time: 0.78 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6709660, upper bound: 743.6736617
time: 0.73 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -137.4890900, 716.1275024, -136.5654449, 712.0169678, -849.5059814, 852.6929321
1: -224.7414246, 850.7014771, -223.1357117, 845.3634644, -1070.1047363, 1073.8370361
2: -158.9876099, 880.9819336, -157.8641815, 876.0595093, -1035.0471191, 1038.8460693
3: -387.2688904, 746.9875488, -384.4993286, 741.7769165, -1129.0457764, 1131.4866943
4: -261.8473816, 755.8291016, -259.9345093, 750.8963623, -1012.7437744, 1015.7636108

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6758500, upper bound: 743.6778177
time: 0.63 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701641, upper bound: 743.6701641
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.49 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -743.6758581, upper bound: 743.6786157
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 3.49
Output dim: 0, lower bound: -743.6709660, upper bound: 743.6736617
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 3.49
Output dim: 0, lower bound: -743.6758500, upper bound: 743.6778177
IS_B2_B2, status: Status.VERIFIED, split count: 2, time: 3.49
Output dim: 0, lower bound: -743.6701641, upper bound: 743.6701641

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -138.5269012, 721.5264282, -123.4315948, 642.8097534, -781.3366699, 844.9579468
1: -226.4326935, 857.1390381, -201.7657318, 763.8026123, -990.2352905, 1058.9047852
2: -160.1910706, 887.5496826, -142.5318909, 790.5067139, -950.6977539, 1030.0815430
3: -390.1859741, 752.6910400, -347.4673157, 670.4584961, -1060.6445312, 1100.1580811
4: -263.8327942, 761.5472412, -234.7127075, 678.4046021, -942.2373657, 996.2599487

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6709660, upper bound: 743.6736617
time: 0.87 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6709660, upper bound: 743.6736617
time: 0.75 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -137.4890900, 716.1275024, -127.0210266, 662.5469360, -800.0359497, 843.1484375
1: -224.7414246, 850.7014771, -207.7908630, 786.8043823, -1011.5457153, 1058.4921875
2: -158.9876099, 880.9819336, -146.7888184, 815.2058716, -974.1934814, 1027.7707520
3: -387.2688904, 746.9875488, -357.9414368, 690.1265259, -1077.3953857, 1104.9289551
4: -261.8473816, 755.8291016, -241.6536560, 698.6527100, -960.5001221, 997.4826050

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701641, upper bound: 743.6701641
time: 0.74 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701641, upper bound: 743.6701641
time: 0.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.73 seconds
IS_B1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.73
Output dim: 0, lower bound: -743.6709660, upper bound: 743.6736617
IS_B1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.73
Output dim: 0, lower bound: -743.6709660, upper bound: 743.6736617
IS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.73
Output dim: 0, lower bound: -743.6701641, upper bound: 743.6701641
IS_B2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.73
Output dim: 0, lower bound: -743.6701641, upper bound: 743.6701641
Binary search (step 5): status=Status.VERIFIED, low=0.2343750, high=0.2500000, mid=0.2343750, abs_max=860.0533447265625
rel_dist={0: [-743.6892492623629, 743.6892492623629]}

## Binary search (step 6) starts
Candidate diff: 0.2421875


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6879498, upper bound: 743.6867335
time: 0.62 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6880046, upper bound: 743.6880046
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -743.6879498, upper bound: 743.6867335
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -743.6880046, upper bound: 743.6880046

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -133.1433258, 692.8184204, -138.5269012, 721.5264282, -854.6695557, 831.3453369
1: -217.3697815, 823.0866089, -226.4326935, 857.1390381, -1074.5087891, 1049.5192871
2: -153.8224335, 852.0822754, -160.1910706, 887.5496826, -1041.3720703, 1012.2733154
3: -374.4883728, 722.8275757, -390.1859741, 752.6910400, -1127.1794434, 1113.0135498
4: -253.3396606, 731.3729858, -263.8327942, 761.5472412, -1014.8869019, 995.2058105

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6787388, upper bound: 743.6759585
time: 0.81 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
time: 0.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -136.5654449, 712.0169678, -137.5650940, 716.5230103, -853.0884399, 849.5820312
1: -223.1357117, 845.3634644, -224.8653259, 851.1728516, -1074.3085938, 1070.2286377
2: -157.8641815, 876.0595093, -159.0757599, 881.4631348, -1039.3272705, 1035.1352539
3: -384.4993286, 741.7769165, -387.4824829, 747.4051514, -1131.9044189, 1129.2593994
4: -259.9345093, 750.8963623, -261.9928284, 756.2476196, -1016.1821289, 1012.8891602

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6779416, upper bound: 743.6759701
time: 0.73 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701745, upper bound: 743.6701745
time: 0.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.28 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -743.6787388, upper bound: 743.6759585
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.28
Output dim: 0, lower bound: -743.6779416, upper bound: 743.6759701
IS_A2_A2, status: Status.VERIFIED, split count: 2, time: 3.28
Output dim: 0, lower bound: -743.6701745, upper bound: 743.6701745

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -138.5269012, 721.5264282, -844.9579468, 781.3366699
1: -201.7657318, 763.8026123, -226.4326935, 857.1390381, -1058.9047852, 990.2352905
2: -142.5318909, 790.5067139, -160.1910706, 887.5496826, -1030.0815430, 950.6977539
3: -347.4673157, 670.4584961, -390.1859741, 752.6910400, -1100.1580811, 1060.6445312
4: -234.7127075, 678.4046021, -263.8327942, 761.5472412, -996.2599487, 942.2373657

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
time: 0.61 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -134.4245758, 702.3953247, -946.0042725, 1436.8714600
1: -401.4721985, 1546.7423096, -219.7473145, 834.1121216, -1235.5842285, 1766.4896240
2: -282.7309570, 1598.5117188, -155.5340271, 863.9214478, -1146.6523438, 1754.0456543
3: -693.1887207, 1357.5162354, -378.8896179, 731.7599487, -1424.9486084, 1736.4058838
4: -466.2039795, 1371.2233887, -256.1381836, 740.8732910, -1207.0772705, 1627.3615723

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
time: 0.66 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -127.0210266, 662.5469360, -137.5650940, 716.5230103, -843.5438843, 800.1119995
1: -207.7908630, 786.8043823, -224.8653259, 851.1728516, -1058.9637451, 1011.6696167
2: -146.7888184, 815.2058716, -159.0757599, 881.4631348, -1028.2519531, 974.2816162
3: -357.9414368, 690.1265259, -387.4824829, 747.4051514, -1105.3465576, 1077.6090088
4: -241.6536560, 698.6527100, -261.9928284, 756.2476196, -997.9012451, 960.6455078

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701745, upper bound: 743.6701745
time: 0.65 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6701745, upper bound: 743.6701745
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.43 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -743.6737892, upper bound: 743.6710002
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.43
Output dim: 0, lower bound: -743.6701745, upper bound: 743.6701745
IS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.43
Output dim: 0, lower bound: -743.6701745, upper bound: 743.6701745

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -128.5790863, 670.3025513, -793.7341309, 771.3887939
1: -201.7657318, 763.8026123, -210.4309387, 796.4016113, -998.1673584, 974.2334595
2: -142.5318909, 790.5067139, -148.6245575, 824.4069214, -966.9387817, 939.1312256
3: -347.4673157, 670.4584961, -362.4870300, 699.0101929, -1046.4772949, 1032.9454346
4: -234.7127075, 678.4046021, -244.7495728, 707.2666626, -941.9793701, 923.1541748

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6733318, upper bound: 743.6725854
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782746, upper bound: 743.6747904
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -250.0492859, 1336.8829346, -1460.3143311, 892.8590088
1: -201.7657318, 763.8026123, -412.1385498, 1587.6175537, -1789.3831787, 1175.9411621
2: -142.5318909, 790.5067139, -290.2971802, 1640.8114014, -1783.3432617, 1080.8039551
3: -347.4673157, 670.4584961, -711.7263184, 1393.3811035, -1740.8480225, 1382.1848145
4: -234.7127075, 678.4046021, -478.7217712, 1407.3099365, -1642.0224609, 1157.1263428

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776998, upper bound: 743.6739186
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782746, upper bound: 743.6747904
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -243.5312500, 1302.0620117, -128.5790863, 670.3025513, -913.8338013, 1430.6411133
1: -401.3452759, 1546.2860107, -210.4309387, 796.4016113, -1197.7467041, 1756.7169189
2: -282.6410522, 1598.0362549, -148.6245575, 824.4069214, -1107.0479736, 1746.6607666
3: -692.9695435, 1357.1099854, -362.4870300, 699.0101929, -1391.9797363, 1719.5969238
4: -466.0551453, 1370.8105469, -244.7495728, 707.2666626, -1173.3217773, 1615.5600586

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6641491, upper bound: 743.6618919
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6619076, upper bound: 743.6592623
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -243.6089783, 1302.4470215, -250.1162415, 1337.2148438, -1580.8238525, 1552.5627441
1: -401.4721985, 1546.7423096, -412.2479858, 1588.0107422, -1989.4829102, 1958.9902344
2: -282.7309570, 1598.5117188, -290.3746338, 1641.2213135, -1923.9520264, 1888.8861084
3: -693.1887207, 1357.5162354, -711.9147949, 1393.7313232, -2086.9194336, 2069.4311523
4: -466.2039795, 1371.2233887, -478.8499146, 1407.6657715, -1873.8697510, 1850.0732422

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6709720, upper bound: 743.6683214
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6619076, upper bound: 743.6592620
time: 0.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.54 seconds
IS_A1_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.54
Output dim: 0, lower bound: -743.6733318, upper bound: 743.6725854
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 0, lower bound: -743.6782746, upper bound: 743.6747904
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 0, lower bound: -743.6776998, upper bound: 743.6739186
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.54
Output dim: 0, lower bound: -743.6782746, upper bound: 743.6747904
IS_A1_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 3.54
Output dim: 0, lower bound: -743.6641491, upper bound: 743.6618919
IS_A1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 3.54
Output dim: 0, lower bound: -743.6619076, upper bound: 743.6592623
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 3.54
Output dim: 0, lower bound: -743.6709720, upper bound: 743.6683214
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 3.54
Output dim: 0, lower bound: -743.6619076, upper bound: 743.6592620

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -127.6417999, 665.4821167, -782.8687134, 739.6878052
1: -191.7628937, 727.0043945, -208.8830872, 790.6538086, -982.4166870, 935.8874512
2: -135.5527344, 752.8033447, -147.5404510, 818.4945679, -954.0473022, 900.3438110
3: -330.2159424, 637.6373901, -359.8313293, 693.9221191, -1024.1380615, 997.4686279
4: -223.1825562, 645.3632812, -242.9667358, 702.1410522, -925.3236084, 888.3299561

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6853730, upper bound: 743.6830629
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6831056, upper bound: 743.6825294
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -123.4315948, 642.8097534, -244.3851166, 1307.7706299, -1431.2020264, 887.1947021
1: -201.7657318, 763.8026123, -402.7041626, 1552.9182129, -1754.6838379, 1166.5068359
2: -142.5318909, 790.5067139, -283.6715088, 1604.8922119, -1747.4240723, 1074.1782227
3: -347.4673157, 670.4584961, -695.5899048, 1362.5427246, -1710.0096436, 1366.0482178
4: -234.7127075, 678.4046021, -467.8361206, 1376.2072754, -1610.9199219, 1146.2407227

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6732081, upper bound: 743.6717137
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6732081, upper bound: 743.6739186
time: 0.88 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -122.4425888, 637.7197876, -245.4833221, 1312.9678955, -1435.4105225, 883.2031250
1: -200.1366119, 757.7338867, -404.5194397, 1559.1007080, -1759.2371826, 1162.2532959
2: -141.3862457, 784.2528687, -284.9915466, 1611.4058838, -1752.7918701, 1069.2442627
3: -344.6659851, 665.0797729, -698.5712280, 1368.0948486, -1712.7608643, 1363.6510010
4: -232.8262634, 672.9862061, -469.9553223, 1381.8250732, -1614.6511230, 1142.9414062

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6748901, upper bound: 743.6715209
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6537800, upper bound: 743.6499360
time: 0.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.94 seconds
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.94
Output dim: 0, lower bound: -743.6853730, upper bound: 743.6830629
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.94
Output dim: 0, lower bound: -743.6831056, upper bound: 743.6825294
IS_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.94
Output dim: 0, lower bound: -743.6732081, upper bound: 743.6717137
IS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.94
Output dim: 0, lower bound: -743.6732081, upper bound: 743.6739186
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.94
Output dim: 0, lower bound: -743.6748901, upper bound: 743.6715209
IS_A1_A1_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.94
Output dim: 0, lower bound: -743.6537800, upper bound: 743.6499360

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -116.8128052, 608.9627686, -122.1394730, 635.9201660, -752.7329712, 731.1021118
1: -190.8185425, 723.3307495, -199.8387604, 755.4221802, -946.2406006, 923.1694946
2: -134.8858795, 749.0592041, -141.1411285, 782.5539551, -917.4398193, 890.2003174
3: -328.5925293, 634.3862305, -344.2393799, 662.7210083, -991.3134766, 978.6254883
4: -222.0782623, 642.1071777, -232.3652649, 670.8918457, -892.9700317, 874.4724121

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830657, upper bound: 743.6824442
time: 0.81 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6830657, upper bound: 743.6824442
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.8150101, 603.6513672, -127.2728500, 663.7396240, -779.5545654, 730.9241943
1: -189.1556396, 717.0230103, -208.1260071, 788.3509521, -977.5065918, 925.1489868
2: -133.7119904, 742.4448853, -147.0796967, 816.4520874, -950.1639404, 889.5245972
3: -325.7151794, 628.8167114, -358.7938538, 691.4874878, -1017.2026367, 987.6105957
4: -220.1329651, 636.4825439, -242.2182617, 699.9373169, -920.0702515, 878.7008057

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6794628, upper bound: 743.6787431
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6642629, upper bound: 743.6629594
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -117.3866196, 612.0460205, -244.3458405, 1307.5758057, -1424.9624023, 856.3917236
1: -191.7628937, 727.0043945, -402.6399231, 1552.6870117, -1744.4497070, 1129.6442871
2: -135.5527344, 752.8033447, -283.6260376, 1604.6513672, -1740.2041016, 1036.4294434
3: -330.2159424, 637.6373901, -695.4790039, 1362.3367920, -1692.5527344, 1333.1163330
4: -223.1825562, 645.3632812, -467.7608643, 1375.9982910, -1599.1805420, 1113.1241455

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6477394, upper bound: 743.6477681
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6434438, upper bound: 743.6427656
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6726764, upper bound: 743.6706251
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6660901, upper bound: 743.6632992
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -122.0635147, 635.7655640, -247.4381409, 1322.8205566, -1444.8840332, 883.2036743
1: -199.5101929, 755.4051514, -407.5706482, 1570.8631592, -1770.3731689, 1162.9757080
2: -140.9466858, 781.8518677, -287.1875916, 1623.4688721, -1764.4155273, 1069.0393066
3: -343.5862427, 663.0192871, -703.8182983, 1378.5524902, -1722.1386719, 1366.8374023
4: -232.1022034, 670.9118042, -473.5455627, 1392.3123779, -1624.4145508, 1144.4573975

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6538064, upper bound: 743.6495487
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6742674, upper bound: 743.6707369
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6732090, upper bound: 743.6697493
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.08 seconds
IS_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -743.6830657, upper bound: 743.6824442
IS_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -743.6830657, upper bound: 743.6824442
IS_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -743.6794628, upper bound: 743.6787431
IS_A1_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 0, lower bound: -743.6642629, upper bound: 743.6629594
IS_A1_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 0, lower bound: -743.6726764, upper bound: 743.6706251
IS_A1_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 0, lower bound: -743.6660901, upper bound: 743.6632992
IS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.08
Output dim: 0, lower bound: -743.6742674, upper bound: 743.6707369
IS_A1_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 5.08
Output dim: 0, lower bound: -743.6732090, upper bound: 743.6697493

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -111.9797897, 583.0902100, -122.1394730, 635.9201660, -747.8999023, 705.2296143
1: -182.8664093, 692.4880371, -199.8387604, 755.4221802, -938.2885132, 892.3267822
2: -129.2702026, 717.6174927, -141.1411285, 782.5539551, -911.8241577, 858.7586060
3: -314.9225159, 607.0635376, -344.2393799, 662.7210083, -977.6435547, 951.3027954
4: -212.7743530, 614.7598267, -232.3652649, 670.8918457, -883.6661377, 847.1251221

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6831591, upper bound: 743.6799735
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6832947, upper bound: 743.6808876
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -116.7978821, 609.1127319, -122.1394730, 635.9201660, -752.7180176, 731.2521362
1: -190.6253662, 723.2798462, -199.8387604, 755.4221802, -946.0475464, 923.1185303
2: -134.8507233, 749.3662109, -141.1411285, 782.5539551, -917.4046631, 890.5073242
3: -328.5602722, 633.9815674, -344.2393799, 662.7210083, -991.2812500, 978.2208252
4: -222.0314026, 641.9342041, -232.3652649, 670.8918457, -892.9232178, 874.2994385

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6807722, upper bound: 743.6786995
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6831591, upper bound: 743.6799735
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6832947, upper bound: 743.6808876
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -115.4984360, 602.0151367, -126.7955475, 661.8461304, -777.3445435, 728.8106689
1: -188.6360931, 715.0719604, -207.1258087, 786.0474243, -974.6835327, 922.1977539
2: -133.3455505, 740.4376831, -146.4456024, 813.9434814, -947.2890015, 886.8832397
3: -324.8172607, 627.0895386, -357.0112610, 689.2674561, -1014.0847168, 984.1007690
4: -219.5283813, 634.7471924, -241.1581573, 697.6276245, -917.1560059, 875.9053345

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6760185, upper bound: 743.6735944
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6792686, upper bound: 743.6787431
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6792686, upper bound: 743.6787431
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -111.8238907, 581.7180786, -246.5728760, 1318.2965088, -1430.1203613, 828.2909546
1: -182.7307129, 691.0246582, -406.1783142, 1565.4688721, -1748.1993408, 1097.2030029
2: -129.0286865, 715.8170166, -286.1877747, 1617.9263916, -1746.9549561, 1002.0047607
3: -314.7500000, 606.2383423, -701.4026489, 1373.7651367, -1688.5151367, 1307.6409912
4: -212.3892975, 613.8373413, -471.8877869, 1387.5156250, -1599.9047852, 1085.7250977

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6718686, upper bound: 743.6705552
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6718686, upper bound: 743.6707369
time: 0.72 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.71 seconds
IS_A1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.71
Output dim: 0, lower bound: -743.6831591, upper bound: 743.6799735
IS_A1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 5.71
Output dim: 0, lower bound: -743.6832947, upper bound: 743.6808876
IS_A1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.71
Output dim: 0, lower bound: -743.6831591, upper bound: 743.6799735
IS_A1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 5.71
Output dim: 0, lower bound: -743.6832947, upper bound: 743.6808876
IS_A1_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.71
Output dim: 0, lower bound: -743.6792686, upper bound: 743.6787431
IS_A1_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.71
Output dim: 0, lower bound: -743.6792686, upper bound: 743.6787431
IS_A1_A1_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.71
Output dim: 0, lower bound: -743.6718686, upper bound: 743.6705552
IS_A1_A1_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.71
Output dim: 0, lower bound: -743.6718686, upper bound: 743.6707369

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -111.5924454, 581.0575562, -114.6612320, 596.6928101, -708.2852783, 695.7188110
1: -182.2236176, 690.0562744, -187.4586334, 708.3845215, -890.6080322, 877.5148926
2: -128.8165131, 715.1417236, -132.3753967, 734.6206055, -863.4371338, 847.5170898
3: -313.8192139, 604.8895874, -322.9160156, 620.6577759, -934.4768677, 927.8055420
4: -212.0219421, 612.5996704, -217.8221283, 629.0731201, -841.0949097, 830.4218140

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754414, upper bound: 743.6744919
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754414, upper bound: 743.6823468
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -110.7889786, 576.7890015, -119.5827637, 623.7321167, -734.5209351, 696.3717651
1: -180.9069672, 684.9938354, -195.5204620, 740.8771362, -921.7839966, 880.5142822
2: -127.8769302, 709.8546143, -138.1561279, 767.3026733, -895.1796265, 848.0107422
3: -311.5156250, 600.4407349, -336.7543640, 649.5301514, -961.0457764, 937.1950073
4: -210.4573975, 608.0972900, -227.4076691, 657.3809814, -867.8383789, 835.5048828

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6601177, upper bound: 743.6574862
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6601177, upper bound: 743.6823799
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -116.4178009, 607.0886230, -114.6612320, 596.6928101, -713.1105957, 721.7498169
1: -189.9922943, 720.8619385, -187.4586334, 708.3845215, -898.3767090, 908.3205566
2: -134.4048615, 746.9046631, -132.3753967, 734.6206055, -869.0253296, 879.2800293
3: -327.4729004, 631.8270874, -322.9160156, 620.6577759, -948.1306152, 954.7430420
4: -221.2924042, 639.7895508, -217.8221283, 629.0731201, -850.3655396, 857.6116943

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6736542, upper bound: 743.6697205
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6533545, upper bound: 743.6497993
time: 0.96 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6533545, upper bound: 743.6799735
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -115.5308151, 602.5333252, -119.5827637, 623.7321167, -739.2628784, 722.1160889
1: -188.5565491, 715.4406738, -195.5204620, 740.8771362, -929.4337158, 910.9611206
2: -133.3775024, 741.2457275, -138.1561279, 767.3026733, -900.6801758, 879.4018555
3: -324.9662170, 627.0258789, -336.7543640, 649.5301514, -974.4963379, 963.7800903
4: -219.5795135, 634.9567871, -227.4076691, 657.3809814, -876.9605103, 862.3643799

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6780710, upper bound: 743.6758442
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6525933, upper bound: 743.6495160
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6525934, upper bound: 743.6495160
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -111.6657639, 581.4688721, -126.7955475, 661.8461304, -773.5119019, 708.2644043
1: -182.3509979, 690.5552368, -207.1258087, 786.0474243, -968.3983154, 897.6810303
2: -128.9067841, 715.6286011, -146.4456024, 813.9434814, -942.8502808, 862.0741577
3: -314.0319519, 605.3522339, -357.0112610, 689.2674561, -1003.2993774, 962.3634644
4: -212.1749573, 613.0396118, -241.1581573, 697.6276245, -909.8025513, 854.1977539

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6727448, upper bound: 743.6722959
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785785, upper bound: 743.6774280
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775227, upper bound: 743.6774379
time: 0.83 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -116.4623489, 607.3715820, -126.7955475, 661.8461304, -778.3084106, 734.1671143
1: -190.0752563, 721.2054443, -207.1258087, 786.0474243, -976.1225586, 928.3311768
2: -134.4624481, 747.2290039, -146.4456024, 813.9434814, -948.4058838, 893.6744995
3: -327.6097107, 632.1482544, -357.0112610, 689.2674561, -1016.8771362, 989.1594849
4: -221.3911133, 640.0907593, -241.1581573, 697.6276245, -919.0187378, 881.2489014

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6727448, upper bound: 743.6735944
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785785, upper bound: 743.6774280
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6775227, upper bound: 743.6774244
time: 0.75 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 7.32 seconds
IS_A1_A1_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6754414, upper bound: 743.6744919
IS_A1_A1_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6754414, upper bound: 743.6823468
IS_A1_A1_B1_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6601177, upper bound: 743.6574862
IS_A1_A1_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6601177, upper bound: 743.6823799
IS_A1_A1_B1_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6533545, upper bound: 743.6497993
IS_A1_A1_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6533545, upper bound: 743.6799735
IS_A1_A1_B1_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6525933, upper bound: 743.6495160
IS_A1_A1_B1_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6525934, upper bound: 743.6495160
IS_A1_A1_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6785785, upper bound: 743.6774280
IS_A1_A1_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6775227, upper bound: 743.6774379
IS_A1_A1_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6785785, upper bound: 743.6774280
IS_A1_A1_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 7.32
Output dim: 0, lower bound: -743.6775227, upper bound: 743.6774244

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -111.5924454, 581.0575562, -111.1765747, 579.0368042, -690.6291504, 692.2341309
1: -182.2236176, 690.0562744, -181.6289368, 687.3578491, -869.5813599, 871.6851807
2: -128.8165131, 715.1417236, -128.2874298, 712.7778931, -841.5944214, 843.4291382
3: -313.8192139, 604.8895874, -312.9619446, 602.0298462, -915.8489380, 917.8515625
4: -212.0219421, 612.5996704, -211.1488190, 610.1497192, -822.1715088, 823.7484741

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6714559, upper bound: 743.6707463
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695253, upper bound: 743.6686826
time: 0.71 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -111.5924454, 581.0575562, -109.5474243, 570.9955444, -682.5878906, 690.6049805
1: -182.2236176, 690.0562744, -178.9551697, 677.4798584, -859.7034302, 869.0114136
2: -128.8165131, 715.1417236, -126.4774704, 703.0489502, -831.8654175, 841.6192017
3: -313.8192139, 604.8895874, -308.3374634, 592.9629517, -906.7820435, 913.2270508
4: -212.0219421, 612.5996704, -208.0803070, 601.2585449, -813.2803345, 820.6798706

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6726657, upper bound: 743.6724628
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6695253, upper bound: 743.6688701
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -110.7889786, 576.7890015, -112.8554459, 589.6697388, -700.4587402, 689.6444092
1: -180.9069672, 684.9938354, -184.3795624, 700.0742188, -880.9811401, 869.3733521
2: -127.8769302, 709.8546143, -130.3926392, 725.5932617, -853.4701538, 840.2472534
3: -311.5156250, 600.4407349, -317.5811768, 613.1365356, -924.6521606, 918.0219116
4: -210.4573975, 608.0972900, -214.5669556, 620.8038940, -831.2612915, 822.6642456

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6597007, upper bound: 743.6710858
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6581292, upper bound: 743.6676712
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -116.4178009, 607.0886230, -109.5474243, 570.9955444, -687.4133301, 716.6360474
1: -189.9922943, 720.8619385, -178.9551697, 677.4798584, -867.4721680, 899.8170776
2: -134.4048615, 746.9046631, -126.4774704, 703.0489502, -837.4536133, 873.3821411
3: -327.4729004, 631.8270874, -308.3374634, 592.9629517, -920.4357910, 940.1645508
4: -221.2924042, 639.7895508, -208.0803070, 601.2585449, -822.5509033, 847.8697510

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6428443, upper bound: 743.6697205
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6525772, upper bound: 743.6625144
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6522460, upper bound: 743.6606325
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -100.7262039, 524.3803101, -125.8797607, 657.0904541, -757.8166504, 650.2600098
1: -164.3929749, 622.2501221, -205.6469727, 780.3735352, -944.7664795, 827.8970947
2: -116.1745453, 645.5622559, -145.3891296, 808.1119995, -924.2865601, 790.9514160
3: -283.2300415, 544.9536743, -354.4484558, 684.2311401, -967.4611816, 899.4020996
4: -191.1242371, 552.3251953, -239.4067993, 692.5831299, -883.7073975, 791.7319336

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6745312, upper bound: 743.6778862
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6757320, upper bound: 743.6771422
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -106.7356644, 555.8107910, -126.3301849, 659.4385376, -766.1741943, 682.1409302
1: -174.2735596, 659.8494873, -206.3672485, 783.1729736, -957.4465332, 866.2167358
2: -123.2007141, 684.2098389, -145.9073792, 810.9898682, -934.1904907, 830.1171875
3: -300.0690308, 578.1328735, -355.6946106, 686.7131348, -986.7821045, 933.8272705
4: -202.7189484, 585.7442017, -240.2668457, 695.0643311, -897.7832642, 826.0110474

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -743.6720322, upper bound: 743.6720904
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770950, upper bound: 743.6796228
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6770950, upper bound: 743.6796228
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -105.5406189, 550.1207275, -125.8797607, 657.0904541, -762.6309204, 676.0004272
1: -172.1035156, 652.7503052, -205.6469727, 780.3735352, -952.4769897, 858.3972168
2: -121.7444382, 677.1962280, -145.3891296, 808.1119995, -929.8564453, 822.5853271
3: -296.8014221, 571.7589111, -354.4484558, 684.2311401, -981.0325317, 926.2072754
4: -200.3808746, 579.3500366, -239.4067993, 692.5831299, -892.9639893, 818.7568359

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6746398, upper bound: 743.6732669
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6782494, upper bound: 743.6768641
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774229
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774229
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -111.6996460, 582.4542847, -126.3301849, 659.4385376, -771.1381226, 708.7844238
1: -182.2803040, 691.4609375, -206.3672485, 783.1729736, -965.4532471, 897.8281860
2: -128.9546509, 716.8013306, -145.9073792, 810.9898682, -939.9445190, 862.7087402
3: -314.1210632, 605.8084106, -355.6946106, 686.7131348, -1000.8342285, 961.5028687
4: -212.2667847, 613.6668091, -240.2668457, 695.0643311, -907.3311157, 853.9336548

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751467, upper bound: 743.6732652
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774244
time: 0.68 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774244
time: 0.74 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.26 seconds
IS_A1_A1_B1_A2_B1_A1_B1_B1_B1, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6714559, upper bound: 743.6707463
IS_A1_A1_B1_A2_B1_A1_B1_B1_B2, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6695253, upper bound: 743.6686826
IS_A1_A1_B1_A2_B1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6726657, upper bound: 743.6724628
IS_A1_A1_B1_A2_B1_A1_B1_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6695253, upper bound: 743.6688701
IS_A1_A1_B1_A2_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6597007, upper bound: 743.6710858
IS_A1_A1_B1_A2_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6581292, upper bound: 743.6676712
IS_A1_A1_B1_A2_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6525772, upper bound: 743.6625144
IS_A1_A1_B1_A2_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6522460, upper bound: 743.6606325
IS_A1_A1_B1_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6745312, upper bound: 743.6778862
IS_A1_A1_B1_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6757320, upper bound: 743.6771422
IS_A1_A1_B1_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6770950, upper bound: 743.6796228
IS_A1_A1_B1_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6770950, upper bound: 743.6796228
IS_A1_A1_B1_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774229
IS_A1_A1_B1_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774229
IS_A1_A1_B1_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774244
IS_A1_A1_B1_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.26
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774244

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -93.8628922, 488.1547546, -125.5025864, 655.0808716, -748.9437256, 613.6573486
1: -153.1079407, 579.0154419, -205.0225983, 777.9741821, -931.0821533, 784.0380249
2: -108.1532745, 601.3243408, -144.9467010, 805.6533203, -913.8065796, 746.2709961
3: -263.6852722, 506.2133179, -353.3683472, 682.0922241, -945.7772827, 859.5816650
4: -177.7759705, 513.9996338, -238.6735535, 690.4520264, -868.2280273, 752.6732178

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743203, upper bound: 743.6778035
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6743203, upper bound: 743.6778862
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -96.0507736, 501.1269836, -124.5684967, 650.2567139, -746.3073730, 625.6954346
1: -156.6433258, 594.4146118, -203.5178375, 772.2363281, -928.8795166, 797.9323120
2: -110.7420120, 616.8745117, -143.8647156, 799.7136230, -910.4556274, 760.7392578
3: -269.7396545, 520.0399780, -350.7276611, 677.0208130, -946.7604980, 870.7676392
4: -182.0628052, 527.1473389, -236.8680420, 685.3407593, -867.4034424, 764.0152588

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6751566, upper bound: 743.6741777
time: 0.94 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754895, upper bound: 743.6770861
time: 0.92 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6754895, upper bound: 743.6771422
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -106.7356644, 555.8107910, -116.6797867, 608.5092163, -715.2448730, 672.4904785
1: -174.2735596, 659.8494873, -190.5769196, 722.4651489, -896.7387085, 850.4263916
2: -123.2007141, 684.2098389, -134.6811676, 748.8215332, -872.0221558, 818.8909912
3: -300.0690308, 578.1328735, -328.5784607, 633.1690674, -933.2380371, 906.7113037
4: -202.7189484, 585.7442017, -221.6945496, 641.2965088, -844.0154419, 807.4387207

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6729967, upper bound: 743.6769177
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741888, upper bound: 743.6768975
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -106.7356644, 555.8107910, -121.5424271, 634.6602783, -741.3959351, 677.3532104
1: -174.2735596, 659.8494873, -198.5716553, 753.5856323, -927.8591919, 858.4211426
2: -123.2007141, 684.2098389, -140.3746643, 780.6516113, -903.8521729, 824.5844727
3: -300.0690308, 578.1328735, -342.1661987, 660.4343262, -960.5032959, 920.2990723
4: -202.7189484, 585.7442017, -231.0986023, 668.7011719, -871.4201050, 816.8427734

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6729967, upper bound: 743.6769177
time: 0.71 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6741888, upper bound: 743.6768975
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -105.5406189, 550.1207275, -116.6797867, 608.5092163, -714.0496826, 666.8004761
1: -172.1035156, 652.7503052, -190.5769196, 722.4651489, -894.5686646, 843.3271484
2: -121.7444382, 677.1962280, -134.6811676, 748.8215332, -870.5659790, 811.8773804
3: -296.8014221, 571.7589111, -328.5784607, 633.1690674, -929.9704590, 900.3374023
4: -200.3808746, 579.3500366, -221.6945496, 641.2965088, -841.6773682, 801.0445557

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6779845, upper bound: 743.6766220
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785278, upper bound: 743.6774280
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785278, upper bound: 743.6774280
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -105.5406189, 550.1207275, -121.5424271, 634.6602783, -740.2007446, 671.6631470
1: -172.1035156, 652.7503052, -198.5716553, 753.5856323, -925.6891479, 851.3219604
2: -121.7444382, 677.1962280, -140.3746643, 780.6516113, -902.3959961, 817.5709229
3: -296.8014221, 571.7589111, -342.1661987, 660.4343262, -957.2357178, 913.9251099
4: -200.3808746, 579.3500366, -231.0986023, 668.7011719, -869.0820312, 810.4486084

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6779845, upper bound: 743.6768641
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785278, upper bound: 743.6774280
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -743.6785278, upper bound: 743.6774280
time: 0.68 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 9.23 seconds
IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6743203, upper bound: 743.6778035
IS_A1_A1_B1_A2_B2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6743203, upper bound: 743.6778862
IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6754895, upper bound: 743.6770861
IS_A1_A1_B1_A2_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6754895, upper bound: 743.6771422
IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6729967, upper bound: 743.6769177
IS_A1_A1_B1_A2_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6741888, upper bound: 743.6768975
IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6729967, upper bound: 743.6769177
IS_A1_A1_B1_A2_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6741888, upper bound: 743.6768975
IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6785278, upper bound: 743.6774280
IS_A1_A1_B1_A2_B2_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6785278, upper bound: 743.6774280
IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6785278, upper bound: 743.6774280
IS_A1_A1_B1_A2_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 9.23
Output dim: 0, lower bound: -743.6785278, upper bound: 743.6774280
IS_A1_A1_B1_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 9.23
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774244
IS_A1_A1_B1_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 9.23
Output dim: 0, lower bound: -743.6776450, upper bound: 743.6774244
Binary search (step 6): status=Status.UNKNOWN, low=0.2343750, high=0.2421875, mid=0.2421875, abs_max=860.0533447265625
rel_dist={0: [-743.6892781222502, 743.6892781222502]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.234375
execution time: 1119.41 seconds
